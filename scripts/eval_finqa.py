import os
import torch
import re
import json
import logging
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoModel,
    Gemma3ForConditionalGeneration
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read the Hugging Face token from file (Ensure this file is manually created and not committed!)
with open("token.txt", "r") as token_file:
    HF_TOKEN = token_file.read().strip()

# Output directories
OUTPUT_DIRECTORY = "../Assets/output/FinQA"
QWEN2VL_DIR = os.path.join(OUTPUT_DIRECTORY, "Qwen2VL")
QWEN2_5VL_DIR = os.path.join(OUTPUT_DIRECTORY, "Qwen2_5VL")
GEMMA_DIR = os.path.join(OUTPUT_DIRECTORY, "gemma-3-4b-it")
GEMMA_DIR_12B = os.path.join(OUTPUT_DIRECTORY, "gemma-3-12b-it")

os.makedirs(QWEN2VL_DIR, exist_ok=True)
os.makedirs(QWEN2_5VL_DIR, exist_ok=True)


def generate_prompt(context: str, question: str) -> str:
    """
    Constructs the prompt in FinQA format using JSON output format.
    """
    return (
        f"Context:\n{context}\n\n"
        f"Given the context, {question} Provide your response in JSON format:\n"
        "{\n"
        '  "explanation": "<model-generated explanation>",\n'
        '  "formatted_answer": "<model-generated float number to two decimal places>"\n'
        "}\n"
    )


def generate_answer(context: str, question: str, processor_name: str, model_name: str, max_pixels: int = 512 * 28 * 28) -> (str, str):
    """
    Generates an answer using the specified model (Qwen2-VL, Qwen2.5-VL, or Gemma).
    """
    prompt = generate_prompt(context, question)
    
    try:
        if "gemma" in model_name.lower():
            processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            model = Gemma3ForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, use_auth_token=HF_TOKEN).eval()
            torch.cuda.empty_cache()
            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)[0][input_len:]
            output_text = processor.decode(generation, skip_special_tokens=True)
        
        elif "Qwen2.5" in model_name or "Qwen2-" in model_name:
            processor = AutoProcessor.from_pretrained(processor_name, max_pixels=max_pixels, trust_remote_code=True)
            messages = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
            model_class = Qwen2_5_VLForConditionalGeneration if "Qwen2.5" in model_name else Qwen2VLForConditionalGeneration
            model = model_class.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda().eval()
            torch.cuda.empty_cache()
            generated_ids = model.generate(**inputs, max_new_tokens=200)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True, use_flash_attn=True).cuda().eval()
            response = model.chat(tokenizer, None, prompt, {"max_new_tokens": 512, "do_sample": False})
            output_text = response
        
        return prompt, output_text
    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return prompt, ""


def extract_formatted_answer(generated_output: str) -> (str, str):
    """
    Extracts the explanation and formatted answer from the model's JSON response.
    """
    try:
        json_output = json.loads(generated_output)
        explanation = json_output.get("explanation", "No explanation provided.")
        formatted_answer = json_output.get("formatted_answer", "No answer provided.")
    except json.JSONDecodeError:
        explanation = "Could not parse explanation."
        formatted_answer = "Could not parse answer."
    
    return explanation, formatted_answer


def evaluate_finqa(num_samples: int = 50):
    """
    Evaluates different models on a subset of the FinQA dataset.
    """
    dataset = load_dataset("Aiera/finqa-verified", split="test")
    results = []
    configurations = [
        ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", QWEN2_5VL_DIR),
        ("Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-2B-Instruct", QWEN2VL_DIR),
        ("google/gemma-3-4b-it", "google/gemma-3-4b-it", GEMMA_DIR),
    ]
    
    for processor_name, model_name, output_dir in configurations:
        correct = 0
        for i, sample in enumerate(dataset.select(range(num_samples))):
            prompt, generated_output = generate_answer(sample["context"], sample["question"], processor_name, model_name)
            explanation, generated_answer = extract_formatted_answer(generated_output)
            target_answer = f"{float(sample["answer"]):.2f}" if isinstance(sample["answer"], float) else str(sample["answer"]).strip()
            if generated_answer == target_answer:
                correct += 1
        accuracy = correct / num_samples
        results.append({"processor": processor_name, "model": model_name, "accuracy": accuracy})
        logger.info("Accuracy for %s: %.2f%%", processor_name, accuracy * 100)
    
    with open(os.path.join(OUTPUT_DIRECTORY, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Evaluation results saved.")


if __name__ == "__main__":
    evaluate_finqa(num_samples=50)