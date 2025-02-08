import os
import torch
import re
import logging
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directories
OUTPUT_DIRECTORY = "./output/FinQA"
QWEN2VL = os.path.join(OUTPUT_DIRECTORY, "Qwen2VL")
QWEN2_5VL = os.path.join(OUTPUT_DIRECTORY, "Qwen2_5VL")
os.makedirs(QWEN2VL, exist_ok=True)
os.makedirs(QWEN2_5VL, exist_ok=True)

def generate_prompt(context: str, question: str) -> str:
    """
    Constructs a prompt from the context and question according to the FinQA format.
    """
    return (
        f"Context:\n{context}\n\n"
        f"Given the context, {question} Report your answer using the following format:\n"
        "Explanation: Explanation of calculation\n"
        "Formatted answer: Float number to two decimal point precision and no units\n"
    )

def generate_answer_from_finqa_sample(context: str, question: str, processor_name: str, model_name: str, max_pixels: int = 512 * 28 * 28) -> (str, str):
    """
    Generates an answer from a FinQA sample using the specified processor and model.
    """
    prompt = generate_prompt(context, question)
    
    try:
        # Load the processor and prepare the prompt
        gen_processor = AutoProcessor.from_pretrained(processor_name, max_pixels=max_pixels)
        messages = [{"role": "user", "content": prompt}]
        text = gen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = gen_processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        
        # Load the generation model
        if "Qwen2.5" in model_name:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16
            ).cuda().eval()
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16
            ).cuda().eval()
        
        torch.cuda.empty_cache()
        generated_ids = model.generate(**inputs, max_new_tokens=200)
        output_text = gen_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return prompt, output_text
    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return prompt, ""

def format_target_answer(target_raw) -> str:
    """
    Ensures the target answer is formatted as a string with 2 decimal places if applicable.
    """
    if isinstance(target_raw, float):
        return f"{target_raw:.2f}"
    return str(target_raw).strip()

def extract_formatted_answer(generated_output: str) -> str:
    """
    Extracts the part after "Formatted answer:" if there is a detailed prompt in the response.
    """
    if "formatted answer:" in generated_output.lower():
        parts = re.split(r"formatted answer:", generated_output, flags=re.IGNORECASE)
        return parts[-1].strip()
    return generated_output.strip()

def save_sample_results(output_dir, sample_id, prompt, target_answer, generated_answer):
    """
    Saves the prompt, target answer, and generated answer for a sample in the specified directory.
    """
    sample_dir = os.path.join(output_dir, f"sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)
    
    with open(os.path.join(sample_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    with open(os.path.join(sample_dir, "target_answer.txt"), "w") as f:
        f.write(target_answer)
    with open(os.path.join(sample_dir, "generated_answer.txt"), "w") as f:
        f.write(generated_answer)

def evaluate_finqa(num_samples: int = 15) -> None:
    """
    Loads a subset of the FinQA benchmark, generates an answer for each sample using two different configurations,
    and compares it with the target answer by performing an exact match. Saves the results in the output folder.
    """
    dataset = load_dataset("Aiera/finqa-verified", split="test")
    results = []

    configurations = [
        ("Qwen/Qwen2-VL-2B-Instruct", "vidore/colqwen2-base", QWEN2VL),
        ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", QWEN2_5VL)
    ]
    
    for processor_name, model_name, output_dir in configurations:
        total = 0
        correct = 0
        config_results = {"processor": processor_name, "model": model_name, "accuracy": 0}
        
        for i, sample in enumerate(dataset.select(range(num_samples))):
            context = sample.get("context", "")
            question = sample.get("question", "")
            target_raw = sample.get("answer", "")
            
            target_answer = format_target_answer(target_raw)
                
            prompt, generated_output = generate_answer_from_finqa_sample(context, question, processor_name, model_name)
            generated_answer = extract_formatted_answer(generated_output)
                
            if generated_answer == target_answer:
                correct += 1
            total += 1

            save_sample_results(output_dir, i + 1, prompt, target_answer, generated_answer)

        accuracy = correct / total if total > 0 else 0
        config_results["accuracy"] = accuracy
        results.append(config_results)
        logger.info("\nExact match accuracy on %d samples for %s: %.2f%%", total, processor_name, accuracy * 100)

    # Save final comparison results to output folder
    output_path = os.path.join(OUTPUT_DIRECTORY, "evaluation_results.txt")
    try:
        with open(output_path, "w") as f:
            for config_result in results:
                f.write(f"Processor: {config_result['processor']}\n")
                f.write(f"Model: {config_result['model']}\n")
                f.write(f"Accuracy: {config_result['accuracy'] * 100:.2f}%\n\n")
        logger.info(f"Evaluation results saved in {output_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")

if __name__ == "__main__":
    evaluate_finqa(num_samples=15)