import os
import torch
import re
import logging
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoModel,
    Gemma3ForConditionalGeneration,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lecture du token Hugging face pour les modèles Gemma depius le fichier scripts/token.txt (A ne pas commiter et créer manuellement!!!)
with open("token.txt", "r") as token_file:
    HF_TOKEN = token_file.read().strip()

# Output directories
OUTPUT_DIRECTORY = "../../Assets/output/FinQA"
QWEN2VL_DIR = os.path.join(OUTPUT_DIRECTORY, "Qwen2VL")
QWEN2_5VL_DIR = os.path.join(OUTPUT_DIRECTORY, "Qwen2_5VL")
INTERN_DIR = os.path.join(OUTPUT_DIRECTORY, "InternVL2_5MPO")
GEMMA_DIR = os.path.join(OUTPUT_DIRECTORY, "gemma-3-4b-it")
GEMMA_DIR_12B = os.path.join(OUTPUT_DIRECTORY, "gemma-3-12b-it")
os.makedirs(QWEN2VL_DIR, exist_ok=True)
os.makedirs(QWEN2_5VL_DIR, exist_ok=True)
os.makedirs(INTERN_DIR, exist_ok=True)


def generate_prompt(context: str, question: str) -> str:
    """
    Construit le prompt à partir du contexte et de la question selon le format FinQA.
    """
    return (
        f"Context:\n{context}\n\n"
        f"Given the context, {question} Report your answer using the following format:\n"
        "Explanation: Explanation of calculation\n"
        "Formatted answer: Float number to two decimal point precision and no units\n"
    )


def generate_answer_from_finqa_sample(
    context: str,
    question: str,
    processor_name: str,
    model_name: str,
    max_pixels: int = 512 * 28 * 28,
) -> (str, str):
    """
    Génère une réponse à partir d'un échantillon FinQA en utilisant
    - Qwen2-VL/Qwen2.5-VL ou Gemma pour la génération
    """
    prompt = generate_prompt(context, question)

    try:
        if "gemma" in model_name.lower():
            # Utilisation du modèle Gemma
            gen_processor = AutoProcessor.from_pretrained(
                processor_name, trust_remote_code=True
            )
            # Construction du message de conversation pour Gemma
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            inputs = gen_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda", dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                use_auth_token=HF_TOKEN,
            ).eval()
            torch.cuda.empty_cache()
            with torch.inference_mode():
                generation = model.generate(
                    **inputs, max_new_tokens=200, do_sample=False
                )
                generation = generation[0][input_len:]
            output_text = gen_processor.decode(generation, skip_special_tokens=True)

        elif "Qwen2.5" in model_name or "Qwen2-" in model_name:
            gen_processor = AutoProcessor.from_pretrained(
                processor_name, max_pixels=max_pixels, trust_remote_code=True
            )
            messages = [{"role": "user", "content": prompt}]
            text = gen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = gen_processor(text=[text], padding=True, return_tensors="pt").to(
                "cuda"
            )

            if "Qwen2.5" in model_name:
                model = (
                    Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_name, torch_dtype=torch.bfloat16
                    )
                    .cuda()
                    .eval()
                )
            else:
                model = (
                    Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name, torch_dtype=torch.bfloat16
                    )
                    .cuda()
                    .eval()
                )

            torch.cuda.empty_cache()
            generated_ids = model.generate(**inputs, max_new_tokens=200)
            output_text = gen_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=False
            )
            model = (
                AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_flash_attn=True,
                )
                .cuda()
                .eval()
            )

            generation_config = {"max_new_tokens": 512, "do_sample": False}
            response = model.chat(tokenizer, None, prompt, generation_config)
            output_text = response

        return prompt, output_text

    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return prompt, ""


def format_target_answer(target_raw) -> str:
    """
    Formate la réponse cible sur 2 décimales si c'est un nombre float.
    """
    if isinstance(target_raw, float):
        return f"{target_raw:.2f}"
    return str(target_raw).strip()


def extract_formatted_answer(generated_output: str) -> str:
    """
    Extrait la partie après "Formatted answer:" pour retrouver la réponse finale.
    """
    if "formatted answer:" in generated_output.lower():
        parts = re.split(r"formatted answer:", generated_output, flags=re.IGNORECASE)
        return parts[-1].strip()
    return generated_output.strip()


def save_sample_results(output_dir, sample_id, prompt, target_answer, generated_answer):
    """
    Sauvegarde localement le prompt, la réponse cible et la réponse générée.
    """
    sample_dir = os.path.join(output_dir, f"sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)

    with open(os.path.join(sample_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    with open(os.path.join(sample_dir, "target_answer.txt"), "w") as f:
        f.write(target_answer)
    with open(os.path.join(sample_dir, "generated_answer.txt"), "w") as f:
        f.write(generated_answer)


def evaluate_finqa(num_samples: int = 50) -> None:
    """
    Charge un sous-ensemble de FinQA, génère une réponse pour chaque échantillon
    avec différentes configurations, compare avec la réponse cible, et enregistre les résultats.
    """
    dataset = load_dataset("Aiera/finqa-verified", split="test")
    results = []

    configurations = [
        # Decommenter ceux qu'on veut pas évaluer!!!!
        # ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", QWEN2_5VL_DIR),
        # ("Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-2B-Instruct", QWEN2VL_DIR),
        # ("google/gemma-3-4b-it", "google/gemma-3-4b-it", GEMMA_DIR),
        ("google/gemma-3-12b-it", "google/gemma-3-12b-it", GEMMA_DIR_12B)
    ]

    for processor_name, model_name, output_dir in configurations:
        total = 0
        correct = 0
        config_results = {
            "processor": processor_name,
            "model": model_name,
            "accuracy": 0,
        }

        # On boucle sur un sous-ensemble (num_samples) du dataset FinQA
        for i, sample in enumerate(dataset.select(range(num_samples))):
            context = sample.get("context", "")
            question = sample.get("question", "")
            target_raw = sample.get("answer", "")

            target_answer = format_target_answer(target_raw)

            prompt, generated_output = generate_answer_from_finqa_sample(
                context, question, processor_name, model_name
            )
            generated_answer = extract_formatted_answer(generated_output)

            if generated_answer == target_answer:
                correct += 1
            total += 1

            save_sample_results(
                output_dir, i + 1, prompt, target_answer, generated_answer
            )

        accuracy = correct / total if total > 0 else 0
        config_results["accuracy"] = accuracy
        results.append(config_results)
        logger.info(
            "\nExact match accuracy on %d samples for %s: %.2f%%",
            total,
            processor_name,
            accuracy * 100,
        )

    # Sauvegarde des résultats d’évaluation
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
    evaluate_finqa(num_samples=50)