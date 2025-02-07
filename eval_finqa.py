import os
import torch
import re
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

def generate_answer_from_finqa_sample(context: str, question: str, max_pixels: int = 512 * 28 * 28) -> str:
    """
    Construit un prompt à partir du contexte et de la question (selon le format FinQA),
    et utilise Qwen pour générer la réponse.
    """
    prompt = (
        f"Context:\n{context}\n\n"
        f"Given the context, {question} Report your answer using the following format:\n"
        "Explanation: Explanation of calculation\n"
        "Formatted answer: Float number to two decimal point precision and no units\n"
    )
    print("Prompt:\n", prompt)
    
    # Charger le processor et préparer le prompt
    gen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", max_pixels=max_pixels)
    messages = [{"role": "user", "content": prompt}]
    text = gen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = gen_processor(text=[text], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    
    # Charger le modèle de génération
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "vidore/colqwen2-base",  # Vérifiez que cet identifiant correspond à un checkpoint valide
        torch_dtype=torch.bfloat16
    ).cuda().eval()
    
    torch.cuda.empty_cache()
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    output_text = gen_processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text

def evaluate_finqa(num_samples: int = 15):
    """
    Charge un sous-ensemble du benchmark FinQA, génère une réponse pour chaque échantillon
    et compare avec la réponse cible en effectuant un exact match.
    """
    dataset = load_dataset("Aiera/finqa-verified", split="test")
    total = 0
    correct = 0
    
    for i, sample in enumerate(dataset.select(range(num_samples))):
        context = sample.get("context", "")
        question = sample.get("question", "")
        target_raw = sample.get("answer", "")
        
        # Assurez-vous que la réponse cible est bien formatée en chaîne avec 2 décimales le cas échéant
        if isinstance(target_raw, float):
            target_answer = f"{target_raw:.2f}"
        else:
            target_answer = str(target_raw).strip()
            
        print(f"\n=== Échantillon {i + 1} ===")
        print("Réponse cible :", target_answer)
        
        generated_output = generate_answer_from_finqa_sample(context, question)
        
        # On extrait la partie après "Formatted answer:" s'il y a un prompt détaillé dans la réponse
        if "formatted answer:" in generated_output.lower():
            parts = re.split(r"formatted answer:", generated_output, flags=re.IGNORECASE)
            generated_answer = parts[-1].strip()
        else:
            generated_answer = generated_output.strip()
            
        print("Réponse générée :", generated_answer)
        
        if generated_answer == target_answer:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nExact match accuracy sur {total} échantillons: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_finqa(num_samples=15)
