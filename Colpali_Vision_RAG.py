
from dotenv import load_dotenv 
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

hf_token = "YOUR_TOKEN" 
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path


RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)

RAG.index(
    input_path="./data/AMEX_EMR_2023.pdf",
    index_name="analyse",
    store_collection_with_index=True, #stockage en représentation base64
    overwrite=True
)

#Charger le modèle VLM et le processeur associé
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.float16
).cuda().eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

#Convertir le PDF en images
images = convert_from_path("./data/AMEX_EMR_2023.pdf")

query = "Quel est le chiffre d'affaire d'Emerson en 2023?"

#Recherche
results = RAG.search(query, k=1) # K=1 retourne la plus adapté souvent 3 est utilisé mais comme pas beaucoup de mémoire

#Vérifier les résultats
if not results:
    raise ValueError("Aucun résultat trouvé pour la requête!")

#Index de page retourné
image_index = results[0]["page_num"] - 1
print(f"Image index : {image_index}")

image_selected = images[image_index]

#Construire les messages modèle Qwen2-VL
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_selected},
            {"type": "text", "text": query},
        ],
    }
]

#Préparer le texte
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

#Préparer les entrées pour le modèle
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

#Générer une réponse
generated_ids = model.generate(**inputs, max_new_tokens=2)

#Décoder la réponse
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

#Afficher la réponse
print("Réponse générée :", output_text[0])
