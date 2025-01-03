import os
import torch
from pdf2image import convert_from_path
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_similarity_map
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

# Répertoires de sortie
output_directory = "./output"
similarity_dir = os.path.join(output_directory, "similarity_maps")
relevant_dir = os.path.join(output_directory, "relevant_documents")
os.makedirs(similarity_dir, exist_ok=True)
os.makedirs(relevant_dir, exist_ok=True)

# Fonction 1 : Indexation et génération des cartes de similarité
def index_and_save_documents(pdf_path: str, query: str, top_k: int = 3):
    # Modèle et processeur ColPALI
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    ).eval()
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

    # Conversion PDF en images
    images = convert_from_path(pdf_path)
    print(f"PDF converti en {len(images)} pages.")

    # Embeddings pour chaque page
    dataloader = DataLoader(
        dataset=images,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    embeddings_list = []
    for batch in tqdm(dataloader, desc="Génération des embeddings"):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embeddings = model(**batch)
        embeddings_list.extend(embeddings.cpu().unbind())

    # Fonction pour récupérer les indices pertinents
    def get_results(query: str):
        batch_queries = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            query_embeddings = model(**batch_queries)
        scores = processor.score_multi_vector(query_embeddings, embeddings_list)
        return scores[0].topk(top_k).indices.tolist()

    top_k_indices = get_results(query)

    # Sauvegarder les documents pertinents et les cartes de similarité
    similarity_scores_path = os.path.join(output_directory, "similarity_scores.txt")
    with open(similarity_scores_path, "w") as score_file:
        for i, idx in enumerate(top_k_indices):
            im = images[idx]
            embeddings = embeddings_list[idx]

            # Sauvegarder l'image pertinente
            relevant_path = os.path.join(relevant_dir, f"relevant_doc_{i + 1}.jpg")
            im.save(relevant_path)

            # Générer et sauvegarder les cartes de similarité
            n_patches = processor.get_n_patches(
                image_size=im.size,
                patch_size=model.patch_size,
            )
            image_mask = processor.get_image_mask(processor.process_images([im]))
            batch_queries = processor.process_queries([query]).to(model.device)
            with torch.no_grad():
                query_embeddings = model(**batch_queries)

            batched_similarity_maps = get_similarity_maps_from_embeddings(
                image_embeddings=embeddings.unsqueeze(0).to("cuda"),
                query_embeddings=query_embeddings,
                n_patches=n_patches,
                image_mask=image_mask,
            )

            query_content = processor.decode(batch_queries.input_ids[0]).replace(
                processor.tokenizer.pad_token, ""
            ).strip()
            query_tokens = processor.tokenizer.tokenize(query_content)

            similarity_maps = batched_similarity_maps[0]
            for token_idx, similarity_map in enumerate(similarity_maps[:len(query_tokens)]):
                max_sim_score = similarity_map.max().item()
                fig, ax = plot_similarity_map(
                    image=im,
                    similarity_map=similarity_map,
                    figsize=(8, 8),
                    show_colorbar=True,
                )
                ax.set_title(
                    f"Token #{token_idx + 1}: `{query_tokens[token_idx].replace('Ġ', '_')}`. MaxSim score: {max_sim_score:.2f}",
                    fontsize=10,
                )
                fig.tight_layout()
                fig_path = os.path.join(similarity_dir, f"doc_{i + 1}_token_{token_idx + 1}.png")
                fig.savefig(fig_path, dpi=100)
                print(f"Saved similarity map to {fig_path}")

                # Sauvegarder le score de similarité dans le fichier texte
                score_file.write(
                    f"Document {i + 1}, Token #{token_idx + 1} (`{query_tokens[token_idx]}`): MaxSim score = {max_sim_score:.2f}\n"
                )
    print(f"Scores de similarité sauvegardés dans {similarity_scores_path}")
    return relevant_dir

def generate_responses(query: str, relevant_dir: str, top_k: int = 3):
    # Charger le modèle et le processeur
    gen_model = Qwen2VLForConditionalGeneration.from_pretrained("vidore/colqwen2-base",torch_dtype=torch.bfloat16).cuda().eval()  
    max_pixels = 512*28*28    
    gen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", max_pixels=max_pixels)

    # Charger les documents pertinents
    relevant_files = sorted(os.listdir(relevant_dir))[:top_k]
    responses = []
    PROMPT = f"""
    Use the page to answer the query:
    {query}
    PDF pages:
    """
    for i, file_name in enumerate(relevant_files):
        im_path = os.path.join(relevant_dir, file_name)
        im = Image.open(im_path)

        # Préparer le prompt et les entrées
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": im},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = gen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = gen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Générer la réponse
        torch.cuda.empty_cache()
        generated_ids = gen_model.generate(**inputs, max_new_tokens=30)
        output_text = gen_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        responses.append(output_text)

        print(f"Réponse générée pour le document {i + 1} : {output_text}")

    # Sauvegarder les réponses générées
    response_path = os.path.join(output_directory, "generated_responses.txt")
    with open(response_path, "w") as f:
        for i, response in enumerate(responses):
            f.write(f"Document {i + 1}:\n{response}\n\n")
    print(f"Réponses générées sauvegardées dans {response_path}")

    
# Exécution
pdf_path = "./data_test/AMEX_EMR_2023.pdf"
query = "What is the operating cash flow of Emerson in 2023?"
relevant_dir = index_and_save_documents(pdf_path, query, top_k=3)
generate_responses(query, relevant_dir, top_k=3)