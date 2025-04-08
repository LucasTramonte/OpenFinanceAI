import os
import torch
import json
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_similarity_map
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sentence_transformers import CrossEncoder
from qwen_vl_utils import process_vision_info
import logging
import matplotlib.pyplot as plt
import pickle
import hashlib
import numpy as np
import re
from typing import List, Tuple, Dict, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directories
OUTPUT_DIRECTORY = "../Assets/output"
SIMILARITY_DIR = os.path.join(OUTPUT_DIRECTORY, "similarity_maps")
RELEVANT_DIR = os.path.join(OUTPUT_DIRECTORY, "relevant_documents")
FILTERED_DIR = os.path.join(OUTPUT_DIRECTORY, "filtered_pages")
CANDIDATES_DIR = os.path.join(OUTPUT_DIRECTORY, "candidate_documents")
os.makedirs(SIMILARITY_DIR, exist_ok=True)
os.makedirs(RELEVANT_DIR, exist_ok=True)
os.makedirs(FILTERED_DIR, exist_ok=True)
os.makedirs(CANDIDATES_DIR, exist_ok=True)

# Configuration des seuils de filtrage
DEFAULT_MIN_TOKENS = 50  # Nombre minimal de tokens pour qu'une page soit considérée comme substantielle
DEFAULT_MIN_TEXT_RATIO = 0.15  # Ratio minimal de texte par rapport à la taille moyenne des pages

# Configuration pour le re-ranking
INITIAL_CANDIDATES = 15  # Nombre de pages candidates à récupérer initialement
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Modèle pour le re-ranking local


def load_model_and_processor():
    """Load the ColPALI model and processor."""
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    ).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
    return model, processor


def load_cross_encoder():
    """Load the cross-encoder model for re-ranking."""
    try:
        model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
        logger.info(f"Cross-encoder model {CROSS_ENCODER_MODEL} loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load cross-encoder model: {e}")
        return None


def convert_pdf_to_images(pdf_path):
    """Convert a PDF file to a list of images."""
    images = convert_from_path(pdf_path)
    logger.info(f"PDF converted to {len(images)} pages.")
    return images


def get_pdf_hash(pdf_path):
    """Generate a unique hash for the given PDF file."""
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_ocr_data(pdf_path):
    """
    Charge les données OCR correspondantes au PDF.
    """
    doc_name = os.path.basename(pdf_path).split('.')[0]
    ocr_path = os.path.join(OUTPUT_DIRECTORY, "ocr_data", f"{doc_name}_ocr_data.json")
    
    if not os.path.exists(ocr_path):
        logger.warning(f"Données OCR non trouvées pour {pdf_path}. Le filtrage textuel ne sera pas optimal.")
        return None
    
    with open(ocr_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    logger.info(f"Données OCR chargées pour {doc_name}: {len(ocr_data['pages'])} pages")
    return ocr_data


def estimate_token_count(text):
    """
    Estime le nombre de tokens dans un texte.
    Une approximation simple est de compter les mots (séparés par des espaces).
    """
    if not text or not isinstance(text, str):
        return 0
    
    # Filtrer les caractères non alphanumériques et compter les mots
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    return len(words)


def filter_pages_by_density(images, ocr_data=None, min_tokens=DEFAULT_MIN_TOKENS, min_ratio=DEFAULT_MIN_TEXT_RATIO):
    """
    Filtre les pages en fonction de leur densité textuelle.
    
    Args:
        images: Liste des images du PDF
        ocr_data: Données OCR du document (optionnel)
        min_tokens: Nombre minimal de tokens pour qu'une page soit conservée
        min_ratio: Ratio minimal par rapport à la moyenne du document
        
    Returns:
        Tuple (filtered_images, filtered_indices, stats)
    """
    if ocr_data is None:
        logger.warning("Pas de données OCR disponibles, le filtrage sera limité.")
        return images, list(range(len(images))), {"filtered_pages": 0, "total_pages": len(images)}
    
    # Calculer la densité textuelle pour chaque page
    token_counts = []
    for page in ocr_data["pages"]:
        text = page.get("text", "")
        token_count = estimate_token_count(text)
        token_counts.append(token_count)
    
    # Calculer la moyenne pour déterminer un seuil dynamique
    avg_tokens = np.mean(token_counts) if token_counts else 0
    dynamic_threshold = max(min_tokens, avg_tokens * min_ratio)
    
    logger.info(f"Densité textuelle moyenne: {avg_tokens:.1f} tokens par page")
    logger.info(f"Seuil de filtrage: {dynamic_threshold:.1f} tokens minimum")
    
    # Filtrer les pages en fonction du seuil
    filtered_indices = []
    filtered_images = []
    
    for i, (image, token_count) in enumerate(zip(images, token_counts)):
        if token_count >= dynamic_threshold:
            filtered_images.append(image)
            filtered_indices.append(i)
        else:
            logger.info(f"Page {i+1} filtrée: seulement {token_count} tokens")
            # Sauvegarder l'image filtrée pour analyse
            image.save(os.path.join(FILTERED_DIR, f"filtered_page_{i+1}.jpg"))
    
    stats = {
        "filtered_pages": len(images) - len(filtered_images),
        "total_pages": len(images),
        "token_counts": token_counts,
        "threshold": dynamic_threshold
    }
    
    # Enregistrer les statistiques de filtrage
    with open(os.path.join(OUTPUT_DIRECTORY, "density_filtering_stats.json"), "w") as f:
        json.dump({
            "avg_tokens_per_page": avg_tokens,
            "threshold_used": dynamic_threshold,
            "pages_kept": len(filtered_images),
            "pages_filtered": len(images) - len(filtered_images),
            "token_counts_by_page": {f"page_{i+1}": count for i, count in enumerate(token_counts)}
        }, f, indent=2)
    
    logger.info(f"{len(filtered_images)}/{len(images)} pages conservées après filtrage par densité textuelle")
    return filtered_images, filtered_indices, stats


def generate_embeddings(model, processor, images, pdf_path, indices=None):
    """Generate embeddings if not already saved for the specific PDF."""
    pdf_hash = get_pdf_hash(pdf_path)
    filtered_suffix = "_filtered" if indices is not None else ""
    index_file = os.path.join(OUTPUT_DIRECTORY, f"document_embeddings_{pdf_hash}{filtered_suffix}.pkl")

    if os.path.exists(index_file):
        try:
            with open(index_file, "rb") as f:
                embeddings_list = pickle.load(f)
            logger.info(f"INFO: Embeddings loaded from cache for {pdf_path}.")
            return embeddings_list
        except Exception as e:
            logger.warning(f"WARNING: Failed to load embeddings for {pdf_path}. Regenerating... (Error: {e})")

    # Compute embeddings if not found or loading failed
    logger.info(f"INFO: Generating new embeddings for {pdf_path}...")
    dataloader = DataLoader(
        dataset=images,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x)
    )

    embeddings_list = []
    for batch in tqdm(dataloader, desc="Generating embeddings"):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embeddings = model(**batch)
        embeddings_list.extend(embeddings.cpu().unbind())

    # Save embeddings specific to this PDF
    with open(index_file, "wb") as f:
        pickle.dump(embeddings_list, f)

    logger.info(f"INFO: Embeddings saved for {pdf_path}.")
    return embeddings_list


def get_candidate_indices(model, processor, query, embeddings_list, k=INITIAL_CANDIDATES, original_indices=None):
    """Retrieve the indices of the top-k candidate pages based on the query."""
    batch_queries = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
    scores = processor.score_multi_vector(query_embeddings, embeddings_list)
    
    # Get top-k indices from the filtered embeddings
    top_indices = scores[0].topk(k).indices.tolist()
    top_scores = scores[0].topk(k).values.tolist()
    
    # Map back to original indices if needed
    if original_indices is not None:
        top_indices = [original_indices[i] for i in top_indices]
        
    return top_indices, top_scores


def extract_numeric_information(text):
    """
    Extrait les informations numériques et pourcentages d'un texte.
    """
    # Extraire les pourcentages
    percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    
    # Extraire les années (2000-2030)
    years = re.findall(r'\b(20[0-2][0-9])\b', text)
    
    # Extraire les valeurs monétaires
    money = re.findall(r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
    
    # Calculer la densité d'information numérique (nombre / longueur)
    num_count = len(percentages) + len(years) + len(money)
    numeric_density = num_count / max(1, len(text.split()))
    
    return {
        "percentages": percentages,
        "years": years,
        "money": money,
        "numeric_count": num_count,
        "numeric_density": numeric_density
    }


def prepare_for_reranking(query, ocr_data, candidate_indices, images):
    """
    Prépare les données pour le re-ranking en extrayant le texte des pages candidates.
    """
    documents = []
    for idx in candidate_indices:
        # Sauvegarder l'image candidate
        image_path = os.path.join(CANDIDATES_DIR, f"candidate_page_{idx+1}.jpg")
        images[idx].save(image_path)
        
        # Extraire le texte de la page
        if ocr_data and idx < len(ocr_data["pages"]):
            page_text = ocr_data["pages"][idx].get("text", "")
            
            # Vérifier si la page contient des tableaux
            tables_info = ""
            if "tables" in ocr_data["pages"][idx] and ocr_data["pages"][idx]["tables"]:
                tables_count = len(ocr_data["pages"][idx]["tables"])
                tables_info = f"[Cette page contient {tables_count} tableau(x)]"
            
            # Extraire les informations numériques
            numeric_info = extract_numeric_information(page_text)
            
            # Ajouter le numéro de page et combiner les informations
            full_text = f"[Page {idx+1}] {tables_info}\n{page_text}"
            documents.append({
                "id": str(idx),
                "text": full_text,
                "page_number": idx + 1,
                "numeric_info": numeric_info
            })
        else:
            documents.append({
                "id": str(idx),
                "text": f"[Page {idx+1}] Texte non disponible",
                "page_number": idx + 1,
                "numeric_info": {"numeric_count": 0, "numeric_density": 0}
            })
    
    return documents


def rerank_with_cross_encoder(query, documents, top_k=3):
    """
    Utilise un cross-encodeur pour ré-ordonner les documents candidats.
    """
    try:
        # Charger le modèle de cross-encodeur
        cross_encoder = load_cross_encoder()
        if cross_encoder is None:
            logger.warning("Impossible de charger le cross-encodeur. Utilisation de l'ordre initial.")
            return [int(doc["id"]) for doc in documents[:top_k]]
        
        # Préparer les paires query-document pour le scoring
        pairs = [(query, doc["text"]) for doc in documents]
        
        # Calculer les scores de pertinence
        logger.info("Application du cross-encodeur pour le re-ranking...")
        scores = cross_encoder.predict(pairs)
        
        # Combiner indices et scores pour le tri
        doc_score_pairs = [(int(documents[i]["id"]), scores[i]) for i in range(len(documents))]
        
        # Trier par score décroissant et prendre les top_k
        sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        reranked_indices = [idx for idx, _ in sorted_pairs[:top_k]]
        rerank_scores = [score for _, score in sorted_pairs[:top_k]]
        
        # Journaliser les résultats
        logger.info(f"Re-ranking effectué. Nouveaux indices: {reranked_indices}")
        logger.info(f"Scores de pertinence: {rerank_scores}")
        
        # Sauvegarder les résultats de re-ranking
        rerank_results = {
            "query": query,
            "results": [
                {
                    "original_page": documents[i]["page_number"],
                    "score": float(scores[i]),
                    "snippet": documents[i]["text"][:200] + "..." if len(documents[i]["text"]) > 200 else documents[i]["text"]
                }
                for i, _ in enumerate(documents)
            ]
        }
        
        with open(os.path.join(OUTPUT_DIRECTORY, "reranking_results.json"), "w") as f:
            json.dump(rerank_results, f, indent=2)
        
        return reranked_indices
        
    except Exception as e:
        logger.error(f"Erreur lors du re-ranking avec cross-encodeur: {e}")
        # En cas d'erreur, revenir aux top_k documents initiaux
        return [int(doc["id"]) for doc in documents[:top_k]]


def save_similarity_scores_and_maps(images, embeddings_list, top_k_indices, query, model, processor, original_indices=None):
    """Save the similarity scores and maps for the top-k relevant pages."""
    similarity_scores_path = os.path.join(OUTPUT_DIRECTORY, "similarity_scores.txt")
    with open(similarity_scores_path, "w") as score_file:
        for i, idx in enumerate(top_k_indices):
            image = images[idx]
            
            # Find the corresponding embedding index
            embedding_idx = idx
            if original_indices is not None:
                try:
                    embedding_idx = original_indices.index(idx)
                except ValueError:
                    logger.warning(f"No embedding found for image index {idx}, skipping similarity maps")
                    continue
                    
            embeddings = embeddings_list[embedding_idx]

            # Save relevant image
            relevant_path = os.path.join(RELEVANT_DIR, f"relevant_doc_{i + 1}.jpg")
            image.save(relevant_path)

            # Generate and save similarity maps
            n_patches = processor.get_n_patches(image_size=image.size, patch_size=model.patch_size, spatial_merge_size=2)
            image_mask = processor.get_image_mask(processor.process_images([image]))
            batch_queries = processor.process_queries([query]).to(model.device)
            with torch.no_grad():
                query_embeddings = model(**batch_queries)

            batched_similarity_maps = get_similarity_maps_from_embeddings(
                image_embeddings=embeddings.unsqueeze(0).to("cuda"),
                query_embeddings=query_embeddings,
                n_patches=n_patches,
                image_mask=image_mask,
            )

            query_tokens = processor.tokenizer.tokenize(
                processor.decode(batch_queries.input_ids[0]).replace(processor.tokenizer.pad_token, "").strip()
            )

            similarity_maps = batched_similarity_maps[0]
            for token_idx, similarity_map in enumerate(similarity_maps[:len(query_tokens)]):
                max_sim_score = similarity_map.max().item()
                fig, ax = plot_similarity_map(
                    image=image,
                    similarity_map=similarity_map,
                    figsize=(8, 8),
                    show_colorbar=True,
                )
                ax.set_title(
                    f"Token #{token_idx + 1}: `{query_tokens[token_idx].replace('Ġ', '_')}`. MaxSim score: {max_sim_score:.2f}",
                    fontsize=10,
                )
                fig.tight_layout()
                fig_path = os.path.join(SIMILARITY_DIR, f"doc_{i + 1}_token_{token_idx + 1}.png")
                fig.savefig(fig_path, dpi=100)
                plt.close(fig)  # Close the figure to free up memory

                # Save similarity score to text file
                score_file.write(
                    f"Document {i + 1}, Token #{token_idx + 1} (`{query_tokens[token_idx]}`): MaxSim score = {max_sim_score:.2f}\n"
                )
    logger.info(f"Similarity scores saved in {similarity_scores_path}")


def index_and_save_documents(pdf_path: str, query: str, top_k: int = 3):
    """Index the PDF document and save the top-k relevant pages and their similarity maps."""
    try:
        # Charger le modèle et le processeur
        model, processor = load_model_and_processor()
        
        # Convertir le PDF en images
        all_images = convert_pdf_to_images(pdf_path)
        
        # Charger les données OCR
        ocr_data = load_ocr_data(pdf_path)
        
        # Filtrer les pages par densité textuelle
        filtered_images, original_indices, density_stats = filter_pages_by_density(all_images, ocr_data)
        
        # Générer les embeddings uniquement pour les pages filtrées
        embeddings_list = generate_embeddings(model, processor, filtered_images, pdf_path, indices=original_indices)
        
        # PHASE 1: Obtenir d'abord les pages les plus pertinentes avec ColQwen2 (récupération initiale)
        initial_indices, initial_scores = get_candidate_indices(
            model, processor, query, embeddings_list, k=top_k, original_indices=original_indices
        )
        
        # Afficher les pages sélectionnées avant le re-ranking
        print("\n" + "="*80)
        print(f"AVANT RE-RANKING - Top {top_k} pages sélectionnées:")
        for i, idx in enumerate(initial_indices):
            print(f"  {i+1}. Page {idx+1} (score: {initial_scores[i]:.4f})")
        print("="*80 + "\n")
        
        # PHASE 2: Obtenir un ensemble plus large de candidats pour le re-ranking
        candidate_indices, _ = get_candidate_indices(
            model, processor, query, embeddings_list, k=INITIAL_CANDIDATES, original_indices=original_indices
        )
        
        # Préparer les données pour le re-ranking
        candidate_documents = prepare_for_reranking(query, ocr_data, candidate_indices, all_images)
        
        # Effectuer le re-ranking pour sélectionner les top_k meilleurs documents
        reranked_indices = rerank_with_cross_encoder(query, candidate_documents, top_k=top_k)
        
        # Afficher les pages sélectionnées après le re-ranking
        print("\n" + "="*80)
        print(f"APRÈS RE-RANKING - Top {top_k} pages sélectionnées:")
        for i, idx in enumerate(reranked_indices):
            print(f"  {i+1}. Page {idx+1}")
        print("="*80 + "\n")
        
        # Log des changements dans l'ordre des pages
        def page_changed(before, after):
            return set(before) != set(after) or before != after
            
        if page_changed(initial_indices, reranked_indices):
            print(f"Le re-ranking a modifié les pages sélectionnées!")
            added = [p for p in reranked_indices if p not in initial_indices]
            removed = [p for p in initial_indices if p not in reranked_indices]
            if added:
                print(f"  Pages ajoutées: {[p+1 for p in added]}")
            if removed:
                print(f"  Pages supprimées: {[p+1 for p in removed]}")
        else:
            print("Le re-ranking n'a pas modifié les pages sélectionnées.")
        
        # Sauvegarder les cartes de similarité et les scores pour les documents après re-ranking
        save_similarity_scores_and_maps(all_images, embeddings_list, reranked_indices, query, model, processor, original_indices)
        
        # Journaliser les résultats du processus complet
        logger.info(f"Filtrage par densité: {density_stats['filtered_pages']} pages filtrées sur {density_stats['total_pages']}")
        logger.info(f"Re-ranking: {len(candidate_indices)} candidats → {len(reranked_indices)} documents finaux")
        
        return RELEVANT_DIR
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


def generate_responses(query: str, relevant_dir: str, top_k: int = 3):
    """Generate responses based on the top-k relevant pages."""
    try:
        # Load the model and processor
        gen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16
        ).cuda().eval()
        max_pixels = 512 * 28 * 28
        gen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=max_pixels)

        # Load the relevant documents
        relevant_files = sorted(os.listdir(relevant_dir))[:top_k]
        image_paths = [f"file://{os.path.abspath(os.path.join(relevant_dir, file_name))}" for file_name in relevant_files]

        logger.info(f"Number of images passed: {len(image_paths)}")

        # Prepare the prompt with the query
        PROMPT = f"Use the following pages to answer the query:\n{query}\n"

        # Prepare the messages with multiple images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_paths[0]},
                    {"type": "image", "image": image_paths[1]},
                    {"type": "image", "image": image_paths[2]},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = gen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = gen_processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        # Generate a unique response
        torch.cuda.empty_cache()
        generated_ids = gen_model.generate(**inputs, max_new_tokens=150)
        output_text = gen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        logger.info("Final response:")
        logger.info(output_text)

        # Save the generated response
        response_path = os.path.join(OUTPUT_DIRECTORY, "generated_responses.txt")
        with open(response_path, "w") as f:
            f.write(output_text)
        logger.info(f"Generated responses saved in {response_path}")

    except Exception as e:
        logger.error(f"An error occurred during response generation: {e}")
        raise

def main():
    """
    Évaluation de la pipeline RAG sur l'ensemble de données ARDIAN.
    Génère les réponses pour chaque question et sauvegarde les résultats.
    """
    # Charger le jeu de données de test
    dataset_path = "../Assets/data_test/ardian_dataset_final.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Préparer la structure pour stocker les documents retournés par question
    retrieved_docs = []
    
    # Définir le répertoire des PDFs
    pdf_directory = "../Assets/data_test/pdfs"
    
    # Traiter chaque question du dataset
    for idx, item in enumerate(dataset):
        question_id = item.get("Question_ID", f"Q{idx+1}")
        question = item.get("Question")
        expected_source = item.get("Expected_source", "")
        
        # Afficher la progression
        print(f"\n\n{'='*80}")
        print(f"Traitement de la question {idx+1}/{len(dataset)}: {question}")
        print(f"Source attendue: {expected_source}")
        print('='*80)
        
        try:
            # Déterminer le chemin du PDF correspondant
            pdf_path = os.path.join(pdf_directory, f"{expected_source}.pdf")
            if not os.path.exists(pdf_path):
                print(f"ATTENTION: PDF non trouvé pour {expected_source}. Utilisation du PDF par défaut.")
                pdf_path = os.path.join(pdf_directory, "AMEX_EMR_2023.pdf")
            
            # Exécuter la récupération de documents avec re-ranking
            relevant_dir = index_and_save_documents(pdf_path, question, top_k=3)
            
            # Récupérer les fichiers des documents pertinents
            doc_files = sorted(os.listdir(relevant_dir))[:3]
            
            # Collecter les informations sur les documents retournés
            current_docs = []
            for i, doc_file in enumerate(doc_files):
                # Le fichier est nommé "relevant_doc_X.jpg" où X est le numéro de position (1, 2, 3)
                # Pour obtenir l'indice réel de la page dans le document, on extrait le numéro de page
                # depuis l'index_and_save_documents
                rank = i + 1  # Le rang dans les résultats (1, 2, 3)
                
                # Pour les indices de page, on définira correctement:
                # Si le fichier est relevant_doc_1.jpg, la page indexée est 0
                page = i  # Par défaut, utiliser l'ordre (0, 1, 2)
                
                doc_path = os.path.join(relevant_dir, doc_file)
                
                # Ajouter à la liste des documents pour cette question
                current_docs.append({
                    "rank": rank,
                    "page": page,
                    "document": os.path.basename(pdf_path),
                    "file_path": doc_path
                })
            
            # Générer la réponse à la question
            generate_responses(question, relevant_dir, top_k=3)
            
            # Lire la réponse générée et extraire seulement la partie de l'assistant
            response_path = os.path.join(OUTPUT_DIRECTORY, "generated_responses.txt")
            with open(response_path, "r", encoding="utf-8") as f:
                full_response = f.read().strip()
            
            # Extraire uniquement la partie après "assistant"
            if "assistant" in full_response:
                # Diviser au mot "assistant" et prendre la dernière partie
                parts = full_response.split("assistant")
                if len(parts) > 1:
                    assistant_response = parts[-1].strip()
                    # Supprimer les caractères non-alphanumériques au début
                    assistant_response = re.sub(r'^[^a-zA-Z0-9]+', '', assistant_response)
                    generated_response = assistant_response
                else:
                    generated_response = full_response
            else:
                generated_response = full_response
            
            # Ajouter la réponse au jeu de données
            dataset[idx]["Answer_PV2"] = generated_response
            
            # Enregistrer les informations de récupération pour cette question
            retrieved_docs.append({
                "Question_ID": question_id,
                "Question": question,
                "Retrieved_Documents": current_docs
            })
            
            print(f"Réponse générée: {generated_response[:100]}...")
            
        except Exception as e:
            print(f"ERREUR lors du traitement de la question {idx+1}: {str(e)}")
            dataset[idx]["Answer_PV2"] = f"ERREUR: {str(e)}"
            retrieved_docs.append({
                "Question_ID": question_id,
                "Question": question,
                "Retrieved_Documents": [],
                "Error": str(e)
            })
        
        # Sauvegarder le dataset après chaque question
        with open("ardian_dataset_with_pv2.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        with open("ardian_retrieved_documents.json", "w", encoding="utf-8") as f:
            json.dump(retrieved_docs, f, indent=2, ensure_ascii=False)
    
    print("\nÉvaluation terminée. Résultats sauvegardés dans ../Assets/output/ardian_dataset_with_pv2.json")
    print("Documents récupérés sauvegardés dans ../Assets/output/ardian_retrieved_documents.json")

# Configuration CUDA pour optimiser l'utilisation mémoire
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    main()