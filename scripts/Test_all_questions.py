import os
import torch
import json
import matplotlib.pyplot as plt
import logging
import pickle
import hashlib
import argparse
import asyncio  # pour gérer le code asynchrone
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_similarity_map,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directories
OUTPUT_DIRECTORY = "../Assets/output"
SIMILARITY_DIR = os.path.join(OUTPUT_DIRECTORY, "similarity_maps")
RELEVANT_DIR = os.path.join(OUTPUT_DIRECTORY, "relevant_documents")
os.makedirs(SIMILARITY_DIR, exist_ok=True)
os.makedirs(RELEVANT_DIR, exist_ok=True)


def load_model_and_processor():
    """Load the ColPALI model and processor."""
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0", torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
    return model, processor


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


def generate_embeddings(model, processor, images, pdf_path):
    """Generate embeddings if not already saved for the specific PDF."""
    pdf_hash = get_pdf_hash(pdf_path)
    index_file = os.path.join(OUTPUT_DIRECTORY, f"document_embeddings_{pdf_hash}.pkl")

    if os.path.exists(index_file):
        try:
            with open(index_file, "rb") as f:
                embeddings_list = pickle.load(f)
            logger.info(f"INFO: Embeddings loaded from cache for {pdf_path}.")
            return embeddings_list
        except Exception as e:
            logger.warning(
                f"WARNING: Failed to load embeddings for {pdf_path}. Regenerating... (Error: {e})"
            )

    logger.info(f"INFO: Generating new embeddings for {pdf_path}...")
    dataloader = DataLoader(
        dataset=images,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )

    embeddings_list = []
    for batch in tqdm(dataloader, desc="Generating embeddings"):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embeddings = model(**batch)
        embeddings_list.extend(embeddings.cpu().unbind())

    with open(index_file, "wb") as f:
        pickle.dump(embeddings_list, f)

    logger.info(f"INFO: Embeddings saved for {pdf_path}.")
    return embeddings_list


def get_relevant_indices(model, processor, query, embeddings_list, top_k):
    """Retrieve the indices of the top-k relevant pages based on the query."""
    batch_queries = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
    scores = processor.score_multi_vector(query_embeddings, embeddings_list)
    return scores[0].topk(top_k).indices.tolist()


def save_similarity_scores_and_maps(
    images, embeddings_list, top_k_indices, query, model, processor
):
    """Save the similarity scores and maps for the top-k relevant pages."""
    similarity_scores_path = os.path.join(OUTPUT_DIRECTORY, "similarity_scores.txt")
    with open(similarity_scores_path, "w") as score_file:
        for i, idx in enumerate(top_k_indices):
            image = images[idx]
            embeddings = embeddings_list[idx]

            relevant_path = os.path.join(RELEVANT_DIR, f"relevant_doc_{i + 1}.jpg")
            image.save(relevant_path)
            logger.info(f"Saved relevant document to {relevant_path}")

            n_patches = processor.get_n_patches(
                image_size=image.size, patch_size=model.patch_size, spatial_merge_size=2
            )
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
                processor.decode(batch_queries.input_ids[0])
                .replace(processor.tokenizer.pad_token, "")
                .strip()
            )

            similarity_maps = batched_similarity_maps[0]
            for token_idx, similarity_map in enumerate(
                similarity_maps[: len(query_tokens)]
            ):
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
                fig_path = os.path.join(
                    SIMILARITY_DIR, f"doc_{i + 1}_token_{token_idx + 1}.png"
                )
                fig.savefig(fig_path, dpi=100)
                plt.close(fig)
                score_file.write(
                    f"Document {i + 1}, Token #{token_idx + 1} (`{query_tokens[token_idx]}`): MaxSim score = {max_sim_score:.2f}\n"
                )
    logger.info(f"Similarity scores saved in {similarity_scores_path}")


def index_and_save_documents(pdf_path: str, query: str, top_k: int = 3):
    """Index the PDF document and save the top-k relevant pages and their similarity maps."""
    try:
        model, processor = load_model_and_processor()
        images = convert_pdf_to_images(pdf_path)
        embeddings_list = generate_embeddings(model, processor, images, pdf_path)
        top_k_indices = get_relevant_indices(
            model, processor, query, embeddings_list, top_k
        )
        save_similarity_scores_and_maps(
            images, embeddings_list, top_k_indices, query, model, processor
        )
        return RELEVANT_DIR
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


def generate_responses(query: str, relevant_dir: str, top_k: int = 3, model_name: str = "gemma-3-4b") -> str:
    """Generate responses using the specified model."""
    try:
        # Configuration based on model selection
        if "gemma" in model_name.lower():
            # Gemma model configuration
            model_id = ("google/gemma-3-4b-it" if "3-4b" in model_name else "google/gemma-3-12b-it")
            gen_model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.bfloat16
            ).eval()
            gen_processor = AutoProcessor.from_pretrained(model_id)
            
            # Load relevant documents
            relevant_files = sorted(os.listdir(relevant_dir))[:top_k]
            image_paths = [
                os.path.abspath(os.path.join(relevant_dir, file_name))
                for file_name in relevant_files
            ]
            logger.info(f"Number of images passed: {len(image_paths)}")

            # Prepare prompts
            PROMPT = f"Use the following pages to answer the query:\n{query}\n"
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful financial assistant. Use the following pages to answer the query",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_paths[0]},
                        {"type": "image", "image": image_paths[1]},
                        {"type": "image", "image": image_paths[2]},
                        {"type": "text", "text": PROMPT},
                    ],
                },
            ]

            # Process inputs for Gemma
            inputs = gen_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(gen_model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = gen_model.generate(
                    **inputs, max_new_tokens=150, do_sample=False
                )
                generation = generation[0][input_len:]
            output_text = gen_processor.decode(generation, skip_special_tokens=True)
            
        elif "qwen2.5" in model_name.lower():
            # Qwen2.5 model configuration
            gen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16
            ).cuda().eval()
            max_pixels = 512 * 28 * 28
            gen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=max_pixels)
            
            # Load relevant documents
            relevant_files = sorted(os.listdir(relevant_dir))[:top_k]
            image_paths = [f"file://{os.path.abspath(os.path.join(relevant_dir, file_name))}" for file_name in relevant_files]
            logger.info(f"Number of images passed: {len(image_paths)}")

            # Prepare prompts
            PROMPT = f"Use the following pages to answer the query:\n{query}\n"
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

            torch.cuda.empty_cache()
            generated_ids = gen_model.generate(**inputs, max_new_tokens=150)
            output_text = gen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
        elif "qwen2" in model_name.lower():
            # Qwen2 model configuration 
            gen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16
            ).cuda().eval()
            gen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            
            # Load relevant documents
            relevant_files = sorted(os.listdir(relevant_dir))[:top_k]
            image_paths = [f"file://{os.path.abspath(os.path.join(relevant_dir, file_name))}" for file_name in relevant_files]
            logger.info(f"Number of images passed: {len(image_paths)}")

            # Prepare prompts
            PROMPT = f"Use the following pages to answer the query:\n{query}\n"
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

            torch.cuda.empty_cache()
            generated_ids = gen_model.generate(**inputs, max_new_tokens=150)
            output_text = gen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        logger.info("Raw response received:")
        logger.info(output_text)

        # Post-processing for different model outputs
        clean_response = ""
        if "assistant" in output_text:
            parts = output_text.split("assistant")
            last_part = parts[-1].strip()
            clean_response = last_part.lstrip(":\n ")
            if "user" in clean_response or "system" in clean_response:
                import re

                assistant_parts = re.findall(
                    r"assistant(.*?)(?:user|system|$)", output_text, re.DOTALL
                )
                if assistant_parts:
                    clean_response = assistant_parts[-1].strip().lstrip(":\n ")
        else:
            clean_response = output_text

        logger.info("Cleaned response:")
        logger.info(clean_response)

        return clean_response

    except Exception as e:
        logger.error(f"An error occurred during response generation: {e}")
        raise


def process_dataset(
    dataset_path: str,
    top_k: int = 3,
    start_question_id: int = 1,
    end_question_id: int = None,
    field_to_update: str = "Answer_Qwen2.5",
    model_name: str = "gemma-3-4b",
):
    """
    Charge le dataset JSON, itère sur chaque échantillon,
    génère la réponse et l'ajoute dans le champ spécifié.
    Écrit les modifications dans le même fichier JSON.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        question_id = sample.get("Question_ID", 0)

        if question_id < start_question_id:
            logger.info(
                f"Skipping Question_ID {question_id} (below start_question_id {start_question_id})"
            )
            continue

        if end_question_id is not None and question_id > end_question_id:
            logger.info(
                f"Skipping Question_ID {question_id} (above end_question_id {end_question_id})"
            )
            continue

        try:
            pdf_name = sample.get("Expected_source", "").strip() + ".pdf"
            pdf_path = os.path.join("../Assets/data_test", pdf_name)
            query = sample.get("Question", "")

            logger.info(f"Processing Question_ID {question_id} with PDF: {pdf_path}")

            relevant_dir = index_and_save_documents(pdf_path, query, top_k=top_k)
            answer = generate_responses(query, relevant_dir, top_k=top_k, model_name=model_name)
            sample[field_to_update] = answer

            logger.info(f"About to write dataset to {os.path.abspath(dataset_path)}")
            with open(dataset_path, "w", encoding="utf-8") as f_out:
                json.dump(data, f_out, indent=2, ensure_ascii=False)
            logger.info(f"Dataset updated with answer for question {question_id}")

        except Exception as e:
            logger.error(f"Error processing Question_ID {question_id}: {e}")
            sample[field_to_update] = f"Error: {str(e)}"
            with open(dataset_path, "w", encoding="utf-8") as f_out:
                json.dump(data, f_out, indent=2, ensure_ascii=False)

    logger.info(f"Dataset fully updated and saved to {dataset_path}")


def main():
    parser = argparse.ArgumentParser(description="Process dataset with Qwen2 model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="../Assets/data_test/ardian_dataset_final.json",
        help="Path to the dataset JSON file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top relevant documents to retrieve",
    )
    parser.add_argument(
        "--start_id", type=int, default=1, help="Start processing from this question ID"
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=None,
        help="Process questions until this ID (inclusive)",
    )
    parser.add_argument(
        "--field",
        type=str,
        help="Field to update in the JSON (e.g., Answer_Qwen2.5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemma-3-4b", "gemma-3-12b-it", "qwen2.5", "qwen2"],
        help="Model to use for generating responses",
    )

    args = parser.parse_args()

    process_dataset(
        dataset_path=args.dataset,
        top_k=args.top_k,
        start_question_id=args.start_id,
        end_question_id=args.end_id,
        field_to_update=args.field,
        model_name=args.model,
    )


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
