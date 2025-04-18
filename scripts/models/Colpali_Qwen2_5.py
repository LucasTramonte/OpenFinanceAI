import os
import torch
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_similarity_map
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
import matplotlib.pyplot as plt
import pickle
import hashlib

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
        "vidore/colqwen2-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
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

def get_relevant_indices(model, processor, query, embeddings_list, top_k):
    """Retrieve the indices of the top-k relevant pages based on the query."""
    batch_queries = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
    scores = processor.score_multi_vector(query_embeddings, embeddings_list)
    return scores[0].topk(top_k).indices.tolist()

def save_similarity_scores_and_maps(images, embeddings_list, top_k_indices, query, model, processor):
    """Save the similarity scores and maps for the top-k relevant pages."""
    similarity_scores_path = os.path.join(OUTPUT_DIRECTORY, "similarity_scores.txt")
    with open(similarity_scores_path, "w") as score_file:
        for i, idx in enumerate(top_k_indices):
            image = images[idx]
            embeddings = embeddings_list[idx]

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
        model, processor = load_model_and_processor()
        images = convert_pdf_to_images(pdf_path)
        embeddings_list = generate_embeddings(model, processor, images,pdf_path)
        top_k_indices = get_relevant_indices(model, processor, query, embeddings_list, top_k)
        save_similarity_scores_and_maps(images, embeddings_list, top_k_indices, query, model, processor)
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
        
        return output_text

    except Exception as e:
        logger.error(f"An error occurred during response generation: {e}")
        raise

def main():
    pdf_path = "../Assets/data_test/pdfs/AMEX_EMR_2023.pdf"
    query = "What percentage of women occupy leadership positions in the company in 2023?"
    relevant_dir = index_and_save_documents(pdf_path, query, top_k=3)
    generate_responses(query, relevant_dir, top_k=3)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    main()