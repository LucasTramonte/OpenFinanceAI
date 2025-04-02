import os
import streamlit as st
import torch
from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader
from tqdm import tqdm
import hashlib
import pickle

# Set environment variable to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Streamlit layout
st.set_page_config(layout="wide")

# Set directories to save files
base_dir = "./streamlit"
pdf_dir = os.path.join(base_dir, "uploaded_pdfs")
temp_dir = os.path.join(base_dir, "temp_files")
output_dir = os.path.join(base_dir, "output")
index_dir = os.path.join(base_dir, "index")

os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

st.title("ColPali and Qwen Multimodal RAG App")
with st.sidebar:
    st.header("Configuration")
    colpali_model_name = st.selectbox("Select ColPali Model", ["vidore/colpali", "vidore/colpali-v1.2"], index=1)
    qwen_model_name = st.selectbox("Select Qwen Model", ["vidore/colqwen2-base", "Qwen/Qwen2.5-VL-7B-Instruct"], index=0)
    pdf_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

def save_uploaded_file(uploaded_file, save_dir):
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def get_pdf_hash(pdf_path):
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def generate_embeddings(model, processor, images, pdf_path):
    pdf_hash = get_pdf_hash(pdf_path)
    index_file = os.path.join(index_dir, f"document_embeddings_{pdf_hash}.pkl")

    if os.path.exists(index_file):
        try:
            with open(index_file, "rb") as f:
                embeddings_list = pickle.load(f)
            st.info(f"Embeddings loaded from cache for {pdf_path}.")
            return embeddings_list
        except Exception as e:
            st.warning(f"Failed to load embeddings for {pdf_path}. Regenerating... (Error: {e})")

    st.info("Generating new embeddings...")
    dataloader = DataLoader(
        dataset=images,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x)
    )

    embeddings_list = []
    for batch in tqdm(dataloader, desc="Generating embeddings"):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embeddings = model(**batch)
        embeddings_list.extend(embeddings.cpu().unbind())

    with open(index_file, "wb") as f:
        pickle.dump(embeddings_list, f)

    st.info(f"Embeddings saved for {pdf_path}.")
    return embeddings_list


def process_query(model, processor, query):
    batch_queries = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
    return query_embeddings

def get_top_k_relevant_images(colpali_processor, embeddings_list, query_embeddings, images, k=3):
    scores = colpali_processor.score_multi_vector(query_embeddings, embeddings_list)
    top_k_indices = scores[0].topk(k).indices.tolist()
    return [images[idx] for idx in top_k_indices]

def save_images(images, save_dir):
    image_paths = []
    for i, img in enumerate(images):
        temp_image_path = os.path.join(save_dir, f"temp_image_{i}.png")
        img.save(temp_image_path)
        image_paths.append(f"file://{os.path.abspath(temp_image_path)}")
    return image_paths

def generate_response(qwen_model, qwen_processor, query, relevant_image_paths):
    prompt = f"Use the following pages to answer the query:\n{query}"
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img_path} for img_path in relevant_image_paths],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_input = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    torch.cuda.empty_cache()
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=150)
    output_text = qwen_processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text

def cleanup_temp_files(file_paths):
    for file_path in file_paths:
        os.remove(file_path)

def main():
    if pdf_file is not None:
        pdf_path = save_uploaded_file(pdf_file, pdf_dir)
        st.success(f"Uploaded file: {pdf_file.name}")

        query = st.text_input("Enter your query:")

        if st.button("Process Query"):
            if query:
                try:
                    st.info("Loading models...")
                    colpali_model = ColPali.from_pretrained(colpali_model_name, torch_dtype=torch.float16, device_map="auto").eval()
                    colpali_processor = ColPaliProcessor.from_pretrained(colpali_model_name)

                    if qwen_model_name == "vidore/colqwen2-base":
                        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen_model_name, torch_dtype=torch.float16).cuda().eval()
                        qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
                    else:
                        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16).cuda().eval()
                        max_pixels = 512 * 28 * 28
                        qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=max_pixels)

                    st.info("Converting PDF to images...")
                    images = convert_pdf_to_images(pdf_path)

                    st.info("Generating embeddings for PDF pages...")
                    embeddings_list = generate_embeddings(colpali_model, colpali_processor, images, pdf_path)

                    st.info("Processing query with ColPali...")
                    query_embeddings = process_query(colpali_model, colpali_processor, query)

                    st.info("Retrieving top-k relevant images...")
                    relevant_images = get_top_k_relevant_images(colpali_processor, embeddings_list, query_embeddings, images)
                    relevant_image_paths = save_images(relevant_images, temp_dir)

                    st.info("Generating response with Qwen...")
                    output_text = generate_response(qwen_model, qwen_processor, query, relevant_image_paths)

                    st.subheader("Generated Response:")
                    st.write(output_text)

                    response_path = os.path.join(output_dir, "response.txt")
                    with open(response_path, "w") as response_file:
                        response_file.write(output_text)
                    st.success(f"Response saved to {response_path}")

                    cleanup_temp_files(relevant_image_paths)

                    del colpali_model, qwen_model
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    st.error("CUDA out of memory. Please try reducing the batch size or using a smaller model.")
                    torch.cuda.empty_cache()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a query!")
    else:
        st.info("Upload a PDF document to get started!")

if __name__ == "__main__":
    main()