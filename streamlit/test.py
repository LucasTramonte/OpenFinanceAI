import os
import streamlit as st
import torch
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader
from tqdm import tqdm
import hashlib
import pickle

# Set environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Page configuration
st.set_page_config(layout="wide", page_title="ColPali & Qwen Chat App")

# Set directories to save files
base_dir = "./streamlit"
OUTPUT_DIRECTORY = "../Assets/output"
pdf_dir = os.path.join(base_dir, "uploaded_pdfs")
temp_dir = os.path.join(base_dir, "temp_files")
output_dir = os.path.join(base_dir, "output")
index_dir = os.path.join(base_dir, "index")
for d in [pdf_dir, temp_dir, output_dir, index_dir]:
    os.makedirs(d, exist_ok=True)

st.title("ColPali and Qwen Multimodal RAG Chat App")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    colpali_model_name = st.selectbox(
        "Select ColPali Model", ["vidore/colqwen2-v1.0"], index=0
    )
    qwen_model_name = st.selectbox(
        "Select Generation Model",
        ["vidore/colqwen2-base", "Qwen/Qwen2.5-VL-3B-Instruct", "google/gemma-3-4b-it"],
        index=0,
    )
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
    index_file = os.path.join(OUTPUT_DIRECTORY, f"document_embeddings_{pdf_hash}.pkl")
    if os.path.exists(index_file):
        try:
            with open(index_file, "rb") as f:
                embeddings_list = pickle.load(f)
            return embeddings_list
        except Exception as e:
            st.warning(
                f"Failed to load embeddings for {pdf_path}. Regenerating... (Error: {e})"
            )
    # Generate embeddings if not cached
    dataloader = DataLoader(
        dataset=images,
        batch_size=1,
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
    return embeddings_list


def process_query(model, processor, query):
    batch_queries = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
    return query_embeddings


def get_top_k_relevant_images(
    processor, embeddings_list, query_embeddings, images, k=3
):
    scores = processor.score_multi_vector(query_embeddings, embeddings_list)
    top_k_indices = scores[0].topk(k).indices.tolist()
    return [images[idx] for idx in top_k_indices]


def save_images(images, save_dir):
    image_paths = []
    for i, img in enumerate(images):
        temp_image_path = os.path.join(save_dir, f"temp_image_{i}.png")
        img.save(temp_image_path)
        image_paths.append(
            temp_image_path
        )  # Simplifier pour l'affichage dans Streamlit
    return image_paths


def generate_response(
    qwen_model, qwen_processor, query, relevant_image_paths, model_name
):
    """Generate a response based on the selected model, query and relevant images."""
    if "gemma" in model_name.lower():
        # Gemma model approach
        prompt = f"Use the following pages to answer the query:\n{query}"
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": img_path}
                        for img_path in relevant_image_paths
                    ],
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = qwen_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(qwen_model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = qwen_model.generate(
                **inputs, max_new_tokens=150, do_sample=False
            )
            generation = generation[0][input_len:]
        output_text = qwen_processor.decode(generation, skip_special_tokens=True)
    else:
        # Qwen models approach
        prompt = f"Use the following pages to answer the query:\n{query}"
        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": img_path}
                        for img_path in relevant_image_paths
                    ],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = qwen_processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        torch.cuda.empty_cache()
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=100)
        output_text = qwen_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return clean_response(output_text)


def clean_response(response_text):
    """
    Nettoie la réponse pour ne garder que la partie finale (adaptable selon vos besoins).
    Par exemple, on peut prendre la dernière ligne non vide.
    """
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
    return lines[-1] if lines else response_text


def cleanup_temp_files(file_paths):
    for file_path in file_paths:
        os.remove(file_path)


# --- Main conversation interface ---
def main():
    # Initialiser l'historique de conversation dans la session
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False

    # Traitement du PDF et chargement des modèles (une seule fois)
    if pdf_file is not None:
        if "pdf_path" not in st.session_state:
            st.session_state.pdf_path = save_uploaded_file(pdf_file, pdf_dir)
            st.success(f"Uploaded file: {pdf_file.name}")

        if not st.session_state.models_loaded:
            with st.spinner("Loading models and processing PDF..."):
                colpali_model = ColQwen2.from_pretrained(
                    colpali_model_name, torch_dtype=torch.bfloat16, device_map="cuda:0"
                ).eval()
                colpali_processor = ColQwen2Processor.from_pretrained(
                    colpali_model_name
                )
                images = convert_pdf_to_images(st.session_state.pdf_path)
                embeddings_list = generate_embeddings(
                    colpali_model, colpali_processor, images, st.session_state.pdf_path
                )
            st.session_state.colpali_model = colpali_model
            st.session_state.colpali_processor = colpali_processor
            st.session_state.images = images
            st.session_state.embeddings_list = embeddings_list

            with st.spinner("Loading response model..."):
                max_pixels = 512 * 28 * 28
                # Charger le modèle de génération sélectionné
                if qwen_model_name == "vidore/colqwen2-base":
                    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        qwen_model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="cuda:0",
                        ignore_mismatched_sizes=True,
                    ).eval()
                    qwen_processor = AutoProcessor.from_pretrained(
                        "Qwen/Qwen2-VL-2B-Instruct", max_pixels=max_pixels
                    )
                elif qwen_model_name == "google/gemma-3-4b-it":
                    qwen_model = Gemma3ForConditionalGeneration.from_pretrained(
                        qwen_model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    ).eval()
                    qwen_processor = AutoProcessor.from_pretrained(qwen_model_name)
                else:
                    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        qwen_model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="cuda:0",
                        low_cpu_mem_usage=True,
                        offload_folder="./offload",
                    ).eval()
                    qwen_processor = AutoProcessor.from_pretrained(
                        qwen_model_name, max_pixels=max_pixels
                    )
            st.session_state.qwen_model = qwen_model
            st.session_state.qwen_processor = qwen_processor
            st.session_state.models_loaded = True

        # Affichage du chat
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.conversation:
                if msg["role"] == "user":
                    st.markdown(
                        f"<div style='text-align: right; background-color:#e3f2fd; padding:10px; border-radius:10px; margin:5px 0;'>"
                        f"<strong>You:</strong> {msg['text']}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='text-align: left; background-color:#f5f5f5; padding:10px; border-radius:10px; margin:5px 0;'>"
                        f"<strong>Assistant:</strong> {msg['text']}</div>",
                        unsafe_allow_html=True,
                    )
                    if "images" in msg and msg["images"]:
                        cols = st.columns(min(3, len(msg["images"])))
                        for i, img_path in enumerate(msg["images"]):
                            with cols[i % len(cols)]:
                                st.image(
                                    img_path,
                                    use_container_width=True,
                                    caption=f"Image {i + 1}",
                                )

        st.markdown("---")

        # Zone de saisie pour la nouvelle question
        if st.session_state.waiting_for_response:
            # Clear any previous components and only show the spinner
            st.markdown("<div style='height: 70px;'></div>", unsafe_allow_html=True)
            progress_container = st.container()
            with progress_container:
                with st.spinner("Generating response..."):
                    query = st.session_state.current_query
                    query_embeddings = process_query(
                        st.session_state.colpali_model,
                        st.session_state.colpali_processor,
                        query,
                    )
                    query_embeddings = process_query(
                        st.session_state.colpali_model,
                        st.session_state.colpali_processor,
                        query,
                    )
                    relevant_images = get_top_k_relevant_images(
                        st.session_state.colpali_processor,
                        st.session_state.embeddings_list,
                        query_embeddings,
                        st.session_state.images,
                    )
                    relevant_image_paths = save_images(relevant_images, temp_dir)
                    output_text = generate_response(
                        st.session_state.qwen_model,
                        st.session_state.qwen_processor,
                        query,
                        relevant_image_paths,
                        qwen_model_name,
                    )

                    # Ajouter à la conversation
                    st.session_state.conversation.append({"role": "user", "text": query})
                    st.session_state.conversation.append(
                        {
                            "role": "assistant",
                            "text": output_text,
                            "images": relevant_image_paths,
                        }
                    )

                    # Réinitialiser l'état d'attente
                    st.session_state.waiting_for_response = False
                    st.session_state.current_query = ""

                    # Forcer la mise à jour de l'interface
                    st.rerun()
        else:
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input("", 
                                          key="chat_input", 
                                          placeholder="Type your question here...")
                submitted = st.form_submit_button("Send")
                if submitted and user_input.strip():
                    st.session_state.waiting_for_response = True
                    st.session_state.current_query = user_input.strip()
                    st.rerun()
    else:
        st.info("Upload a PDF document to get started!")


if __name__ == "__main__":
    main()
