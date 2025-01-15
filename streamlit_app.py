import os
import streamlit as st
import torch
from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

#Streamlit layout
st.set_page_config(layout="wide")

#Set directories to save files
base_dir = "./streamlit"
pdf_dir = os.path.join(base_dir, "uploaded_pdfs")
temp_dir = os.path.join(base_dir, "temp_files")
output_dir = os.path.join(base_dir, "output")

os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

st.title("ColPali and Qwen Multimodal RAG App")
with st.sidebar:
    st.header("Configuration")
    colpali_model_name = st.selectbox("Select ColPali Model", ["vidore/colpali", "vidore/colpali-v1.2"], index=1)
    qwen_model_name = st.selectbox("Select Qwen Model", ["vidore/colqwen2-base"], index=0)
    pdf_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

#Main app
if pdf_file is not None:
    #Saving the uploaded PDF
    pdf_path = os.path.join(pdf_dir, pdf_file.name)
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    st.success(f"Uploaded file: {pdf_file.name}")

    query = st.text_input("Enter your query:")

    if st.button("Process Query"):
        if query:
            try:
                #Initializing models and processors
                st.info("Loading models...")
                colpali_model = ColPali.from_pretrained(colpali_model_name, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
                colpali_processor = ColPaliProcessor.from_pretrained(colpali_model_name)

                qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(qwen_model_name, torch_dtype=torch.bfloat16).cuda().eval()
                qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

                #Converting PDF to images
                st.info("Converting PDF to images...")
                images = convert_from_path(pdf_path)
                embeddings_list = []

                #Generating embeddings for each page
                st.info("Generating embeddings for PDF pages...")
                for image in images:
                    image_input = colpali_processor.process_images([image]).to(colpali_model.device)
                    with torch.no_grad():
                        embeddings = colpali_model(**image_input)
                    embeddings_list.append(embeddings.cpu())

                #Processing the query
                st.info("Processing query with ColPali...")
                query_input = colpali_processor.process_queries([query]).to(colpali_model.device)
                with torch.no_grad():
                    query_embeddings = colpali_model(**query_input)

                #Similarity scores
                embeddings_tensor = torch.cat(embeddings_list, dim=0)
                if query_embeddings.dim() != 3:
                    query_embeddings = query_embeddings.unsqueeze(0)

                scores = colpali_processor.score_multi_vector(query_embeddings, embeddings_tensor)
                top_k_indices = scores[0].topk(3).indices.tolist()

                #Retrieving top-k relevant images
                relevant_images = [images[idx] for idx in top_k_indices]
                relevant_image_paths = []
                for i, img in enumerate(relevant_images):
                    temp_image_path = os.path.join(temp_dir, f"temp_image_{i}.png")
                    img.save(temp_image_path)
                    relevant_image_paths.append(f"file://{os.path.abspath(temp_image_path)}")

                st.info("Generating response with Qwen...")
                prompt = f"""
                Use the following pages to answer the query:
                {query}
                """
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

                #Displaying the response
                st.subheader("Generated Response:")
                st.write(output_text)

                #Saving the response
                response_path = os.path.join(output_dir, "response.txt")
                with open(response_path, "w") as response_file:
                    response_file.write(output_text)
                st.success(f"Response saved to {response_path}")

                #Cleaning up temporary images
                for img_path in relevant_image_paths:
                    os.remove(img_path)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query!")
else:
    st.info("Upload a PDF document to get started!")
