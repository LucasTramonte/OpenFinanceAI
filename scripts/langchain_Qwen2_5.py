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
import tempfile

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directories
OUTPUT_DIRECTORY = "../Assets/output"
SIMILARITY_DIR = os.path.join(OUTPUT_DIRECTORY, "similarity_maps")
RELEVANT_DIR = os.path.join(OUTPUT_DIRECTORY, "relevant_documents")
os.makedirs(SIMILARITY_DIR, exist_ok=True)
os.makedirs(RELEVANT_DIR, exist_ok=True)


def langchain_retrieve_top_docs(pdf_path: str, query: str, top_k: int = 3):
    logger.info("Running LangChain-based retrieval...")

    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    relevant_docs = retriever.get_relevant_documents(query)

    return [doc.page_content for doc in relevant_docs]

def generate_response_from_text(query: str, context_chunks: list[str]):
    try:
        # Load Qwen2.5 model and processor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16
        ).cuda().eval()

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        # Build prompt
        context = "\n\n".join(context_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # Text-only input
        inputs = processor(text=prompt, return_tensors="pt").to("cuda")

        # Generate response
        torch.cuda.empty_cache()
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract only the final answer (remove echo)
        # Look for 'Answer:' or just grab the last paragraph/sentence
        if "Answer:" in output_text:
            final_answer = output_text.split("Answer:")[-1].strip()
        elif "Therefore," in output_text:
            final_answer = output_text.split("Therefore,")[-1].strip()
        else:
            final_answer = output_text.strip()

        logger.info("Final answer:")
        logger.info(final_answer)

        # Save only final answer to file
        response_path = os.path.join(OUTPUT_DIRECTORY, "langchain_qwen_answer.txt")
        with open(response_path, "w") as f:
            f.write(final_answer)

        logger.info(f"Answer saved at: {response_path}")

    except Exception as e:
        logger.error(f"Error in response generation: {e}")
        raise


def main():
    pdf_path = "../Assets/data_test/pdfs/AMEX_EMR_2023.pdf"
    query = "What percentage of women occupy leadership positions in the company in 2023?"

    top_chunks = langchain_retrieve_top_docs(pdf_path, query, top_k=3)
    generate_response_from_text(query, top_chunks)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    main()
