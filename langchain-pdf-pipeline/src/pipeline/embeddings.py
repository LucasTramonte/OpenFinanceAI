import os
import pickle
import hashlib
from pdf2image import convert_from_path
from langchain_community.embeddings import HuggingFaceEmbeddings

OUTPUT_DIRECTORY = "../Assets/output"

def get_pdf_hash(pdf_path):
    """Generate a unique hash for the given PDF file."""
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def convert_pdf_to_images(pdf_path):
    """Convert a PDF file to a list of images."""
    images = convert_from_path(pdf_path)
    return images

def generate_embeddings(images, pdf_path, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Generate embeddings for the given images using HuggingFaceEmbeddings."""
    pdf_hash = get_pdf_hash(pdf_path)
    index_file = os.path.join(OUTPUT_DIRECTORY, f"document_embeddings_{pdf_hash}.pkl")

    # Load cached embeddings if they exist
    if os.path.exists(index_file):
        with open(index_file, "rb") as f:
            embeddings_list = pickle.load(f)
        return embeddings_list

    # Initialize the HuggingFaceEmbeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    # Generate embeddings for each image
    embeddings_list = []
    for image in images:
        # Convert the image to text (if OCR is needed, implement it here)
        text = image_to_text(image)  # Placeholder function for OCR
        embedding = embeddings_model.embed_query(text)
        embeddings_list.append(embedding)

    # Save embeddings to a file for future use
    with open(index_file, "wb") as f:
        pickle.dump(embeddings_list, f)

    return embeddings_list

def image_to_text(image):
    """Placeholder function to convert an image to text using OCR."""
    # Implement OCR logic here if needed (e.g., using pytesseract)
    return "Extracted text from image"

def process_pdf_embeddings(pdf_path, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Process a PDF file and generate embeddings for its content."""
    images = convert_pdf_to_images(pdf_path)
    embeddings_list = generate_embeddings(images, pdf_path, model_name)
    return embeddings_list