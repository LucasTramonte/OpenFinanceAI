import os
import logging
from pdf2image import convert_from_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_pdf_to_images(pdf_path):
    """Convert a PDF file to a list of images."""
    try:
        images = convert_from_path(pdf_path)
        logger.info(f"PDF converted to {len(images)} pages.")
        return images
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        raise

def get_pdf_hash(pdf_path):
    """Generate a unique hash for the given PDF file."""
    import hashlib
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()