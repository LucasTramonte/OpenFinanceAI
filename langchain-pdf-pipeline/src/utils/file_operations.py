import os
import hashlib
import logging

logger = logging.getLogger(__name__)

def generate_pdf_hash(pdf_path):
    """Generate a unique hash for a given PDF file."""
    try:
        hasher = hashlib.md5()
        with open(pdf_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    except FileNotFoundError:
        logger.error(f"File not found: {pdf_path}")
        return None

def save_to_file(file_path, data):
    """Save data to a file."""
    try:
        with open(file_path, "w") as f:
            f.write(data)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}", exc_info=True)

def read_from_file(file_path):
    """Read data from a file."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None

def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Directory created: {directory_path}")
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}", exc_info=True)
