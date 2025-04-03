import os
import hashlib

def generate_pdf_hash(pdf_path):
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def save_to_file(file_path, data):
    with open(file_path, "w") as f:
        f.write(data)

def read_from_file(file_path):
    with open(file_path, "r") as f:
        return f.read()

def create_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)