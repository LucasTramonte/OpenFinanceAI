import os
import importlib
from pathlib import Path
from pdf2image import convert_from_path
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
ASSETS_DIR = "./Assets"
PDF_INPUT_DIR = os.path.join(ASSETS_DIR, "data_test/pdfs")
OUTPUT_DIR = os.path.join(ASSETS_DIR, "output")
MODELS_DIR = "./scripts/models"
LOG_FILE = os.path.join(OUTPUT_DIR, "query_logs.json")  # Log file for storing queries and feedback

# Ensure necessary directories exist
os.makedirs(PDF_INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_models():
    """List all available models in the models directory."""
    model_files = [
        f.replace(".py", "") for f in os.listdir(MODELS_DIR) 
        if f.endswith(".py") and f != "__init__.py"  # Exclude __init__.py
    ]
    return model_files

def load_model_script(model_name):
    """Dynamically load the selected model script."""
    try:
        model_module = importlib.import_module(f"scripts.models.{model_name}")
        return model_module
    except ModuleNotFoundError:
        logger.error(f"Error: The model '{model_name}' was not found in the models directory.")
        exit(1)

def log_query(query, response, feedback):
    """Log the query, response, and user feedback to a JSON file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "feedback": feedback
    }
    
    # Check if the log file exists and is not empty
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            # Handle the case where the file exists but is invalid
            logs = []
    else:
        logs = []

    # Append the new log entry
    logs.append(log_entry)

    # Write the updated logs back to the file
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

def main():
    print("Welcome to the OpenFinanceAI Terminal Interface!")
    print("This program retrieves the top-3 relevant documents from a PDF and generates a response using a generative model.")
    
    # Step 1: Choose a generative model
    print("\nAvailable generative models:")
    models = list_models()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    model_choice = input("\nEnter the number corresponding to the model you want to use: ")
    try:
        model_choice = int(model_choice)
        if model_choice < 1 or model_choice > len(models):
            raise ValueError
        selected_model = models[model_choice - 1]
    except ValueError:
        print("Invalid choice. Please enter a valid number.")
        exit(1)
    
    print(f"\nYou selected the model: {selected_model}")
    model_script = load_model_script(selected_model)
    
    # Step 2: Ask the user to place the PDF in the input directory
    print(f"\nPlease place your PDF file in the following directory: {PDF_INPUT_DIR}")
    input("Press Enter once you have placed the PDF file...")
    
    # List available PDFs
    pdf_files = [f for f in os.listdir(PDF_INPUT_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {PDF_INPUT_DIR}. Please add a PDF and try again.")
        exit(1)
    
    print("\nAvailable PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"{i}. {pdf}")
    
    pdf_choice = input("\nEnter the number corresponding to the PDF you want to process: ")
    try:
        pdf_choice = int(pdf_choice)
        if pdf_choice < 1 or pdf_choice > len(pdf_files):
            raise ValueError
        selected_pdf = pdf_files[pdf_choice - 1]
    except ValueError:
        print("Invalid choice. Please enter a valid number.")
        exit(1)
    
    pdf_path = os.path.join(PDF_INPUT_DIR, selected_pdf)
    print(f"\nYou selected the PDF: {selected_pdf}")
    
    # Step 3: Ask the user for a query
    query = input("\nEnter your query: ")
    print(f"\nYour query: {query}")
    
    # Step 4: Perform retrieval and response generation
    print("\nRetrieving the top-3 relevant documents...")
    relevant_dir = model_script.index_and_save_documents(pdf_path, query, top_k=3)
    
    print("\nGenerating a response using the selected model...")
    response = model_script.generate_responses(query, relevant_dir, top_k=3)
    
    print("\nProcess complete. Check the output directory for results.")
    print("\nGenerated Response:")
    print(response)
    
    # Step 5: Collect user feedback
    feedback = input("\nWas the response helpful? (👍 / 👎): ")
    while feedback not in ["👍", "👎"]:
        print("Invalid input. Please enter 👍 or 👎.")
        feedback = input("\nWas the response helpful? (👍 / 👎): ")
    
    # Log the query, response, and feedback
    log_query(query, response, feedback)
    print("\nThank you for your feedback! It has been logged.")

if __name__ == "__main__":
    main()