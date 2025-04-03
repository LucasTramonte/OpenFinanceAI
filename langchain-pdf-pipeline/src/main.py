import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.query_handling import QueryHandler
from pipeline.response_generation import generate_responses
from utils.logging import setup_logging

def main():
    setup_logging()
    
    pdf_path = "../Assets/data_test/AMEX_EMR_2023.pdf"
    query = "What percentage of women occupy leadership positions in the company in 2023?"
    
    # Initialize QueryHandler
    query_handler = QueryHandler(pdf_path, top_k=3)
    
    # Get relevant documents
    relevant_docs = query_handler.get_relevant_indices(query)
    
    # Map relevant_docs to valid file paths (if they are indices or invalid references)
    relevant_images = []
    for doc in relevant_docs:
        if isinstance(doc, str) and os.path.exists(doc):  # Check if it's a valid file path
            relevant_images.append(doc)
        elif isinstance(doc, int):  # If it's an index, map it to a file path
            image_path = f"../Assets/output/page_{doc}.png"  # Adjust this path based on your setup
            if os.path.exists(image_path):
                relevant_images.append(image_path)
            else:
                print(f"Invalid document reference: {doc}")
        else:
            print(f"Invalid document reference: {doc}")
    
    # Generate responses based on relevant images
    response = generate_responses(query, relevant_images)
    
    # Print the generated response
    print("Generated Response:")
    print(response)

if __name__ == "__main__":
    main()