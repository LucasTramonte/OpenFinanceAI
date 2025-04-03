from pipeline.query_handling import QueryHandler
from pipeline.response_generation import generate_responses
from utils.logging import setup_logging
from pipeline.embeddings import process_pdf_embeddings

def main():
    setup_logging()
    
    pdf_path = "../Assets/data_test/AMEX_EMR_2023.pdf"
    query = "What percentage of women occupy leadership positions in the company in 2023?"

    # Get embeddings + images in memory (instead of saving images)
    embeddings_list, image_dict = process_pdf_embeddings(pdf_path)

    # Initialize QueryHandler
    query_handler = QueryHandler(pdf_path, top_k=3)
    
    # Get relevant documents
    relevant_images = query_handler.handle_query(query, image_dict)  # Pass images in-memory

    if not relevant_images:
        print("No relevant images found for the query.")
        return

    response = generate_responses(query, relevant_images)
    
    print("Generated Response:")
    print(response)


if __name__ == "__main__":
    main()