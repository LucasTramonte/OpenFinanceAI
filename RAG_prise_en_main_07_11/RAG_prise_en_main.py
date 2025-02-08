from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
import os
import logging 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with your Hugging Face API key
MYHFKEY = os.getenv("MYHFKEY_HUGGINGFACEHUB_API_TOKEN", "YOUR_DEFAULT_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = MYHFKEY

def read_pdfs_from_folder(folder_path: str):
    """Read all PDF files from the specified folder and extract text."""
    all_content = []
    metadata = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            try:
                reader = PdfReader(pdf_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():  # If the text is not empty
                        all_content.append(text)
                        metadata.append({"file_name": file_name, "page_number": i + 1, "pdf_path": pdf_path})
            except Exception as e:
                logging.error(f"Error reading file {file_name}: {e}")
    return all_content, metadata

def split_text_into_chunks(texts: list, metadata: list, chunk_size: int = 200, chunk_overlap: int = 20):
    """Split texts into smaller chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    final_content = []
    chunk_metadata = []
    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        final_content.extend(chunks)
        chunk_metadata.extend([metadata[i]] * len(chunks))  # Associate metadata with chunks
    return final_content, chunk_metadata

def create_embeddings(model_name: str):
    """Create embeddings using the specified model."""
    return HuggingFaceEmbeddings(model_name=model_name)

def create_vector_store(texts: list, embeddings):
    """Create a vector store with FAISS."""
    return FAISS.from_texts(texts, embeddings)

def configure_llm(repo_id: str, temperature: float = 1e-10):
    """Configure the language model (LLM)."""
    return HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": temperature})

def load_qa_chain_with_llm(llm):
    """Load the question-answering chain with the specified LLM."""
    return load_qa_chain(llm, chain_type="stuff")

def ask_questions(questions: list, doc_search, qa_chain):
    """Ask a list of questions and print the answers."""
    for question in questions:
        docs = doc_search.similarity_search(question)
        answer = qa_chain.run(input_documents=docs, question=question)
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}\n")

def main():
    data_folder = os.getenv("DATA_FOLDER", "../data/")
    all_content, metadata = read_pdfs_from_folder(data_folder)

    if not all_content:
        raise ValueError("No valid content found in the PDF files in the folder.")

    final_content, chunk_metadata = split_text_into_chunks(all_content, metadata)

    embeddings = create_embeddings("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    doc_search = create_vector_store(final_content, embeddings)

    llm = configure_llm(repo_id="google/flan-t5-large")
    qa_chain = load_qa_chain_with_llm(llm)

    questions = [
        "What was the growth in workforce in 2023?",
        "What is the revenue of EDF Renewables in 2023?"
    ]

    ask_questions(questions, doc_search, qa_chain)

if __name__ == "__main__":
    main()

# Key components of LangChain for RAG
# To build a RAG system with LangChain, we need the following components:
#
# - `Document Loaders`: Load documents from various sources (PDFs, websites, databases, etc.).
# - `Text Splitters`: Split documents into smaller chunks for more efficient handling.
# - `Embeddings`: Convert text into numerical vectors to enable semantic search.
# - `Vector Stores`: Store embeddings for quick retrieval.
# - `Retrievers`: Retrieve relevant documents based on a query.
# - `LLMs (Language Models)`: Generate responses based on the retrieved documents.
# - `Chains`: Orchestrate the flow of information between components.