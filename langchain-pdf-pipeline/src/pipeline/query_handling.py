import os
import logging
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub  
#from langchain_huggingface import HuggingFaceEndpoint
from utils.logging import setup_logging

# TO DO  : this code is incorrect : results = retrieval_qa.invoke({"query": query}) is the model's answer, and not a document to be retrieved.

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()
class QueryHandler:
    def __init__(self, pdf_path: str, top_k: int = 3):
        self.pdf_path = pdf_path
        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.vector_store = self.create_vector_store()
        self.llm = self.initialize_llm()

    def create_vector_store(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Split documents: {len(split_docs)} chunks created.")
        vector_store = FAISS.from_documents(split_docs, self.embeddings)
        return vector_store

    def initialize_llm(self):
        """Initialize a Hugging Face LLM with the token read from a file."""
        token_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "token.txt"))
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"Hugging Face token file not found at: {token_path}")
        
        with open(token_path, "r") as token_file:
            HF_TOKEN = token_file.read().strip()

        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

        return HuggingFaceHub(repo_id="google/flan-t5-base" ,model_kwargs={"temperature": 0.5, "max_length": 250})

    def get_relevant_indices(self, query: str):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        retrieval_qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever)

        try:
            results = retrieval_qa.invoke({"query": query})
            logger.info(f"Raw retrieval results: {results}")  # Log raw output for debugging

            if not results or 'result' not in results:
                logger.warning("No results found from retrieval.")
                return []

            # Attempt to parse the result into a list of integers or valid references
            retrieved_docs = results.get('result', [])
            if isinstance(retrieved_docs, str):
                # Split string into potential page numbers or references
                retrieved_docs = [int(x.strip()) for x in retrieved_docs.split() if x.strip().isdigit()]
            
            logger.info(f"Processed retrieved documents: {retrieved_docs}")
            return retrieved_docs
        except Exception as e:
            logger.error(f"Error during query retrieval: {e}")
            return []


    def handle_query(self, query: str, image_dict):
        """Find relevant images directly from memory instead of the filesystem."""
        relevant_docs = self.get_relevant_indices(query)
        relevant_images = []

        for doc in relevant_docs:
            if isinstance(doc, int) and doc in image_dict:
                relevant_images.append(image_dict[doc])  
            else:
                logger.warning(f"Page {doc} not found in memory.")

        return relevant_images
