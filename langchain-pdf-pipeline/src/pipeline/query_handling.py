import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub  # Use Hugging Face LLMs via Hugging Face Hub

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
        vector_store = FAISS.from_documents(split_docs, self.embeddings)
        return vector_store

    def initialize_llm(self):
        """Initialize a Hugging Face LLM with the token read from a file."""
        # Adjusted relative path to the token file
        token_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "token.txt"))
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"Hugging Face token file not found at: {token_path}")
        
        with open(token_path, "r") as token_file:
            HF_TOKEN = token_file.read().strip()

        # Set the token as an environment variable
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

        # Initialize the Hugging Face LLM
        return HuggingFaceHub(
            repo_id="google/flan-t5-base",  # Replace with your preferred Hugging Face model
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )

    def get_relevant_indices(self, query: str):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        retrieval_qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever)
        results = retrieval_qa({"query": query})
        return results['result']

    def handle_query(self, query: str):
        relevant_docs = self.get_relevant_indices(query)
        return relevant_docs