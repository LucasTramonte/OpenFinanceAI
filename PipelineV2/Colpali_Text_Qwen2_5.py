import os
import torch
import json
import numpy as np
import re
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_similarity_map
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
import matplotlib.pyplot as plt
import pickle
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directories
OUTPUT_DIRECTORY = "../Assets/output"
SIMILARITY_DIR = os.path.join(OUTPUT_DIRECTORY, "similarity_maps")
RELEVANT_DIR = os.path.join(OUTPUT_DIRECTORY, "relevant_documents")
os.makedirs(SIMILARITY_DIR, exist_ok=True)
os.makedirs(RELEVANT_DIR, exist_ok=True)

# Financial terms dictionary with weights
FINANCIAL_TERMS = {
    # Accounting/financial metrics
    "revenue": 2.5, "sales": 2.0, "profit": 2.5, "margin": 2.0, "ebitda": 3.0, 
    "earnings": 2.5, "eps": 3.0, "income": 2.0, "cash flow": 2.5, "fcf": 2.5,
    "roi": 2.5, "roa": 2.5, "roe": 2.5, "dividend": 2.0, "yield": 1.5,
    "debt": 2.0, "liability": 2.0, "asset": 2.0, "equity": 2.0, "capital": 1.8,
    "balance sheet": 2.5, "p/e": 2.5, "gross profit": 2.5,
    
    # Financial statements sections
    "statement": 1.8, "annual report": 2.0, "quarterly": 1.8, "fiscal year": 1.8,
    "consolidated": 1.5, "segment": 2.0, "operating": 1.5,
    
    # Percentages and numeric indicators
    "%": 2.0, "percent": 2.0, "ratio": 2.0, "growth": 2.0, "increase": 1.5, 
    "decrease": 1.5, "million": 1.5, "billion": 1.5, "dollar": 1.5,
    
    # Company specific terms
    "emerson": 1.5, "copeland": 1.8, "aspentech": 2.0, "ni": 1.5,
}

# Words to downweight
DOWNWEIGHTED_TERMS = {
    "what": 0.3, "how": 0.3, "why": 0.3, "when": 0.3, "where": 0.3, "who": 0.3,
    "the": 0.2, "a": 0.2, "an": 0.2, "of": 0.3, "in": 0.3, "on": 0.3, "at": 0.3,
    "to": 0.3, "for": 0.3, "with": 0.3, "by": 0.3, "as": 0.3, "that": 0.3,
}

def load_model_and_processor():
    """Load the ColPALI model and processor."""
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    ).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
    return model, processor

def convert_pdf_to_images(pdf_path):
    """Convert a PDF file to a list of images."""
    images = convert_from_path(pdf_path)
    logger.info(f"PDF converted to {len(images)} pages.")
    return images

def load_ocr_data(pdf_path):
    """Load OCR data from processed JSON if available."""
    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = f"../Assets/output/ocr_data/ocr_data/{base_name}_ocr_data.json"
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                ocr_data = json.load(f)
            logger.info(f"OCR data loaded from {json_path}")
            return ocr_data
        except Exception as e:
            logger.warning(f"Failed to load OCR data: {e}")
    
    logger.warning(f"OCR data file not found at {json_path}")
    return None

def extract_financial_data(text):
    """Extract financial data patterns like percentages, dollar amounts, ratios, etc."""
    if not text:
        return []
        
    financial_data = []
    
    # Pattern for percentages - don't capture groups, match whole pattern
    percentage_pattern = r'\b\d+(\.\d+)?%'
    percentages = [match for match in re.findall(percentage_pattern, text)]
    financial_data.extend([(match, 'percentage') for match in percentages])
    
    # Pattern for dollar amounts
    dollar_pattern = r'\$\d+(\.\d+)?\s*(million|billion|trillion|M|B|T)?'
    dollars = [match for match in re.findall(dollar_pattern, text)]
    financial_data.extend([(match, 'dollar') for match in dollars])
    
    # Pattern for numeric ratios
    ratio_pattern = r'\b\d+(\.\d+)?[:/]\d+(\.\d+)?\b'
    ratios = [match for match in re.findall(ratio_pattern, text)]
    financial_data.extend([(match, 'ratio') for match in ratios])
    
    return financial_data

def get_pdf_hash(pdf_path):
    """Generate a unique hash for the given PDF file."""
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def simple_tokenize(text):
    """Simple tokenization function that doesn't rely on NLTK."""
    if not text:
        return ["placeholder"]  # Ensure at least one token
    
    # Remove punctuation and lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace and filter empty strings
    tokens = [t for t in text.split() if t]
    
    # Return at least one token to avoid empty documents
    return tokens if tokens else ["placeholder"]

class SimpleBM25:
    """A simplified BM25 implementation that avoids NumPy array boolean evaluation issues."""
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.doc_freqs = {}
        self.doc_len = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_len) / max(1, len(self.corpus))
        self.N = len(corpus)
        
        # Calculate document frequencies
        for doc in corpus:
            for term in set(doc):
                if term not in self.doc_freqs:
                    self.doc_freqs[term] = 0
                self.doc_freqs[term] += 1
    
    def get_scores(self, query):
        """Score each document based on query terms."""
        scores = [0] * self.N
        
        # If empty corpus or query, return zeros
        if len(query) == 0 or self.N == 0:
            return scores
        
        k1 = 1.5
        b = 0.75
        
        for term in query:
            if term not in self.doc_freqs:
                continue
                
            df = self.doc_freqs[term]
            # Use standard log formulation for IDF
            idf = max(0, np.log((self.N - df + 0.5) / (df + 0.5)))
            
            for i, doc in enumerate(self.corpus):
                term_freq = doc.count(term)
                if term_freq == 0:
                    continue
                    
                doc_len = self.doc_len[i]
                numerator = idf * term_freq * (k1 + 1)
                denominator = term_freq + k1 * (1 - b + b * doc_len / max(1, self.avg_doc_len))
                scores[i] += numerator / max(0.001, denominator)  # Avoid division by zero
        
        return scores

def build_bm25_index(pages_text):
    """Build a BM25 index for the document pages."""
    try:
        # Ensure all pages have at least some content
        tokenized_pages = [simple_tokenize(page) for page in pages_text]
        
        # Try using the rank_bm25 library first
        try:
            from rank_bm25 import BM25Okapi
            return BM25Okapi(tokenized_pages)
        except Exception as e:
            logger.warning(f"Failed to use BM25Okapi: {e}. Using SimpleBM25 fallback.")
            return SimpleBM25(tokenized_pages)
            
    except Exception as e:
        logger.error(f"Error building BM25 index: {e}")
        # Return a dummy BM25 that returns zeros
        return SimpleBM25([[]])

def generate_hybrid_embeddings(model, processor, images, ocr_data, pdf_path):
    """
    Generate hybrid embeddings (dense embeddings + financial metadata) for document pages.
    """
    pdf_hash = get_pdf_hash(pdf_path)
    index_file = os.path.join(OUTPUT_DIRECTORY, f"hybrid_embeddings_{pdf_hash}.pkl")

    if os.path.exists(index_file):
        try:
            with open(index_file, "rb") as f:
                hybrid_data = pickle.load(f)
            logger.info(f"Hybrid embeddings loaded from cache for {pdf_path}.")
            return hybrid_data
        except Exception as e:
            logger.warning(f"WARNING: Failed to load hybrid embeddings. Regenerating... (Error: {e})")

    # Step 1: Generate dense embeddings
    logger.info(f"Generating dense embeddings for {pdf_path}...")
    dataloader = DataLoader(
        dataset=images,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x)
    )

    dense_embeddings = []
    for batch in tqdm(dataloader, desc="Generating dense embeddings"):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embeddings = model(**batch)
        dense_embeddings.extend(embeddings.cpu().unbind())

    # Step 2: Extract text for each page for BM25 indexing
    pages_text = []
    financial_data_by_page = []
    tables_by_page = []
    
    if ocr_data and "pages" in ocr_data:
        for i, page in enumerate(ocr_data["pages"]):
            if i >= len(dense_embeddings):
                break  # Don't process more pages than we have embeddings for
            
            # Ensure we have text for every page, fallback to empty string if missing
            page_text = page.get("text", "")
            pages_text.append(page_text)
            
            # Extract financial data from page text
            financial_data_by_page.append(extract_financial_data(page_text))
            
            # Get tables if available
            tables_by_page.append(page.get("tables", []))
    else:
        # Fallback if OCR data not available
        pages_text = ["placeholder"] * len(dense_embeddings)
        financial_data_by_page = [[] for _ in range(len(dense_embeddings))]
        tables_by_page = [[] for _ in range(len(dense_embeddings))]
    
    # Ensure we have text for each page with embeddings
    while len(pages_text) < len(dense_embeddings):
        pages_text.append("placeholder")
        financial_data_by_page.append([])
        tables_by_page.append([])
    
    # Step 3: Build BM25 index
    bm25_index = build_bm25_index(pages_text)
    
    # Create hybrid embeddings structure
    hybrid_data = {
        "dense_embeddings": dense_embeddings,
        "pages_text": pages_text,
        "bm25_index": bm25_index,
        "financial_data": financial_data_by_page,
        "tables": tables_by_page
    }
    
    # Save hybrid embeddings
    with open(index_file, "wb") as f:
        pickle.dump(hybrid_data, f)
    
    logger.info(f"Hybrid embeddings saved for {pdf_path}.")
    return hybrid_data

def get_relevant_indices_hybrid(model, processor, query, hybrid_data, top_k, 
                              dense_weight=0.6, bm25_weight=0.3, financial_weight=0.1):
    """
    Retrieve the indices of the top-k relevant pages using a hybrid approach.
    """
    # 1. Get dense embedding scores
    batch_queries = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
    
    dense_scores = processor.score_multi_vector(
        query_embeddings, 
        hybrid_data["dense_embeddings"]
    )[0].tolist()
    
    # 2. Get BM25 scores with improved token weighting
    query_tokens = simple_tokenize(query)
    
    # Apply financial term weighting to query with better handling
    weighted_query = []
    for token in query_tokens:
        token_lower = token.lower()
        
        # Handle multi-word financial terms
        found_multiword = False
        for term in [t for t in FINANCIAL_TERMS if " " in t]:  # Check multi-word terms first
            if term in query.lower():
                term_tokens = simple_tokenize(term)
                if token_lower in term_tokens:
                    weight = int(FINANCIAL_TERMS[term] * 10)
                    weighted_query.extend([token] * weight)
                    found_multiword = True
                    break
        
        if found_multiword:
            continue
            
        # Handle single-word terms
        if token_lower in FINANCIAL_TERMS:
            weight = int(FINANCIAL_TERMS[token_lower] * 10)
            weighted_query.extend([token] * weight)
        elif token_lower not in DOWNWEIGHTED_TERMS:
            weighted_query.append(token)
    
    if not weighted_query:  # Fallback if no tokens remain after filtering
        weighted_query = query_tokens
    
    # Safely get BM25 scores - wrap in try/except for better error handling
    try:
        bm25_scores = hybrid_data["bm25_index"].get_scores(weighted_query)
        
        # Convert numpy arrays to list if needed
        if hasattr(bm25_scores, 'tolist'):
            bm25_scores = bm25_scores.tolist()
            
        # Ensure we have scores for all documents
        if len(bm25_scores) < len(dense_scores):
            bm25_scores.extend([0] * (len(dense_scores) - len(bm25_scores)))
        
        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores) if bm25_scores else 1
        if max_bm25 > 0:
            bm25_scores = [score / max_bm25 for score in bm25_scores]
        else:
            bm25_scores = [0] * len(bm25_scores)
    except Exception as e:
        logger.warning(f"Error getting BM25 scores: {e}. Using zeros.")
        bm25_scores = [0] * len(dense_scores)
    
    # 3. Calculate financial relevance scores with more nuance
    financial_scores = [0] * len(dense_scores)
    
    # Enhanced financial term detection
    query_lower = query.lower()
    
    # Check for percentage questions specifically
    percentage_question = any(term in query_lower for term in ["percent", "%", "proportion", "ratio", "share"])
    
    # Check for financial terms in query (include multi-word terms)
    query_financial_terms = []
    for term in FINANCIAL_TERMS:
        if term.lower() in query_lower:
            query_financial_terms.append(term)
    
    if query_financial_terms or percentage_question:
        for i, (page_text, financial_data) in enumerate(zip(hybrid_data["pages_text"], hybrid_data["financial_data"])):
            if i >= len(financial_scores):
                break
            
            page_lower = page_text.lower()
            
            # Score based on financial data types
            percentage_data = [d for d in financial_data if d[1] == 'percentage']
            dollar_data = [d for d in financial_data if d[1] == 'dollar']
            ratio_data = [d for d in financial_data if d[1] == 'ratio']
            
            # Give higher scores to pages with percentages if question asks about percentages
            if percentage_question and percentage_data:
                financial_scores[i] += min(1.0, len(percentage_data) * 0.2)
            
            # Score based on all financial data
            financial_scores[i] += min(1.0, len(financial_data) * 0.1)  # Cap at 1.0
            
            # Score based on financial terms
            for term in query_financial_terms:
                if term.lower() in page_lower:
                    financial_scores[i] += FINANCIAL_TERMS.get(term, 1.0) * 0.2
                    
                    # Bonus for terms appearing near financial data
                    term_pos = page_lower.find(term.lower())
                    if term_pos > 0:
                        context = page_lower[max(0, term_pos-50):min(len(page_lower), term_pos+50)]
                        # Fixed check for financial data in context
                        for f in financial_data:
                            # Make sure we're dealing with strings
                            match_text = str(f[0])
                            if match_text in context:
                                financial_scores[i] += 0.3  # Bonus for term-data proximity
                                break
            
            # Check for tables (finance documents often contain important tables)
            if i < len(hybrid_data["tables"]) and len(hybrid_data["tables"][i]) > 0:
                financial_scores[i] += 0.3
    
    # 4. Combine scores
    combined_scores = []
    for i in range(len(dense_scores)):
        score = (
            dense_weight * dense_scores[i] +
            bm25_weight * (bm25_scores[i] if i < len(bm25_scores) else 0) +
            financial_weight * (financial_scores[i] if i < len(financial_scores) else 0)
        )
        combined_scores.append(score)
    
    # 5. Get top-k indices (ensure we don't exceed the number of pages)
    top_k = min(top_k, len(combined_scores))
    top_indices = np.argsort(combined_scores)[-top_k:][::-1].tolist()
    
    # Log scores for debugging
    logger.info(f"Top {top_k} pages with combined scores:")
    for i, idx in enumerate(top_indices):
        logger.info(f"Rank {i+1}: Page {idx+1} - Dense: {dense_scores[idx]:.3f}, "
                   f"BM25: {bm25_scores[idx] if idx < len(bm25_scores) else 0:.3f}, "
                   f"Financial: {financial_scores[idx] if idx < len(financial_scores) else 0:.3f}, "
                   f"Combined: {combined_scores[idx]:.3f}")
    
    return top_indices

def save_similarity_scores_and_maps(images, hybrid_data, top_k_indices, query, model, processor):
    """Save the similarity scores and maps for the top-k relevant pages."""
    similarity_scores_path = os.path.join(OUTPUT_DIRECTORY, "similarity_scores.txt")
    with open(similarity_scores_path, "w") as score_file:
        for i, idx in enumerate(top_k_indices):
            # Skip if index is out of range
            if idx >= len(images):
                logger.warning(f"Index {idx} out of range for images list (length {len(images)})")
                continue
                
            image = images[idx]
            
            # Skip if index is out of range for embeddings
            if idx >= len(hybrid_data["dense_embeddings"]):
                logger.warning(f"Index {idx} out of range for embeddings list")
                continue
                
            embeddings = hybrid_data["dense_embeddings"][idx]

            # Save relevant image
            relevant_path = os.path.join(RELEVANT_DIR, f"relevant_doc_{i + 1}.jpg")
            image.save(relevant_path)

            # Generate and save similarity maps
            try:
                n_patches = processor.get_n_patches(image_size=image.size, patch_size=model.patch_size, spatial_merge_size=2)
                image_mask = processor.get_image_mask(processor.process_images([image]))
                batch_queries = processor.process_queries([query]).to(model.device)
                
                with torch.no_grad():
                    query_embeddings = model(**batch_queries)

                batched_similarity_maps = get_similarity_maps_from_embeddings(
                    image_embeddings=embeddings.unsqueeze(0).to("cuda"),
                    query_embeddings=query_embeddings,
                    n_patches=n_patches,
                    image_mask=image_mask,
                )

                query_tokens = processor.tokenizer.tokenize(
                    processor.decode(batch_queries.input_ids[0]).replace(processor.tokenizer.pad_token, "").strip()
                )

                similarity_maps = batched_similarity_maps[0]
                for token_idx, similarity_map in enumerate(similarity_maps[:min(len(query_tokens), len(similarity_maps))]):
                    max_sim_score = similarity_map.max().item()
                    fig, ax = plot_similarity_map(
                        image=image,
                        similarity_map=similarity_map,
                        figsize=(8, 8),
                        show_colorbar=True,
                    )
                    ax.set_title(
                        f"Token #{token_idx + 1}: `{query_tokens[token_idx].replace('Ġ', '_')}`. MaxSim score: {max_sim_score:.2f}",
                        fontsize=10,
                    )
                    fig.tight_layout()
                    fig_path = os.path.join(SIMILARITY_DIR, f"doc_{i + 1}_token_{token_idx + 1}.png")
                    fig.savefig(fig_path, dpi=100)
                    plt.close(fig)  # Close the figure to free up memory

                    # Save similarity score to text file
                    score_file.write(
                        f"Document {i + 1}, Token #{token_idx + 1} (`{query_tokens[token_idx]}`): MaxSim score = {max_sim_score:.2f}\n"
                    )
            except Exception as e:
                logger.error(f"Error generating similarity maps for page {idx+1}: {e}")
                
    logger.info(f"Similarity scores saved in {similarity_scores_path}")

def index_and_save_documents(pdf_path: str, query: str, top_k: int = 3,
                          dense_weight=0.6, bm25_weight=0.3, financial_weight=0.1):
    """Index the PDF document and save the top-k relevant pages and their similarity maps."""
    try:
        model, processor = load_model_and_processor()
        images = convert_pdf_to_images(pdf_path)
        
        # Load OCR data if available
        ocr_data = load_ocr_data(pdf_path)
        
        # Generate hybrid embeddings
        hybrid_data = generate_hybrid_embeddings(model, processor, images, ocr_data, pdf_path)
        
        # Get relevant indices using hybrid approach
        top_k_indices = get_relevant_indices_hybrid(
            model, processor, query, hybrid_data, top_k,
            dense_weight, bm25_weight, financial_weight
        )
        
        # Save similarity scores and maps
        save_similarity_scores_and_maps(images, hybrid_data, top_k_indices, query, model, processor)
        
        return RELEVANT_DIR
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

def generate_responses(query: str, relevant_dir: str, top_k: int = 3):
    """Generate responses based on the top-k relevant pages."""
    try:
        # Load the model and processor
        gen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16
        ).cuda().eval()
        max_pixels = 512 * 28 * 28
        gen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=max_pixels)

        # Load the relevant documents
        relevant_files = sorted(os.listdir(relevant_dir))[:top_k]
        image_paths = [f"file://{os.path.abspath(os.path.join(relevant_dir, file_name))}" for file_name in relevant_files]

        logger.info(f"Number of images passed: {len(image_paths)}")

        # Enhance prompt with financial context awareness
        financial_terms_in_query = [term for term in FINANCIAL_TERMS if term.lower() in query.lower()]
        financial_context = ""
        if financial_terms_in_query or any(term in query.lower() for term in ["percent", "%", "proportion", "ratio"]):
            financial_context = "\nPay special attention to financial metrics, percentages, tables with data, and performance indicators. "
            financial_context += "Ensure any numerical values are accurately reported from the documents."
            
        PROMPT = f"Use the following pages to answer the query:{financial_context}\n{query}\n"

        # Prepare the messages with multiple images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_paths[0]},
                    {"type": "image", "image": image_paths[1]},
                    {"type": "image", "image": image_paths[2]},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = gen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = gen_processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        # Generate a unique response
        torch.cuda.empty_cache()
        generated_ids = gen_model.generate(**inputs, max_new_tokens=150)
        output_text = gen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        logger.info("Final response:")
        logger.info(output_text)

        # Save the generated response
        response_path = os.path.join(OUTPUT_DIRECTORY, "generated_responses.txt")
        with open(response_path, "w") as f:
            f.write(output_text)
        logger.info(f"Generated responses saved in {response_path}")

    except Exception as e:
        logger.error(f"An error occurred during response generation: {e}")
        raise

def main():
    pdf_path = "../Assets/data_test/pdfs/AMEX_EMR_2023.pdf"
    query = "What percentage of women occupy leadership positions in the company in 2023?"
    
    # Configurable weights for the hybrid retrieval approach
    dense_weight = 0.6    # Weight for dense embedding scores
    bm25_weight = 0.3     # Weight for sparse BM25 scores
    financial_weight = 0.1 # Weight for financial terms and structures
    
    relevant_dir = index_and_save_documents(
        pdf_path, query, top_k=3,
        dense_weight=dense_weight, 
        bm25_weight=bm25_weight, 
        financial_weight=financial_weight
    )
    generate_responses(query, relevant_dir, top_k=3)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    main()