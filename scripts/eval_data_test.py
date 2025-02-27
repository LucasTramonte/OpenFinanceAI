import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import BartForConditionalGeneration, BartTokenizer
import numpy as np
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
file_path = '../Assets/data_test/ardian_dataset_test.csv'
df = pd.read_csv(file_path)

# Initialize ROUGE scorer and BART model
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Function to evaluate answers using all metrics
def evaluate_answers(row):
    results = {}
    expected = str(row['Expected_Answer'])
    
    # TF-IDF Vectorizer for cosine similarity
    vectorizer = TfidfVectorizer().fit_transform([expected] + [str(row[model]) for model in ['Answer_Qwen2', 'Answer_Qwen2.5', 'Answer_OpenGVLab']])
    vectors = vectorizer.toarray()
    expected_vector = vectors[0]
    
    for i, model in enumerate(['Answer_Qwen2', 'Answer_Qwen2.5', 'Answer_OpenGVLab']):
        candidate = str(row[model])
        metrics = {}
        
        # Skip if no candidate answer
        if not candidate.strip():
            metrics.update({
                f'{model}_cosine': 0,
                f'{model}_rouge': 0,
                f'{model}_bert': 0,
                f'{model}_bart': 0
            })
            results.update(metrics)
            continue
        
        # 1. Cosine Similarity 
        model_vector = vectors[i + 1]
        cosine_sim = cosine_similarity([expected_vector], [model_vector])[0][0]
        metrics[f'{model}_cosine'] = cosine_sim
        
        # 2. ROUGE Score
        rouge_scores = rouge.score(expected, candidate)
        metrics[f'{model}_rouge'] = rouge_scores['rougeL'].fmeasure
        
        # 3. BERTScore
        _, _, bert_f1 = bert_score([candidate], [expected], lang='en', model_type='microsoft/deberta-large-mnli')
        metrics[f'{model}_bert'] = bert_f1.numpy()[0]
        
        # 4. BARTScore (using inverse perplexity)
        input_ids = bart_tokenizer.encode(expected, return_tensors='pt', max_length=1024, truncation=True)
        candidate_ids = bart_tokenizer.encode(candidate, return_tensors='pt', max_length=1024, truncation=True)
        
        with torch.no_grad():
            score = bart_model(input_ids=input_ids, labels=candidate_ids).loss
        metrics[f'{model}_bart'] = np.exp(-score.item())  # Higher score = better alignment
        
        results.update(metrics)
    return results

# Apply the evaluation function to each row
df = df.join(pd.DataFrame(df.apply(evaluate_answers, axis=1).tolist()))

# Calculate average scores for all metrics
average_metrics = {
    'Cosine Similarity (Qwen2)': df['Answer_Qwen2_cosine'].mean(),
    'ROUGE-L (Qwen2)': df['Answer_Qwen2_rouge'].mean(),
    'BERTScore (Qwen2)': df['Answer_Qwen2_bert'].mean(),
    'BARTScore (Qwen2)': df['Answer_Qwen2_bart'].mean(),
    'Cosine Similarity (Qwen2.5)': df['Answer_Qwen2.5_cosine'].mean(),
    'ROUGE-L (Qwen2.5)': df['Answer_Qwen2.5_rouge'].mean(),
    'BERTScore (Qwen2.5)': df['Answer_Qwen2.5_bert'].mean(),
    'BARTScore (Qwen2.5)': df['Answer_Qwen2.5_bart'].mean(),
    'Cosine Similarity (OpenGVLab)': df['Answer_OpenGVLab_cosine'].mean(),
    'ROUGE-L (OpenGVLab)': df['Answer_OpenGVLab_rouge'].mean(),
    'BERTScore (OpenGVLab)': df['Answer_OpenGVLab_bert'].mean(),
    'BARTScore (OpenGVLab)': df['Answer_OpenGVLab_bart'].mean(),
}

# Log results
logging.info("Enhanced Model Evaluation Metrics:")
for metric, value in average_metrics.items():
    logging.info(f"{metric}: {value * 100:.2f}%")

# Save results to a new CSV file
output_file_path = '../Assets/data_test/ardian_dataset_test_evaluation.csv'
df.to_csv(output_file_path, index=False)