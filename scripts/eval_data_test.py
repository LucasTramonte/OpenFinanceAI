import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
file_path = '../Assets/data_test/ardian_dataset_test.csv'
df = pd.read_csv(file_path)

#Initialize ROUGE scorer and Flan-T5 model
rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
t5_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')

#Function to evaluate answers using all metrics
def evaluate_answers(row):
    expected = str(row['Expected_Answer']).strip()
    results = {}
    
    #Extract candidate answers
    model_answers = {model: str(row[model]).strip() for model in ['Answer_Qwen2', 'Answer_Qwen2.5', 'Answer_OpenGVLab']}
    
    #Skip processing if expected answer is empty
    if not expected:
        return {f'{model}_{metric}': 0 for model in model_answers for metric in ['rouge1', 'rouge2', 'rougeL', 'bert', 'flan-t5']}
    
    for model, candidate in model_answers.items():
        metrics = {}
        
        #Skip empty candidate answers
        if not candidate:
            results.update({f'{model}_{metric}': 0 for metric in ['rouge1', 'rouge2', 'rougeL', 'bert', 'flan-t5']})
            continue
        
        #ROUGE Score
        rouge_scores = rouge.score(expected, candidate)
        metrics[f'{model}_rouge1'] = rouge_scores['rouge1'].fmeasure
        metrics[f'{model}_rouge2'] = rouge_scores['rouge2'].fmeasure
        metrics[f'{model}_rougeL'] = rouge_scores['rougeL'].fmeasure
                
        #BERTScore
        _, _, bert_f1 = bert_score([candidate], [expected], lang='en', model_type='microsoft/deberta-large-mnli')
        metrics[f'{model}_bert'] = bert_f1.numpy()[0]
        
        #Flan-T5 Score (inverse perplexity estimation)
        input_ids = t5_tokenizer(expected, return_tensors='pt', truncation=True, max_length=1024).input_ids
        candidate_ids = t5_tokenizer(candidate, return_tensors='pt', truncation=True, max_length=1024).input_ids
        
        with torch.no_grad():
            outputs = t5_model(input_ids=input_ids, labels=candidate_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        metrics[f'{model}_flan-t5'] = 1 / perplexity  # Inverse perplexity

        results.update(metrics)
    
    return results

# Apply the evaluation function to each row
df = df.join(pd.DataFrame(df.apply(evaluate_answers, axis=1).tolist()))

# Compute average scores
metrics_list = ['rouge1', 'rouge2', 'rougeL', 'bert', 'flan-t5']
models = ['Answer_Qwen2', 'Answer_Qwen2.5', 'Answer_OpenGVLab']

# Compute mean scores
average_metrics = []
for model in models:
    for metric in metrics_list:
        mean_score = df[f'{model}_{metric}'].mean()
        average_metrics.append({'Model': model, 'Metric': metric, 'Mean': mean_score})


# Save aggregation results to CSV
aggregation_df = pd.DataFrame(average_metrics)
aggregation_df.to_csv('../Assets/data_test/agreggation_metrics.csv', index=False)

# Log results
logging.info("Model Evaluation Metrics:")
for _, row in aggregation_df.iterrows():
    logging.info(f"{row['Metric']} ({row['Model']}): {row['Mean'] * 100:.2f}%")

# Save results
df.to_csv('../Assets/data_test/ardian_dataset_test_evaluation.csv', index=False)
logging.info("Evaluation results saved successfully.")
