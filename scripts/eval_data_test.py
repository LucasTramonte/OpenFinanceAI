import pandas as pd
import re
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
import numpy as np
from tqdm import tqdm

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function for numerical extraction
def extract_numbers(text):
    numbers = re.findall(
        r'(?:\$|€|£)?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+%?', 
        str(text)
    )
    parsed = []
    for num in numbers:
        try:
            value = float(re.sub(r'[^\d.]', '', num))
            if '%' in num:
                value /= 100
            parsed.append(value)
        except ValueError:
            continue
    return parsed

def evaluate_answers(row):
    expected = str(row['Expected_Answer']).strip()
    question = str(row['Question']).strip()
    results = {}
    model_answers = {
        model: str(row[model]).strip() 
        for model in ['Answer_Qwen2', 'Answer_Qwen2.5']
    }

    if not expected:
        return results

    for model, candidate in model_answers.items():
        metrics = {}
        if not candidate:
            continue

        try:
            # ROUGE Scores
            rouge_scores = rouge_scorer.score(expected, candidate)
            metrics.update({
                f'{model}_rouge1': rouge_scores['rouge1'].fmeasure,
                f'{model}_rouge2': rouge_scores['rouge2'].fmeasure,
                f'{model}_rougeL': rouge_scores['rougeL'].fmeasure,
            })
        except Exception as e:
            logging.error(f"ROUGE error: {str(e)}")
            metrics.update({f'{model}_rouge{n}': 0 for n in ['1', '2', 'L']})

        try:
            # BERTScore
            _, _, bert_f1 = bert_score([candidate], [expected], lang='en', 
                                     model_type='microsoft/deberta-large-mnli')
            metrics[f'{model}_bert'] = bert_f1.numpy()[0]
        except Exception as e:
            logging.error(f"BERTScore error: {str(e)}")
            metrics[f'{model}_bert'] = 0

        try:
            # Flan-T5 Perplexity
            inputs = t5_tokenizer(question, return_tensors='pt', 
                                truncation=True, max_length=512)
            targets = t5_tokenizer(candidate, return_tensors='pt', 
                                 truncation=True, max_length=512)
            with torch.no_grad():
                outputs = t5_model(**inputs, labels=targets.input_ids)
                loss = outputs.loss
                metrics[f'{model}_flan-t5'] = 1 / torch.exp(loss).item()
        except Exception as e:
            logging.error(f"Flan-T5 error: {str(e)}")
            metrics[f'{model}_flan-t5'] = 0

        try:
            # Numerical Accuracy
            if row['Type'] in ['Tabular', 'Chart']:
                expected_nums = extract_numbers(expected)
                candidate_nums = extract_numbers(candidate)
                
                match = 0
                tolerance = 0.05
                for e in expected_nums:
                    for c in candidate_nums:
                        if abs(e - c) <= max(tolerance * abs(e), 0.01):
                            match = 1
                            break
                    if match: break
                metrics[f'{model}_numerical_acc'] = match
            else:
                metrics[f'{model}_numerical_acc'] = np.nan
        except Exception as e:
            logging.error(f"Numerical accuracy error: {str(e)}")
            metrics[f'{model}_numerical_acc'] = 0

        results.update(metrics)
    
    return results

# Borda Count implementation
def calculate_borda_scores(agg_df, metrics):
    models = ['Answer_Qwen2', 'Answer_Qwen2.5']
    borda_scores = {model: 0 for model in models}
    
    for metric in metrics:
        # Rank models for each metric (higher is better)
        ranked = agg_df[agg_df['Metric'] == metric].sort_values('Average', ascending=False)
        # Assign Borda points (2 for 1st, 1 for 2nd, 0 for 3rd)
        for i, model in enumerate(ranked['Model']):
            borda_scores[model] += (2 - i)
    
    return borda_scores

# Initialize components
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

# Load data
df = pd.read_csv('../Assets/data_test/ardian_dataset_test.csv')

# Apply evaluation with tqdm progress bar
tqdm.pandas(desc="Evaluating Answers")
results = df.progress_apply(evaluate_answers, axis=1)
df = df.join(pd.DataFrame(results.tolist()))

# Calculate averages
metrics = ['rouge1', 'rouge2', 'rougeL', 'bert', 'flan-t5', 'numerical_acc']
aggregation = []

for model in tqdm(['Answer_Qwen2', 'Answer_Qwen2.5'], desc="Aggregating Metrics"):
    for metric in metrics:
        if metric == 'numerical_acc':
            valid_rows = df[df['Type'].isin(['Tabular', 'Chart'])]
            avg = valid_rows[f'{model}_{metric}'].mean()
        else:
            avg = df[f'{model}_{metric}'].mean()
        aggregation.append({
            'Model': model,
            'Metric': metric,
            'Average': avg
        })

# Create aggregation DataFrame
agg_df = pd.DataFrame(aggregation)

# Calculate Borda scores
borda_scores = calculate_borda_scores(agg_df, metrics)

# Add Borda scores to the aggregation DataFrame
agg_df['Borda'] = agg_df['Model'].map(borda_scores)

# Save results
agg_df.to_csv('../Assets/data_test/aggregated_metrics.csv', index=False)
df.to_csv('../Assets/data_test/evaluated_dataset.csv', index=False)

logging.info("Final Aggregated Metrics:")
logging.info(agg_df.to_string())