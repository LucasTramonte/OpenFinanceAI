import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
import math
import asyncio
import time
import numpy as np
from requests.exceptions import ChunkedEncodingError
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import StringPresence
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_numbers(text):
    numbers = re.findall(r"(?:\$|€|£)?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+%?", str(text))
    parsed = []
    for num in numbers:
        try:
            value = float(re.sub(r"[^\d.]", "", num))
            if "%" in num: value /= 100
            parsed.append(value)
        except ValueError: continue
    return parsed

def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"Fichier JSON chargé avec succès: {file_path}")
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier JSON {file_path}: {str(e)}")

file_path = "../../Assets/data_test/ardian_dataset_final.json"
df = load_json_file(file_path)
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
string_presence_scorer = StringPresence()

def get_string_presence(candidate, reference, max_retries=5, delay=2):
    sample = SingleTurnSample(response=candidate, reference=reference.strip())
    for attempt in range(max_retries):
        try:
            score = asyncio.run(string_presence_scorer.single_turn_ascore(sample))
            return score
        except ChunkedEncodingError as ce:
            logging.error(f"ChunkedEncodingError on attempt {attempt + 1} for candidate '{candidate[:30]}...': {ce}")
            time.sleep(delay)
        except Exception as e:
            logging.error(f"Error on attempt {attempt + 1} for candidate '{candidate[:30]}...': {e}")
            time.sleep(delay)
    return 0

def evaluate_answers(row):
    expected = str(row["Expected_Answer"]).strip()
    question_type = row["Question_Type"].strip().lower() if ("Question_Type" in row and pd.notna(row["Question_Type"]) and row["Question_Type"]) else "short"
    results = {}
    model_answers = {model: str(row[model]).strip() for model in ["Answer_Qwen2", "Answer_Qwen2.5", "Answer_Gemma_4B", "Answer_Gemma_12B"]}
    
    if not expected:
        for model in model_answers:
            for m in ["rouge1", "rouge2", "rougeL", "bert", "flan-t5", "string_presence", "numerical_acc"]:
                results[f"{model}_{m}"] = None
        return results
    
    for model, candidate in model_answers.items():
        metrics = {}
        if not candidate:
            for m in ["rouge1", "rouge2", "rougeL", "bert", "flan-t5", "string_presence", "numerical_acc"]:
                metrics[f"{model}_{m}"] = None
            results.update(metrics)
            continue
        
        # ROUGE scores
        rouge_scores = rouge.score(expected, candidate)
        metrics[f"{model}_rouge1"] = rouge_scores["rouge1"].fmeasure
        metrics[f"{model}_rouge2"] = rouge_scores["rouge2"].fmeasure
        metrics[f"{model}_rougeL"] = rouge_scores["rougeL"].fmeasure
        
        if question_type == "short":
            if "References" in row and pd.notna(row["References"]) and row["References"].strip():
                references = row["References"].split(";")
                string_presence_score = 0
                for reference in references:
                    score = get_string_presence(candidate, reference)
                    if score == 1:
                        string_presence_score = 1
                        break
                metrics[f"{model}_string_presence"] = string_presence_score
            else:
                metrics[f"{model}_string_presence"] = None
                
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
            metrics[f"{model}_numerical_acc"] = match
            metrics[f"{model}_bert"] = None
            metrics[f"{model}_flan-t5"] = None
        else:
            _, _, bert_f1 = bert_score([candidate], [expected], lang="en", model_type="microsoft/deberta-large-mnli")
            metrics[f"{model}_bert"] = bert_f1.numpy()[0]
            
            input_ids = t5_tokenizer(expected, return_tensors="pt", truncation=True, max_length=1024).input_ids
            candidate_ids = t5_tokenizer(candidate, return_tensors="pt", truncation=True, max_length=1024).input_ids
            with torch.no_grad():
                outputs = t5_model(input_ids=input_ids, labels=candidate_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            metrics[f"{model}_flan-t5"] = 1 / perplexity
            metrics[f"{model}_string_presence"] = None
            metrics[f"{model}_numerical_acc"] = None
        
        results.update(metrics)
    return results

tqdm.pandas(desc="Evaluating Answers")
df_results = df.progress_apply(evaluate_answers, axis=1).tolist()
df_results = pd.DataFrame(df_results)
df = df.join(df_results)

# Agrégation des métriques
metrics_list_common = ["rouge1", "rouge2", "rougeL", "faithfulness"]
metrics_list_short = ["string_presence", "numerical_acc"]
metrics_list_long = ["bert", "flan-t5"]
models = ["Answer_Qwen2", "Answer_Qwen2.5", "Answer_Gemma_4B", "Answer_Gemma_12B"]

average_metrics = []
for model in models:
    # Common metrics
    for metric in metrics_list_common:
        col_name = f"{model}_{metric}"
        overall_avg = df[col_name].dropna().mean()
        short_avg = df[df["Question_Type"].str.lower() == "short"][col_name].dropna().mean()
        long_avg = df[df["Question_Type"].str.lower() == "long"][col_name].dropna().mean()
        average_metrics.extend([
            {"Model": model, "Metric": metric, "Type": "Overall", "Mean": overall_avg},
            {"Model": model, "Metric": metric, "Type": "Short", "Mean": short_avg},
            {"Model": model, "Metric": metric, "Type": "Long", "Mean": long_avg}
        ])
    
    # Short-only metrics
    for metric in metrics_list_short:
        col_name = f"{model}_{metric}"
        short_avg = df[df["Question_Type"].str.lower() == "short"][col_name].dropna().mean()
        average_metrics.append({"Model": model, "Metric": metric, "Type": "Short", "Mean": short_avg})
    
    # Long-only metrics
    for metric in metrics_list_long:
        col_name = f"{model}_{metric}"
        long_avg = df[df["Question_Type"].str.lower() == "long"][col_name].dropna().mean()
        average_metrics.append({"Model": model, "Metric": metric, "Type": "Long", "Mean": long_avg})
    
    # Faithfulness based on existing scores
    faithfulness_col = "Faithfulness_Score_" + model.replace("Answer_", "")
    faithfulness_avg_value = df[faithfulness_col].dropna().mean()
    average_metrics.append({"Model": model, "Metric": "faithfulness", "Type": "Overall", "Mean": faithfulness_avg_value})

aggregation_df = pd.DataFrame(average_metrics)

# Calculate Borda scores
def calculate_borda_scores_by_type(agg_df):
    models = ["Answer_Qwen2", "Answer_Qwen2.5", "Answer_Gemma_4B", "Answer_Gemma_12B"]
    global_scores = {model: 0 for model in models}
    short_scores = {model: 0 for model in models}
    long_scores = {model: 0 for model in models}

    global_groups = [
        ("rouge1", "Overall"), ("rouge2", "Overall"), ("rougeL", "Overall"), ("faithfulness", "Overall"),
        ("rouge1", "Short"), ("rouge2", "Short"), ("rougeL", "Short"),
        ("rouge1", "Long"), ("rouge2", "Long"), ("rougeL", "Long"),
        ("string_presence", "Short"), ("numerical_acc", "Short"),
        ("bert", "Long"), ("flan-t5", "Long")
    ]
    short_groups = [
        ("rouge1", "Short"), ("rouge2", "Short"), ("rougeL", "Short"), ("faithfulness", "Short"),
        ("string_presence", "Short"), ("numerical_acc", "Short")
    ]
    long_groups = [
        ("rouge1", "Long"), ("rouge2", "Long"), ("rougeL", "Long"), ("faithfulness", "Long"),
        ("bert", "Long"), ("flan-t5", "Long")
    ]

    def assign_borda_scores(score_dict, groups):
        for metric, mtype in groups:
            subset = agg_df[(agg_df["Metric"] == metric) & (agg_df["Type"] == mtype)]
            if not subset.empty:
                ranked = subset.sort_values("Mean", ascending=False)
                for i, model in enumerate(ranked["Model"]):
                    score_dict[model] += len(models) - 1 - i

    assign_borda_scores(global_scores, global_groups)
    assign_borda_scores(short_scores, short_groups)
    assign_borda_scores(long_scores, long_groups)
    return {"global": global_scores, "short": short_scores, "long": long_scores}

borda_dict = calculate_borda_scores_by_type(aggregation_df)

# Create Borda scores DataFrame
borda_scores_df = pd.DataFrame({
    "Model": models,
    "Borda_Global": [borda_dict["global"][model] for model in models],
    "Borda_Short": [borda_dict["short"][model] for model in models],
    "Borda_Long": [borda_dict["long"][model] for model in models],
})

# Visualize Borda scores
x = range(len(models))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([p - width for p in x], borda_scores_df["Borda_Global"], width=width, label="Borda Global")
ax.bar(x, borda_scores_df["Borda_Short"], width=width, label="Borda Short")
ax.bar([p + width for p in x], borda_scores_df["Borda_Long"], width=width, label="Borda Long")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Borda Score")
ax.set_title("Scores Borda par modèle")
ax.legend()
plt.tight_layout()
plt.savefig("../../Assets/output/metrics/borda_scores.png")
plt.show()

# Calculate faithfulness averages
faithfulness_avg = {}
for model in models:
    faithfulness_col = "Faithfulness_Score_" + model.replace("Answer_", "")
    faithfulness_avg[model] = df[faithfulness_col].dropna().mean()
    logging.info(f"Moyenne de Faithfulness pour {model}: {faithfulness_avg[model] * 100:.2f}%")

# Visualize faithfulness scores
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(models, [faithfulness_avg[model] for model in models], color="skyblue")
ax.set_ylabel("Moyenne Faithfulness")
ax.set_title("Moyenne des scores Faithfulness par modèle")
plt.tight_layout()
plt.savefig("../../Assets/output/metrics/faithfulness_scores.png")
plt.show()

# Create metrics tables by type
short_df = aggregation_df[aggregation_df["Type"] == "Short"].pivot(index="Model", columns="Metric", values="Mean").reset_index()
long_df = aggregation_df[aggregation_df["Type"] == "Long"].pivot(index="Model", columns="Metric", values="Mean").reset_index()

# Create global table
global_table = borda_scores_df[["Model", "Borda_Global"]].copy()
global_table["Faithfulness_Avg"] = global_table["Model"].map(faithfulness_avg)

# Save tables to Markdown
global_md = "## Tableau des métriques Global\n\n" + global_table.to_markdown(index=False)
short_md = "## Tableau des métriques Short\n\n" + short_df.to_markdown(index=False)
long_md = "## Tableau des métriques Long\n\n" + long_df.to_markdown(index=False)

final_md = global_md + "\n\n" + short_md + "\n\n" + long_md
with open("../../Assets/data_test/tableau_final.md", "w", encoding="utf-8") as f:
    f.write(final_md)
logging.info("Tableaux récapitulatifs sauvegardés dans tableau_final.md")

# Save aggregation metrics to JSON
with open("../Assets/data_test/agreggation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(aggregation_df.to_dict("records"), f, ensure_ascii=False, indent=2)
logging.info("Métriques d'agrégation sauvegardées en JSON.")

# Log model evaluation metrics
logging.info("Model Evaluation Metrics:")
for _, row in aggregation_df.iterrows():
    if row["Mean"] is not None:
        logging.info(f"{row['Metric']} ({row['Model']}, {row['Type']}): {row['Mean'] * 100:.2f}%")
    else:
        logging.info(f"{row['Metric']} ({row['Model']}, {row['Type']}): None")

# String presence metrics
df_string_presence = df[(df["References"].notna()) & (df["References"].str.strip() != "")]
total_string_presence = df_string_presence.shape[0]
logging.info(f"Nombre total de questions avec référence: {total_string_presence}")

for model in models:
    count_presence = df_string_presence[df_string_presence[f"{model}_string_presence"] == 1].shape[0]
    logging.info(f"Pour le modèle {model}, {count_presence} questions sur {total_string_presence} ont un score string_presence de 1.")

# Save all evaluation results to JSON
with open("../../Assets/data_test/ardian_dataset_final_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict("records"), f, ensure_ascii=False, indent=2)
logging.info("Résultats d'évaluation sauvegardés en JSON.")
