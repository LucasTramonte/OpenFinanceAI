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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_numbers(text):
    numbers = re.findall(r"(?:\$|€|£)?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+%?", str(text))
    parsed = []
    for num in numbers:
        try:
            value = float(re.sub(r"[^\d.]", "", num))
            if "%" in num:
                value /= 100
            parsed.append(value)
        except ValueError:
            continue
    return parsed

def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"Fichier JSON chargé avec succès: {file_path}")
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier JSON {file_path}: {str(e)}")


file_path = "../Assets/data_test/ardian_dataset_eval.json"
df = load_json_file(file_path)
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
string_presence_scorer = StringPresence()

def get_string_presence(candidate, reference, max_retries=5, delay=2):
    """
    Retourne le score (1 ou 0) de StringPresence pour une paire (candidate, reference)
    en exécutant la méthode asynchrone single_turn_ascore de manière synchrone.
    """
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
            # Si la réponse candidate est vide, toutes les métriques prennent la valeur None
            for m in ["rouge1", "rouge2", "rougeL", "bert", "flan-t5", "string_presence", "numerical_acc"]:
                metrics[f"{model}_{m}"] = None
            results.update(metrics)
            continue

        # Calcul commun : scores ROUGE
        rouge_scores = rouge.score(expected, candidate)
        metrics[f"{model}_rouge1"] = rouge_scores["rouge1"].fmeasure
        metrics[f"{model}_rouge2"] = rouge_scores["rouge2"].fmeasure
        metrics[f"{model}_rougeL"] = rouge_scores["rougeL"].fmeasure

        if question_type == "short":
            # Pour les questions short : évaluer string_presence et numerical_acc
            if ("References" in row and pd.notna(row["References"]) and row["References"].strip()):
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
                if match:
                    break
            metrics[f"{model}_numerical_acc"] = match

            # Pour short, on ne calcule pas ces métriques
            metrics[f"{model}_bert"] = None
            metrics[f"{model}_flan-t5"] = None

        else:  # question_type == "long"
            # Pour les questions long : calcul de BERTScore et Flan-T5
            _, _, bert_f1 = bert_score([candidate], [expected], lang="en", model_type="microsoft/deberta-large-mnli")
            metrics[f"{model}_bert"] = bert_f1.numpy()[0]

            input_ids = t5_tokenizer(expected, return_tensors="pt", truncation=True, max_length=1024).input_ids
            candidate_ids = t5_tokenizer(candidate, return_tensors="pt", truncation=True, max_length=1024).input_ids
            with torch.no_grad():
                outputs = t5_model(input_ids=input_ids, labels=candidate_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            metrics[f"{model}_flan-t5"] = 1 / perplexity  # Inverse perplexity

            # Pour long, ces métriques ne sont pas évaluées
            metrics[f"{model}_string_presence"] = None
            metrics[f"{model}_numerical_acc"] = None

        results.update(metrics)
    return results


tqdm.pandas(desc="Evaluating Answers")
df_results = df.progress_apply(evaluate_answers, axis=1).tolist()
df_results = pd.DataFrame(df_results)
df = df.join(df_results)

# -----------------------------
# Agrégation des métriques
# -----------------------------
# Trois groupes de métriques :
# - Métriques communes (ROUGE) : on calcule une moyenne "Overall" ainsi que par type.
# - Métriques short-only : string_presence, numerical_acc
# - Métriques long-only : bert, flan-t5
metrics_list_common = ["rouge1", "rouge2", "rougeL"]
metrics_list_short = ["string_presence", "numerical_acc"]
metrics_list_long = ["bert", "flan-t5"]
models = ["Answer_Qwen2", "Answer_Qwen2.5", "Answer_Gemma_4B", "Answer_Gemma_12B"]

average_metrics = []
for model in models:
    for metric in metrics_list_common:
        col_name = f"{model}_{metric}"
        overall_avg = df[col_name].dropna().mean()
        # On filtre sur les valeurs de "Question_Type" en minuscules
        short_avg = df[df["Question_Type"].str.lower() == "short"][col_name].dropna().mean()
        long_avg = df[df["Question_Type"].str.lower() == "long"][col_name].dropna().mean()
        average_metrics.append({"Model": model, "Metric": metric, "Type": "Overall", "Mean": overall_avg})
        average_metrics.append({"Model": model, "Metric": metric, "Type": "Short", "Mean": short_avg})
        average_metrics.append({"Model": model, "Metric": metric, "Type": "Long", "Mean": long_avg})

# Pour les métriques short-only
for model in models:
    for metric in metrics_list_short:
        col_name = f"{model}_{metric}"
        short_avg = df[df["Question_Type"].str.lower() == "short"][col_name].dropna().mean()
        average_metrics.append({"Model": model, "Metric": metric, "Type": "Short", "Mean": short_avg})

# Pour les métriques long-only
for model in models:
    for metric in metrics_list_long:
        col_name = f"{model}_{metric}"
        long_avg = df[df["Question_Type"].str.lower() == "long"][col_name].dropna().mean()
        average_metrics.append({"Model": model, "Metric": metric, "Type": "Long", "Mean": long_avg})

aggregation_df = pd.DataFrame(average_metrics)


def calculate_borda_scores(agg_df):
    models = ["Answer_Qwen2", "Answer_Qwen2.5", "Answer_Gemma_4B", "Answer_Gemma_12B"]
    borda_scores = {model: 0 for model in models}
    # Groupes de métriques et leur type pour l'agrégation
    metric_groups = [
        ("rouge1", "Overall"),
        ("rouge2", "Overall"),
        ("rougeL", "Overall"),
        ("rouge1", "Short"),
        ("rouge2", "Short"),
        ("rougeL", "Short"),
        ("rouge1", "Long"),
        ("rouge2", "Long"),
        ("rougeL", "Long"),
        ("string_presence", "Short"),
        ("numerical_acc", "Short"),
        ("bert", "Long"),
        ("flan-t5", "Long")
    ]
    for metric, mtype in metric_groups:
        subset = agg_df[(agg_df["Metric"] == metric) & (agg_df["Type"] == mtype)]
        if not subset.empty:
            ranked = subset.sort_values("Mean", ascending=False)
            # Application de la formule de Borda (2 - index)
            for i, model in enumerate(ranked["Model"]):
                borda_scores[model] += 2 - i
    return borda_scores

borda_scores = calculate_borda_scores(aggregation_df)
aggregation_df["Borda"] = aggregation_df["Model"].map(borda_scores)


with open("../Assets/data_test/agreggation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(aggregation_df.to_dict("records"), f, ensure_ascii=False, indent=2)
logging.info("Métriques d'agrégation sauvegardées en JSON.")

logging.info("Model Evaluation Metrics:")
for _, row in aggregation_df.iterrows():
    if row["Mean"] is not None:
        logging.info(f"{row['Metric']} ({row['Model']}, {row['Type']}): {row['Mean'] * 100:.2f}%")
    else:
        logging.info(f"{row['Metric']} ({row['Model']}, {row['Type']}): None")

df_string_presence = df[(df["References"].notna()) & (df["References"].str.strip() != "")]
total_string_presence = df_string_presence.shape[0]
logging.info(f"Nombre total de questions avec référence: {total_string_presence}")
for model in models:
    count_presence = df_string_presence[df_string_presence[f"{model}_string_presence"] == 1].shape[0]
    logging.info(f"Pour le modèle {model}, {count_presence} questions sur {total_string_presence} ont un score string_presence de 1.")

with open("../Assets/data_test/ardian_dataset_test_evaluation_final.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict("records"), f, ensure_ascii=False, indent=2)
logging.info("Résultats d'évaluation sauvegardés en JSON.")