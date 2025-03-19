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

# Helper function for numerical extraction
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

# Load the dataset from JSON (dataset de base)
file_path = "../Assets/data_test/ardian_dataset_test.json"
df = load_json_file(file_path)

# Initialisation des scorers
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
            # Exécute la coroutine de manière synchrone
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
    question = str(row["Question"]).strip()
    results = {}

    # Extraction des réponses candidates pour chaque modèle
    model_answers = {model: str(row[model]).strip() for model in ["Answer_Qwen2", "Answer_Qwen2.5","Answer_Gemma_4B"]}

    # Si la réponse attendue est vide, on renvoie 0 pour toutes les métriques
    if not expected:
        return {
            f"{model}_{metric}": 0
            for model in model_answers
            for metric in ["rouge1","rouge2","rougeL","bert","flan-t5","string_presence","numerical_acc"]}

    for model, candidate in model_answers.items():
        metrics = {}

        # Si la réponse candidate est vide, on assigne 0 pour ce modèle
        if not candidate:
            results.update(
                {
                    f"{model}_{metric}": 0 for metric in ["rouge1","rouge2","rougeL","bert","flan-t5","string_presence","numerical_acc"]})
            continue

        # ROUGE Score
        rouge_scores = rouge.score(expected, candidate)
        metrics[f"{model}_rouge1"] = rouge_scores["rouge1"].fmeasure
        metrics[f"{model}_rouge2"] = rouge_scores["rouge2"].fmeasure
        metrics[f"{model}_rougeL"] = rouge_scores["rougeL"].fmeasure

        # BERTScore
        _, _, bert_f1 = bert_score([candidate],[expected],lang="en",model_type="microsoft/deberta-large-mnli",)
        metrics[f"{model}_bert"] = bert_f1.numpy()[0]

        # Flan-T5 Score (inverse perplexity estimation)
        input_ids = t5_tokenizer(expected, return_tensors="pt", truncation=True, max_length=1024).input_ids
        candidate_ids = t5_tokenizer(candidate, return_tensors="pt", truncation=True, max_length=1024).input_ids
        with torch.no_grad():
            outputs = t5_model(input_ids=input_ids, labels=candidate_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        metrics[f"{model}_flan-t5"] = 1 / perplexity  # Inverse perplexity

        # String Presence Score : on évalue uniquement si la question contient des références
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
            metrics[f"{model}_string_presence"] = "None"  # Les questions sans références ne sont pas évaluées
            

        # Numerical Accuracy
        if row["Type"] in ["Tabular", "Chart"]:
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
        else:
            metrics[f"{model}_numerical_acc"] = np.nan

        results.update(metrics)

    return results

# Borda Count implementation
def calculate_borda_scores(agg_df, metrics):
    models = ["Answer_Qwen2", "Answer_Qwen2.5","Answer_Gemma_4B"]
    borda_scores = {model: 0 for model in models}

    for metric in metrics:
        # Rank models for each metric (higher is better)
        ranked = agg_df[agg_df["Metric"] == metric].sort_values("Mean", ascending=False)
        # Assign Borda points (2 pour la 1ère place, 1 pour la 2ème, etc.)
        for i, model in enumerate(ranked["Model"]):
            borda_scores[model] += 2 - i

    return borda_scores

# Appliquer la fonction d'évaluation sur chaque ligne du DataFrame
tqdm.pandas(desc="Evaluating Answers")
df_results = df.progress_apply(evaluate_answers, axis=1).tolist()
df_results = pd.DataFrame(df_results)
df = df.join(df_results)

# Pour le calcul de la métrique string_presence, on ne garde que les questions disposant de références
df_string_presence = df[df["References"].notna() & (df["References"].str.strip() != "")].copy()

# Calcul des scores moyens pour chaque modèle et chaque métrique
metrics_list = ["rouge1","rouge2","rougeL","bert","flan-t5","string_presence","numerical_acc"]
models = ["Answer_Qwen2", "Answer_Qwen2.5","Answer_Gemma_4B"]
average_metrics = []
for model in models:
    for metric in metrics_list:
        if metric == "string_presence":
            # Calculer la moyenne sur le sous-ensemble où la métrique n'est pas "None"
            valid_values = df_string_presence[f"{model}_{metric}"].dropna()
            mean_score = valid_values.mean() if not valid_values.empty else 0
        else:
            mean_score = df[f"{model}_{metric}"].mean()
        average_metrics.append({"Model": model, "Metric": metric, "Mean": mean_score})
aggregation_df = pd.DataFrame(average_metrics)

# Calcul des scores Borda
borda_scores = calculate_borda_scores(aggregation_df, metrics_list)

# Ajout des scores Borda au DataFrame d'agrégation
aggregation_df["Borda"] = aggregation_df["Model"].map(borda_scores)

# Sauvegarde des résultats d'agrégation en JSON
with open("../Assets/data_test/agreggation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(aggregation_df.to_dict("records"), f, ensure_ascii=False, indent=2)
logging.info("Métriques d'agrégation sauvegardées en JSON.")

# Affichage des résultats moyens dans le log
logging.info("Model Evaluation Metrics:")
for _, row in aggregation_df.iterrows():
    logging.info(f"{row['Metric']} ({row['Model']}): {row['Mean'] * 100:.2f}%")

# Affichage des statistiques spécifiques pour string_presence
total_string_presence = df_string_presence.shape[0]
logging.info(f"Nombre total de questions avec référence: {total_string_presence}")
for model in models:
    count_presence = df_string_presence[df_string_presence[f"{model}_string_presence"] == 1].shape[0]
    logging.info(f"Pour le modèle {model}, {count_presence} questions sur {total_string_presence} ont un score string_presence de 1.")

# Sauvegarde des résultats finaux en JSON
with open("../Assets/data_test/ardian_dataset_test_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict("records"), f, ensure_ascii=False, indent=2)
logging.info("Résultats d'évaluation sauvegardés en JSON.")