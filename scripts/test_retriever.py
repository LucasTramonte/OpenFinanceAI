import os
import torch
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pdf2image import convert_from_path
from colpali_engine.models import ColQwen2, ColQwen2Processor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import hashlib
from collections import defaultdict
import itertools
from typing import List, Dict, Any, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = "../Assets/output"
RETRIEVER_EVAL_DIR = os.path.join(OUTPUT_DIRECTORY, "retriever_evaluation")
EMBEDDINGS_DIR = os.path.join(OUTPUT_DIRECTORY, "embeddings")

# Créer les répertoires nécessaires
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(RETRIEVER_EVAL_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def load_model_and_processor():
    """Load the ColPALI model and processor."""
    logger.info("Loading ColQwen2 model...")
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    ).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
    return model, processor

def convert_pdf_to_images(pdf_path):
    """Convert a PDF file to a list of images."""
    logger.info(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=200)
    logger.info(f"PDF converted to {len(images)} pages.")
    return images

def get_pdf_hash(pdf_path):
    """Generate a unique hash for the given PDF file."""
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def generate_embeddings(model, processor, images, pdf_path):
    """Generate embeddings if not already saved for the specific PDF."""
    pdf_hash = get_pdf_hash(pdf_path)
    index_file = os.path.join(EMBEDDINGS_DIR, f"document_embeddings_{pdf_hash}.pkl")

    if os.path.exists(index_file):
        try:
            with open(index_file, "rb") as f:
                embeddings_list = pickle.load(f)
            logger.info(f"Embeddings loaded from cache for {pdf_path}.")
            return embeddings_list
        except Exception as e:
            logger.warning(f"Failed to load embeddings for {pdf_path}. Regenerating... (Error: {e})")

    # Compute embeddings if not found or loading failed
    logger.info(f"Generating new embeddings for {pdf_path}...")
    dataloader = DataLoader(
        dataset=images,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x)
    )

    embeddings_list = []
    for batch in tqdm(dataloader, desc="Generating embeddings"):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embeddings = model(**batch)
        embeddings_list.extend(embeddings.cpu().unbind())

    # Save embeddings specific to this PDF
    with open(index_file, "wb") as f:
        pickle.dump(embeddings_list, f)

    logger.info(f"Embeddings saved for {pdf_path}.")
    return embeddings_list

def compute_embedding_similarity(
    question_embeddings: torch.Tensor, 
    image_embeddings: torch.Tensor
) -> Dict[str, float]:
    """
    Computes similarity metrics between question embeddings and image embeddings.
    
    Args:
        question_embeddings: Tensor of shape (n_questions, embedding_dim)
        image_embeddings: Tensor of shape (n_images, embedding_dim)
        
    Returns:
        Dictionary containing similarity metrics:
        - cosine_sim_mean: Mean cosine similarity
        - cosine_sim_median: Median cosine similarity
        - euclidean_dist_mean: Mean euclidean distance
        - matching_score: Percentage of questions where highest similarity is with correct image
    """
    # Normalize embeddings for cosine similarity
    q_norm = torch.nn.functional.normalize(question_embeddings, p=2, dim=1)
    i_norm = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
    
    # Calculate cosine similarities between all questions and images
    cosine_sim = torch.matmul(q_norm, i_norm.T)  # shape: (n_questions, n_images)
    
    # Calculate Euclidean distances
    q_expanded = question_embeddings.unsqueeze(1).expand(-1, image_embeddings.size(0), -1)
    i_expanded = image_embeddings.unsqueeze(0).expand(question_embeddings.size(0), -1, -1)
    euclidean_dist = torch.sqrt(torch.sum((q_expanded - i_expanded) ** 2, dim=2))
    
    # Assuming diagonal elements should match (question i corresponds to image i)
    diag_cosine = torch.diagonal(cosine_sim)
    diag_euclidean = torch.diagonal(euclidean_dist)
    
    # Calculate metrics
    metrics = {
        "cosine_sim_mean": float(diag_cosine.mean().item()),
        "cosine_sim_median": float(torch.median(diag_cosine).item()),
        "euclidean_dist_mean": float(diag_euclidean.mean().item()),
        "matching_score": float((torch.argmax(cosine_sim, dim=1) == torch.arange(len(question_embeddings))).float().mean().item())
    }
    
    return metrics

def get_relevant_indices(model, processor, query, embeddings_list, top_k):
    """Retrieve the indices of the top-k relevant pages based on the query."""
    batch_queries = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
    scores = processor.score_multi_vector(query_embeddings, embeddings_list)
    top_indices = scores[0].topk(top_k).indices.tolist()
    top_scores = scores[0].topk(top_k).values.tolist()
    return top_indices, top_scores

def parse_document_ids(doc_ids_str):
    """Convertit une chaîne d'IDs de documents en liste."""
    if not doc_ids_str or pd.isna(doc_ids_str):
        return []
    return [id.strip() for id in doc_ids_str.split(",")]

class RetrieverEvaluator:
    """Évaluateur pour le retriever de documents visuels."""
    
    def __init__(self, data_path):
        """
        Initialise l'évaluateur avec le fichier de données.
        
        Args:
            data_path: Chemin vers le fichier JSON contenant les questions et réponses attendues
        """
        self.data_path = data_path
        self.data = self._load_data()
        self.results = {}
        
    def _load_data(self):
        """Charge les données depuis le fichier JSON."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate(self, retriever_results, k_values=[1, 3, 5], embedding_metrics=None):
        """
        Évalue le retriever en calculant diverses métriques.
        
        Args:
            retriever_results: Dictionnaire {question_id: [doc_id1, doc_id2, ...]} 
                             contenant les documents récupérés par le retriever
            k_values: Liste des valeurs de k pour les métriques @k
            embedding_metrics: Métriques de similarité des embeddings (optionnel)
            
        Returns:
            Dictionnaire contenant les résultats des métriques
        """
        metrics = defaultdict(list)
        question_level_metrics = defaultdict(dict)
        question_types = defaultdict(list)
        question_subjects = defaultdict(list)
        
        for item in tqdm(self.data, desc="Évaluation des questions"):
            question_id = item["Question_ID"]
            
            # Ignorer les questions sans résultat du retriever
            if question_id not in retriever_results:
                continue
                
            # Documents de référence
            expected_docs_top1 = parse_document_ids(item.get("Expected_documents_top1", ""))
            expected_docs_top3 = parse_document_ids(item.get("Expected_documents_top3", ""))
            
            # Vérifier s'il y a des documents attendus
            if not expected_docs_top3:
                continue
                
            retrieved_docs = [str(doc_id) for doc_id in retriever_results[question_id]]
            
            # Stocker le type et le sujet de la question
            question_type = item.get("Question_Type", "Unknown")
            question_subject = item.get("Question_subject", "Unknown")
            
            # Calculer les métriques spécifiques à la question
            question_metrics = {
                "question_type": question_type,
                "question_subject": question_subject
            }
            
            # MRR (Mean Reciprocal Rank)
            mrr = 0
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in expected_docs_top3:
                    # i is a zero-based index (0,1,2...), but MRR formula requires ranks starting at 1
                    # This converts index to rank position (e.g., index 0 → rank 1)
                    mrr = 1 / (i + 1)
                    break
            metrics["mrr"].append(mrr)
            question_metrics["mrr"] = mrr
            question_types[question_type].append(("mrr", mrr))
            question_subjects[question_subject].append(("mrr", mrr))
            
            # Pour chaque valeur de k
            for k in k_values:
                if k <= len(retrieved_docs):
                    retrieved_at_k = retrieved_docs[:k]
                    
                    # Intersection entre documents récupérés et attendus
                    relevant_retrieved = set(retrieved_at_k) & set(expected_docs_top3)
                    
                    # Precision@k
                    precision_k = len(relevant_retrieved) / len(retrieved_at_k) if retrieved_at_k else 0
                    metrics[f"precision@{k}"].append(precision_k)
                    question_metrics[f"precision@{k}"] = precision_k
                    question_types[question_type].append((f"precision@{k}", precision_k))
                    question_subjects[question_subject].append((f"precision@{k}", precision_k))
                    
                    # Recall@k
                    recall_k = len(relevant_retrieved) / len(expected_docs_top3) if expected_docs_top3 else 0
                    metrics[f"recall@{k}"].append(recall_k)
                    question_metrics[f"recall@{k}"] = recall_k
                    question_types[question_type].append((f"recall@{k}", recall_k))
                    question_subjects[question_subject].append((f"recall@{k}", recall_k))
            
            # Exact Match pour le top1
            if retrieved_docs and expected_docs_top1:
                exact_match = 1 if retrieved_docs[0] in expected_docs_top1 else 0
                metrics["exact_match_top1"].append(exact_match)
                question_metrics["exact_match_top1"] = exact_match
                question_types[question_type].append(("exact_match_top1", exact_match))
                question_subjects[question_subject].append(("exact_match_top1", exact_match))
            
            question_level_metrics[question_id] = question_metrics
        
        # Calculer les moyennes des métriques
        overall_results = {metric: np.mean(values) for metric, values in metrics.items()}
        
        # Ajouter les métriques d'embedding si disponibles
        if embedding_metrics:
            overall_results["embedding_similarity"] = embedding_metrics
        
        # Calculer les moyennes par type de question
        type_results = {}
        for q_type, values in question_types.items():
            type_results[q_type] = {}
            for metric_name, group in itertools.groupby(sorted(values, key=lambda x: x[0]), key=lambda x: x[0]):
                type_results[q_type][metric_name] = np.mean([score for _, score in group])
        
        # Calculer les moyennes par sujet de question
        subject_results = {}
        for subject, values in question_subjects.items():
            subject_results[subject] = {}
            for metric_name, group in itertools.groupby(sorted(values, key=lambda x: x[0]), key=lambda x: x[0]):
                subject_results[subject][metric_name] = np.mean([score for _, score in group])
        
        self.results = {
            "overall": overall_results,
            "by_question_type": type_results,
            "by_question_subject": subject_results,
            "per_question": question_level_metrics
        }
        
        return self.results
    
    def plot_overall_results(self, output_path=None):
        """Génère un graphique des métriques globales."""
        if not self.results:
            logger.error("Aucun résultat à afficher. Exécutez d'abord evaluate().")
            return
            
        if not output_path:
            output_path = os.path.join(RETRIEVER_EVAL_DIR, "overall_metrics.png")
            
        overall_metrics = self.results["overall"]
        
        # Filtrer les métriques d'embedding pour un graphique séparé
        embedding_metrics = {}
        if "embedding_similarity" in overall_metrics:
            embedding_metrics = overall_metrics["embedding_similarity"]
            overall_metrics = {k: v for k, v in overall_metrics.items() if k != "embedding_similarity"}
        
        # Créer un DataFrame pour la visualisation
        df = pd.DataFrame({
            "Métrique": list(overall_metrics.keys()),
            "Score": list(overall_metrics.values())
        })
        
        # Trier pour regrouper les métriques similaires
        df["Groupe"] = df["Métrique"].apply(lambda x: x.split("@")[0] if "@" in x else x)
        df = df.sort_values(["Groupe", "Métrique"])
        
        # Créer le graphique
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x="Métrique", y="Score", data=df, palette="viridis")
        
        # Ajouter les étiquettes et le titre
        plt.title("Évaluation du Retriever Visuel", fontsize=16)
        plt.xlabel("Métrique", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Ajouter les valeurs sur chaque barre
        for i, v in enumerate(df["Score"]):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Graphique sauvegardé dans {output_path}")
        plt.close()
        
        # Si des métriques d'embedding existent, créer un graphique séparé pour elles
        if embedding_metrics:
            embed_output_path = os.path.join(RETRIEVER_EVAL_DIR, "embedding_metrics.png")
            plt.figure(figsize=(10, 6))
            
            # Créer un DataFrame pour les métriques d'embedding
            embed_df = pd.DataFrame({
                "Métrique": list(embedding_metrics.keys()),
                "Score": list(embedding_metrics.values())
            })
            
            # Créer le graphique
            ax = sns.barplot(x="Métrique", y="Score", data=embed_df, palette="viridis")
            
            # Ajouter les étiquettes et le titre
            plt.title("Métriques de Similarité des Embeddings", fontsize=16)
            plt.xlabel("Métrique", fontsize=14)
            plt.ylabel("Score", fontsize=14)
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            
            # Ajouter les valeurs sur chaque barre
            for i, v in enumerate(embed_df["Score"]):
                ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
            
            plt.tight_layout()
            plt.savefig(embed_output_path)
            logger.info(f"Graphique des métriques d'embedding sauvegardé dans {embed_output_path}")
            plt.close()
    
    def plot_by_type(self, metric="recall@3", output_path=None):
        """Génère un graphique des performances par type de question."""
        if not self.results or "by_question_type" not in self.results:
            logger.error("Aucun résultat à afficher. Exécutez d'abord evaluate().")
            return
            
        if not output_path:
            output_path = os.path.join(RETRIEVER_EVAL_DIR, f"by_type_{metric}.png")
        
        # Extraire les données
        types = []
        scores = []
        for q_type, metrics in self.results["by_question_type"].items():
            if metric in metrics:
                types.append(q_type)
                scores.append(metrics[metric])
        
        # Créer le graphique
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=types, y=scores, palette="viridis")
        
        # Ajouter les étiquettes et le titre
        plt.title(f"{metric} par Type de Question", fontsize=16)
        plt.xlabel("Type de Question", fontsize=14)
        plt.ylabel(f"Score ({metric})", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Ajouter les valeurs sur chaque barre
        for i, v in enumerate(scores):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Graphique sauvegardé dans {output_path}")
        plt.close()
    
    def plot_by_subject(self, metric="recall@3", output_path=None):
        """Génère un graphique des performances par sujet de question."""
        if not self.results or "by_question_subject" not in self.results:
            logger.error("Aucun résultat à afficher. Exécutez d'abord evaluate().")
            return
            
        if not output_path:
            output_path = os.path.join(RETRIEVER_EVAL_DIR, f"by_subject_{metric}.png")
        
        # Extraire les données
        subjects = []
        scores = []
        for subject, metrics in self.results["by_question_subject"].items():
            if metric in metrics:
                subjects.append(subject)
                scores.append(metrics[metric])
        
        # Trier par score pour une meilleure lisibilité
        sorted_data = sorted(zip(subjects, scores), key=lambda x: x[1], reverse=True)
        subjects = [s for s, _ in sorted_data]
        scores = [s for _, s in sorted_data]
        
        # Créer le graphique
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x=subjects, y=scores, palette="viridis")
        
        # Ajouter les étiquettes et le titre
        plt.title(f"{metric} par Sujet de Question", fontsize=16)
        plt.xlabel("Sujet de Question", fontsize=14)
        plt.ylabel(f"Score ({metric})", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Ajouter les valeurs sur chaque barre
        for i, v in enumerate(scores):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Graphique sauvegardé dans {output_path}")
        plt.close()
    
    def save_results(self, output_path=None):
        """Sauvegarde les résultats au format JSON."""
        if not self.results:
            logger.error("Aucun résultat à sauvegarder. Exécutez d'abord evaluate().")
            return
            
        if not output_path:
            output_path = os.path.join(RETRIEVER_EVAL_DIR, "retriever_results.json")
            
        # Convertir les valeurs numpy en float pour assurer la sérialisation JSON
        results_serializable = {
            "overall": {k: float(v) if not isinstance(v, dict) else v for k, v in self.results["overall"].items()},
            "by_question_type": {
                q_type: {k: float(v) for k, v in metrics.items()}
                for q_type, metrics in self.results["by_question_type"].items()
            },
            "by_question_subject": {
                subject: {k: float(v) for k, v in metrics.items()}
                for subject, metrics in self.results["by_question_subject"].items()
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Résultats sauvegardés dans {output_path}")
    
    def generate_report(self, output_path=None):
        """Génère un rapport Markdown avec les résultats de l'évaluation."""
        if not self.results:
            logger.error("Aucun résultat pour générer un rapport. Exécutez d'abord evaluate().")
            return
            
        if not output_path:
            output_path = os.path.join(RETRIEVER_EVAL_DIR, "retriever_report.md")
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Rapport d'Évaluation du Retriever Visuel ColQwen2\n\n")
            
            # Section résumé global
            f.write("## 1. Résumé des performances globales\n\n")
            f.write("| Métrique | Score |\n")
            f.write("|----------|------:|\n")
            
            # Séparer les métriques d'embedding
            embedding_metrics = {}
            standard_metrics = {}
            for metric, score in self.results["overall"].items():
                if metric == "embedding_similarity":
                    embedding_metrics = score
                else:
                    standard_metrics[metric] = score
            
            # Afficher les métriques standard
            for metric, score in sorted(standard_metrics.items()):
                f.write(f"| {metric} | {score:.4f} |\n")
            
            # Section métriques de similarité d'embedding
            if embedding_metrics:
                f.write("\n## 2. Métriques de similarité des embeddings\n\n")
                f.write("| Métrique | Score |\n")
                f.write("|----------|------:|\n")
                
                for metric, score in sorted(embedding_metrics.items()):
                    f.write(f"| {metric} | {score:.4f} |\n")
                
                # Ajouter une analyse des métriques d'embedding
                f.write("\n### Analyse des métriques d'embedding\n\n")
                
                # Analyser la similarité cosinus
                cosine_sim = embedding_metrics.get("cosine_sim_mean", 0)
                if cosine_sim > 0.8:
                    f.write("✅ **Forte similarité cosinus** : Les embeddings des questions et des images correspondantes montrent une forte similarité, indiquant que le modèle comprend bien le lien entre le texte et le contenu visuel.\n\n")
                elif cosine_sim > 0.6:
                    f.write("✓ **Bonne similarité cosinus** : Les embeddings des questions et des images correspondantes montrent une bonne similarité, mais il reste une marge d'amélioration.\n\n")
                else:
                    f.write("❌ **Faible similarité cosinus** : Les embeddings des questions et des images correspondantes ne sont pas suffisamment similaires, suggérant que le modèle pourrait avoir du mal à aligner correctement le texte et l'image.\n\n")
                
                # Analyser le matching score
                matching = embedding_metrics.get("matching_score", 0)
                if matching > 0.8:
                    f.write("✅ **Excellent matching score** : Pour la majorité des questions, l'image la plus similaire est effectivement celle qui correspond à la question.\n\n")
                elif matching > 0.6:
                    f.write("✓ **Bon matching score** : Pour une bonne partie des questions, l'image la plus similaire correspond à la question, mais il reste une marge d'amélioration.\n\n")
                else:
                    f.write("❌ **Faible matching score** : Le modèle a du mal à associer les questions avec les images correspondantes, ce qui peut indiquer un problème dans l'encodage des relations texte-image.\n\n")
            
            # Section par type de question (décalée à cause de la section embedding)
            f.write("\n## 3. Performances par type de question\n\n")
            
            for q_type, metrics in sorted(self.results["by_question_type"].items()):
                f.write(f"### Type: {q_type}\n\n")
                f.write("| Métrique | Score |\n")
                f.write("|----------|------:|\n")
                
                for metric, score in sorted(metrics.items()):
                    f.write(f"| {metric} | {score:.4f} |\n")
                
                f.write("\n")
            
            # Section par sujet de question
            f.write("\n## 4. Performances par sujet de question\n\n")
            
            for subject, metrics in sorted(self.results["by_question_subject"].items()):
                f.write(f"### Sujet: {subject}\n\n")
                f.write("| Métrique | Score |\n")
                f.write("|----------|------:|\n")
                
                for metric, score in sorted(metrics.items()):
                    f.write(f"| {metric} | {score:.4f} |\n")
                
                f.write("\n")
            
            # Section analyse
            f.write("\n## 5. Analyse et recommandations\n\n")
            
            # MRR global
            mrr = standard_metrics.get("mrr", 0)
            f.write(f"### Mean Reciprocal Rank (MRR): {mrr:.4f}\n\n")
            if mrr > 0.7:
                f.write("✅ **Excellent**: Le retriever place généralement un document pertinent très haut dans les résultats.\n\n")
            elif mrr > 0.5:
                f.write("✓ **Bon**: Le retriever place souvent un document pertinent assez haut dans les résultats.\n\n")
            else:
                f.write("❌ **À améliorer**: Le retriever pourrait mieux classer les documents pertinents.\n\n")
            
            # Precision@1
            p1 = standard_metrics.get("precision@1", 0)
            f.write(f"### Precision@1: {p1:.4f}\n\n")
            if p1 > 0.7:
                f.write("✅ **Excellent**: Le premier document récupéré est généralement pertinent.\n\n")
            elif p1 > 0.5:
                f.write("✓ **Bon**: Le premier document récupéré est souvent pertinent.\n\n")
            else:
                f.write("❌ **À améliorer**: Le premier document récupéré n'est pas suffisamment pertinent.\n\n")
            
            # Recall@3
            r3 = standard_metrics.get("recall@3", 0)
            f.write(f"### Recall@3: {r3:.4f}\n\n")
            if r3 > 0.7:
                f.write("✅ **Excellent**: Les trois premiers documents récupérés contiennent la majorité des documents pertinents.\n\n")
            elif r3 > 0.5:
                f.write("✓ **Bon**: Les trois premiers documents récupérés contiennent une bonne partie des documents pertinents.\n\n")
            else:
                f.write("❌ **À améliorer**: Les trois premiers documents récupérés manquent plusieurs documents pertinents.\n\n")
        
        logger.info(f"Rapport généré dans {output_path}")

def evaluate_retriever(pdf_path, json_path):
    """
    Évalue les performances du retriever visuel sur un dataset.
    
    Args:
        pdf_path: Chemin vers le PDF à analyser
        json_path: Chemin vers le fichier JSON contenant les questions et références
        
    Returns:
        Les résultats de l'évaluation
    """
    # Charger le modèle et le processor
    model, processor = load_model_and_processor()
    
    # Convertir le PDF en images
    images = convert_pdf_to_images(pdf_path)
    
    # Générer les embeddings des images
    embeddings_list = generate_embeddings(model, processor, images, pdf_path)
    
    # Charger les données d'évaluation
    with open(json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    
    # Récupérer les documents pertinents pour chaque question
    retriever_results = {}
    retriever_similarities = {}
    
    # Collecter tous les embeddings de questions et d'images pour l'analyse de similarité
    question_embeddings = []
    image_embeddings = []
    questions_with_embeddings = set()
    
    for item in tqdm(eval_data, desc="Evaluation du retriever sur chaque question"):
        question_id = item["Question_ID"]
        question = item["Question"]
        
        # Préparer les inputs pour la question
        question_inputs = processor.process_queries([question]).to(model.device)
        
        # Obtenir l'embedding de la question
        with torch.no_grad():
            question_embedding = model(**question_inputs)
        
        # Récupérer les top-5 documents pour cette question
        scores = processor.score_multi_vector(question_embedding, embeddings_list)
        top_k_indices = scores[0].topk(5).indices.tolist()
        top_k_sims = scores[0].topk(5).values.tolist()
        
        # Stocker les résultats
        retriever_results[question_id] = top_k_indices
        retriever_similarities[question_id] = top_k_sims
        
        # Pour les métriques d'embedding, associer chaque question avec le document attendu
        expected_docs_top1 = parse_document_ids(item.get("Expected_documents_top1", ""))
        if expected_docs_top1 and question_id not in questions_with_embeddings:
            # Calculer la moyenne sur la dimension de séquence pour obtenir un vecteur uniforme
            question_emb_mean = question_embedding[0].mean(dim=0).cpu()
            question_embeddings.append(question_emb_mean)
            
            # Utiliser le premier document attendu comme référence
            try:
                doc_idx = int(expected_docs_top1[0])
                if 0 <= doc_idx < len(embeddings_list):
                    # Calculer la moyenne pour l'embedding de l'image également
                    image_emb_mean = embeddings_list[doc_idx].mean(dim=0).cpu()
                    image_embeddings.append(image_emb_mean)
                    questions_with_embeddings.add(question_id)
            except (ValueError, IndexError):
                logger.warning(f"Document ID invalide pour la question {question_id}: {expected_docs_top1[0]}")
    
    # Calculer les métriques de similarité d'embedding si nous avons des paires
    embedding_similarity_metrics = None
    if question_embeddings and image_embeddings:
        try:
            # Vérifier les dimensions
            logger.info(f"Nombre d'embeddings de questions: {len(question_embeddings)}")
            logger.info(f"Nombre d'embeddings d'images: {len(image_embeddings)}")
            
            # Regrouper les embeddings en tenseurs
            q_embeddings = torch.stack(question_embeddings)
            i_embeddings = torch.stack(image_embeddings)
            
            logger.info(f"Forme des embeddings de questions: {q_embeddings.shape}")
            logger.info(f"Forme des embeddings d'images: {i_embeddings.shape}")
            
            # S'assurer que nous avons le même nombre de questions et d'images
            min_count = min(len(q_embeddings), len(i_embeddings))
            q_embeddings = q_embeddings[:min_count]
            i_embeddings = i_embeddings[:min_count]
            
            # Calculer les métriques de similarité
            embedding_similarity_metrics = compute_embedding_similarity(q_embeddings, i_embeddings)
            logger.info(f"Métriques de similarité calculées sur {min_count} paires question-image")
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques de similarité: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Sauvegarder les résultats bruts du retriever
    raw_results_path = os.path.join(RETRIEVER_EVAL_DIR, "raw_retriever_results.json")
    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": {str(k): [int(i) for i in v] for k, v in retriever_results.items()},
            "similarities": {str(k): [float(s) for s in v] for k, v in retriever_similarities.items()}
        }, f, indent=2)
    logger.info(f"Résultats bruts du retriever sauvegardés dans {raw_results_path}")
    
    # Initialiser l'évaluateur
    evaluator = RetrieverEvaluator(json_path)
    
    # Évaluer le retriever
    results = evaluator.evaluate(retriever_results, k_values=[1, 3, 5], embedding_metrics=embedding_similarity_metrics)
    
    # Visualiser les résultats globaux
    evaluator.plot_overall_results()
    
    # Visualiser les résultats par type de question
    evaluator.plot_by_type(metric="recall@3")
    evaluator.plot_by_type(metric="precision@1")
    
    # Visualiser les résultats par sujet
    evaluator.plot_by_subject(metric="recall@3")
    
    # Sauvegarder les résultats
    evaluator.save_results()
    
    # Générer un rapport détaillé
    evaluator.generate_report()
    
    return results, retriever_results, retriever_similarities

if __name__ == "__main__":
    # Chemins des fichiers
    PDF_PATH = "/Users/rayanebouaita/Documents/CentraleSupélec/PFE/OpenFinanceAI/Assets/data_test/pdfs/AMEX_EMR_2023.pdf"  
    JSON_PATH = "/Users/rayanebouaita/Documents/CentraleSupélec/PFE/OpenFinanceAI/Assets/data_test/ardian_dataset_final.json"  
    
    # Évaluer le retriever
    results, retriever_results, similarities = evaluate_retriever(PDF_PATH, JSON_PATH)
    
    # Afficher les résultats globaux
    print("\nRésultats globaux du retriever:")
    for metric, score in results["overall"].items():
        if metric == "embedding_similarity":
            print("\nMétriques de similarité des embeddings:")
            for embed_metric, embed_score in score.items():
                print(f"  {embed_metric}: {embed_score:.4f}")
        else:
            print(f"{metric}: {score:.4f}")
    
    # Afficher les meilleurs résultats par type de question
    print("\nMeilleurs résultats par type de question (Recall@3):")
    for q_type, metrics in results["by_question_type"].items():
        if "recall@3" in metrics:
            print(f"{q_type}: {metrics['recall@3']:.4f}")