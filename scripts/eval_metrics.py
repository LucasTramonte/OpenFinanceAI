import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON file with the correct encoding
json_file_path = "../Assets/data_test/ardian_dataset_test_evaluation_final.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Create output directories
output_dir = "../Assets/output/metrics"
os.makedirs(output_dir, exist_ok=True)

models = ["Qwen2", "Qwen2.5", "Gemma_4B", "Gemma_12B"]
metrics = ["rouge1", "rouge2", "rougeL", "string_presence", "numerical_acc", "bert", "flan-t5"]
personal_scores = [f"Score_perso_{model}" for model in models]

# Prepare a DataFrame for analysis
rows = []
for entry in data[:20]:  # Process only the first 20 questions
    for model in models:
        row = {"Question_ID": entry.get("Question_ID", None), "Model": model}
        row["Question_Type"] = entry.get("Question_Type", None)
        row["Question_subject"] = entry.get("Question_subject", None)
        row["Personal_Score"] = entry.get(f"Score_perso_{model}", None)
        for metric in metrics:
            row[metric] = entry.get(f"Answer_{model}_{metric}", None)
        rows.append(row)

df = pd.DataFrame(rows)

# Handle missing or NaN values
#df.fillna(0.0, inplace=True)

# Filter rows with valid personal scores
df = df[df["Personal_Score"].notnull()]

# Normalize metrics for comparability
for metric in metrics:
    if df[metric].notnull().any():
        df[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

# Rank metrics by correlation with Personal_Score
correlations = {}
for metric in metrics:
    correlations[metric] = df[["Personal_Score", metric]].corr(method="spearman").iloc[0, 1]

ranked_metrics = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

# Save correlation heatmap as an image
plt.figure(figsize=(12, 8))
sns.heatmap(df[metrics + ["Personal_Score"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

# Save correlations to a JSON file
correlation_json_path = os.path.join(output_dir, "correlations.json")
with open(correlation_json_path, "w", encoding="utf-8") as f:
    json.dump(correlations, f, indent=4)

# Save ranked metrics to a JSON file
ranked_metrics_json_path = os.path.join(output_dir, "ranked_metrics.json")
with open(ranked_metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(ranked_metrics, f, indent=4)

# Print results
print("Correlation between Personal Score and Metrics:")
for metric, corr in correlations.items():
    print(f"{metric}: {corr:.2f}")

print("\nRanked Metrics by Correlation with Personal Score:")
for rank, (metric, corr) in enumerate(ranked_metrics, start=1):
    print(f"{rank}. {metric}: {corr:.2f}")


print(f"\nHeatmap saved to: {heatmap_path}")
print(f"Correlations saved to: {correlation_json_path}")
print(f"Ranked metrics saved to: {ranked_metrics_json_path}")