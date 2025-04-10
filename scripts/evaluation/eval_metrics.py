import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the JSON file with the correct encoding
json_file_path = "../../Assets/data_test/ardian_dataset_final_evaluation.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Create output directories
output_dir = "../../Assets/output/metrics"
os.makedirs(output_dir, exist_ok=True)

models = ["Qwen2", "Qwen2.5", "Gemma_4B", "Gemma_12B"]
metrics = ["rouge1", "rouge2", "rougeL", "string_presence", "numerical_acc", "bert", "flan-t5"]
personal_scores = [f"Score_perso_{model}" for model in models]

# Prepare a DataFrame for analysis
rows = []
for entry in data[:52]:  # Process only the first 52 entries
    for model in models:
        row = {"Question_ID": entry.get("Question_ID", None), "Model": model}
        row["Question_Type"] = entry.get("Question_Type", None)
        row["Question_subject"] = entry.get("Question_subject", None)
        row["Personal_Score"] = entry.get(f"Score_perso_{model}", None)
        for metric in metrics:
            row[metric] = entry.get(f"Answer_{model}_{metric}", None)
        rows.append(row)

df = pd.DataFrame(rows)

# Filter rows with valid personal scores
df = df[df["Personal_Score"].notnull()]

# Normalize metrics for comparability
for metric in metrics:
    if df[metric].notnull().any():
        df[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

# Function to calculate and save results for a given correlation method
def calculate_and_save_results(method):
    # Create a subfolder for the method
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    # Rank metrics by correlation with Personal_Score
    correlations = {}
    for metric in metrics:
        correlations[metric] = df[["Personal_Score", metric]].corr(method=method).iloc[0, 1]

    ranked_metrics = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    # Save correlation heatmap as an image
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[metrics + ["Personal_Score"]].corr(method=method), annot=True, cmap="coolwarm")
    plt.title(f"Correlation Heatmap - {method.capitalize()} method")
    plt.tight_layout()
    heatmap_path = os.path.join(method_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    # Save correlations to a JSON file
    correlation_json_path = os.path.join(method_dir, "correlations.json")
    with open(correlation_json_path, "w", encoding="utf-8") as f:
        json.dump(correlations, f, indent=4)

    # Save ranked metrics to a JSON file
    ranked_metrics_json_path = os.path.join(method_dir, "ranked_metrics.json")
    with open(ranked_metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(ranked_metrics, f, indent=4)

    # Print results
    print(f"\nCorrelation between Personal Score and Metrics ({method.capitalize()}):")
    for metric, corr in correlations.items():
        print(f"{metric}: {corr:.2f}")

    print(f"\nRanked Metrics by Correlation with Personal Score ({method.capitalize()}):")
    for rank, (metric, corr) in enumerate(ranked_metrics, start=1):
        print(f"{rank}. {metric}: {corr:.2f}")

    print(f"\nHeatmap saved to: {heatmap_path}")
    print(f"Correlations saved to: {correlation_json_path}")
    print(f"Ranked metrics saved to: {ranked_metrics_json_path}")

# Calculate and save results for Spearman and Pearson methods
calculate_and_save_results("spearman")
calculate_and_save_results("pearson")

# Recompute correlations for the entire df
pearson_corr = df[["Personal_Score", "string_presence"]].corr(method="pearson").iloc[0, 1]
spearman_corr = df[["Personal_Score", "string_presence"]].corr(method="spearman").iloc[0, 1]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

# --- Plot 1: Pearson ---
sns.scatterplot(
    data=df,
    x="Personal_Score",
    y="string_presence",
    alpha=0.7,
    ax=axes[0]
)
# Add a regression line (without additional scatter points)
sns.regplot(
    data=df,
    x="Personal_Score",
    y="string_presence",
    scatter=False,
    line_kws={"color": "red"},
    ci=None,
    ax=axes[0]
)
axes[0].set_title(f"Pearson Correlation: {pearson_corr:.2f}", fontsize=12)
axes[0].set_xlabel("Personal Score", fontsize=11)
axes[0].set_ylabel("String Presence", fontsize=11)

# --- Plot 2: Spearman ---
sns.scatterplot(
    data=df,
    x="Personal_Score",
    y="string_presence",
    alpha=0.7,
    ax=axes[1],
    color="green"
)
# Add a regression line
sns.regplot(
    data=df,
    x="Personal_Score",
    y="string_presence",
    scatter=False,
    line_kws={"color": "black"},
    ci=None,
    ax=axes[1]
)
axes[1].set_title(f"Spearman Correlation: {spearman_corr:.2f}", fontsize=12)
axes[1].set_xlabel("Personal Score", fontsize=11)
axes[1].set_ylabel("String Presence", fontsize=11)

plt.suptitle("Comparing Pearson vs. Spearman (Personal Score vs. String Presence)", fontsize=14)
plt.tight_layout()

# Save and/or show
comparison_plot_path = os.path.join(output_dir, "correlation_string_presence_comparison.png")
plt.savefig(comparison_plot_path, dpi=150)
plt.show()

print(f"Comparison scatterplot saved to: {comparison_plot_path}")

# Create a copy with random jitter
df_jittered = df.copy()
df_jittered["Personal_Score"] = df_jittered["Personal_Score"] + np.random.uniform(-0.01, 0.01, size=len(df_jittered))
df_jittered["string_presence"] = df_jittered["string_presence"] + np.random.uniform(-0.01, 0.01, size=len(df_jittered))

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Pearson with jitter
sns.scatterplot(
    data=df_jittered,
    x="Personal_Score",
    y="string_presence",
    alpha=0.6,
    ax=axes[0]
)
sns.regplot(
    data=df_jittered,
    x="Personal_Score",
    y="string_presence",
    scatter=False,
    line_kws={"color": "red"},
    ci=None,
    ax=axes[0]
)
axes[0].set_title(f"Pearson Corr (Jittered): {pearson_corr:.2f}")

# Spearman with jitter
sns.scatterplot(
    data=df_jittered,
    x="Personal_Score",
    y="string_presence",
    alpha=0.6,
    ax=axes[1],
    color="green"
)
sns.regplot(
    data=df_jittered,
    x="Personal_Score",
    y="string_presence",
    scatter=False,
    line_kws={"color": "black"},
    ci=None,
    ax=axes[1]
)
axes[1].set_title(f"Spearman Corr (Jittered): {spearman_corr:.2f}")

jitter_plot_path = os.path.join(output_dir, "personal_score_vs_string_presence_jittered.png")
plt.savefig(jitter_plot_path, dpi=150)
plt.close()

print(f"Jittered scatter plot saved to: {jitter_plot_path}")

plt.figure(figsize=(8, 6))
sns.violinplot(x="string_presence", y="Personal_Score", data=df, inner="quartile")
plt.xlabel("String Presence (0 ou 1)")
plt.ylabel("Personal Score")
plt.title("Violin plot du Personal Score en fonction de la présence de la chaîne")
plt.show()
plt.savefig(
    os.path.join(output_dir, "violin_plot_string_presence_vs_personal_score.png"),
    dpi=150,
)


plt.figure(figsize=(8, 6))
sns.barplot(x="string_presence", y="Personal_Score", data=df, errorbar=("ci", 95))
plt.xlabel("String Presence (0 ou 1)")
plt.ylabel("Personal Score (moyenne)")
plt.title("Moyenne du Personal Score par catégorie de String Presence")
plt.show()
plt.savefig(
    os.path.join(output_dir, "barplot_string_presence_vs_personal_score.png"), dpi=150
)