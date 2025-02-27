import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
file_path = '../Assets/data_test/ardian_dataset_test.csv'
df = pd.read_csv(file_path)

# Function to evaluate the answers using cosine similarity
def evaluate_answers(row):
    results = {}
    vectorizer = TfidfVectorizer().fit_transform([str(row['Expected_Answer'])] + [str(row[model]) for model in ['Answer_Qwen2', 'Answer_Qwen2.5', 'Answer_OpenGVLab']])
    vectors = vectorizer.toarray()
    expected_vector = vectors[0]
    for i, model in enumerate(['Answer_Qwen2', 'Answer_Qwen2.5', 'Answer_OpenGVLab']):
        model_vector = vectors[i + 1]
        similarity = cosine_similarity([expected_vector], [model_vector])[0][0]
        results[model] = similarity
    return results

# Apply the evaluation function to each row
df['Evaluation'] = df.apply(evaluate_answers, axis=1)

# Calculate the average similarity for each model
average_similarity = {
    'Answer_Qwen2': df['Evaluation'].apply(lambda x: x['Answer_Qwen2']).mean(),
    'Answer_Qwen2.5': df['Evaluation'].apply(lambda x: x['Answer_Qwen2.5']).mean(),
    'Answer_OpenGVLab': df['Evaluation'].apply(lambda x: x['Answer_OpenGVLab']).mean(),
}

# Print the average similarity results
logging.info("Model Evaluation Average Similarity:")
for model, similarity in average_similarity.items():
    logging.info(f"{model}: {similarity * 100:.2f}%")

# Save the evaluation results to a new CSV file
output_file_path = '../Assets/data_test/ardian_dataset_test_evaluation.csv'
df.to_csv(output_file_path, index=False)