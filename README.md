# OpenFinanceAI

![Example Streamlit Interface](Assets/Images/Argimi.png)


Open-source project focused on developing a generative AI agent for financial analysis and responsible investment recommendations.

---

<div style="display: flex; gap: 10px;">
  <a href="https://huggingface.co/datasets/artefactory/Argimi-Ardian-Finance-10k-text">[Dataset]</a>
  <a href="https://github.com/LucasTramonte/OpenFinanceAI/blob/main/Assets/report/final_report.pdf">[Report]</a>
</div>

## Table of Contents
- [Usage Instructions](#usage-instructions)
  - [Main Script](#main-script)
  - [Evaluation Scripts](#evaluation-scripts)
  - [Retriever Evaluation](#retriever-evaluation)
  - [FinQA Evaluation](#finqa-evaluation)
- [Output Files Explained](#output-files-explained)
- [Metrics for Evaluation](#metrics-for-evaluation)
  - [Retriever](#retriever)
  - [Generator](#generator)
- [Ranking Methods](#ranking-methods)
  - [Borda Count](#borda-count)
- [Aggregation](#aggregation)
  - [JSON Structure](#json-structure)
  - [Evaluation Summary](#evaluation-summary)
    - [Global Metrics Table](#global-metrics-table)
    - [Short Metrics](#short-metrics)
    - [Long Metrics](#long-metrics)
- [Streamlit Application](#streamlit-application)
- [Authors](#authors)
- [References](#references)

## Usage Instructions

### Main script

We are utilizing the [DCE](https://dce.pages.centralesupelec.fr/) GPU (GPU RAM : 24 Gb) provided by CentraleSupélec for training our models .

Install the required packages:

```bash
pip install -r requirements.txt
```

Execute the main script as follows

```bash
python main.py
```

### Evaluation Scripts
1. Evaluates the dataset using various metrics (e.g., ROUGE, BERTScore, numerical accuracy).

```bash
python eval_data_test.py
```
Results are saved in Assets\data_test\ardian_dataset_test_evaluation.json

2. Computes aggregated metrics (e.g., Borda scores) for model evaluation.

```bash
python generate_metrics.py
```
Results are saved in Assets\data_test\agreggation_metrics.json

3. Evaluates the FinQA dataset

```bash
python eval_finqa.py
```

Results are saved in Assets\output\FinQA

4. Analyzes metrics and generates visualizations for evaluation.

```bash
python eval_metrics.py
```
Visualizations are saved in Assets\output\metrics


### Retriever Evaluation

To evaluate the retrievers, we followed the official instructions provided in the [Vidore Benchmark Retriever README](https://github.com/illuin-tech/vidore-benchmark/blob/main/src/vidore_benchmark/retrievers/README.md). 

1. Install the required package:
   ```bash
   pip install "vidore-benchmark[all-retrievers]"
   ```
2. Ensure all necessary files are saved manually, as automatic saving is not handled by the process.
3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
4. Execute the retriever evaluation command:
   ```bash
   vidore-benchmark evaluate-retriever \
       --model-class colqwen2_personal \
       --model-name vidore/colqwen2-v1.0 \
       --dataset-name vidore/docvqa_test_subsampled \
       --split test
   ```

### FinQA Evaluation

In addition to evaluating the retriever, we also assess our model’s performance using [FinQA](https://finqasite.github.io/), which focuses on evaluating the Visual Language Model (VLM) directly rather than retrieval effectiveness. 

1. **Load the dataset** containing questions, answers, and associated PDF documents.
2. **Run the model** on the dataset without retrieval, directly processing the PDF input.
3. **Assess the generated answers** qualitatively to ensure coherence and accuracy.

Since FinQA does not rely on retrieval metrics, the objective is to verify if the model generates meaningful and contextually correct responses from the provided PDFs. This evaluation complements the retriever assessment by testing the end-to-end question-answering capability of our system.

#### FinQA Accuracy Results:

| Model | Accuracy |
|-------|----------|
| OpenGVLab/InternVL2_5-8B-MPO | 20.00% |
| Qwen/Qwen2.5-VL-7B-Instruct | 28.00% |
| Qwen/Qwen2-VL-2B-Instruct | 0.00% |
| google/gemma-3-4b-it | 26.00% |

## Output Files Explained

The output directory contains the following key files and folders:

1. **`relevant_documents/`**
   - **Description**: Contains the top-K most relevant images extracted from the input PDF.
   - **Details**: These images are determined based on similarity scores with the provided query.

2. **`similarity_maps/`**
   - **Description**: Contains visualizations of token similarity maps.
   - **Details**: Each map highlights the most relevant areas of a document for each token in the query.

3. **`generated_responses.txt`**
   - **Description**: A text file containing the final generated responses based on the query.
   - **Details**: Combines information from all top-K relevant images to produce a single, coherent answer.

4. **`similarity_scores.txt`**
   - **Description**: A detailed log of similarity scores for each token in the query and the corresponding document sections.
   - **Format**:
     ```plaintext
     Document 1, Token #1 (`<bos>`): MaxSim score = 0.28
     Document 1, Token #2 (`Query`): MaxSim score = 0.24
     ...
     ```

5. **`FinQA/`**
   - **Description**: Contains the results of the FinQA evaluation.
   - **Details**: This directory includes subfolders with a prompt, target, and generated answer for each sample and for each model evaluated, as well as a final comparison results file.
   - **Subfolders**:
     - **`Qwen2VL/`**: Contains the results for the Qwen2VL model.
     - **`Qwen2_5VL/`**: Contains the results for the Qwen2_5VL model.
     - **`evaluation_results.txt`**: Final comparison results of the models' accuracy.

## Metrics for Evaluation

To evaluate the performance of our models, we use a combination of automatic metrics and ranking methods. These metrics help us assess the quality of generated responses and compare different models effectively.

### Retriever

We evaluated our personal retriever on the ViDoRe benchmark as stated above in order to confirm the results obtained in the orignial paper.

Furthermore, we are using 4 metrics to evaluate its performance with our datatset of questions:

#### MMR (Mean Reciprocal Rank)
- Measures the effectiveness of the retriever in returning relevant documents at the top of the ranked list.
- Calculated as the reciprocal of the rank of the first relevant document.
- Higher values indicate better performance (range: 0-1).

#### Precision at 1
- Measures the relevance of the top retrieved document.
- Calculated as the proportion of relevant documents among the first retrieved document.
- Higher values indicate better precision (range: 0-1).

#### Recall at 3
- Measures the ability of the retriever to find all relevant documents among the top 3 retrieved.
- Calculated as the proportion of relevant documents found among the top 3 retrieved.
- Higher values indicate better recall (range: 0-1).

#### Embeddings Comparison
- Compares the embeddings of the query with those of the retrieved documents.
- Uses cosine similarity to measure the semantic closeness.
- Higher similarity values indicate better relevance of retrieved documents.

#### Retriever Evaluation Results

| Métrique | Standard | Scale à 10 PDFs | Différence |
|----------|--------:|----------------:|-----------:|
| exact_match_top1 | 0.3462 | 0.2308 | -0.1154 🔻 |
| mrr | 0.5814 | 0.4913 | -0.0901 🔻 |
| precision@1 | 0.4808 | 0.4231 | -0.0577 🔻 |
| precision@3 | 0.3205 | 0.2564 | -0.0641 🔻 |
| precision@5 | 0.2308 | 0.2000 | -0.0308 🔻 |
| recall@1 | 0.1603 | 0.1410 | -0.0193 🔻 |
| recall@3 | 0.3138 | 0.2497 | -0.0641 🔻 |
| recall@5 | 0.3699 | 0.3202 | -0.0497 🔻 |

For more detailed results, please refer to `comparative_report.md`.

### Generator

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1**: Measures overlap of unigrams between generated and reference text.  
- **ROUGE-2**: Measures overlap of bigrams between generated and reference text.  
- **ROUGE-L**: Measures the longest common subsequence between generated and reference text.  

#### BERTScore
- Computes similarity between generated and reference text using contextual embeddings from BERT.

#### Flan-T5 Perplexity
- Measures how well the generated text aligns with the expected output using the Flan-T5 model. The score is calculated as the inverse of perplexity (1 / perplexity)

#### Numerical Accuracy
- Specifically for tabular/chart questions, checks if numerical values in the generated response match the expected values within a **5% tolerance**.

#### Multi-modal faithfulness
- Binary score (0 or 1) that evaluates whether the model's response contains information present in the retrieved documents.
- 1: The response is faithful and only uses information present in the context (top 3 retrieved documents).
- 0: The response contains hallucinated information not present in the retrieved documents.
- This metric helps identify when the model generates information not grounded in the provided visual and textual context.
- Evaluation is performed by comparing the model's response against the information contained in the 3 retrieved documents to ensure factual accuracy.

## Ranking Methods
To compare models across multiple metrics, we use the **Borda Count method**:

### Borda Count
1. For each metric, models are ranked (1st, 2nd, 3rd).  
2. Points are assigned:  
   - 3 points for 1st place  
   - 2 point for 2nd place  
   - 1 point for 3rd place 
   - 0 point for 4th place 
3. The total Borda score is the sum of points across all metrics.

To provide a comprehensive evaluation of our models, we calculate three different Borda scores:

1. **Global Borda Score**
   - Uses all metrics across all question types
   - Provides an overall performance indicator

2. **Short-Answer Borda Score**
   - Uses metrics specific to short-answer questions:
     - ROUGE-1 (Short)
     - ROUGE-2 (Short)
     - ROUGE-L (Short)
     - String Presence (Short)
     - Numerical Accuracy (Short)
     - Faithfulness Score
   - Helps determine which model performs best for concise, factual answers

3. **Long-Answer Borda Score**
   - Uses metrics specific to long-answer questions:
     - ROUGE-1 (Long)
     - ROUGE-2 (Long)
     - ROUGE-L (Long)
     - BERTScore (Long)
     - Flan-T5 Perplexity (Long)
     - Faithfulness Score
   - Helps determine which model performs best for detailed explanations

This multi-faceted evaluation approach allows us to determine which model performs best overall, as well as which models excel at specific types of questions. The faithfulness score is integrated into both short and long answer evaluations to ensure that all models are assessed for their ability to avoid hallucinations and stick to the retrieved context.

## 3. Aggregation

### JSON Structure:
Each entry in the JSON file contains the following fields:
- **Model**: The name of the model being evaluated (e.g., `Answer_Qwen2`, `Answer_Qwen2.5`, `Answer_Gemma_4B`, etc.).
- **Metric**: The evaluation metric (e.g., `rouge1`, `rouge2`, `rougeL`, `bert`, `flan-t5`, `string_presence`, `numerical_acc`).
- **Type**: The type of question being evaluated:
  - `Overall`: Aggregated results across all question types.
  - `Short`: Results for short-answer questions.
  - `Long`: Results for long-answer questions.
- **Mean**: The average score for the given model, metric, and question type.
- **Borda**: The Borda score for the model, calculated based on its ranking across all metrics.

### Evaluation Summary

#### Global Metrics Table

| Model                     |   Borda_Global | Faithfulness_Avg |
|:--------------------------|---------------:|------------------|
| Answer_Qwen2              |             60 | 84.62%           |
| Answer_Qwen2.5            |             58 | 84.62%           |
| Answer_Gemma_4B           |             21 | 88.46%           |
| Answer_Gemma_12B          |             22 | 82.69%           |
| Answer_PV2                |             43 | 78.43%           |
| Answer_langchain_pipeline |              6 | 65.38%           |

#### Short Metrics

| Model                     | faithfulness | numerical_acc | rouge1  | rouge2  | rougeL  | string_presence |
|:--------------------------|:-------------|:--------------|:--------|:--------|:--------|:---------------|
| Answer_Gemma_12B          | 66.67%       | 44.44%        | 21.75%  | 11.16%  | 20.17%  | 27.78%         |
| Answer_Gemma_4B           | 77.78%       | 50.00%        | 23.68%  | 11.94%  | 21.74%  | 33.33%         |
| Answer_PV2                | 82.35%       | 61.11%        | 26.36%  | 15.16%  | 23.34%  | 38.89%         |
| Answer_Qwen2              | 77.78%       | 55.56%        | 34.94%  | 22.14%  | 32.79%  | 38.89%         |
| Answer_Qwen2.5            | 88.89%       | 66.67%        | 29.45%  | 17.33%  | 26.82%  | 72.22%         |
| Answer_langchain_pipeline | 77.78%       | 44.44%        | 19.64%  | 7.87%   | 18.21%  | 22.22%         |

#### Long metrics

| Model                     | bert    | faithfulness | flan-t5 | rouge1  | rouge2  | rougeL  |
|:--------------------------|:--------|:-------------|:--------|:--------|:--------|:--------|
| Answer_Gemma_12B          | 53.43%  | 91.18%       | 6.13%   | 25.35%  | 6.03%   | 16.58%  |
| Answer_Gemma_4B           | 51.24%  | 94.12%       | 5.69%   | 23.00%  | 4.02%   | 14.51%  |
| Answer_PV2                | 57.76%  | 76.47%       | 8.21%   | 28.66%  | 9.78%   | 19.51%  |
| Answer_Qwen2              | 59.09%  | 88.24%       | 8.26%   | 28.45%  | 8.53%   | 19.83%  |
| Answer_Qwen2.5            | 58.28%  | 82.35%       | 8.17%   | 29.38%  | 9.80%   | 20.43%  |
| Answer_langchain_pipeline | 51.65%  | 58.82%       | 8.92%   | 18.42%  | 3.65%   | 12.28%  |

For more detailed evaluation results, please refer to `tableau_metrics.md`.

## Streamlit Application

1. Go to  the [interactive desktop](https://dev.dce-cs.fr/pun/sys/dashboard/batch_connect/sys/bc_desktop/dce/session_contexts/new) from DCE 

2. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

Example screenshot of the Streamlit interface:

![Example Streamlit Interface](Assets/Images/example_streamlit_interface.png)

![Click here to watch the demo video](Assets/Images/demo_streamlit.mp4)


## Authors
- **Gabriel Trier**  
- **Lucas Tramonte**  
- **Rayane Bouaita**  

## References

- [1] Ragas:library to evaluate llm.
- [2] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema
Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al. Finqa: A dataset of numerical
reasoning over financial data. arXiv preprint arXiv:2109.00122, 2022. Available at https:
//arxiv.org/pdf/2109.00122.
- [3] Manuel Faysse. Analysis of the colpali: Efficient document retrieval with vision language models
paper. https://huggingface.co/blog/manu/colpali, 2024. Hugging Face Blog, accessed on
2024-12-18.
- [4] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and
Pierre Colombo. Colpali: Efficient document retrieval with vision language models. arXiv
preprint arXiv:2407.01449, 2024. Available at https://arxiv.org/abs/2407.01449.
- [5] N. Bradley Fox, Benjamin Bruyns, et al. Borda count: An evaluation of borda count variations
using ranked choice voting data. arXiv preprint arXiv:2501.00618v2, 2022. Available at https:
//arxiv.org/html/2501.00618v2.
- [6] Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via con-
textualized late interaction over bert. arXiv preprint arXiv:2004.12832, 2020. Available at
https://arxiv.org/abs/2004.12832.
- [7] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational
Linguistics.
- [8] Chen Ling, Xujiang Zhao, Jiaying Lu, Chengyuan Deng, Can Zheng, Junxiang Wang, Tanmoy
Chowdhury, Yun Li, Hejie Cui, Xuchao Zhang, et al. Domain specialization as the key to make
large language models disruptive: A comprehensive survey. arXiv preprint arXiv:2305.18703, 2023.


