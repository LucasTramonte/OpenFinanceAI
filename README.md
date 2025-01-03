# OpenFinanceAI
This repository contains the code and resources for an open-source project focused on developing a generative AI agent for financial analysis and responsible investment recommendations.

---

## Table of Contents
- [Overview](#openfinanceai)
- [Project Structure](#project-structure)
- [Usage Instructions](#usage-instructions)
- [Output Files Explained](#output-files-explained)
- [Problems](#actual-problems)
- [Authors](#authors) 

## Project Structure

```plaintext
OpenFinanceAI/

├── Colpali_Qwen.py                 # Main script to process PDFs and generate responses
├── Colpali_Qwen_ResponsePerPage.py # Alternate script for per-page responses (under testing)
├── output/                         # Directory for generated results
│   ├── relevant_documents/         # Stores the top-K most relevant document images
│   ├── similarity_maps/            # Stores similarity map visualizations
│   ├── generated_responses.txt     # Final generated responses
│   └── similarity_scores.txt       # Similarity scores for each token and document
├── data/                           # Dataset provided
├── data_test/                      # Smaller dataset to run to test PDFs
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## Usage Instructions

### GPU Access
To request a GPU session:
```bash
srun -p gpu_inter -t 00:30:00 --pty bash
```
To request a specific GPU node:

```bash
srun -p gpu_inter -t 00:30:00 --nodelist=sh03 --pty bash
```

Execute the main script as follows

```bash
python Colpali_Qwen.py
```
Please note that there is another vesrion that we are testing ```Colpali_Qwen_ResponsePerPage.py``` that generates a response for every single K documents. 
### File Management
Use [WinSCP](https://winscp.net/eng/download.php) for file transfer between local and remote systems.

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

## Actual problems : 

This problem occurs when a lot of jobs are being launched at the same time in the DCE! However we managed to run our code and obtain generation results with Qwen2-VL-2B-Instruct. 

Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.48it/s]
Image index : 12
Traceback (most recent call last):
  File "/usr/users/openfinanceai/tramonte_luc/Colpali_Vision_RAG.py", line 77, in <module>
    generated_ids = model.generate(**inputs, max_new_tokens=2)
    .
    .
    .
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 19.68 GiB. GPU 0 has a total capacity of 23.68 GiB of which 11.88 GiB is free. Including non-PyTorch memory, this process has 11.79 GiB memory in use. Of the allocated memory 11.44 GiB is allocated by PyTorch, and 43.59 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## Authors
- **Gabriel Trier**  
- **Lucas Tramonte**  
- **Rayane Bouaita**  

