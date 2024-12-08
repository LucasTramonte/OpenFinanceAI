# OpenFinanceAI
This repository contains the code and resources for an open-source project focused on developing a generative AI agent for financial analysis and responsible investment recommendations.

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
python Colpali_Vision_RAG.py
```

### File Management
Use [WinSCP](https://winscp.net/eng/download.php) for file transfer between local and remote systems.

## Actual problem : 

Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.48it/s]
Image index : 12
Traceback (most recent call last):
  File "/usr/users/openfinanceai/tramonte_luc/Colpali_Vision_RAG.py", line 77, in <module>
    generated_ids = model.generate(**inputs, max_new_tokens=2)
    .
    .
    .
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 19.68 GiB. GPU 0 has a total capacity of 23.68 GiB of which 11.88 GiB is free. Including non-PyTorch memory, this process has 11.79 GiB memory in use. Of the allocated memory 11.44 GiB is allocated by PyTorch, and 43.59 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)