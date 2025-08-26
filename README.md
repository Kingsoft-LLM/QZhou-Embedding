---
license: apache-2.0
tags:
  - sentence-transformers
  - sentence-similarity
  - mteb
  - retriever
  - text-embeddings-inference
---
# QZhou-Embedding
<div align="center">
<img src="assets/image-1.png" width="800" height="300"></img>
</div>


## Introduction
We present <a href="https://huggingface.co/Kingsoft-LLM/QZhou-Embedding">QZhou-Embedding</a> (called "Qingzhou Embedding"), a general-purpose contextual text embedding model with exceptional text representation capabilities. Built upon the <a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct">Qwen2.5-7B-Instruct</a> foundation model, we designed a unified multi-task framework and developed a data synthesis pipeline leveraging LLM APIs, effectively enhancing model's text embedding capabilities. Additionally, we employ a two-stage training strategy, comprising initial retrieval-focused pretraining followed by full-task fine-tuning, enabling the embedding model to extend its capabilities based on robust retrieval performance. Our model achieves state-of-the-art results on the MTEB and CMTEB benchmarks, ranking first on both leaderboards(August 27, 2025).

**<span style="color:red">We will promptly release our technical report—stay tuned!</span>**  

## Basic Features

- Powerful text embedding capabilities；
- Long context: up to 8k context length；
- 7B parameter size

## Usage
### Completely reproduce the benchmark results
We provide detailed parameters and environment configurations so that you can run results that are completely consistent with the mteb leaderboard on your own machine, including configurations such as environment dependencies and model arguments.
#### Requirements
- Python: 3.10.12
- Sentence Transformers: 3.4.1
- Transformers: 4.51.1
- PyTorch: 2.7.1
- Accelerate: 1.3.0
- Datasets: 3.2.0
- Tokenizers: 0.21.2
- mteb: 1.38.30
#### Transformers model load arguments
torch_dtype=torch.bfloat16<br>
attn_implementation='sdpa'<br>
**NOTE：** The ranking results use the sdpa mode. Other modes ('eager', 'flash_attention_2') may have deviations in results, but still keep the overall performance consistent.
#### Instruction Adding Rules
Details can be found on our <a href="https://github.com/Kingsoft-LLM/QZhou-Embedding">GitHub</a>.
#### Evaluation code usage
Find our benchmark evaluation code on <a href="https://github.com/Kingsoft-LLM/QZhou-Embedding">GitHub</a>. The mteb benchmark script is **run_mteb_all_v2.py**, and the cmteb benchmark script is **run_cmteb_all.py**. Run the following command:
```bash
POOLING_MODE=mean
normalize=true
use_instruction=true
export TOKENIZERS_PARALLELISM=true

model_name_or_path=<model dir>

python3 ./run_cmteb_all.py \
    --model_name_or_path ${model_name_or_path}  \
    --pooling_mode ${POOLING_MODE} \
    --normalize ${normalize} \
    --use_instruction ${use_instruction} \
    --output_dir <output dir>

python3 ./run_mteb_all_v2.py \
    --model_name_or_path ${model_name_or_path}  \
    --pooling_mode ${POOLING_MODE} \
    --normalize ${normalize} \
    --use_instruction ${use_instruction} \
    --output_dir <output dir>
```
The "<>" should be replaced with your actual setting.<br>
This is a general script that can be used to evaluate other huggingface embedding models, but you need to ensure that the pooling and other configurations are correct.

### Sentence-transformers

```py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "Kingsoft-LLM/QZhou-Embedding",
    model_kwargs={"device_map": "cuda", "trust_remote_code": True},
    tokenizer_kwargs={"padding_side": "left", "trust_remote_code": True},
    trust_remote_code=True
)

queries = [
    "What is photosynthesis?",
    "Who invented the telephone?",
]
documents = [
    "Photosynthesis is the process by which green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen. This biochemical reaction occurs in chloroplasts.",
    "Alexander Graham Bell is credited with inventing the first practical telephone in 1876, receiving US patent number 174,465 for his device."
]

query_embeddings = model.encode(queries, prompt_name="query", normalize_embeddings=True)
document_embeddings = model.encode(documents, normalize_embeddings=True)

similarity = model.similarity(query_embeddings, document_embeddings)
```

### Huggingface Transformers

```py
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def mean_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:

    seq_lengths = attention_mask.sum(dim=-1)
    return torch.stack(
                [
                    last_hidden_states[i, -length:, :].sum(dim=0) / length
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
            )


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = [
    get_detailed_instruct(task, 'What is photosynthesis?'),
    get_detailed_instruct(task, 'Who invented the telephone?')
]

documents = [
    "Photosynthesis is the process by which green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen. This biochemical reaction occurs in chloroplasts.",
    "Alexander Graham Bell is credited with inventing the first practical telephone in 1876, receiving US patent number 174,465 for his device."
]

input_texts = queries + documents

tokenizer = AutoTokenizer.from_pretrained('Kingsoft-LLM/QZhou-Embedding', padding_side='left', trust_remote_code=True)
model = AutoModel.from_pretrained('Kingsoft-LLM/QZhou-Embedding', trust_remote_code=True, device_map='cuda')

batch_dict = tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=8192,
    return_tensors="pt",
)
batch_dict.to(model.device)
outputs = model(**batch_dict)
embeddings = mean_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T)
```
