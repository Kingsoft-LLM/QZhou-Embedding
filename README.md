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
We have released <a href="https://huggingface.co/Kingsoft-LLM/QZhou-Embedding">QZhou-Embedding</a> (called "Qingzhou Embedding"), a large-scale text embedding model designed for general use，excelling at various text embedding tasks (retrieval, re-ranking, sentence similarity, and classification). We modified the model structure based on Qwen2Model by changing the causal attention mechanism to bidirectional attention, so that each token can capture the global context semantics. The new module is named QZhouModel.Leveraging the general language capabilities of its underlying model, and pre-trained on massive amounts of text, QZhou-Embedding achieves even more powerful text embedding representations. QZhou-Embedding is continuously trained using millions of high-quality open-source embedding datasets and over 5 million high-quality synthetic data (using two synthetic techniques: rewriting and expansion). Initial retrieval training provides the model with a foundation for query-doc semantic matching capabilities. Later, multi-dimensional training such as STS and clustering, helps the model achieve continuous breakthroughs in various tasks. QZhou-Embedding is a 7B model and can embed long text vectors up to 8k in size. It achieved the highest average score on the mteb/cmteb evaluation benchmarks. In terms of various task scores, its clustering, sentence pair classification, rearrangement, and STS task achieved the highest average scores.
## Basic Features

- Powerful text embedding capabilities；
- Long context: up to 8k context length；
- 7B parameter size


## Technical Introduction
### Unified Task Modeling Framework
We unify the text embedding objectives into three major modeling optimization issues and propose a unified training data structured solution and corresponding training mechanism. This approach can integrate most open source data as retrieval training sets. The structured data can be as follows:
- Retrieval
  - title-body
  - title-abstract
  - Question Answering Dataset
  - Reading comprehension
  - ...

- STS
  - text pair + label in {true, false}、{yes, no}
  - text pair + score（such as 0.2, 3.1. 4.8, etc.）
  - NLI dataset：text pair + label in {'entailment', 'neutral', 'contradiction'}

- CLS
  - text+CLS label

<div align="center"><img src="assets/image-18.png" width="1000" height="600"></img></div>
<div align="center"><img src="assets/image-16.png" width="1000" height="550"></img></div>

### Training Objectives

- Retrieval: Apply InfoNCE contrastive loss function, and follow the gte/qwen3-embedding to add the query-query negative as part of the denominator.<br>
<div align="center"><img src="assets/formula1.png" width="700" height="110"></img></div>

- STS：Apply Cosent loss：
<div align="center"><img src="assets/formula2.png" width="700" height="110"></img></div>

- CLS: Apply the same InfoNCE loss as retrieval, but for In-Batch Negative, due to the high probability of same-class conflicts, a mask mechanism is used to cover up similar samples in negative examples shared by different samples.
<div align="center"><img src="assets/formula3.png" width="1100" height="180"></img></div>
Where $C_{t_i}$ represents the class label of sample $t_i$ , and $n$ is the number of negative samples for a single data point.

### Feature Enhancement Data Synthesis Technology
In the context of powerful languages and writing capabilities in LLMs, we've fully leveraged the LLMs API to propose a data synthesis technology. To address issues like limited data and narrow topics/features in training sets, we've proposed rewriting and expanding synthesis techniques. Furthermore, to increase the difficulty of negative examples during training, we've designed a hard negative example synthesis technology based on big models, combined with existing strong retriever-based hard negative examples sampling. Several of these technologies are described below:
<div align="center"><img src="assets/image-9.png" width="930" height="290"></img></div>
<div align="center"><img src="assets/image-10.png" width="880" height="220"></img></div>
<div align="center"><img src="assets/image-11.png" width="880" height="210"></img></div>

For more details, including reproduction of evaluation results, Instruction content and adding method, please refer to our <a href="https://github.com/Kingsoft-LLM/QZhou-Embedding">GitHub</a> repo, thanks! 

## Evaluation Results
### mteb details
<div align="center"><img src="assets/image-7.png" width="1100" height="260"></img></div>

### cmteb details
<div align="center"><img src="assets/image-8.png" width="1000" height="260"></img></div>

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
```
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

```
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Kingsoft-LLM/QZhou-Embedding")

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

```
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


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
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T)
```
