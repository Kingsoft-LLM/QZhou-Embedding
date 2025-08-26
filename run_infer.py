import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


### Sentence-transformers
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


### Huggingface Transformers
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

