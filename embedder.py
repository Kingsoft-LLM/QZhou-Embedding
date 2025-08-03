import logging
import os
from typing import Dict, List, Optional, Union

import math
import numpy as np
import queue
import torch
import torch.multiprocessing as mp
from torch import Tensor, device, nn
from tqdm.autonotebook import trange
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class General_Embedder(nn.Module):
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        pooling_mode: str = "mean",
        normalize=False,
        max_length: int = 512,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_mode = pooling_mode
        self.max_length = max_length
        self.normalize = normalize
        self.pool = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        **kwargs,
    ):
        keys = ["pooling_mode", "normalize", "max_length"]
        encoder_args = {
            key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None
        }

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=kwargs['trust_remote_code'])
        tokenizer.padding_side = "left"
        model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
      
        return cls(model=model, tokenizer=tokenizer, **encoder_args)

    def tokenize(self, sents, prompt):

        sents = [prompt + x for x in sents]
        model_inputs = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return model_inputs


    def forward(self, sentence_feature: Dict[str, Tensor]):
        reps = self.model(**sentence_feature)
        if self.normalize:
            return F.normalize(self.get_pooling(sentence_feature, reps.last_hidden_state), p=2, dim=-1)
        else:
            return self.get_pooling(sentence_feature, reps.last_hidden_state)

    def get_pooling(self, features, last_hidden_states): 
        
        seq_lengths = features["attention_mask"].sum(dim=-1)

        if self.pooling_mode == "mean":
            return torch.stack(
                [
                    last_hidden_states[i, -length:, :].sum(dim=0) / length
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
            )
        elif self.pooling_mode == "weighted_mean":
            bs, l, _ = last_hidden_states.shape
            complete_weights = torch.zeros(bs, l, device=last_hidden_states.device)
            for i, seq_l in enumerate(seq_lengths):
                if seq_l > 0:
                    complete_weights[i, -seq_l:] = torch.arange(seq_l) + 1
                    complete_weights[i] /= torch.clamp(
                        complete_weights[i].sum(), min=1e-9
                    )
            return torch.sum(last_hidden_states * complete_weights.unsqueeze(-1), dim=1)
        elif self.pooling_mode == "eos_token" or self.pooling_mode == "last_token":
            return last_hidden_states[:, -1]
        elif self.pooling_mode == "bos_token":
            return last_hidden_states[
                features["input_ids"] == self.tokenizer.bos_token_id
            ]
        else:
            raise ValueError(f"{self.pooling_mode} is not implemented yet.")

    def encode(self, 
                sentences: Union[str, List[str]],
                **kwargs):
        
        try:
            target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            ctx = mp.get_context('spawn')
            input_queue = ctx.Queue()
            output_queue = ctx.Queue()
            processes = []

            for cuda_id in target_devices:
                p = ctx.Process(
                    target=self._encode_multi_process_worker,
                    args=(self, cuda_id, input_queue, output_queue),
                    daemon=True
                )
                p.start()
                processes.append(p)

            part_size = math.ceil(len(sentences) / len(processes))
            chunk_size = part_size if part_size < 3200 else 3200

            print(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

            last_chunk_id = 0
            chunk = []

            for sentence in sentences:
                chunk.append(sentence)
                if len(chunk) >= chunk_size:
                    input_queue.put([last_chunk_id, chunk, kwargs])
                    last_chunk_id += 1
                    chunk = []

            if len(chunk) > 0:
                input_queue.put([last_chunk_id, chunk, kwargs])
                last_chunk_id += 1

            results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
            embeddings = np.concatenate([result[1] for result in results_list])
            
            for p in processes:
                p.terminate()
            for p in processes:
                p.join()
                p.close()
            input_queue.close()
            output_queue.close()
            torch.cuda.empty_cache()

            return embeddings
        except RuntimeError as e:
            return self._encode(sentences, **kwargs)
    
    @staticmethod
    def _encode_multi_process_worker(self, target_device, input_queue, results_queue):

        while True:
            try:
                last_chunk_id, sentences, kwargs = input_queue.get()
                kwargs.update(device=target_device)
                embeddings = self._encode(sentences, **kwargs)
                results_queue.put([last_chunk_id, embeddings])
            except queue.Empty:
                break
    
    @torch.no_grad()
    def _encode(
        self,
        sentences: Union[str, List[str]],
        prompt: str = '',
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(device)

        if convert_to_tensor:
            convert_to_numpy = False

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []

        for start_index in trange(
            0,
            len(sentences),
            batch_size,
            desc=f"Batches_{device}",
            disable=not show_progress_bar,
            position=int(device[-1]) if device[-1].isdigit() else 0
        ):
            sentences_batch = sentences_sorted[
                start_index : start_index + batch_size
            ]
            
            sentences_batch = [(x if x != '' else (x + 'Null')) for x in sentences_batch]

            features = self.tokenize(sentences_batch, prompt)
            features = batch_to_device(features, device)
            with torch.no_grad():
                embeddings = self.forward(features)
                embeddings = embeddings.detach()
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        all_embeddings = all_embeddings.to(torch.float32)
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
 
        return len(text)
