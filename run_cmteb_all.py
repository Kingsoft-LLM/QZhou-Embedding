import argparse
from typing import Any
import mteb
from mteb.encoder_interface import PromptType
import numpy as np
import json
import torch

import numpy as np

from embedder import General_Embedder

instruction_map = {
        'EcomRetrieval': 'Given a user query from an e-commerce website, retrieve description sentences of relevant products',
        'MedicalRetrieval': 'Given a medical question, retrieve user replies that best answer the question',
        'CMedQAv1-reranking': 'Given a Chinese community medical question, retrieve replies that best answer the question',
        'CMedQAv2-reranking': 'Given a Chinese community medical question, retrieve replies that best answer the question',
        'ATEC': 'Retrieve semantically similar text',
        'BQ': 'Retrieve semantically similar text',
        'LCQMC': 'Retrieve semantically similar text',
        'PAWSX': 'Retrieve semantically similar text',
        'STSB': 'Retrieve semantically similar text',
        'AFQMC': 'Retrieve semantically similar text',
        'QBQTC': 'Retrieve semantically similar text',
        'TNews': 'Classify the fine-grained category of the given news title',
        'IFlyTek': 'Given an App description text, find the appropriate fine-grained category',
        'MultilingualSentiment': 'Classify sentiment of the customer review into positive, neutral, or negative',
        'JDReview': 'Classify the customer review for iPhone on e-commerce platform into positive or negative',
        'OnlineShopping': 'Classify the customer review for online shopping into positive or negative',
        'Waimai': 'Classify the customer review from a food takeaway platform into positive or negative',
        'Ocnli': 'Retrieve semantically similar text.',
        'Cmnli': 'Retrieve semantically similar text.',
        'CLSClusteringP2P': 'Identify the main category of scholar papers based on the titles and abstracts',
        'CLSClusteringS2S': 'Identify the main category of scholar papers based on the titles',
        'ThuNewsClusteringS2S': 'Identify the topic or theme of the given news articles based on the titles',
        'T2Retrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
        'CmedqaRetrieval': 'Given a Chinese community medical question, retrieve replies that best answer the question',
        'CovidRetrieval': 'Given a question on COVID-19, retrieve news articles that answer the question',
        'DuRetrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
        'MMarcoReranking': 'Given a Chinese search query, retrieve web passages that answer the question',
        'MMarcoRetrieval': 'Given a web search query, retrieve relevant passages that answer the query',
        'T2Reranking': 'Given a Chinese search query, retrieve web passages that answer the question',
        'ThuNewsClusteringP2P': 'Identify the topic or theme of the given news articles based on the titles and contents',
        'VideoRetrieval': 'Given a video search query, retrieve the titles of relevant videos'
        }

def get_detailed_instruct(instruction: str) -> str:
    if not instruction: return ''

    return 'Instruct: {}\nQuery: '.format(instruction)

query_passage_instruction = ["AFQMC", "ATEC", "BQ", "LCQMC", 'STSB', \
    "QBQTC", 'PAWSX', 'CLSClusteringP2P', 'CLSClusteringS2S', 'ThuNewsClusteringS2S', 'ThuNewsClusteringP2P', 'Cmnli', \
    'Ocnli', 'IFlyTek', 'JDReview', 'MultilingualSentiment', 'OnlineShopping', 'Waimai', 'TNews', 'MMarcoReranking', \
    'CMedQAv1-reranking', 'CMedQAv2-reranking', 'T2Reranking']

all_cmteb_tasks = ['AFQMC', 'ATEC', 'BQ', 'CLSClusteringP2P', 'CLSClusteringS2S', 'CMedQAv1-reranking', 'CMedQAv2-reranking', 'CmedqaRetrieval', 'Cmnli', \
                'CovidRetrieval', 'DuRetrieval', 'EcomRetrieval', 'IFlyTek', 'JDReview', 'LCQMC', 'MMarcoReranking', 'MMarcoRetrieval', \
                'MedicalRetrieval', 'MultilingualSentiment', 'Ocnli', 'OnlineShopping', 'PAWSX', 'QBQTC', 'STSB', 'T2Reranking', \
                'T2Retrieval', 'TNews', 'ThuNewsClusteringP2P', 'ThuNewsClusteringS2S', 'VideoRetrieval', 'Waimai']

class EmbedderWrapper:
    def __init__(self, model=None, use_instruction=True):

        self.model = model
        self.use_instruction = use_instruction

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        
        if self.use_instruction and (prompt_type == PromptType.query or task_name in query_passage_instruction):
            instruction = get_detailed_instruct(instruction_map[task_name])
        else:
            instruction = ''
   
        return self.model.encode(sentences=sentences, prompt=instruction, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None, 
    )
    parser.add_argument("--pooling_mode", type=str, default="mean")
    parser.add_argument("--normalize", type=str, choices=['true', 'false'], default='true')
    parser.add_argument("--use_instruction", type=str, choices=['true', 'false'], default='false')
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    emb_model = General_Embedder.from_pretrained(
        model_name_or_path=args.model_name_or_path,
        pooling_mode=args.pooling_mode,
        max_length=1024,
        normalize=True if args.normalize == 'true' else False,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16)

    model = EmbedderWrapper(model=emb_model, use_instruction=True if args.use_instruction == 'true' else False)
    model.model.eval()
    print('MODEL_NAME: ', args.model_name_or_path)
    benchmark = mteb.get_benchmark("MTEB(cmn, v1)")
    evaluation = mteb.MTEB(tasks=benchmark)
    evaluation.run(model, output_folder=args.output_dir)

    ### Select tasks to run individually. 
    # for task in all_cmteb_tasks:
    #     tasks = mteb.get_tasks(tasks=[task])
    #     evaluation = mteb.MTEB(tasks=tasks)
    #     evaluation.run(model, output_folder=args.output_dir)
