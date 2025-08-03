import argparse
from typing import Any
import mteb
from mteb.encoder_interface import PromptType
import numpy as np
import json
import torch

import numpy as np

from embedder import General_Embedder

def get_detailed_instruct(instruction: str) -> str:
    if not instruction: return ''

    return 'Instruct: {}\nQuery: '.format(instruction)

query_passage_instruction = ['AmazonCounterfactualClassification', 'ArXivHierarchicalClusteringP2P', 'ArXivHierarchicalClusteringS2S', \
    'BIOSSES', 'Banking77Classification', 'BiorxivClusteringP2P.v2', 'ImdbClassification', 'MTOPDomainClassification', \
    'MassiveIntentClassification', 'MassiveScenarioClassification', 'MedrxivClusteringP2P', 'MedrxivClusteringS2S', \
    'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS17', 'STS22.v2', 'STSBenchmark', 'SprintDuplicateQuestions', \
    'StackExchangeClustering.v2', 'StackExchangeClusteringP2P.v2', 'ToxicConversationsClassification', \
    'TweetSentimentExtractionClassification', 'TwentyNewsgroupsClustering.v2', 'TwitterSemEval2015', 'TwitterURLCorpus', 'AskUbuntuDupQuestions', 'MindSmallReranking', \
    'CQADupstackGamingRetrieval', 'CQADupstackUnixRetrieval', 'ArguAna', 'SummEvalSummarization.v2']

all_mteb_tasks = ['AmazonCounterfactualClassification', 'ArXivHierarchicalClusteringP2P', 'ArXivHierarchicalClusteringS2S', 'ArguAna', 'AskUbuntuDupQuestions', \
                'BIOSSES', 'Banking77Classification', 'BiorxivClusteringP2P.v2', 'CQADupstackGamingRetrieval', 'CQADupstackUnixRetrieval', \
                'ClimateFEVERHardNegatives', 'FEVERHardNegatives', 'FiQA2018', 'HotpotQAHardNegatives', 'ImdbClassification', 'MTOPDomainClassification', \
                'MedrxivClusteringP2P', 'MedrxivClusteringS2S', 'MindSmallReranking', \
                'SCIDOCS', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS17', 'STS22.v2', 'STSBenchmark', 'SprintDuplicateQuestions', \
                'StackExchangeClustering.v2', 'StackExchangeClusteringP2P.v2', 'SummEvalSummarization.v2', 'TRECCOVID', 'ToxicConversationsClassification', \
                'TweetSentimentExtractionClassification', 'TwentyNewsgroupsClustering.v2', 'TwitterSemEval2015', 'TwitterURLCorpus', 'Touche2020Retrieval.v3', \
                'MassiveIntentClassification', 'MassiveScenarioClassification']

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
            instruction = get_detailed_instruct(task_to_instructions[task_name])
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

    task_to_instructions = None
    with open("./task_to_instructions_v2.json", "r") as f:
        task_to_instructions = json.load(f)

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
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")
    evaluation = mteb.MTEB(tasks=benchmark)
    evaluation.run(model, output_folder=args.output_dir)

    ### Select tasks to run individually. 
    # for task in all_mteb_tasks:
    #     tasks = mteb.get_tasks(tasks=[task])
    #     evaluation = mteb.MTEB(tasks=tasks)
    #     evaluation.run(model, output_folder=args.output_dir)
