import os
from typing import Any, Dict, List

import numpy as np
import torch
import vllm
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from openai import OpenAI

from transformers import AutoTokenizer

from htmlrag import clean_html, build_block_tree, EmbedHTMLPruner, BM25HTMLPruner, GenHTMLPruner

#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000
# Set the maximum context window (in tokens) for pruning the HTML blocks.
MAX_CONTEXT_WINDOW = 2048  # Adjust this value as needed.

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters
VLLM_TENSOR_PARALLEL_SIZE = 1  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENCE_TRANSFORMER_BATCH_SIZE = 32  # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """

    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        self.initialize_models(llm_name, is_server, vllm_server)
        embed_model_name = "BAAI/bge-large-en-v1.5"
        query_instruction = """
            Instruct: Given the document, retrieve relevant passages that directly answer the query.\nQuery: 
            """
        self.html_pruner_embed = EmbedHTMLPruner(
            embed_model=embed_model_name,
            local_inference=True,
            query_instruction_for_retrieval=query_instruction, 
        )
        self.html_pruner_bm25 = BM25HTMLPruner()

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # Initialize the model with vllm server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            # Initialize the tokenizer for htmlrag
            self.chat_tokenizer = AutoTokenizer.from_pretrained(self.llm_name, trust_remote_code=True)
        else:
            # Initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # Note: bfloat16 is not supported on NVIDIA T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()
            # Initialize the tokenizer for htmlrag
            self.chat_tokenizer = AutoTokenizer.from_pretrained(self.llm_name, trust_remote_code=True)

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.

        Returns:
            int: The batch size, an integer between 1 and 16.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries.

        Returns:
            List[str]: A list of plain text responses for each query in the batch.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        batch_retrieval_results = []
        EXIT_FLAG = 0
        for idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[idx]
            query_time = query_times[idx]
            search_results = batch_search_results[idx]  # List of search results for this query

            pruned_htmls = []
            for i, search_result in enumerate(search_results):
                html_content = search_result['page_result']
                # Process html_content with embed
                simplified_html = clean_html(html_content)
                if '<html' not in simplified_html.lower():
                        #print(f"empty html at index {i}")
                        simplified_html = f"<html>{simplified_html}</html>"

                pruned_html = simplified_html
                block_tree, pruned_html = build_block_tree(pruned_html, max_node_words=200)
                block_rankings = self.html_pruner_embed.calculate_block_rankings(query, pruned_html, block_tree)

                pruned_html = self.html_pruner_embed.prune_HTML(
                    pruned_html, block_tree, block_rankings, self.chat_tokenizer, 500
                )
                pruned_htmls.append(f"HTML {i+1}: {pruned_html}")

            if EXIT_FLAG == 1:
                print(pruned_htmls)
                print(query)
                exit()
            if pruned_htmls:
                retrieval_results = pruned_htmls
            else:
                retrieval_results = []

            batch_retrieval_results.append(retrieval_results)

        # Prepare formatted prompts for the LLM
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)

        # Generate responses via vllm
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # Randomness of the sampling
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # Randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )
            answers = []
            for response in responses:
                answers.append(response.outputs[0].text)

        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times, and retrieval results using the chat_template of the model.

        Parameters:
            - queries (List[str]): A list of queries to be formatted into prompts.
            - query_times (List[str]): A list of query_time strings corresponding to each query.
            - batch_retrieval_results (List[List[str]]): A list of retrieval results for each query.
        """
        system_prompt = "You are provided with partially cleaned HTML-like references. Treat `<h2>` or `##` as headings, `<ul>` or `-` as lists, and `<p>` as paragraphs. Ignore attributes and focus on the text content. If something looks like code or a style tag, ignore it. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""

            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the pruned texts as references in the model's prompt template.
                for snippet in retrieval_results:
                    references += f"- {snippet.strip()}\n"

            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message
            user_message += f"Using only the HTML formatted references listed above, answer the following question: \n"
            #user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # There is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts