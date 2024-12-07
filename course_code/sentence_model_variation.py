import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, QueryBundle
import newspaper
import html
import threading

from openai import OpenAI

from tqdm import tqdm

#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 10 # 20
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 1950

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---

class ChunkExtractor:
    # Testing idea from: https://github.com/USTCAGI/CRAG-in-KDD-Cup2024/blob/master/models/retrieve/retriever.py
    # CRAG 2nd place team
    # Use the newspaper3k package to parse text from HTML.
    def get_text(self, html):
        if html is None or html.strip() == "":
            return ""
        # soup = BeautifulSoup(
        #     html, features="lxml"
        # )
        # return soup.get_text(" ", strip=True)
        article = newspaper.Article('')
        article.set_html(html)
        try:
            article.parse()
            return article.text
        except:
            soup = BeautifulSoup(
                html, features="lxml"
            )
            return soup.get_text(" ", strip=True)
            return text.replace("\n", " ")

    def get_html_text_threaded(self, html, timeout=20):
        text = ""
        def timeout_function():
            nonlocal text
            text = self.get_text(html)
        thread = threading.Thread(target=timeout_function)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            print("Timeout occurred")
        return text

    @ray.remote
    def _extract_chunks(self, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        text = self.get_html_text_threaded(html_source["page_result"])
        snippet = html.unescape(html_source["page_snippet"]) # also extract snippet context to get shorter summary
        return text, snippet 

    def extract_chunks(self, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        # inspect page_url to make sure they are all unique
        page_urls = set()
        search_res = []
        for result in batch_search_results:
            if not result["page_url"] in page_urls:
                search_res.append(result)
                page_urls.add(result["page_url"])

        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                html_source
            )
            for html_source in search_res
        ]

        docs = []
        for response_ref in ray_response_refs:
            text, snippet = ray.get(response_ref)
            new_doc = Document(text=text)
            # For some reason checking the length of text and snippet doesn't work correctly
            if len(new_doc.text) > 0:
                docs.append(new_doc)
            new_doc = Document(text=snippet)
            if len(new_doc.text) > 0:
                docs.append(new_doc)
        
        return docs


class SentenceModelVariationRAG:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # initialize the model with vllm server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                max_model_len=2048,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = HuggingFaceEmbedding(
            # "all-MiniLM-L6-v2",
            # "BAAI/bge-m3",
            "thenlper/gte-large",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.rerank_model = SentenceTransformerRerank(
            top_n=5, 
            model="BAAI/bge-reranker-v2-m3",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        batch_results = []
        for _idx, _ in enumerate(batch_interaction_ids):
            query = queries[_idx]
            search_result = batch_search_results[_idx]

            docs = self.chunk_extractor.extract_chunks(search_result)
            node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
            nodes = node_parser.get_nodes_from_documents(docs)
            index = VectorStoreIndex(nodes, embed_model=self.sentence_model)
            retriever = index.as_retriever(similarity_top_k=NUM_CONTEXT_SENTENCES)
            nodes = retriever.retrieve(query)
            # batch_results.append([node.get_text().strip() for node in nodes])

            reranked_nodes = self.rerank_model.postprocess_nodes(
                nodes,
                query_bundle=QueryBundle(query_str=query)
            )
            batch_results.append([node.get_text().strip() for node in reranked_nodes])
            
        # Prepare formatted prompts from the LLM        
        formatted_prompts = self.format_prompts(queries, query_times, batch_results)

        # Generate responses via vllm
        # note that here self.batch_size = 1
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                max_tokens=150,  # Maximum number of tokens to generate per output sequence.
            )
            answers = [response.choices[0].message.content]
            print(f"answer: {answers[0]}")
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0, #0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=100,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )
            answers = []
            for i, response in enumerate(responses):
                print(f"Q: {queries[i]}")
                print(f"A: {response.outputs[0].text}")
                answers.append(response.outputs[0].text)

        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        # system_prompt = "You are a helpful assistant"
        formatted_prompts = []
        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
                    if len(references) > MAX_CONTEXT_REFERENCES_LENGTH:
                        break
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            # user_message += f"{references}\n------\n\n"
            # user_message 
            user_message += f"Using only the references listed above, answer the following question. Think step by step and then provide the final answer. Note: - If the question contains ANY factual errors or is inherently incorrect, you MUST reply `invalid question`.\n - For the final answer, use as few words as possible  \n"
            # user_message += f"Using only the references listed above, answer the following question: \n"
            # user_message += "Note: \n - For the final answer, use as few words as possible.\n - If the question contains ANY factual errors, you MUST reply `invalid question`.\n - If you don't know the answer, you MUST reply `I don't know`.\n - The output format MUST meet the requirements: Start with `# Thought process\n` and then output the thought process regarding how you answer the question. You MUST not repeat statements. After you finish thinking, you must reply with the final answer on the last line, starting with `# Final Answer\n` and use as few words as possible."
            user_message += f"{references}\n------\n\n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
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
