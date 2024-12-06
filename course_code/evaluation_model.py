import os
from typing import Any, Dict, List

import vllm

from openai import OpenAI
from openai import APIConnectionError, OpenAI, RateLimitError
from templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS

from loguru import logger

#### CONFIG PARAMETERS ---

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

#### CONFIG PARAMETERS END---

class EvaluationModel:
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None, max_retries=10):
        """
        Initialize your model(s) here if necessary.
        This is the constructor for your DummyModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
        self.initialize_models(llm_name, is_server, vllm_server, max_retries)

    def initialize_models(self, llm_name, is_server, vllm_server, max_retries):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server
        self.max_retries = max_retries

        if self.is_server:
            # initialize the model with vllm server
            #openai_api_key = "EMPTY"
            #openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=os.environ.get("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()


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

        formatted_prompts = self.format_prommpts(queries, query_times)

        # note that here self.batch_size = 1
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                    model = self.llm_name,
                    messages = formatted_prompts[0],
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm = False
            )
            answers = []
            for response in responses:
                answers.append(response.outputs[0].text)

        return answers

    def get_system_message(self):
        """Returns the system message containing instructions and in context examples."""
        return INSTRUCTIONS + IN_CONTEXT_EXAMPLES


    def format_prompts(self, query, ground_truth, prediction):
        system_message = self.get_system_message()
        user_message = f"{system_message}\nQuestion: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n"

        formatted_prompts = []

        if self.is_server:
            # there is no need to wrap the messages into chat when using the server
            # because we use the chat API: chat.completions.create
            formatted_prompts.append(
                [
                  {"role": "user", "content": user_message},
                  {"role": "assistant", "content": "Accuracy:"},
                ]
            )
        else:
            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": "Accuracy:"},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        return formatted_prompts


    def evaluate(self, query, ground_truth, prediction):

        formatted_prompts = self.format_prompts(query, ground_truth, prediction)

        for attempt in range(self.max_retries):
            try:
                if self.is_server:
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_name,
                        messages=formatted_prompts[0]
                    )
                    answers = [response.choices[0].message.content]
                else:
                    responses = self.llm.generate(
                        formatted_prompts,
                        vllm.SamplingParams(n=1),
                        use_tqdm=False
                    )
                    answers = []
                    for response in responses:
                        answers.append(response.outputs[0].text)
                return answers[0]
            except (APIConnectionError, RateLimitError):
                logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
        return None
