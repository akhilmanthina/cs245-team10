# Pipeline Setup

Since we have multiple models that focus on different aspects of improvement, we have multiple pipelines. The following are separate pipelines for each model:

---

## HyDE

1. **Query Translation (HyDE)**:
   - The original query is passed to a large language model (LLM), `llama-3.2-1B-instruct`, to generate a hypothetical document that could plausibly answer the query.
   - The hypothetical document is embedded using a sentence transformer model.

2. **Embedding Comparison**:
   - The embeddings of the hypothetical documents are compared with the stored reference embeddings using cosine similarity.
   - In the hybrid HyDE version, both embeddings of the original query and HyDE are used for the final cosine similarity score comparison as a weighted average. The weighted value can be adjusted using the alpha variable. 

3. **Top-K Selection**
	 - The top k highest cosine similarity score references are kept while the others are discarded. 
	 
4. **Reference Insertion**
   - The references from the previous step are then inserted into the prompt and formatted. 

5. **Answer Generation**:
   - The LLM generates an answer based on the top-k references. 

## Prompt Engineering

1. **Embedding Comparison**:
	 - The original query and references are embedded using a sentence transformer model.
   - The embeddings of the query are compared with the stored reference embeddings using cosine similarity.

2. **Cosine Similarity Score Retrieval**: 
	 - The top-k references are kept along with their cosine similarity scores. 
   - These scores reflect the relevance of each reference to the query.
   - In the threshold variation, a threshold value creates a lower bound of acceptable cosine similarity scores. So scores below the threshold cannot be in the final references. 

3. **Incorporating Scores into the Prompt**:  
   - Each reference is appended with its corresponding cosine similarity score in the final prompt.
   - The score acts as a relevancy weight, explicitly guiding the LLM on the importance of each reference.

4. **Modified System Prompt**:  
   - The system prompt instructs the LLM to utilize the cosine similarity scores as weights when generating the answer.

5. **Answer Generation**:  
   - The LLM processes the query and the references (now weighted by cosine scores) and generates an answer.


[YOU GUYS CAN ADD YOUR PIPELINES HERE]

# Setup Instructions & Execution Steps 

## Environment Setup

After cloning into the repository, run the following in the terminal with your huggingface token:

```bash
huggingface-cli login --token "YOUR HF_TOKEN"
```

Then run the following to setup the environment:

```bash
conda create -n crag python=3.10
conda activate crag
pip install -r requirements.txt
pip install --upgrade openai
export CUDA_VISIBLE_DEVICES=0
```

The dataset will also need to be downloaded from: https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/retrieval-summarization/dataset_files
You will need to make an account. 
Place the file into the `./data` directory. 

### Note: a powerful GPU is required to execute the following models!

## Execution Steps

To generate predictions use:

```bash
python generate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --split 1 --model_name "{model_name_here}" --llm_name "meta-llama/Llama-3.2-1B-Instruct"
```

The generated predictions are saved under the `./output/data/{model_name` directory.

Different models can be used by changing the model name.

# List of different models:
  - vanilla_baseline
  - rag_baseline
  - rag_HyDE
  - rag_HyDE_hybrid
  - prompt_eng
  - reduced_top_k
  - prompt_eng_threshold
  [ ADD UR MODELS HERE]
  
 To evaluate the model performance use:
 
 ```bash
 python evaluate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --model_name "{same_model_as_above}" --llm_name "meta-llama/Llama-3.2-1B-Instruct" --max_retries 10
 ```
 
 Insert the same model name as for the generations. 
