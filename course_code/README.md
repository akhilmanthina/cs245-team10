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


## Sentence Model Variation

**Note: There are many different outputs under different folders for this model underneath `/output/data/sentence_model_variation/Llama-3.2-1B-Instruct/...` because we ran this same model many (14) times to test small configuration changes. These included the HTML parser, chunk size, top k retrieve number, max context length, and the sentence models. Thus, it didn't make sense to make 14 different models to go along with all of the outputs we reference in our report, so there is just one model file to refer to that relates to all the outputs.**

1. **Text Parsing**
   - The webpages are passed to a BeautifulSoup or Newspaper3k parser which extracts the plaintext from the HTML document.
   - The page snippet (summary) is also stored.
   - These are stored in a LlamaCore `Document` object to be used later in the pipeline.

2. **Text Chunking**
   - We use the LlamaCore `SentenceSplitter` to split up the text into chunks. We use this because `SentenceSplitter` does its best to chunk sentences and paragraphs together, avoiding leaving a dangling sentence/paragraph. This seems to be the best middle ground between complete token-level and sentence-level chunking.

3. **VectorDB and Dense Embedding Comparison**
   - We use a LlamaCore `VectorDB` to store the text embeddings which are generated with the sentence_model. We ran multiple tests with different sentence models including all-MiniLM-L6-v2, BAAI/bge-m3, and thenlper/gte-large.
   - Based on embedding similarity, the top `similarity_top_k` chunks are pulled. We tested 10, 15, and 20 for this value in our tests.

4. **Semantic Comparison**
   - We use a BAAI/bge-reranker-v2-m3 reranker model to do a semantic similarity comparison between the top `similarity_top_k` nodes and the query. The top 5 most related objects are returned and used for the QA procedure.

5. **Answer Generation**
   - From here, the procedure is the same as the RAG baseline. The top 5 nodes are fed into the `formatted_prompts` function, which is mostly similar. 
      - There is a small alteration to the user_message which tells the model to "think step by step and then provide the final answer" to promote CoT reasoning. However, we did not notice a significant enough difference to justify including it in our report, which is already dense with tables.

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
# Run this if you want to run sentence_model_variation. This includes the LlamaCore package and the relevant dependencies for newspaper3k
pip install llama-index llama-index-embeddings-huggingface newspaper3k lxml[html_clean]
export CUDA_VISIBLE_DEVICES=0
```

The dataset will also need to be downloaded from: https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/retrieval-summarization/dataset_files
You will need to make an account. 
Place the file into the `./data` directory. 

### Note: a powerful GPU is required to execute the following models!
Specifically, the `sentence_model_variation` model will *need* an L4 GPU from GCP rather than a T4 GPU. This is because the more complex sentence models we used require upwards of 22-23gb of memory. And this was just testing on the 1B parameter model...

## Execution Steps

To generate predictions use:

```bash
python generate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --split 1 --model_name "{model_name_here}" --llm_name "meta-llama/Llama-3.2-1B-Instruct"
```

The generated predictions are saved under the `./output/data/{model_name` directory.

Different models can be used by changing the model name.

# List of different models:v
  - vanilla_baseline
  - rag_baseline
  - rag_HyDE
  - rag_HyDE_hybrid
  - prompt_eng
  - reduced_top_k
  - prompt_eng_threshold
  - sentence_model_variation
  
 To evaluate the model performance use:
 
 ```bash
 python evaluate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --model_name "{same_model_as_above}" --llm_name "meta-llama/Llama-3.2-1B-Instruct" --max_retries 10
 ```
 
 Insert the same model name as for the generations. 
 To evaluate existing results (our submission) for `sentence_model_variation`, you will need to pick out the exact `predictions.json` file you wish to evaluate from the many test runs and move it to the outer directory. From there, you can run evaluate as normal.