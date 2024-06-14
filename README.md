
<div align="center">
    <div><img src="assets/redis_logo.svg" style="width: 130px"> </div>
    <div style="display: inline-block; text-align: center; margin-bottom: 10px;">
        <span style="font-size: 36px;"><b>Redis Vector Search: Financial Examples</b></span>
        <br />
    </div>
    <br />
</div>



*A detailed set of notebooks to teach semantic search and RAG patterns over public financial 10k filings, metadata and earning calls of some of the top Russel 3000 index with different Redis clients and integrations including: [redis-py](https://redis-py.readthedocs.io/en/stable/index.html), [redisvl](https://redisvl.com), and [langchain](https://python.langchain.com/docs/integrations/vectorstores/redis).*

# ⚡ Introduction to Vector Search in Redis
[Redis](https://redis.com), widely recognized for its low-latency performance, extends beyond traditional noSQL databases. It's uniquely suited for tasks like caching, session management, job queuing, and JSON storage. With enhanced Search+Query features, Redis emerges as a performant [Vector Database](https://redis.com/solutions/use-cases/vector-database) supporting Vector Search over unstructured data encoded as embeddings.

### Installation

#### Local LLM
All the notebooks here runs on local LLMs served by Ollama, and we specifically use `llama3` model in all of our examples. 
So make sure you install Ollama and pull `llama3` model on your local machine where you run these notebooks.

(1) Download [Ollama app](https://ollama.ai/) and install.
(2) run `ollama pull llama3`

#### Local Embedding models

Throughout the notebooks in this repo, we are using `SentenceTransformerEmbeddings` and to be able to run it locally, 
we specify a local cache folder for `SentenceTransformer` models. 
If you already downloaded the models in a local file system, set this folder in the notebook via

```code
#setting the local downloaded sentence transformer models folder
os.environ["TRANSFORMERS_CACHE"] = f"{path_to_your_local_cache_folder}/models"
```

otherwise the underlying library tries to download the models from HuggingFace if this folder is not available locally.
In particular, we use `sentence-transformers--all-MiniLM-L6-v2` which we have made it available locally in the `models` 
folder of this repo.

#### Provide NLTK Data
in the `.env` file please add the full path for `NLTK_DATA` and point it to the directory that we have provided in the 
`{path_to_this_repo}models/nltk_data`



### Notes
(1) It must be noted that much of the quality of the results depends very much on the embedding model that is used 
for representing the data in vector space in your domain. Here we use a very simple embedding model that represents text 
in a vector of `384` dimensions. Other BERT-driven embedding models would try to capture more nuances of text in `768` 
dimensions. And OpenAI embedding models are in `1536` dimension. We recommend using a proper domain-specific 
embedding model for your use case for better results.

(2) The same advice applies for the LLMs you use for specific tasks. Here we use a general llama3 model for generation 
and tasks that we need. But for better results in a specific domain, it is recommended to fine-tune or
even train your own foundational models if you have the resources to do so. A successful, foundational model 
in the finance space was [BloombergGPT](https://arxiv.org/abs/2303.17564).  

(3) And finally, be advised that we have used the base `llama3` LLM for all of our tasks, including some classification 
and planning for next action and not just only generation part of the RAG patterns. 
At this point in the evolution of LLMs this is not advised.

<img src="assets/GenAI-usecases.png" alt="GenAI" width="400"/>







