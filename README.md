
<div align="center">
    <div><img src="assets/redis_logo.svg" style="width: 130px"> </div>
    <div style="display: inline-block; text-align: center; margin-bottom: 10px;">
        <span style="font-size: 36px;"><b>Redis Vector Search: Financial Examples</b></span>
        <br />
    </div>
    <br />
</div>



*A detailed notebook to teach semantic search and RAG patterns over public financial 10k documents with different Redis clients and integrations including: [redis-py](https://redis-py.readthedocs.io/en/stable/index.html), [redisvl](https://redisvl.com), and [langchain](https://python.langchain.com/docs/integrations/vectorstores/redis).*

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

otherwise the underlying library tries to download the models from HuggingFace if this folder if not available locally.
In particular, we use `sentence-transformers--all-MiniLM-L6-v2` which we have made it available locally in the `models` 
folder of this repo.





