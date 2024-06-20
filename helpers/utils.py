import yaml
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import VLLMOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
USE_VLLM = False
LOCAL_OLLAMA_MODEL = 'llama3'
LOCAL_VLLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

"""
To be able to use VLLM you must have the VLLM server installed on the local machine in this setup
please refer to documentation in ReadMe on how to dod so:
for example:

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=hf_token" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3-8B-Instruct
    
"""


def get_llm():
    if USE_VLLM:
        vllm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1",
            model_name=LOCAL_VLLM_MODEL,
            model_kwargs={"stop": ["."]},
        )
        return vllm
    #We default to Ollama
    else:
        llm = Ollama(model=LOCAL_OLLAMA_MODEL)
        return llm


def get_chat_llm(temperature=0, format=None):
    if USE_VLLM:
        inference_server_url = "http://localhost:8000/v1"

        chatVLLM = ChatOpenAI(
            model=LOCAL_VLLM_MODEL,
            openai_api_key="EMPTY",
            openai_api_base=inference_server_url,
            max_tokens=5,
            temperature=temperature,
        )
        return chatVLLM
    #We default to Ollama
    else:
        if format is None:
            chat_llm = ChatOllama(model=LOCAL_OLLAMA_MODEL, temperature=temperature)
            return chat_llm
        else:
            chat_llm = ChatOllama(model=LOCAL_OLLAMA_MODEL, temperature=temperature, format=format)
            return chat_llm


def format_docs(docs):
    clean_docs = []
    for doc in docs:
        clean_docs.append(str(doc.page_content).replace("\n", " "))
    return clean_docs


def create_langchain_schemas_from_redis_schema(redis_yschema):
    with open(redis_yschema, "r") as yschema:
        schema = yaml.load(yschema, Loader=yaml.Loader)

    vector_schema = {
        "name": None,
        "algorithm": None,
        "dims": None,
        "distance_metric": None,
        "datatype": None,
    }

    index_schema = {
        "vector": [],
        "text": [],
        "tag": [],
        "numeric": [],
        "content_vector_key": None  # name of the vector field in langchain
    }

    for f in schema['fields']:
        # print(f'f={f}')
        if f['type'] == 'tag':
            index_schema['tag'] = index_schema['tag'] + [{'name': f['name']}]
        elif f['type'] == 'numeric':
            index_schema['numeric'] = index_schema['numeric'] + [{'name': f['name']}]
        elif f['type'] == 'text':
            index_schema['text'] = index_schema['text'] + [{'name': f['name']}]
        # TODO: add and test rest of the index field types
        elif f['type'] == 'vector':
            vector_schema = {
                "name": f['name'],
                "algorithm": str(f['attrs']['algorithm']).upper(),
                "dims": int(f['attrs']['dims']),
                "distance_metric": str(f['attrs']['distance_metric']).upper(),
                "datatype": str(f['attrs']['type']).upper(),
            }
            index_schema['vector'] = index_schema['vector'] + [vector_schema]
            index_schema['content_vector_key'] = f['name']  # langchain only accepts one vector name?
    return vector_schema, index_schema
