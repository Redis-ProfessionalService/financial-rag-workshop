import yaml

from collections import defaultdict
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import VLLMOpenAI
from langchain_openai import ChatOpenAI


def get_llm(
    local_llm_engine='vllm',
    vllm_url="http://localhost:8000/v1",
    vllm_model="meta-llama/Meta-Llama-3-8B-Instruct",
    ollama_model='llama3',
    temperature=0,
    model_kwargs=None
):
    if model_kwargs is None:
        model_kwargs = {}

    if local_llm_engine == 'vllm':
        vllm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=vllm_url,
            model_name=vllm_model,
            temperature= temperature,
            model_kwargs=model_kwargs
        )
        print(f"Created VLLMOpenAI on using {vllm_model} served from {vllm_url}")
        return vllm

    # default to Ollama
    else:
        llm = Ollama(model=ollama_model)
        print(f"Created Ollama using {ollama_model} served locally")
        return llm


def get_chat_llm(
    local_llm_engine='vllm',
    vllm_url="http://localhost:8000/v1",
    vllm_model="meta-llama/Meta-Llama-3-8B-Instruct",
    ollama_model='llama3',
    temperature=0,
    format=None
):
    if local_llm_engine == 'vllm':
        chatVLLM = ChatOpenAI(
            model=vllm_model,
            openai_api_key="EMPTY",
            openai_api_base=vllm_url,
            temperature=temperature
        )
        print(f"Created VLLM ChatOpenAI using {vllm_model} served from {vllm_url}")
        return chatVLLM

    # default to Ollama
    else:
        if format is None:
            chat_llm = ChatOllama(model=ollama_model, temperature=temperature)
            print(f"Created ChatOllama using {ollama_model} served locally")
            return chat_llm
        else:
            chat_llm = ChatOllama(model=ollama_model, temperature=temperature, format=format)
            print(f"Created ChatOllama using {ollama_model} with format={format} served locally")
            return chat_llm


def format_docs(docs):
    companies = defaultdict(list)
    for doc in docs:
        company_name = doc.metadata.get("company_name", "Additional Context")
        companies[company_name].append(str(doc.page_content).replace("\n", " "))

    result = []
    for company, contents in companies.items():
        if company == "Additional Context":
            result.append(company)
        else:
            result.append(f"Company: {company}")
        result.append("\n".join(contents))

    return "\n\n".join(result)


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
