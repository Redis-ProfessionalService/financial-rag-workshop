import os
import warnings
import json
import json
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from unstructured.partition.pdf import partition_pdf
import nltk

warnings.filterwarnings("ignore")
dir_path = os.getcwd()
parent_directory = os.path.dirname(dir_path)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ROOT_DIR"] = parent_directory
print(dir_path)
print(parent_directory)

filings_data_path = f"{parent_directory}/resources/filings"
earning_calls_data_path = f"{parent_directory}/resources/earning_calls"
filings_dirs = os.listdir(filings_data_path)
earning_calls_dirs = os.listdir(earning_calls_data_path)


def get_sec_data():
    sec_data = {}
    dirs = []
    for filings_dir in filings_dirs:
        dirs.append(os.path.join(filings_data_path, filings_dir))
    for earning_calls_dir in earning_calls_dirs:
        dirs.append(os.path.join(earning_calls_data_path, earning_calls_dir))

    for path in dirs:
        ticker = str(path).strip().split("/")[len(str(path).strip().split("/")) - 1]
        if sec_data.get(ticker) is None:
            sec_data[ticker] = {
                "10K_files": [],
                "metadata_file": [],
                "transcript_files": [],
            }

        if not os.path.isdir(path):
            continue
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if '10K' in str(file):
                sec_data[ticker]["10K_files"] = sec_data[ticker]["10K_files"] + [full_path]
            elif 'metadata' in str(file):
                sec_data[ticker]["metadata_file"] = sec_data[ticker]["metadata_file"] + [full_path]
            elif 'earning_calls' in str(full_path):
                sec_data[ticker]["transcript_files"] = sec_data[ticker]["transcript_files"] + [full_path]

    print(f" ✅ Loaded doc info for  {len(sec_data.keys())} tickers...")
    return sec_data


def load_json_metadata(path):
    obj = {}
    with open(path, 'r') as json_file:
        meta_dict = json.load(json_file)
        obj['ticker'] = meta_dict['Ticker']
        obj['company_name'] = meta_dict['Name']
        obj['sector'] = meta_dict['Sector']
        obj['asset_class'] = meta_dict['Asset Class']
        obj['market_value'] = float(str(meta_dict['Market Value']).replace(",", ""))
        obj['weight'] = meta_dict['Weight (%)']
        obj['notional_value'] = float(str(meta_dict['Notional Value']).replace(",", ""))
        obj['shares'] = float(str(meta_dict['Shares']).replace(",", ""))
        obj['location'] = meta_dict['Location']
        obj['price'] = float(str(meta_dict['Price']).replace(",", ""))
        obj['exchange'] = meta_dict['Exchange']
        obj['currency'] = meta_dict['Currency']
        obj['fx_rate'] = meta_dict['FX Rate']
        obj['market_currency'] = meta_dict['Market Currency']
        obj['accrual_date'] = meta_dict['Accrual Date']

    return obj


def get_chunks(file_name, chunk_size=2500, chunk_overlap=0):
    if not os.path.isfile(file_name):
        print(f"Error: File {file_name} does not exist- no chunks extracted")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        loader = UnstructuredFileLoader(
            file_name, mode="single", strategy="fast"
        )
        chunks = loader.load_and_split(text_splitter)
        return chunks
    except Exception as e:
        print(f"Error chunking {file_name} skipping")
        return []

import uuid
def add_embeddings_for_chunks(chunks_to_process, shared_obj_to_load, doc_type, embeddings):
    chunk_objs_to_load = []
    for i, chunk in enumerate(chunks_to_process):
        obj_to_load = shared_obj_to_load.copy()
        content = str(chunk.page_content)
        source_doc_full_path = str(chunk.metadata['source'])
        source_doc = str(source_doc_full_path).split("/")[len(str(source_doc_full_path).split("/")) - 1]
        obj_to_load['chunk_id'] = f"{source_doc}-{str(uuid.uuid4())}"
        obj_to_load['source_doc'] = f"{source_doc}"
        obj_to_load['content'] = content
        obj_to_load['doc_type'] = doc_type
        emb = embeddings.embed_query(content)
        obj_to_load['text_embedding'] = np.array(emb).astype(np.float32).tobytes()
        chunk_objs_to_load.append(obj_to_load)
    return chunk_objs_to_load


def redis_bulk_upload(data_dict, index, embeddings, tickers=None):
    total_chunks_count = 0
    total_10K_count = 0
    total_earning_calls_count = 0
    total_keys = []
    if tickers is None:
        tickers = list(data_dict.keys())

    for ticker in tickers:
        if len(data_dict[ticker]["metadata_file"]) > 0:
            shared_metadata = load_json_metadata(data_dict[ticker]["metadata_file"][0])

        for filing_file in data_dict[ticker]["10K_files"]:
            filing_file_filename = str(filing_file).split("/")[len(str(filing_file).split("/")) - 1]
            fchunks = get_chunks(filing_file)
            doc_type = '10K'
            filing_chunk_objs_to_load = add_embeddings_for_chunks(fchunks, shared_metadata.copy(), doc_type=doc_type,
                                                                  embeddings=embeddings)
            keys = index.load(filing_chunk_objs_to_load, id_field="chunk_id")
            total_keys = total_keys + keys
            print(f"✅ Loaded {len(keys)} {doc_type} chunks for ticker={ticker} from {filing_file_filename}")
            total_chunks_count = total_chunks_count + len(keys)
            total_10K_count = total_10K_count + 1

        for earning_file in data_dict[ticker]["transcript_files"]:
            earning_file_filename = str(earning_file).split("/")[len(str(earning_file).split("/")) - 1]
            echunks = get_chunks(earning_file)
            doc_type = 'earning_call'
            earning_chunk_objs_to_load = add_embeddings_for_chunks(echunks,
                                                                   shared_metadata.copy(),
                                                                   doc_type=doc_type,
                                                                   embeddings=embeddings)
            keys = index.load(earning_chunk_objs_to_load, id_field="chunk_id")
            total_keys = total_keys + keys
            print(f"✅ Loaded {len(keys)} {doc_type} chunks for ticker={ticker} from {earning_file_filename}")
            total_chunks_count = total_chunks_count + len(keys)
            total_earning_calls_count = total_earning_calls_count + 1

    print(
        f"✅✅✅Loaded a total of {total_chunks_count} chunks from {total_10K_count} 10Ks and {total_earning_calls_count} earning calls for {len(tickers)} tickers.")
