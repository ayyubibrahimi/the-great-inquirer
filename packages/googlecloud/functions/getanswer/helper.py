from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain_openai import OpenAIEmbeddings

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def parse_field(request_json, field: str):
    if request_json and field in request_json:
        return request_json[field]
    else:
        raise ValueError(f"JSON is invalid, or missing a '${field}' property")


def get_dbs():
    dir = Path(__file__).parent.absolute()
    general_embeddings, in_depth_embeddings = create_embeddings()

   # New FAISS indices paths for each document type
    faiss_fc_index_path = dir.joinpath("cache/faiss_index_in_depth_fc")
    faiss_cj_index_path = dir.joinpath("cache/faiss_index_in_depth_cj")
    faiss_pdf_index_path = dir.joinpath("cache/faiss_index_in_depth_pdf")
    faiss_pc_index_path = dir.joinpath("cache/faiss_index_in_depth_pc")
    faiss_news_index_path = dir.joinpath("cache/faiss_index_in_depth_news")

    # Loading new FAISS indices for each document type
    db_fc = FAISS.load_local(faiss_fc_index_path, in_depth_embeddings, allow_dangerous_deserialization=True)
    db_cj = FAISS.load_local(faiss_cj_index_path, in_depth_embeddings, allow_dangerous_deserialization=True)
    db_pdf = FAISS.load_local(faiss_pdf_index_path, in_depth_embeddings,  allow_dangerous_deserialization=True)
    db_pc = FAISS.load_local(faiss_pc_index_path, in_depth_embeddings,  allow_dangerous_deserialization=True)
    db_news = FAISS.load_local(faiss_news_index_path, in_depth_embeddings,  allow_dangerous_deserialization=True)

    voting_roll_df_path = dir.joinpath("cache/parsed_voting_rolls.csv")
    voting_roll_df = pd.read_csv(voting_roll_df_path)

    logger.info("Loaded databases from specific FAISS indices")
    return db_fc, db_cj, db_pdf, db_pc, db_news, voting_roll_df


def create_embeddings():
    base_embeddings = OpenAIEmbeddings()
    return base_embeddings, base_embeddings


def sort_retrieved_documents(doc_list):
    docs = sorted(doc_list, key=lambda x: x[1], reverse=True)

    third = len(docs) // 3

    highest_third = docs[:third]
    middle_third = docs[third : 2 * third]
    lowest_third = docs[2 * third :]

    highest_third = sorted(highest_third, key=lambda x: x[1], reverse=True)
    middle_third = sorted(middle_third, key=lambda x: x[1], reverse=True)
    lowest_third = sorted(lowest_third, key=lambda x: x[1], reverse=True)

    docs = highest_third + lowest_third + middle_third
    return docs
