from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from modules.llm import get_llm_chain
from modules.query_handlers import query_chain

from langchain_core.documents import Document
from langchain.schema import BaseRetriever

from pinecone import Pinecone

from typing import List

from logger import logger

import os
import requests


router = APIRouter()


HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

HF_API_URL = (
    "https://api-inference.huggingface.co/"
    "pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}


def get_embedding(text: str):

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": text},
        timeout=30
    )

    response.raise_for_status()

    return response.json()


pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

index = pc.Index(
    os.environ["PINECONE_INDEX_NAME"]
)


class SimpleRetriever(BaseRetriever):

    docs: List[Document]

    def _get_relevant_documents(
        self,
        query: str
    ) -> List[Document]:

        return self.docs


@router.post("/ask/")
async def ask_question(
    question: str = Form(...)
):

    try:

        logger.info(f"user query: {question}")

        embedded_query = get_embedding(question)

        res = index.query(
            vector=embedded_query,
            top_k=3,
            include_metadata=True
        )

        docs = [
            Document(
                page_content=match["metadata"].get(
                    "text",
                    ""
                ),
                metadata=match["metadata"]
            )
            for match in res["matches"]
        ]

        retriever = SimpleRetriever(
            docs=docs
        )

        chain = get_llm_chain(
            retriever
        )

        result = query_chain(
            chain,
            question
        )

        logger.info("query successful")

        return result

    except Exception as e:

        logger.exception(
            "Error processing question"
        )

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )