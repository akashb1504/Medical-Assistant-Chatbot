from fastapi import APIRouter, Form

from fastapi.responses import JSONResponse

from modules.llm import get_llm_chain

from modules.query_handlers import query_chain

from langchain_core.documents import Document

from langchain.schema import BaseRetriever

from pinecone import Pinecone

from fastembed import TextEmbedding

from typing import List

from logger import logger

import os


router = APIRouter()


embedding_model = None


def get_embedding_model():

    global embedding_model

    if embedding_model is None:

        embedding_model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    return embedding_model


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

        logger.info(
            f"user query: {question}"
        )

        model = get_embedding_model()

        embedded_query = list(
            model.embed([question])
        )[0].tolist()

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

        logger.info(
            "query successful"
        )

        return result

    except Exception as e:

        logger.exception(
            "Error processing question"
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )