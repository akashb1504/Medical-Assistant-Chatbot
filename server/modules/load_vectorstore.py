import os

import time

import gc

from pathlib import Path

from dotenv import load_dotenv

from tqdm.auto import tqdm

from pinecone import (
    Pinecone,
    ServerlessSpec
)

from modules.embedding import embedding_model

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)


load_dotenv()


PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_ENV = "us-east-1"

PINECONE_INDEX_NAME = "medicalindex"

UPLOAD_DIR = "./uploaded_docs"

MAX_CHUNKS = 50

EMBED_BATCH_SIZE = 8


os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


pc = Pinecone(
    api_key=PINECONE_API_KEY
)

spec = ServerlessSpec(
    cloud="aws",
    region=PINECONE_ENV
)

existing_indexes = [
    i["name"]
    for i in pc.list_indexes()
]

if PINECONE_INDEX_NAME not in existing_indexes:

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=spec
    )

    while not pc.describe_index(
        PINECONE_INDEX_NAME
    ).status["ready"]:

        time.sleep(1)

index = pc.Index(
    PINECONE_INDEX_NAME
)


def load_vectorstore(uploaded_files):

    file_paths = []

    for file in uploaded_files:

        save_path = (
            Path(UPLOAD_DIR)
            / file.filename
        )

        with open(save_path, "wb") as f:

            f.write(file.file.read())

        file_paths.append(str(save_path))

    for file_path in file_paths:

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
        )

        chunks = splitter.split_documents(
            documents
        )

        chunks = chunks[:MAX_CHUNKS]

        print(
            f"Processing {len(chunks)} chunks..."
        )

        for i in tqdm(
            range(0, len(chunks), EMBED_BATCH_SIZE)
        ):

            batch_chunks = chunks[
                i:i + EMBED_BATCH_SIZE
            ]

            texts = [
                chunk.page_content
                for chunk in batch_chunks
            ]

            embeddings = [
                embedding.tolist()
                for embedding in embedding_model.embed(
                    texts
                )
            ]

            vectors = []

            for j, chunk in enumerate(
                batch_chunks
            ):

                metadata = chunk.metadata

                metadata["text"] = (
                    chunk.page_content
                )

                vector_id = (
                    f"{Path(file_path).stem}-{i+j}"
                )

                vectors.append(
                    (
                        vector_id,
                        embeddings[j],
                        metadata
                    )
                )

            index.upsert(
                vectors=vectors
            )

            del texts
            del embeddings
            del vectors

            gc.collect()

        del documents
        del chunks

        gc.collect()

        print(
            f"Upload complete for {file_path}"
        )