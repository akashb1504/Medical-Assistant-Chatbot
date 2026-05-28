import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm

from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex"

UPLOAD_DIR = "./uploaded_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# LOAD ONCE
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# PINECONE INIT ONCE
pc = Pinecone(api_key=PINECONE_API_KEY)

spec = ServerlessSpec(
    cloud="aws",
    region=PINECONE_ENV
)

existing_indexes = [
    i["name"] for i in pc.list_indexes()
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

index = pc.Index(PINECONE_INDEX_NAME)


def load_vectorstore(uploaded_files):

    file_paths = []

    for file in uploaded_files:

        save_path = Path(UPLOAD_DIR) / file.filename

        with open(save_path, "wb") as f:
            f.write(file.file.read())

        file_paths.append(str(save_path))

    for file_path in file_paths:

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        metadatas = []

        for chunk in chunks:

            metadata = chunk.metadata

            metadata["text"] = chunk.page_content

            metadatas.append(metadata)

        ids = [
            f"{Path(file_path).stem}-{i}"
            for i in range(len(chunks))
        ]

        print(f"Embedding {len(texts)} chunks...")

        embeddings = embed_model.encode(texts).tolist()

        vectors = list(
            zip(ids, embeddings, metadatas)
        )

        print("Uploading to Pinecone...")

        with tqdm(
            total=len(vectors),
            desc="Upserting to Pinecone"
        ) as progress:

            batch_size = 50

            for i in range(0, len(vectors), batch_size):

                batch = vectors[i:i + batch_size]

                index.upsert(vectors=batch)

                progress.update(len(batch))

        print(f"Upload complete for {file_path}")