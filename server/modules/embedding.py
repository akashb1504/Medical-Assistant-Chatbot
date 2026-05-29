from fastembed import TextEmbedding


embedding_model = None


def get_embedding_model():

    global embedding_model

    if embedding_model is None:

        embedding_model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    return embedding_model