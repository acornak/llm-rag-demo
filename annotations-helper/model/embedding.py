from langchain_community.embeddings import OllamaEmbeddings


def get_embedding_function() -> OllamaEmbeddings:
    """
    Get the embedding function.

    Available models:
    - "mxbai-embed-large"
    - "nomic-embed-text"
    - "all-minilm"

    You need to pull them first using ollama pull {model_name} command.

    :return: Embedding function
    """
    return OllamaEmbeddings(model="all-minilm")
