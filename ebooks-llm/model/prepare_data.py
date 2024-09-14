"""
Prepare the data for the chatbot.
"""

import os
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import openai


# Load the documents from the directory
def load_documents() -> list[Document]:
    """
    Load the documents from the directory.

    :return: List of documents
    """
    DATA_PATH = os.environ["DATA_PATH"]

    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    return documents


def create_chunks(documents: list[Document]) -> list[Document]:
    """
    Create chunks from the documents.

    :param documents: List of documents
    :return: List of chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 300 characters
        chunk_overlap=100,  # 100 characters
        length_function=len,  # Use the len() function to calculate the length of a chunk
        add_start_index=True,  # Add the start index to the chunk
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def create_chroma(from_documents: bool, chunks: list[Document]) -> Chroma:
    """
    Create the Chroma vector store.

    :param documents: List of documents
    :return:
    """
    CHROMA_PATH = os.environ["CHROMA_PATH"]

    if from_documents:
        db = Chroma.from_documents(
            chunks,
            OpenAIEmbeddings(),  # Use the OpenAIEmbeddings for the vectorization
            persist_directory=CHROMA_PATH,
        )
    else:
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings()
        )

    return db


def prepare_data() -> None:
    """
    Prepare the data for the chatbot.

    :return:
    """
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
        CHROMA_PATH = os.environ["CHROMA_PATH"]
        os.environ["DATA_PATH"]
    except KeyError:
        print(
            "Please set the OPENAI_API_KEY, CHROMA_PATH and DATA_PATH environment variable."
        )
        return

    if not os.path.exists(CHROMA_PATH):
        print("Chroma database not found. Creating new one. This may take a while.")
        documents = load_documents()
        chunks = create_chunks(documents)
        db = create_chroma(True, chunks)
    else:
        db = create_chroma(False, [])

    return db
