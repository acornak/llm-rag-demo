"""
Handling the response generation.
"""

from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def search_db(db: Chroma, query: str) -> list[Document]:
    """
    Search the database for the query.

    :param db: Chroma database
    :param query: Query string
    :return: List of documents
    """
    results = db.similarity_search_with_relevance_scores(query, k=3)

    # Filter out the results with a relevance score less than 0.6
    if len(results) == 0 or results[0][1] < 0.6:
        return "Sorry, I couldn't find any relevant information."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    model = ChatOpenAI()
    response_text = model.invoke(prompt).content

    source = [doc.metadata.get("source", None) for doc, _ in results]

    return f"{response_text}\n\nSource: {source[0]}"
