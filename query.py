from get_embedding_function import get_embeddding_function
from langchain_community.vectorstores import Chroma
from prompt import PROMPT_TEMPLATE
from langchain_community.chat_models import ChatOllama

CHROMA_PATH = 'chroma'

def query_rag():
    db = Chroma(
    persist_directory=CHROMA_PATH, 
    embedding_function=get_embeddding_function()
    )
    query_text = input("What question would you like to ask about the files in the database?: ")

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    #print(f"Prompt: \n\n {prompt}")

    llm = ChatOllama(model="llama3")
    response = llm.invoke(prompt)
    response_text = response.content
    print(response_text)

if __name__ == "__main__":
    query_rag()