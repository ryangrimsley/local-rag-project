from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embeddding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
