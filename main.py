from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader 
from langchain.schema.document import Document
from get_embedding_function import get_embeddding_function
import os

#name of directory where data for rag is stored
DATA_DIR = "docs"
#full path to directory where data to be used for rag is stored
PATH_TO_DATA = os.path.join(os.path.dirname(__file__), DATA_DIR)

#function to load text files from a directory into documents
def load_documents_from_dir(path) -> list[Document]:
    loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

#function to split documents recursively and return the list of chunks
def split_documents(documents:list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    state_of_the_union = load_documents_from_dir(PATH_TO_DATA)
    texts = split_documents(state_of_the_union)
    print(texts[0])