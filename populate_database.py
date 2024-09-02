import os
import shutil
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader 
from langchain.schema.document import Document
from get_embedding_function import get_embeddding_function
from langchain_community.vectorstores import Chroma

CHROMA_PATH = 'chroma'
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
        chunk_size = 600,
        chunk_overlap = 60,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks:list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddding_function()
    )
    existing_items = db.get(include=[]) #IDs are always included by default
    existing_ids = set(existing_items["ids"])
    new_chunks = []
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    #get chunks and their ids
    chunks_with_ids = calculate_chunk_ids(chunks)

    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    #if there are new chunks, add them to the database
    if len(new_chunks):
        print(f"Adding {len(new_chunks)} new documents")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("Documents Added!")
    else:
        print("No new documents to add.")

#function to give each chunk a unique id solely based on which chunk it is
# i.e. the first chunk will be 0, second will be 1, etc.
def calculate_chunk_ids(chunks:list[Document]):
    for index, chunk in enumerate(chunks):
        source = chunk.metadata['source']
        chunk_id = f"{source}:{index}"
        chunk.metadata["id"] = chunk_id
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database...")
        clear_database()

    docs = load_documents_from_dir(PATH_TO_DATA)
    chunks = split_documents(docs)

    add_to_chroma(chunks)

    