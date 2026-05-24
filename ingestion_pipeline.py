import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv




load_dotenv(dotenv_path=r"E:\SOUPTIK PAL\\TEST RAG\\.env",override=True)

print(f"Current Directory: {os.getcwd()}")
print(f"Is .env file here? {os.path.exists('.env')}")
print(f"Is API Key loaded? {bool(os.getenv('GOOGLE_API_KEY'))}")

api_key = os.getenv("GOOGLE_API_KEY")


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

#Check if dors directory exists
    if not os.path.exists(docs_path):
            raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your files.")



#Load all .txt files from the docs directory

    loader =DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    if len(documents)==0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your documents.")

    for i, doc in enumerate(documents [:4]): 

        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters") 
        print(f"Content preview: {doc.page_content [:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents



def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n",","," "],
        
    )
        

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store using Google Gemini"""
    print("Creating Gemini embeddings and storing in ChromaDB...")

 
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_document",
        google_api_key = os.getenv("GOOGLE_API_KEY")
    )

    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
      
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore    

def main():
    print("main fuction")
    documents= load_documents(docs_path="E:\\SOUPTIK PAL\\test Rag\\docs")
    chunks=split_documents(documents,chunk_size=600)
    vector_store=create_vector_store(chunks)


if __name__=="__main__":
    main()