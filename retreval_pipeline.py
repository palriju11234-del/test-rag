from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


load_dotenv(dotenv_path=r"E:\SOUPTIK PAL\\TEST RAG\\.env",override=True)

persistent_directory = "db/chroma_db"


embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
query ="Does the text claim that nuclear energy is a renewable energy source?"

#retriever = db.as_retriever(search_kwargs={"k": 3})

# Alternative retriever (commented out)
retriever=db.as_retriever(
     search_type="similarity_score_threshold",
     search_kwargs={
         "k": 5,
         "score_threshold":0.5
     }
 )


relevant_docs = retriever.invoke(query)


print(f"User Query: {query}")

print("\n--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:")
    print(doc.page_content)