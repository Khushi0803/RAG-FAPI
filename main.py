from pathlib import Path
from rag_pipeline import (
    process_all_pdfs,
    split_documents,
    EmeddingManager,
    VectorStore,
    RAGRetriever,
    rag_simple,
    llm
)

BASE_DIR = Path(__file__).parent
VECTOR_STORE_PATH = str(BASE_DIR / "chroma_db")

def build_and_save_vectorstore():
    all_pdf_documents = process_all_pdfs(".")
    chunks = split_documents(all_pdf_documents)

    emedding_manager = EmeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = emedding_manager.generate_emeddings(texts)

    vectorstore = VectorStore(persist_directory=VECTOR_STORE_PATH)  # ✅
    vectorstore.add_documents(chunks, embeddings)
    print(f"✅ Vector store saved to {VECTOR_STORE_PATH}")


def load_retriever() -> RAGRetriever:
    print("Loading retriever from existing vector store...")
    emedding_manager = EmeddingManager()
    vectorstore = VectorStore(persist_directory=VECTOR_STORE_PATH)  
    
    if vectorstore.collection.count() == 0:
        print("ChromaDB is empty,Building vector store now")
        build_and_save_vectorstore()
        
    rag_retriever = RAGRetriever(vectorstore, emedding_manager)
    print("✅ Retriever loaded successfully")
    return rag_retriever

if __name__ == "__main__":
    build_and_save_vectorstore()
