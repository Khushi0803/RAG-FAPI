__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
# ✅ Text splitters (moved to separate package)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


### Read all the pdf's inside the directory
def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

"""# Process all PDFs in the data directory
all_pdf_documents = process_all_pdfs("data") -"""



### text splitting

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

     # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs

""'chunks=split_documents(all_pdf_documents)'""



### emedding and vector store 
import numpy as np
#from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple   
from sklearn.metrics.pairwise import cosine_similarity


# Emedding Manager
class EmeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding manager ready")

    def generate_emeddings(self, texts):
        import numpy as np
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings).astype(np.float32)
      
  

        """class EmeddingManager:
            #Handles document embedding generation using SentenceTransformer
            def __init__(self,model_name:str = "all-MiniLM-L6-v2"):

                Initialize the emedding manager
                    Argss:Hugging face model name for sentence emeddings
                
                self.model_name = model_name
                self.model = None
                self._load_model()

            def _load_model(self):
                try:
                    print(f"Loading emedding model :{self.model_name}")
                    self.model = SentenceTransformer(self.model_name)
                    print(f"Model loaded successfully.Embedding dimension : {self.model.get_sentence_embedding_dimension()}")
                except Exception as e:
                    print(f"Error loading model {self.model_name}:{e}")
                    raise
        
    def generate_emeddings(self, texts : List[str]) -> np.ndarray :
        
        if not self.model:
            raise ValueError("Model not Loaded")

        print(f"Generating emeddings for {len(texts)} texts...")
        emeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated emeddings with shape: {emeddings.shape}")
        return emeddings

        ## initialize the emedding manager":

        emedding_manager=EmeddingManager()
        emedding_manager"""
        

# VECTOR STORE

class VectorStore:
   # Manages document emeddings in a ChromaDB vector store
    
    def __init__(self, collection_name: str = "pdf_documents", 
                 persist_directory: str = "C:/chroma_db"): 
        
        """Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store"""
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        #Initialize ChromaDB client and collection
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document emeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], emeddings: np.ndarray):
        
        """Add documents and their emeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            emeddings: Corresponding emeddings for the documents"""
        
        if len(documents) != len(emeddings):
            raise ValueError("Number of documents must match number of emeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        emeddings_list = []
        
        for i, (doc, emedding) in enumerate(zip(documents, emeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # emedding
            emeddings_list.append(emedding.tolist())
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=emeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise


"""vectorstore=VectorStore()"""

        

"""
    chunks

    #convert the text into emeddings"""


"""convert emeddings
    emeddings = emedding_manager.generate_emeddings(texts)
 

    vectorstore.add_documents(chunks,emeddings)"""

# Retrieval Pipleline
class RAGRetriever:

    def __init__(self,vector_store:VectorStore , emedding_manager : EmeddingManager):


        self.vector_store=vector_store
        self.emedding_manager=emedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents from query :'{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        #generate query emedding
        query_emedding=self.emedding_manager.generate_emeddings([query])[0]

        #Search in vector store:
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_emedding.tolist()],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]


                for i ,(doc_id,document,metadata,distance) in enumerate (zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1-distance

                    if similarity_score>= score_threshold :
                     retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

"""rag_retriever = RAGRetriever(vectorstore ,emedding_manager)"""

            
#rag_retriever.retrieve("f the company sponsors an employee’s certification, what service commitment may be required?")


# RAG Pipeline with Groq LLM

### Simple RAG pipeline with Groq LLM
# rag_pipeline.py - bottom section

from langchain_groq import ChatGroq
import os
import httpx
from dotenv import load_dotenv

# ✅ Load .env first
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file!")

print(f"✅ Groq key loaded: {groq_api_key[:10]}...")


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024,
    http_client=httpx.Client(verify=False)
)

## 2. Simple RAG function: retrieve context + generate response
def rag_simple(query,retriever,llm,top_k=3):
    ## retriever the context
    results=retriever.retrieve(query,top_k=top_k)
    context="\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."
    
    ## generate the answwer using GROQ LLM
    prompt=f"""Use the following context to answer the question concisely.
        Context:
        {context}

        Question: {query}

        Answer:"""
    
    #response=llm.invoke([prompt.format(context=context,query=query)])
    response = llm.invoke(prompt.format(context=context, query=query))

    return response.content

"""answer=rag_simple("Why is onboarding training mandatory for new employees and how does it affect probation confirmation?",rag_retriever,llm)
print(answer)"""







