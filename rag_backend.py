import os
import torch
import logging
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import time
import json
import pickle
import hashlib

# Langchain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_workers: int = mp.cpu_count()
    batch_size: int = 100
    index_dir: str = "faiss_index"
    embedding_model: str = "intfloat/e5-small"
    cache_embeddings: bool = True

class GPUManager:
    @staticmethod
    def is_available() -> tuple[bool, str]:
        try:
            if torch.cuda.is_available():
                return True, torch.cuda.get_device_name(0)
            return False, "No GPU found"
        except Exception as e:
            logger.error(f"GPU check failed: {e}")
            return False, str(e)
    
    @staticmethod
    def get_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        if torch.cuda.is_available():
            return {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(0),
                "cached": torch.cuda.memory_reserved(0)
            }
        return {"total": 0, "allocated": 0, "cached": 0}

class EmbeddingManager:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = GPUManager.get_device()
        self._model = None
        self._cache_dir = Path("embedding_cache")
        self._cache_dir.mkdir(exist_ok=True)
        logger.info(f"Using device for embeddings: {self.device}")
    
    @property
    def model(self):
        if self._model is None:
            self._model = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._model
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for embeddings"""
        content = "".join(texts)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cached_embeddings(self, cache_key: str) -> Optional[List[List[float]]]:
        """Load cached embeddings if available"""
        if not self.config.cache_embeddings:
            return None
        
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        return None
    
    def _save_embeddings_cache(self, cache_key: str, embeddings: List[List[float]]):
        """Save embeddings to cache"""
        if not self.config.cache_embeddings:
            return
        
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
        except Exception as e:
            logger.warning(f"Failed to save embeddings cache: {e}")

class FastPDFProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_single_pdf(self, file_path: str) -> List[Document]:
        """Process a single PDF file"""
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            
            # Add metadata
            filename = os.path.basename(file_path)
            for doc in docs:
                doc.metadata.update({
                    "source": filename,
                    "file_path": file_path,
                    "processed_at": time.time()
                })
            
            return docs
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
    
    def process_multiple_pdfs(self, file_paths: List[str], 
                            progress_callback: Optional[Callable] = None) -> List[Document]:
        """Process multiple PDFs concurrently"""
        all_docs = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_pdf, path): path 
                for path in file_paths
            }
            
            completed = 0
            total = len(file_paths)
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    completed += 1
                    
                    if progress_callback:
                        progress = completed / total * 0.3  # 30% for PDF processing
                        progress_callback(progress, f"📄 Processed {completed}/{total} PDFs")
                        
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
        
        return all_docs
    
    def split_documents(self, docs: List[Document], 
                       progress_callback: Optional[Callable] = None) -> List[Document]:
        """Split documents into chunks"""
        if progress_callback:
            progress_callback(0.35, "🔪 Splitting documents into chunks...")
        
        # Process in batches for better memory management
        all_chunks = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_chunks = self.text_splitter.split_documents(batch)
            all_chunks.extend(batch_chunks)
            
            if progress_callback:
                progress = 0.35 + (i / len(docs)) * 0.15  # 15% for splitting
                progress_callback(progress, f"🔪 Split {i + len(batch)}/{len(docs)} documents")
        
        return all_chunks

class VectorStoreManager:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.embedding_manager = EmbeddingManager(config)
        self.pdf_processor = FastPDFProcessor(config)
    
    def create_vectorstore(self, file_paths: List[str], 
                          progress_callback: Optional[Callable] = None) -> int:
        """Create FAISS vector store from PDF files"""
        try:
            # Step 1: Process PDFs
            if progress_callback:
                progress_callback(0.1, "🚀 Starting PDF processing...")
            
            all_docs = self.pdf_processor.process_multiple_pdfs(file_paths, progress_callback)
            
            if not all_docs:
                raise ValueError("No documents were processed successfully")
            
            # Step 2: Split documents
            chunks = self.pdf_processor.split_documents(all_docs, progress_callback)
            
            if progress_callback:
                progress_callback(0.5, f"📊 Created {len(chunks)} chunks")
            
            # Step 3: Create embeddings and vector store
            if progress_callback:
                progress_callback(0.6, "🧠 Generating embeddings...")
            
            vectordb = FAISS.from_documents(chunks, self.embedding_manager.model)
            
            if progress_callback:
                progress_callback(0.9, "💾 Saving vector store...")
            
            # Save vector store
            index_path = Path(self.config.index_dir)
            index_path.mkdir(exist_ok=True)
            vectordb.save_local(str(index_path))
            
            # Save metadata
            metadata = {
                "num_chunks": len(chunks),
                "num_docs": len(all_docs),
                "files": [os.path.basename(f) for f in file_paths],
                "config": self.config.__dict__,
                "created_at": time.time()
            }
            
            with open(index_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if progress_callback:
                progress_callback(1.0, "✅ Vector store created successfully!")
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vectorstore(self) -> FAISS:
        """Load existing vector store"""
        index_path = Path(self.config.index_dir)
        if not index_path.exists():
            raise FileNotFoundError(f"Vector store not found at {index_path}")
        
        return FAISS.load_local(
            str(index_path), 
            self.embedding_manager.model,
            allow_dangerous_deserialization=True
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get vector store metadata"""
        metadata_path = Path(self.config.index_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

class OllamaManager:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._llm = None
    
    @property
    def llm(self):
        if self._llm is None:
            self._llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                num_gpu=1 if torch.cuda.is_available() else 0,
                temperature=0.1
            )
        return self._llm
    
    def check_status(self) -> tuple[bool, str]:
        """Check if Ollama is running and accessible"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if self.model_name in model_names:
                    return True, f"✅ {self.model_name} is available"
                else:
                    return False, f"❌ {self.model_name} not found. Available: {model_names}"
            return False, f"❌ Ollama API returned status {response.status_code}"
        except Exception as e:
            return False, f"❌ Cannot connect to Ollama: {str(e)}"

class RAGChain:
    def __init__(self, config: ProcessingConfig, model_name: str = "mistral:7b-instruct"):
        self.config = config
        self.vector_manager = VectorStoreManager(config)
        self.ollama_manager = OllamaManager(model_name)
        self._chain = None
        self._retriever = None
    
    def initialize(self):
        """Initialize the RAG chain"""
        try:
            # Load vector store
            vectordb = self.vector_manager.load_vectorstore()
            self._retriever = vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.5}
            )
            
            # Create prompt template
            prompt_template = """You are a helpful AI assistant. Use ONLY the following context to answer the question.
If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Provide a clear, concise answer based only on the context above:"""
            
            # Create QA chain using the modern approach
            from langchain.chains import RetrievalQA
            self._chain = RetrievalQA.from_chain_type(
                llm=self.ollama_manager.llm,
                chain_type="stuff",
                retriever=self._retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {e}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self._chain:
            raise ValueError("RAG chain not initialized. Call initialize() first.")
        
        try:
            # Get answer from chain
            result = self._chain({"query": question})
            
            # Calculate confidence based on similarity scores
            docs = result.get("source_documents", [])
            scores = [doc.metadata.get("score", 0.5) for doc in docs]
            avg_confidence = sum(scores) / len(scores) if scores else 0.0
            
            return {
                "result": result["result"],
                "source_documents": docs,
                "confidence": avg_confidence,
                "num_sources": len(docs)
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG chain: {e}")
            return {
                "result": f"Error processing query: {str(e)}",
                "source_documents": [],
                "confidence": 0.0
            }

# Main interface functions
def create_rag_system(file_paths: List[str], config: ProcessingConfig = None, 
                     progress_callback: Optional[Callable] = None) -> RAGChain:
    """Create a complete RAG system from PDF files"""
    if config is None:
        config = ProcessingConfig()
    
    # Create vector store
    vector_manager = VectorStoreManager(config)
    vector_manager.create_vectorstore(file_paths, progress_callback)
    
    # Create and initialize RAG chain
    rag_chain = RAGChain(config)
    rag_chain.initialize()
    
    return rag_chain

def load_existing_rag_system(config: ProcessingConfig = None) -> RAGChain:
    """Load an existing RAG system"""
    if config is None:
        config = ProcessingConfig()
    
    rag_chain = RAGChain(config)
    rag_chain.initialize()
    
    return rag_chain

# Export main functions
__all__ = [
    'GPUManager',
    'ProcessingConfig', 
    'RAGChain',
    'create_rag_system',
    'load_existing_rag_system',
    'OllamaManager'
]