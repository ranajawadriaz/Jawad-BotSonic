import os
import re
import tempfile
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin
import pandas as pd
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from datetime import datetime
from config import settings
from models import Bot, BotSource


class BotRAGEngine:
    """RAG Engine for individual bots with ChromaDB support based on working reference"""
    
    def __init__(self, bot: Bot):
        """Initialize RAG engine for a specific bot based on working reference"""
        self.bot = bot
        self.collection_name = f"bot_{bot.id}_documents"
        
        # Disable ChromaDB telemetry to avoid error messages
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        # Initialize Groq client with API key
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        
        # Initialize ChromaDB client for local vector storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection for storing document vectors for this bot
        # Use default embedding function to avoid conflicts
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        
        # Initialize embeddings model for converting text to vectors
        # Using the new langchain-huggingface package
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✅ HuggingFace embeddings loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load HuggingFace embeddings: {e}")
            # Fallback to simpler approach if needed
            self.embeddings = None
        
        # Initialize text splitter based on working reference
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250,  # Increased overlap to preserve critical data
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text content from PDF file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        finally:
            os.unlink(temp_file_path)
    
    def extract_text_from_excel(self, file_content: bytes) -> str:
        """Extract text content from Excel file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            df = pd.read_excel(temp_file_path, sheet_name=None)
            text = ""
            for sheet_name, sheet_df in df.items():
                text += f"Sheet: {sheet_name}\n"
                text += sheet_df.to_string(index=False)
                text += "\n\n"
            return text
        finally:
            os.unlink(temp_file_path)
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text content from TXT file"""
        return file_content.decode('utf-8')
    
    def scrape_website_simple(self, url: str) -> str:
        """Simple and reliable web scraping"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            # Try to get main content
            main_content = ""
            
            # Look for main content areas
            for selector in ['main', 'article', '.content', '.post', '.entry-content', '#content', '.main-content']:
                elements = soup.select(selector)
                if elements:
                    main_content = " ".join([elem.get_text(separator=' ', strip=True) for elem in elements])
                    break
            
            # Fallback to body text
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text(separator=' ', strip=True)
                else:
                    main_content = soup.get_text(separator=' ', strip=True)
            
            # Clean up text
            main_content = re.sub(r'\s+', ' ', main_content.strip())
            
            if len(main_content) < 50:
                raise Exception("No meaningful content found")
            
            return main_content
            
        except Exception as e:
            raise Exception(f"Failed to scrape {url}: {str(e)}")
    
    def extract_text_from_url(self, url: str) -> str:
        """Extract text from URL"""
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        return self.scrape_website_simple(url)
    
    def process_and_store_document(self, file_content: bytes = None, filename: str = None, url: str = None, db: Session = None) -> Dict[str, Any]:
        """Process and store document/URL content in bot's ChromaDB collection"""
        try:
            # Determine input type and extract text
            if url:
                text = self.extract_text_from_url(url)
                content_type = "web"
                source_name = url
            elif filename and file_content:
                if filename.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_content)
                    content_type = "pdf"
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    text = self.extract_text_from_excel(file_content)
                    content_type = "excel"
                elif filename.lower().endswith('.txt'):
                    text = self.extract_text_from_txt(file_content)
                    content_type = "txt"
                else:
                    return {"error": "Unsupported file type"}
                source_name = filename
            else:
                return {"error": "Either file content with filename or URL must be provided"}
            
            # Validate content
            if not text or len(text.strip()) < 10:
                return {"error": "No meaningful content extracted"}
            
            # Clean text
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Store chunks in ChromaDB - basic approach without custom embeddings
            if self.collection:
                documents = chunks
                metadatas = [
                    {
                        "source": source_name,
                        "content_type": content_type,
                        "chunk_id": i,
                        "bot_id": str(self.bot.id)  # Convert to string to avoid type issues
                    } 
                    for i, chunk in enumerate(chunks)
                ]
                ids = [f"{self.collection_name}_chunk_{i}_{abs(hash(chunk)) % 1000000}" for i, chunk in enumerate(chunks)]
                
                # Store documents in ChromaDB collection
                if self.embeddings:
                    # Use custom embeddings if available
                    embeddings = []
                    for chunk in chunks:
                        # Convert text chunk to vector using embeddings model
                        embedding = self.embeddings.embed_query(chunk)
                        embeddings.append(embedding)
                    
                    # Store vectors in ChromaDB collection with custom embeddings
                    self.collection.add(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                else:
                    # Use ChromaDB's default embedding function
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                
                print(f"Successfully stored {len(chunks)} chunks in ChromaDB for bot {self.bot.id}")
            else:
                return {"error": "ChromaDB collection not available"}
            
            # Update database if provided
            if db and url:
                # Update the source status to completed
                source = db.query(BotSource).filter(
                    BotSource.bot_id == self.bot.id,
                    BotSource.source_url == url
                ).first()
                
                if source:
                    source.processing_status = "completed"
                    source.chunk_count = len(chunks)
                    source.processed_at = datetime.utcnow()
                    db.commit()
            
            elif db and filename:
                # Update file source status  
                source = db.query(BotSource).filter(
                    BotSource.bot_id == self.bot.id,
                    BotSource.source_name == filename
                ).first()
                
                if source:
                    source.processing_status = "completed"
                    source.chunk_count = len(chunks)
                    source.processed_at = datetime.utcnow()
                    db.commit()
            
            return {
                "message": f"Successfully processed {content_type} content",
                "chunks_created": len(chunks),
                "source_name": source_name,
                "content_type": content_type
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Update database with error if provided
            if db and url:
                source = db.query(BotSource).filter(
                    BotSource.bot_id == self.bot.id,
                    BotSource.source_url == url
                ).first()
                
                if source:
                    source.processing_status = "failed"
                    source.error_message = error_msg
                    db.commit()
            
            elif db and filename:
                source = db.query(BotSource).filter(
                    BotSource.bot_id == self.bot.id,
                    BotSource.source_name == filename
                ).first()
                
                if source:
                    source.processing_status = "failed"
                    source.error_message = error_msg
                    db.commit()
            
            return {"error": error_msg}
    
    def get_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in ChromaDB based on query - using working reference approach"""
        try:
            if not self.collection:
                print("ChromaDB collection not available")
                return []
                
            if self.embeddings:
                # Convert user query to vector for semantic search using custom embeddings 
                query_embedding = self.embeddings.embed_query(query)
                
                # Search for most similar chunks in ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where={"bot_id": str(self.bot.id)}  # Filter by bot_id
                )
            else:
                # Use ChromaDB's default embedding for search
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where={"bot_id": str(self.bot.id)}  # Filter by bot_id
                )
            
            # Format results - extract relevant document chunks
            chunks = []
            if results.get('documents') and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]
                
                for i, doc in enumerate(documents):
                    if not doc:  # Skip empty documents
                        continue
                        
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    distance = distances[i] if i < len(distances) else 1.0
                    
                    # Convert distance to similarity score
                    similarity_score = max(0, 1 - distance)
                    
                    chunks.append({
                        "content": doc,
                        "source": metadata.get("source", "unknown"),
                        "content_type": metadata.get("content_type", "unknown"),
                        "similarity_score": similarity_score,
                        "chunk_id": metadata.get("chunk_id", 0)
                    })
            
            return chunks
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []

    def generate_response(self, user_message: str, session_id: str = None) -> Dict[str, Any]:
        """Generate response using RAG"""
        try:
            # Get relevant chunks
            relevant_chunks = self.get_relevant_chunks(user_message, n_results=5)
            
            # Prepare context from chunks
            context = ""
            sources = []
            
            for chunk in relevant_chunks:
                context += f"Source: {chunk['source']}\n{chunk['content']}\n\n"
                sources.append(chunk)
            
            # Create system prompt
            system_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say so politely.

Context:
{context}

Please provide a helpful and accurate response based on the context provided."""
            
            # Generate response using Groq
            response = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=1,
                max_tokens=1000
            )
            
            ai_response = response.choices[0].message.content
            
            return {
                "response": ai_response,
                "sources": sources
            }
            
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "sources": []
            }


def get_bot_rag_engine(bot: Bot) -> BotRAGEngine:
    """Get RAG engine instance for a bot"""
    return BotRAGEngine(bot)
