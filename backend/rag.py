import os
import re
import tempfile
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin
import pandas as pd
import PyPDF2
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None

from groq import Groq
import requests
from bs4 import BeautifulSoup
try:
    import html2text
except ImportError:
    html2text = None

try:
    import nltk
except ImportError:
    nltk = None

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from datetime import datetime
from config import settings
from models import Bot, BotSource


class BotRAGEngine:
    """RAG Engine for individual bots with multi-tenant ChromaDB support"""
    
    def __init__(self, bot: Bot):
        """Initialize RAG engine for a specific bot"""
        # Disable ChromaDB telemetry
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        self.bot = bot
        self.collection_name = bot.collection_name
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        
        # Get or create bot-specific collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        
        # Initialize embeddings model
        try:
            if HuggingFaceEmbeddings:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL_NAME
                )
            else:
                self.embeddings = None
        except Exception as e:
            print(f"Warning: Could not load HuggingFace embeddings: {e}")
            self.embeddings = None
        
        # Initialize text splitters
        self.text_splitters = {
            'standard': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=250,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            ),
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=250,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
            ),
            'web': RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
            ),
            'contact': RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        }
        
        # Initialize HTML to text converter
        if html2text:
            self.html_converter = html2text.HTML2Text()
            self.html_converter.ignore_links = False
            self.html_converter.ignore_images = True
        else:
            self.html_converter = None
        
        # Download NLTK data
        try:
            if nltk:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
        except:
            pass
    
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
    
    def scrape_website_beautiful_soup(self, url: str) -> str:
        """Scrape website content using BeautifulSoup"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        ]
        
        for attempt, user_agent in enumerate(user_agents):
            try:
                headers = {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                
                session = requests.Session()
                session.headers.update(headers)
                
                response = session.get(url, timeout=(10, 30), allow_redirects=True, verify=True)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '').lower()
                if 'html' not in content_type and 'text' not in content_type:
                    raise Exception(f"URL returned non-HTML content: {content_type}")
                
                try:
                    soup = BeautifulSoup(response.content, 'lxml')
                except:
                    soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "noscript"]):
                    script.decompose()
                
                # Extract content from various selectors
                content_selectors = [
                    'main', 'article', '.content', '.post', '.entry-content', 
                    '.article-body', '.post-content', '#content', '.main-content',
                    '.blog-post', '.news-content', '.page-content', '.post-body',
                    '.contact', '.contact-info', '.contact-section', '#contact',
                    '.footer', 'footer', '.site-footer', '#footer',
                    '.address', '.location', '.office-info', '.company-info',
                    'nav', '.navigation', '.navbar', '.menu'
                ]
                
                main_content = ""
                content_found = []
                
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        section_content = " ".join([elem.get_text(separator=' ', strip=True) for elem in elements])
                        if section_content and len(section_content.strip()) > 10:
                            content_found.append(f"[{selector}] {section_content}")
                
                if content_found:
                    main_content = " ".join(content_found)
                else:
                    body = soup.find('body')
                    if body:
                        main_content = body.get_text(separator=' ', strip=True)
                    else:
                        main_content = soup.get_text(separator=' ', strip=True)
                
                # Clean the extracted text
                main_content = re.sub(r'\s+', ' ', main_content)
                main_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', main_content)
                
                if main_content.strip():
                    return main_content.strip()
                else:
                    raise Exception("No meaningful content extracted")
                
            except Exception as e:
                if attempt == len(user_agents) - 1:
                    raise Exception(f"BeautifulSoup scraping failed: {str(e)}")
                continue
    
    def scrape_website_selenium(self, url: str) -> str:
        """Scrape website content using Selenium for dynamic content"""
        driver = None
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            try:
                driver = webdriver.Chrome(options=chrome_options)
                driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            except Exception as e:
                raise Exception(f"Failed to initialize Chrome driver: {str(e)}")
            
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(10)
            
            driver.get(url)
            
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Scroll to load lazy content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'lxml')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                element.decompose()
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '.post', '.entry-content', 
                '.article-body', '.post-content', '#content', '.main-content',
                '.blog-post', '.news-content', '.page-content', '.post-body'
            ]
            
            main_content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    main_content = " ".join([elem.get_text(strip=True) for elem in elements])
                    break
            
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text(separator=' ', strip=True)
                else:
                    main_content = soup.get_text(separator=' ', strip=True)
            
            # Clean the text
            main_content = re.sub(r'\s+', ' ', main_content)
            main_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', main_content)
            
            if main_content.strip():
                return main_content.strip()
            else:
                raise Exception("No meaningful content extracted with Selenium")
                
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def extract_text_from_url(self, url: str) -> str:
        """Extract text from URL with fallback options"""
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        # Try BeautifulSoup first
        try:
            return self.scrape_website_beautiful_soup(url)
        except Exception as e:
            # Fallback to Selenium
            try:
                return self.scrape_website_selenium(url)
            except Exception as e2:
                raise Exception(f"Both scraping methods failed. BeautifulSoup: {e}, Selenium: {e2}")
    
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
                    content_type = "general"
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
            
            # Choose appropriate text splitter
            critical_patterns = [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
                r'\$\d+(?:\.\d{2})?',  # Prices
                r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b'  # Addresses
            ]
            
            critical_info_count = sum(len(re.findall(pattern, cleaned_text, re.IGNORECASE)) for pattern in critical_patterns)
            critical_density = critical_info_count / (len(cleaned_text) / 1000) if len(cleaned_text) > 0 else 0
            
            if critical_density > 2:
                splitter = self.text_splitters['contact']
            elif content_type == "web":
                splitter = self.text_splitters['web']
            elif content_type in ["pdf", "excel"]:
                splitter = self.text_splitters['recursive']
            else:
                splitter = self.text_splitters['standard']
            
            # Split text into chunks
            chunks = splitter.split_text(cleaned_text)
            
            if not chunks:
                return {"error": "No content chunks generated"}
            
            # Create metadata for each chunk
            documents = chunks
            metadatas = [
                {
                    "source": source_name,
                    "content_type": content_type,
                    "chunk_id": i,
                    "bot_id": self.bot.id,
                    "word_count": len(chunk.split()),
                    "char_count": len(chunk),
                } 
                for i, chunk in enumerate(chunks)
            ]
            ids = [f"{self.bot.id}_{source_name}_{i}" for i in range(len(chunks))]
            
            # Store in ChromaDB
            if self.embeddings:
                embeddings = []
                for chunk in chunks:
                    embedding = self.embeddings.embed_query(chunk)
                    embeddings.append(embedding)
                
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            # Update bot source record in database if provided
            if db:
                source = db.query(BotSource).filter(
                    BotSource.bot_id == self.bot.id,
                    BotSource.source_name == source_name
                ).first()
                
                if source:
                    source.processing_status = "completed"
                    source.chunk_count = len(chunks)
                    source.processed_at = func.now()
                    db.commit()
            
            return {
                "message": f"Successfully processed and stored {len(chunks)} chunks",
                "chunks_count": len(chunks),
                "content_type": content_type,
                "source_name": source_name
            }
            
        except Exception as e:
            # Update source record with error if database session provided
            if db:
                source = db.query(BotSource).filter(
                    BotSource.bot_id == self.bot.id,
                    BotSource.source_name == source_name if 'source_name' in locals() else filename or url
                ).first()
                
                if source:
                    source.processing_status = "failed"
                    source.error_message = str(e)
                    db.commit()
            
            return {"error": f"Error processing document: {str(e)}"}
    
    def search_similar_documents(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents in bot's ChromaDB collection"""
        try:
            if self.embeddings:
                query_embedding = self.embeddings.embed_query(query)
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
            
            # Format results
            if not results['documents'] or not results['documents'][0]:
                return []
            
            formatted_results = []
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []
    
    def generate_response(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Generate response using RAG with bot-specific guidelines"""
        import time
        start_time = time.time()
        
        # Search for relevant documents
        relevant_docs = self.search_similar_documents(query, n_results=5)
        
        if not relevant_docs:
            return {
                "response": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                "sources_used": [],
                "response_time_ms": int((time.time() - start_time) * 1000)
            }
        
        # Create context from retrieved documents
        context_parts = []
        sources_used = []
        
        for doc in relevant_docs:
            context_parts.append(doc['content'])
            sources_used.append({
                'source': doc['metadata'].get('source', 'Unknown'),
                'content_type': doc['metadata'].get('content_type', 'Unknown'),
                'relevance_score': 1 - doc.get('distance', 0)  # Convert distance to relevance
            })
        
        context = "\n\n".join(context_parts)
        
        # Get bot guidelines if any
        guidelines_text = ""
        if hasattr(self.bot, 'guidelines') and self.bot.guidelines:
            active_guidelines = [g for g in self.bot.guidelines if g.is_active]
            if active_guidelines:
                guidelines_text = "\n".join([f"- {g.content}" for g in active_guidelines])
        
        # Create prompt with context, guidelines, and query
        prompt = f"""You are an AI assistant for {self.bot.widget_name or 'AI Assistant'}. Based on the following context from the uploaded documents, please answer the user's question.

{f'''Guidelines to follow:
{guidelines_text}

''' if guidelines_text else ''}Context:
{context}

Question: {query}

Please provide a helpful and accurate answer based only on the information provided in the context above."""
        
        # Generate response using Groq
        try:
            completion = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            response = completion.choices[0].message.content
            response_time_ms = int((time.time() - start_time) * 1000)
            
            return {
                "response": response,
                "sources_used": sources_used if self.bot.show_sources else [],
                "response_time_ms": response_time_ms
            }
            
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "sources_used": [],
                "response_time_ms": int((time.time() - start_time) * 1000)
            }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about bot's document collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "bot_id": self.bot.id
            }
        except Exception as e:
            return {"error": f"Error getting collection info: {str(e)}"}
    
    def clear_collection(self) -> Dict[str, str]:
        """Clear all documents from bot's collection"""
        try:
            # Delete and recreate collection
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
            return {"message": "Collection cleared successfully", "status": "success"}
        except Exception as e:
            return {"error": f"Error clearing collection: {str(e)}"}


def get_bot_rag_engine(bot: Bot) -> BotRAGEngine:
    """Factory function to create RAG engine for a bot"""
    return BotRAGEngine(bot)
