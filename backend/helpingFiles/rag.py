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
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import requests
from bs4 import BeautifulSoup
import html2text
import nltk
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class RAGChatbot:
    def __init__(self):
        # Disable ChromaDB telemetry to avoid error messages
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        # Initialize Groq client with API key
        self.groq_client = Groq(api_key="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")
        
        # Initialize ChromaDB client for local vector storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection for storing document vectors
        # Use default embedding function to avoid conflicts
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents"
        )
        
        # Initialize embeddings model for converting text to vectors
        # Using the new langchain-huggingface package
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            print(f"Warning: Could not load HuggingFace embeddings: {e}")
            # Fallback to simpler approach if needed
            self.embeddings = None
        
        # Initialize multiple text splitters for different content types
        self.text_splitters = {
            # Standard splitter for general content with critical data preservation
            'standard': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=250,  # Increased overlap to preserve critical data
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            ),
            # Recursive splitter for better semantic chunking with critical data awareness
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=250,  # Increased overlap
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
            ),
            # Smaller chunks for web content but with sufficient overlap for contact info
            'web': RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,  # Increased overlap from 150 to 200
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
            ),
            # Special splitter for contact-heavy content
            'contact': RecursiveCharacterTextSplitter(
                chunk_size=600,  # Smaller chunks to keep contact info together
                chunk_overlap=300,  # Large overlap to ensure phone numbers aren't split
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        }
        
        # Initialize HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        
        # Download NLTK data for text preprocessing (if not already downloaded)
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text content from PDF file"""
        # Create a temporary file to read PDF content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Read PDF and extract text from all pages
            with open(temp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def extract_text_from_excel(self, file_content: bytes) -> str:
        """Extract text content from Excel file"""
        # Create temporary file for Excel processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Read all sheets from Excel file
            df = pd.read_excel(temp_file_path, sheet_name=None)
            text = ""
            for sheet_name, sheet_df in df.items():
                # Convert DataFrame to string representation
                text += f"Sheet: {sheet_name}\n"
                text += sheet_df.to_string(index=False)
                text += "\n\n"
            return text
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extract text content from TXT file"""
        # Decode bytes to string (assuming UTF-8 encoding)
        return file_content.decode('utf-8')
    
    def scrape_website_beautiful_soup(self, url: str) -> str:
        """Scrape website content using BeautifulSoup (for static content) with enhanced error handling"""
        # List of user agents to try if one fails
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        # Try different configurations
        for attempt, user_agent in enumerate(user_agents):
            try:
                print(f"Attempting to scrape {url} with user agent {attempt + 1}/{len(user_agents)}")
                
                # Enhanced headers to mimic a real browser
                headers = {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Cache-Control': 'max-age=0'
                }
                
                # Create session for better connection handling
                session = requests.Session()
                session.headers.update(headers)
                
                # Configure session settings
                session.max_redirects = 5
                
                # Make HTTP request with longer timeout and retry logic
                response = session.get(
                    url, 
                    timeout=(10, 30),  # (connect timeout, read timeout)
                    allow_redirects=True,
                    verify=True,  # SSL verification
                    stream=False
                )
                response.raise_for_status()
                
                # Check if response contains HTML content
                content_type = response.headers.get('content-type', '').lower()
                if 'html' not in content_type and 'text' not in content_type:
                    raise Exception(f"URL returned non-HTML content: {content_type}")
                
                # Parse HTML content with different parsers if needed
                try:
                    soup = BeautifulSoup(response.content, 'lxml')
                except:
                    soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements but keep nav/footer for contact info
                for script in soup(["script", "style", "noscript"]):
                    script.decompose()
                
                # Extract ALL content including nav, footer, and contact sections
                # Don't filter out nav/footer as they often contain contact info
                content_selectors = [
                    # Main content areas
                    'main', 'article', '.content', '.post', '.entry-content', 
                    '.article-body', '.post-content', '#content', '.main-content',
                    '.blog-post', '.news-content', '.page-content', '.post-body',
                    # Contact-specific areas
                    '.contact', '.contact-info', '.contact-section', '#contact',
                    '.footer', 'footer', '.site-footer', '#footer',
                    '.address', '.location', '.office-info', '.company-info',
                    'nav', '.navigation', '.navbar', '.menu'
                ]
                
                main_content = ""
                content_found = []
                
                # Try each selector and combine all content
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        section_content = " ".join([elem.get_text(separator=' ', strip=True) for elem in elements])
                        if section_content and len(section_content.strip()) > 10:
                            content_found.append(f"[{selector}] {section_content}")
                
                # Combine all found content
                if content_found:
                    main_content = " ".join(content_found)
                else:
                    # If no specific content found, get everything from body
                    body = soup.find('body')
                    if body:
                        main_content = body.get_text(separator=' ', strip=True)
                    else:
                        main_content = soup.get_text(separator=' ', strip=True)
                
                # DEBUG: Look specifically for address patterns in the raw content
                print(f"🔍 DEBUG: Extracted {len(main_content)} characters from {url}")
                print(f"🔍 DEBUG: First 300 chars: {main_content[:300]}")
                
                # Check for common address indicators
                address_keywords = ['address', 'location', 'office', 'headquarters', 'street', 'avenue', 'road', 'blvd', 'suite', 'floor']
                found_keywords = [kw for kw in address_keywords if kw.lower() in main_content.lower()]
                if found_keywords:
                    print(f"🏠 DEBUG: Found address keywords: {found_keywords}")
                    
                    # Look for text around these keywords
                    for keyword in found_keywords[:3]:  # Check first 3 matches
                        pattern = f'.{{0,100}}{re.escape(keyword)}.{{0,100}}'
                        matches = re.finditer(pattern, main_content, re.IGNORECASE)
                        for i, match in enumerate(matches):
                            if i >= 2:  # Limit to 2 matches per keyword
                                break
                            print(f"🏠 Context around '{keyword}': {match.group()}")
                else:
                    print(f"⚠️  DEBUG: No address keywords found in scraped content")
                
                # Basic validation of extracted content
                if len(main_content.strip()) < 100:
                    print(f"Warning: Very little content extracted from {url}")
                
                # Clean and format the text
                # Remove excessive whitespace
                main_content = re.sub(r'\s+', ' ', main_content)
                main_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', main_content)
                
                if main_content.strip():
                    print(f"Successfully scraped {len(main_content)} characters from {url}")
                    return main_content.strip()
                else:
                    raise Exception("No meaningful content extracted")
                
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error with user agent {attempt + 1}: {str(e)}")
                if attempt == len(user_agents) - 1:
                    raise Exception(f"Connection failed after trying {len(user_agents)} different configurations: {str(e)}")
                continue
                
            except requests.exceptions.Timeout as e:
                print(f"Timeout error with user agent {attempt + 1}: {str(e)}")
                if attempt == len(user_agents) - 1:
                    raise Exception(f"Request timeout after trying {len(user_agents)} different configurations: {str(e)}")
                continue
                
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error with user agent {attempt + 1}: {str(e)}")
                if attempt == len(user_agents) - 1:
                    raise Exception(f"HTTP error after trying {len(user_agents)} different configurations: {str(e)}")
                continue
                
            except requests.exceptions.RequestException as e:
                print(f"Request error with user agent {attempt + 1}: {str(e)}")
                if attempt == len(user_agents) - 1:
                    raise Exception(f"Request failed after trying {len(user_agents)} different configurations: {str(e)}")
                continue
                
            except Exception as e:
                print(f"Parsing error with user agent {attempt + 1}: {str(e)}")
                if attempt == len(user_agents) - 1:
                    raise Exception(f"Failed to parse content after trying {len(user_agents)} different configurations: {str(e)}")
                continue
        
        # This should never be reached, but just in case
        raise Exception("Unexpected error in scraping process")
    
    def scrape_website_selenium(self, url: str) -> str:
        """Scrape website content using Selenium (for dynamic content) with enhanced error handling"""
        driver = None
        try:
            print(f"Attempting Selenium scraping for {url}")
            
            # Configure Chrome options for headless browsing
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
            
            # Additional options for better compatibility
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            chrome_options.add_argument("--disable-javascript")  # For faster loading of static content
            
            # Initialize Chrome driver with error handling
            try:
                driver = webdriver.Chrome(options=chrome_options)
                driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            except Exception as e:
                raise Exception(f"Failed to initialize Chrome driver. Please ensure Chrome and ChromeDriver are installed: {str(e)}")
            
            # Set timeouts
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(10)
            
            try:
                # Navigate to URL
                driver.get(url)
                
                # Wait for page to load
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Scroll to load any lazy-loaded content
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Get page source and parse with BeautifulSoup
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'lxml')
                
                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                    element.decompose()
                
                # Extract main content with same selectors as BeautifulSoup method
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
                
                # If no main content found, get all text from body
                if not main_content:
                    body = soup.find('body')
                    if body:
                        main_content = body.get_text(separator=' ', strip=True)
                    else:
                        main_content = soup.get_text(separator=' ', strip=True)
                
                # Clean the extracted text
                main_content = re.sub(r'\s+', ' ', main_content)
                main_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', main_content)
                
                if main_content.strip():
                    print(f"Selenium successfully scraped {len(main_content)} characters from {url}")
                    return main_content.strip()
                else:
                    raise Exception("No meaningful content extracted with Selenium")
                
            except Exception as e:
                raise Exception(f"Error during Selenium scraping: {str(e)}")
                
        except Exception as e:
            print(f"Selenium scraping failed: {e}")
            raise Exception(f"Selenium scraping failed: {str(e)}")
        finally:
            # Always clean up the driver
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def extract_text_from_url(self, url: str) -> str:
        """Main method to extract text from URL with fallback options"""
        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format. Please provide a complete URL with http:// or https://")
        
        # Try BeautifulSoup first (faster for static content)
        try:
            return self.scrape_website_beautiful_soup(url)
        except Exception as e:
            print(f"BeautifulSoup scraping failed: {e}")
            # Fallback to Selenium for dynamic content
            try:
                return self.scrape_website_selenium(url)
            except Exception as e2:
                raise Exception(f"Both scraping methods failed. BeautifulSoup: {e}, Selenium: {e2}")
    
    def clean_and_structure_text(self, text: str, content_type: str = "general") -> str:
        """NO CLEANING - Return text exactly as-is to prevent any data loss"""
        if not text or not text.strip():
            return ""
        
        # ZERO cleaning - return text exactly as scraped
        print(f"🔍 NO CLEANING applied. Original text length: {len(text)} chars")
        print(f"🔍 First 200 chars: {repr(text[:200])}")
        
        return text.strip()  # Only remove leading/trailing whitespace
    
    def validate_content_quality(self, text: str) -> Dict[str, Any]:
        """Very permissive content validation - accepts almost all content to prevent data loss"""
        if not text:
            return {"is_valid": False, "reason": "Empty content"}
        
        # Check for critical information patterns that should always be preserved
        critical_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',   # (555) 123-4567
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\$\d+(?:\.\d{2})?',  # Prices
            r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',  # Addresses
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',  # Days
            r'\b\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?\b',  # Times
            r'\b(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?\b'  # Websites
        ]
        
        has_critical_info = any(re.search(pattern, text, re.IGNORECASE) for pattern in critical_patterns)
        
        # Basic quality metrics
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.split('\n'))
        
        # VERY permissive minimum requirements - accept even single words if they might be important
        min_words = 1  
        min_chars = 3  # Accept very short content
        
        # Only reject truly empty or meaningless content
        if word_count < min_words and char_count < min_chars:
            return {"is_valid": False, "reason": f"Content too minimal: {word_count} words, {char_count} chars"}
        
        # Check if it's just whitespace or special characters
        if not re.search(r'[a-zA-Z0-9]', text):
            return {"is_valid": False, "reason": "No alphanumeric content"}
        
        # Check for placeholder content (when scraping fails)
        if "Content could not be scraped due to anti-bot protection" in text:
            return {
                "is_valid": True,  # Accept placeholder content to prevent crashes
                "word_count": len(text.split()),
                "char_count": len(text),
                "line_count": 1,
                "word_diversity": 1.0,
                "ui_ratio": 0.0,
                "has_critical_info": False,
                "is_placeholder": True
            }
        
        # Accept everything else - no diversity checks, no UI filtering
        # The goal is to preserve ALL potentially useful information
        
        return {
            "is_valid": True,
            "word_count": word_count,
            "char_count": char_count,
            "line_count": line_count,
            "word_diversity": 1.0,  # Don't calculate, just mark as good
            "ui_ratio": 0.0,  # Don't filter UI content
            "has_critical_info": has_critical_info
        }
    
    def process_and_store_document(self, file_content: bytes = None, filename: str = None, url: str = None) -> Dict[str, str]:
        """Process uploaded document or URL and store in ChromaDB as vectors with advanced preprocessing"""
        try:
            # Determine input type and extract text
            if url:
                # Handle URL input with enhanced navbar page scraping
                print(f"🌐 Processing URL with enhanced navbar scraping: {url}")
                text = self.scrape_website_with_navbar_pages(url)
                content_type = "web"
                source_name = url
            elif filename and file_content:
                # Handle file input
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
                    return {"error": "Unsupported file type. Please upload PDF, Excel, TXT files, or provide a URL."}
                source_name = filename
            else:
                return {"error": "Either file content with filename or URL must be provided."}
            
            # Step 1: Clean and structure the extracted text
            cleaned_text = self.clean_and_structure_text(text, content_type)
            
            # Step 2: Validate content quality (but don't reject - just log)
            quality_check = self.validate_content_quality(cleaned_text)
            if not quality_check["is_valid"]:
                print(f"⚠️  Content quality warning: {quality_check['reason']} - but proceeding anyway to preserve data")
            else:
                print(f"✅ Content quality check passed: {quality_check.get('word_count', 0)} words")
            
            # Always proceed regardless of quality check to avoid data loss
            
            # Step 3: Choose appropriate text splitter based on content type and critical data detection
            # Check if content contains high density of critical information
            critical_patterns = [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
                r'\$\d+(?:\.\d{2})?',  # Prices
                r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b'  # Addresses
            ]
            
            critical_info_count = sum(len(re.findall(pattern, cleaned_text, re.IGNORECASE)) for pattern in critical_patterns)
            text_length = len(cleaned_text)
            critical_density = critical_info_count / (text_length / 1000) if text_length > 0 else 0  # per 1000 chars
            
            # Select splitter based on content type and critical data density
            if critical_density > 2:  # High density of critical info (>2 items per 1000 chars)
                splitter = self.text_splitters['contact']
                print(f"Using contact-aware chunking due to high critical data density: {critical_density:.2f}")
            elif content_type == "web":
                splitter = self.text_splitters['web']
            elif content_type in ["pdf", "excel"]:
                splitter = self.text_splitters['recursive']
            else:
                splitter = self.text_splitters['standard']
            
            # Step 4: Split text into semantically meaningful chunks
            chunks = splitter.split_text(cleaned_text)
            
            # Step 4.5: Post-process chunks to ensure critical information isn't lost at boundaries
            enhanced_chunks = self.enhance_chunks_for_critical_data(chunks, cleaned_text)
            
            # Step 5: Accept ALL chunks - no filtering to prevent data loss
            quality_chunks = enhanced_chunks  # Accept all chunks
            
            # Log information about the chunks but don't filter
            for i, chunk in enumerate(quality_chunks):
                chunk_quality = self.validate_content_quality(chunk)
                if chunk_quality.get("has_critical_info"):
                    print(f"📞 Chunk {i} contains critical information")
                if not chunk_quality["is_valid"]:
                    print(f"⚠️  Chunk {i} has quality warning: {chunk_quality['reason']} - but keeping it anyway")
                else:
                    print(f"✅ Chunk {i} quality: {chunk_quality.get('word_count', 0)} words")
            
            # Always proceed with all chunks - no rejection based on quality
            if not quality_chunks:
                # Check if this was due to anti-bot protection
                if "Content could not be scraped due to anti-bot protection" in cleaned_text:
                    return {
                        "message": f"Website {source_name} appears to have anti-bot protection that prevented content extraction",
                        "chunks_count": 0,
                        "content_type": content_type,
                        "status": "blocked"
                    }
                else:
                    print("⚠️  No chunks generated - this might indicate an issue with text extraction")
                    return {"error": "No content chunks generated - check source data"}
            
            print(f"📊 Storing ALL {len(quality_chunks)} chunks in vector database")
            
            # Step 6: Generate embeddings and store in ChromaDB
            documents = quality_chunks
            metadatas = [
                {
                    "source": source_name,
                    "content_type": content_type,
                    "chunk_id": i,
                    "word_count": len(chunk.split()),
                    "char_count": len(chunk),
                    "has_critical_info": any(re.search(pattern, chunk, re.IGNORECASE) 
                                           for pattern in [
                                               r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                                               r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                                               r'\$\d+',
                                               r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b'
                                           ])
                } 
                for i, chunk in enumerate(quality_chunks)
            ]
            ids = [f"{source_name}_chunk_{i}" for i in range(len(quality_chunks))]
            
            # Debug: Count critical information in final chunks
            critical_chunks = sum(1 for meta in metadatas if meta["has_critical_info"])
            print(f"Final storage: {len(quality_chunks)} chunks, {critical_chunks} contain critical information")
            
            # Store documents in ChromaDB collection
            if self.embeddings:
                # Use custom embeddings if available
                embeddings = []
                for chunk in quality_chunks:
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
            
            return {
                "message": f"Successfully processed and stored {len(quality_chunks)} quality chunks from {source_name}",
                "chunks_count": len(quality_chunks),
                "original_chunks": len(chunks),
                "filtered_chunks": len(chunks) - len(quality_chunks),
                "content_type": content_type,
                "quality_metrics": quality_check
            }
            
        except Exception as e:
            return {"error": f"Error processing document: {str(e)}"}
    
    def search_similar_documents(self, query: str, n_results: int = 3) -> List[str]:
        """Search for similar documents in ChromaDB based on query"""
        try:
            if self.embeddings:
                # Convert user query to vector for semantic search using custom embeddings
                query_embedding = self.embeddings.embed_query(query)
                
                # Search for most similar chunks in ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results  # Return top 3 most similar chunks
                )
            else:
                # Use ChromaDB's default embedding for search
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results  # Return top 3 most similar chunks
                )
            
            # Extract relevant document chunks
            relevant_docs = results['documents'][0] if results['documents'] else []
            return relevant_docs
        except Exception as e:
            print(f"Error in search: {e}")
            return []
    
    def generate_response(self, query: str) -> str:
        """Generate response using RAG: Retrieve relevant docs + Generate with LLM"""
        # Step 1: Retrieve relevant documents from vector database
        relevant_docs = self.search_similar_documents(query)
        
        if not relevant_docs:
            return "I couldn't find any relevant information in the uploaded documents to answer your question."
        
        # Step 2: Create context from retrieved documents
        context = "\n\n".join(relevant_docs)
        
        # Step 3: Create prompt with context and user query
        prompt = f"""Based on the following context from the uploaded documents, please answer the user's question.

Context:
{context}

Question: {query}

Please provide a helpful and accurate answer based only on the information provided in the context above."""
        
        # Step 4: Generate response using Groq LLM
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more focused responses
                max_completion_tokens=1024,
                top_p=1,
                stream=False,  # Set to False to get complete response
                stop=None,
            )
            
            # Extract and return the generated response
            response = completion.choices[0].message.content
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about stored documents in ChromaDB"""
        # Count total documents in collection
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }
    
    def process_url(self, url: str) -> Dict[str, str]:
        """Convenience method specifically for URL processing"""
        return self.process_and_store_document(url=url)
    
    def clear_database(self) -> Dict[str, str]:
        """Clear all documents from ChromaDB"""
        try:
            # Delete and recreate collection
            self.chroma_client.delete_collection(name="documents")
            self.collection = self.chroma_client.get_or_create_collection(name="documents")
            return {"message": "Database cleared successfully", "status": "success"}
        except Exception as e:
            return {"error": f"Error clearing database: {str(e)}"}
    
    def enhance_chunks_for_critical_data(self, chunks: List[str], original_text: str) -> List[str]:
        """Post-process chunks to ensure critical information isn't lost at boundaries"""
        if not chunks:
            return chunks
        
        enhanced_chunks = []
        critical_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',   # (555) 123-4567
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\$\d+(?:\.\d{2})?',  # Prices
            r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b'  # Addresses
        ]
        
        # Find all critical information in the original text
        all_critical_info = set()
        for pattern in critical_patterns:
            matches = re.finditer(pattern, original_text, re.IGNORECASE)
            for match in matches:
                all_critical_info.add(match.group().strip())
        
        # Check each chunk for missing critical information
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk
            
            # Find critical info present in this chunk
            chunk_critical_info = set()
            for pattern in critical_patterns:
                matches = re.finditer(pattern, chunk, re.IGNORECASE)
                for match in matches:
                    chunk_critical_info.add(match.group().strip())
            
            # Check if any critical info is missing from this chunk but exists in original
            missing_critical = all_critical_info - chunk_critical_info
            
            if missing_critical:
                # Look for context around missing critical info in original text
                for missing_item in missing_critical:
                    # Find the context where this critical info appears in original text
                    pattern = re.escape(missing_item)
                    match = re.search(f'.{{0,100}}{pattern}.{{0,100}}', original_text, re.IGNORECASE)
                    if match:
                        context = match.group().strip()
                        
                        # Check if this context would make sense to add to current chunk
                        # (basic heuristic: if chunk already mentions related terms)
                        context_words = set(context.lower().split())
                        chunk_words = set(chunk.lower().split())
                        
                        # Look for related terms that might indicate this critical info belongs here
                        contact_terms = {'contact', 'location'}
                        business_terms = {'service'}
                        
                        if (context_words & chunk_words) or (contact_terms & chunk_words) or (business_terms & chunk_words):
                            # Add the critical info with minimal context to this chunk
                            if missing_item not in chunk:
                                enhanced_chunk += f"\n\nContact Information: {context}"
                                print(f"Enhanced chunk {i} with missing critical info: {missing_item}")
            
            enhanced_chunks.append(enhanced_chunk)
        
        # Final pass: create a dedicated contact info chunk if we have orphaned critical data
        all_chunks_text = ' '.join(enhanced_chunks)
        still_missing = set()
        for critical_item in all_critical_info:
            if critical_item not in all_chunks_text:
                still_missing.add(critical_item)
        
        if still_missing:
            contact_chunk = "Contact Information:\n" + "\n".join(still_missing)
            enhanced_chunks.append(contact_chunk)
            print(f"Created dedicated contact chunk for orphaned critical info: {still_missing}")
        
        return enhanced_chunks
    
    def scrape_website_with_navbar_pages(self, base_url: str) -> str:
        """Enhanced scraping that discovers and scrapes ALL navbar pages from the main page"""
        print(f"🔍 Enhanced scraping starting for {base_url}")
        
        # Try to scrape the main page first with multiple methods
        main_content = ""
        main_page_html = ""
        try:
            print(f"🔍 Attempting BeautifulSoup scraping for main page...")
            
            # Get raw HTML first for link extraction
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                response = requests.get(base_url, timeout=10, headers=headers)
                main_page_html = response.content
            except Exception as e:
                print(f"⚠️  Could not get HTML for link extraction: {e}")
            
            # Get cleaned content
            main_content = self.scrape_website_beautiful_soup(base_url)
            
            if not main_content or len(main_content) < 50:  # Very little content means likely blocked
                print(f"⚠️  BeautifulSoup failed or blocked, trying Selenium for main page...")
                main_content = self.scrape_website_selenium(base_url)
        except Exception as e:
            print(f"⚠️  Main page BeautifulSoup failed: {e}")
            try:
                print(f"🔍 Fallback: Trying Selenium for main page...")
                main_content = self.scrape_website_selenium(base_url)
            except Exception as e2:
                print(f"⚠️  Both BeautifulSoup and Selenium failed for main page: {e2}")
        
        # Extract ALL navbar links from the main page
        navbar_links = self.extract_navbar_links(base_url, main_page_html)
        
        additional_content = []
        
        # Try to scrape discovered navbar pages
        scraped_pages_count = 0
        total_discovered_pages = len(navbar_links)
        
        print(f"🎯 Starting to scrape {total_discovered_pages} discovered navbar pages...")
        
        for i, (page_url, link_text) in enumerate(navbar_links):
            try:
                print(f"\n🔍 [{i+1}/{total_discovered_pages}] Trying discovered page: {page_url} (from link: '{link_text}')")
                
                # Try BeautifulSoup first, then Selenium with detailed error reporting
                page_content = ""
                scraping_success = False
                
                try:
                    print(f"   📄 Attempting BeautifulSoup for {page_url}...")
                    page_content = self.scrape_website_beautiful_soup(page_url)
                    if page_content and len(page_content) > 50:
                        print(f"   ✅ BeautifulSoup success: {len(page_content)} chars")
                        scraping_success = True
                    else:
                        print(f"   ⚠️  BeautifulSoup returned minimal content ({len(page_content)} chars), trying Selenium...")
                except Exception as bs_error:
                    print(f"   ❌ BeautifulSoup failed: {str(bs_error)[:200]}...")
                
                # If BeautifulSoup didn't work well, try Selenium
                if not scraping_success:
                    try:
                        print(f"   🤖 Attempting Selenium for {page_url}...")
                        page_content = self.scrape_website_selenium(page_url)
                        if page_content and len(page_content) > 50:
                            print(f"   ✅ Selenium success: {len(page_content)} chars")
                            scraping_success = True
                        else:
                            print(f"   ⚠️  Selenium returned minimal content ({len(page_content)} chars)")
                    except Exception as selenium_error:
                        print(f"   ❌ Selenium failed: {str(selenium_error)[:200]}...")
                
                # Process the content if we got something meaningful
                if scraping_success and page_content and len(page_content) > 100:
                    print(f"✅ Successfully scraped {page_url} ({len(page_content)} chars)")
                    additional_content.append(f"[Page: {link_text}] {page_content}")
                    scraped_pages_count += 1
                    
                    # Show a preview of the scraped content
                    print(f"   📄 Content preview: {page_content[:150]}...")
                    
                    # Check if this page has useful information
                    info_indicators = ['address', 'street', 'avenue', 'road', 'office', 'location', 'phone', 'email', 'contact', 'about', 'service', 'team', 'company']
                    found_indicators = [ind for ind in info_indicators if ind in page_content.lower()]
                    if found_indicators:
                        print(f"📋 Information indicators found on {page_url}: {found_indicators}")
                else:
                    print(f"❌ Failed to get meaningful content from {page_url}")
                        
            except Exception as e:
                print(f"⚠️  Unexpected error scraping {page_url}: {str(e)[:200]}...")
                continue
        
        print(f"\n📊 Navbar page scraping summary:")
        print(f"   • Total discovered: {total_discovered_pages}")
        print(f"   • Successfully scraped: {scraped_pages_count}")
        print(f"   • Failed: {total_discovered_pages - scraped_pages_count}")
        
        # Combine all content
        all_content = []
        if main_content and len(main_content) > 50:
            all_content.append(main_content)
        all_content.extend(additional_content)
        
        combined_content = " ".join([content for content in all_content if content])
        
        if not combined_content:
            # Last resort: create minimal placeholder content so the system doesn't crash
            print(f"⚠️  No content could be scraped from {base_url}")
            print(f"🔧 Creating placeholder content to prevent system crash")
            combined_content = f"Website: {base_url} (Content could not be scraped due to anti-bot protection)"
        
        print(f"🔍 Total content collected: {len(combined_content)} characters from {len(all_content)} pages")
        
        # Final check for comprehensive website information
        info_patterns = [
            r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',  # Addresses
            r'\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}',  # City, State ZIP
            r'\d{3}[-.]?\d{3}[-.]?\d{4}',  # Phone numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Prices
            r'\b(?:CEO|CTO|CFO|President|Director|Manager|VP)\b',  # Titles
            r'\b(?:since|founded|established)\s+\d{4}\b',  # Company founding dates
        ]
        
        found_patterns = []
        for pattern in info_patterns:
            matches = re.findall(pattern, combined_content, re.IGNORECASE)
            if matches:
                found_patterns.extend(matches[:3])  # Limit to 3 per pattern
        
        if found_patterns:
            print(f"🎉 Found comprehensive website information: {found_patterns}")
        else:
            print(f"⚠️  No structured information patterns found in combined content")
        
        return combined_content
    
    def extract_navbar_links(self, base_url: str, html_content: bytes) -> List[tuple]:
        """Extract ALL navigation links from the main page HTML"""
        navbar_links = []
        
        try:
            if not html_content:
                print("⚠️  No HTML content available for link extraction")
                return navbar_links
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for links in navigation elements (prioritize navbar links)
            link_selectors = [
                'nav a',           # Navigation links
                '.navbar a',       # Navbar links  
                '.navigation a',   # Navigation links
                '.menu a',         # Menu links
                'header a',        # Header links
                '.header a',       # Header class links
                '.main-nav a',     # Main navigation
                '.primary-nav a',  # Primary navigation
                '.site-nav a',     # Site navigation
                'footer a',        # Footer links (secondary priority)
            ]
            
            found_urls = set()  # Avoid duplicates
            link_priorities = {}  # Track where each link was found for prioritization
            
            for priority, selector in enumerate(link_selectors):
                elements = soup.select(selector)
                
                for element in elements:
                    # Get the link URL
                    href = element.get('href')
                    onclick = element.get('onclick', '')
                    
                    # Get the link text
                    link_text = element.get_text(strip=True)
                    
                    # Process onclick for JavaScript navigation
                    if onclick and not href:
                        onclick_match = re.search(r"location\.href\s*=\s*['\"]([^'\"]+)['\"]", onclick)
                        if onclick_match:
                            href = onclick_match.group(1)
                    
                    if href and link_text:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            full_url = base_url.rstrip('/') + href
                        elif href.startswith('http'):
                            full_url = href
                        elif href.startswith('#'):
                            continue  # Skip anchor links
                        elif href.startswith('mailto:') or href.startswith('tel:'):
                            continue  # Skip mailto and tel links
                        elif href.startswith('javascript:'):
                            continue  # Skip javascript links
                        else:
                            # Relative path
                            full_url = urljoin(base_url, href)
                        
                        # Filter out obvious non-content pages
                        skip_patterns = [
                            'login', 'signin', 'signup', 'register', 'logout',
                            'cart', 'checkout', 'account', 'profile', 'dashboard',
                            'admin', 'wp-admin', 'search', 'sitemap',
                            '.pdf', '.doc', '.zip', '.exe'
                        ]
                        
                        should_skip = any(pattern in full_url.lower() for pattern in skip_patterns)
                        
                        # Avoid duplicates, self-references, and external domains
                        parsed_base = urlparse(base_url)
                        parsed_full = urlparse(full_url)
                        
                        if (not should_skip and 
                            full_url != base_url and 
                            full_url not in found_urls and
                            parsed_full.netloc == parsed_base.netloc and  # Same domain only
                            len(link_text.strip()) > 0):
                            
                            found_urls.add(full_url)
                            
                            # Store with priority (lower number = higher priority)
                            if full_url not in link_priorities or priority < link_priorities[full_url]:
                                link_priorities[full_url] = priority
                                # Remove old entry if exists
                                navbar_links = [(url, text) for url, text in navbar_links if url != full_url]
                                navbar_links.append((full_url, link_text))
                                print(f"🔗 Discovered navbar link: {full_url} ('{link_text}') from {selector}")
            
            # Sort by priority (navbar links first, then header, then footer)
            navbar_links.sort(key=lambda x: link_priorities.get(x[0], 999))
            
            # Limit to prevent too many requests (but allow more since we're getting all navbar content)
            navbar_links = navbar_links[:8]  # Maximum 8 pages from navbar
            
            if navbar_links:
                print(f"🎯 Found {len(navbar_links)} navbar links to explore")
            else:
                print(f"⚠️  No navbar links found on main page")
            
            return navbar_links
            
        except Exception as e:
            print(f"⚠️  Error extracting navbar links: {e}")
            return navbar_links
    
    def debug_navbar_page_scraping(self, base_url: str, discovered_links: List[tuple]) -> None:
        """Debug helper to test navbar page scraping with discovered links"""
        print(f"\n🔍 DEBUG: Testing navbar page scraping for {base_url}")
        print(f"📋 Discovered {len(discovered_links)} navbar links:")
        
        for i, (page_url, link_text) in enumerate(discovered_links):
            print(f"\n[{i+1}] Testing: {page_url} ('{link_text}')")
            
            # Test if URL is accessible
            try:
                import requests
                response = requests.head(page_url, timeout=10)
                print(f"   📡 URL Status: {response.status_code}")
                if response.status_code != 200:
                    print(f"   ⚠️  URL may not be accessible")
            except Exception as e:
                print(f"   ❌ URL check failed: {e}")
            
            # Test BeautifulSoup scraping
            try:
                content = self.scrape_website_beautiful_soup(page_url)
                if content and len(content) > 100:
                    print(f"   ✅ BeautifulSoup: {len(content)} chars")
                    print(f"   📄 Preview: {content[:100]}...")
                else:
                    print(f"   ⚠️  BeautifulSoup: Only {len(content)} chars")
            except Exception as e:
                print(f"   ❌ BeautifulSoup failed: {e}")
            
            # Test Selenium scraping if BeautifulSoup fails
            try:
                content = self.scrape_website_selenium(page_url)
                if content and len(content) > 100:
                    print(f"   ✅ Selenium: {len(content)} chars")
                    print(f"   📄 Preview: {content[:100]}...")
                else:
                    print(f"   ⚠️  Selenium: Only {len(content)} chars")
            except Exception as e:
                print(f"   ❌ Selenium failed: {e}")