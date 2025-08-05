"""
Simplified RAG engine for text search and response generation.
This implementation uses basic text matching without vector embeddings.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import re

logger = logging.getLogger(__name__)

class SimpleRAGEngine:
    def __init__(self, bot_id: str):
        self.bot_id = bot_id
        self.documents = []
        self.data_dir = Path(f"data/bot_{bot_id}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.load_documents()
    
    def load_documents(self):
        """Load documents from the data directory."""
        try:
            doc_file = self.data_dir / "documents.json"
            if doc_file.exists():
                with open(doc_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents for bot {self.bot_id}")
            else:
                self.documents = []
                logger.info(f"No documents file found for bot {self.bot_id}")
        except Exception as e:
            logger.error(f"Error loading documents for bot {self.bot_id}: {e}")
            self.documents = []
    
    def save_documents(self):
        """Save documents to the data directory."""
        try:
            doc_file = self.data_dir / "documents.json"
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.documents)} documents for bot {self.bot_id}")
        except Exception as e:
            logger.error(f"Error saving documents for bot {self.bot_id}: {e}")
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """Add a document to the collection."""
        try:
            doc = {
                "content": content,
                "metadata": metadata or {},
                "id": len(self.documents)
            }
            self.documents.append(doc)
            self.save_documents()
            logger.info(f"Added document to bot {self.bot_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding document to bot {self.bot_id}: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents using simple text matching."""
        try:
            if not self.documents:
                logger.warning(f"No documents available for bot {self.bot_id}")
                return []
            
            # Simple keyword-based scoring
            query_words = set(re.findall(r'\w+', query.lower()))
            scored_docs = []
            
            for doc in self.documents:
                content_words = set(re.findall(r'\w+', doc["content"].lower()))
                
                # Calculate simple intersection score
                intersection = query_words.intersection(content_words)
                score = len(intersection) / len(query_words) if query_words else 0
                
                # Boost score if query appears as a phrase
                if query.lower() in doc["content"].lower():
                    score += 0.5
                
                if score > 0:
                    scored_docs.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": score
                    })
            
            # Sort by score and return top_k
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            results = scored_docs[:top_k]
            
            logger.info(f"Found {len(results)} relevant documents for query in bot {self.bot_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents for bot {self.bot_id}: {e}")
            return []
    
    def generate_response(self, query: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate a response based on the query and context."""
        try:
            if not context:
                context = self.search(query)
            
            if not context:
                return "I don't have any relevant information to answer your question. Please try rephrasing or ask about something else."
            
            # Create a simple response based on the most relevant document
            best_match = context[0]
            content = best_match["content"]
            
            # Extract relevant sentences that might contain the answer
            sentences = re.split(r'[.!?]+', content)
            relevant_sentences = []
            
            query_words = set(re.findall(r'\w+', query.lower()))
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_words = set(re.findall(r'\w+', sentence.lower()))
                if query_words.intersection(sentence_words):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                # Return the most relevant sentences
                response = ". ".join(relevant_sentences[:2])
                if not response.endswith('.'):
                    response += '.'
                return response
            else:
                # Fallback to a portion of the content
                words = content.split()
                if len(words) > 50:
                    return " ".join(words[:50]) + "..."
                return content
                
        except Exception as e:
            logger.error(f"Error generating response for bot {self.bot_id}: {e}")
            return "I encountered an error while processing your question. Please try again."

def get_bot_rag_engine(bot_id: str) -> SimpleRAGEngine:
    """Get or create a RAG engine for a specific bot."""
    return SimpleRAGEngine(bot_id)
