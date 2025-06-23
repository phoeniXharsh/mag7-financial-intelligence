"""
Robust Conversational Agent using the new RobustRAGPipeline
This replaces the old agent with proper mixed prompt handling
"""

import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from implement_robust_rag_pipeline import RobustRAGPipeline

class RobustConversationalAgent:
    """Enhanced conversational agent with robust mixed prompt handling"""
    
    def __init__(self, rag_pipeline: RobustRAGPipeline):
        """Initialize with robust RAG pipeline"""
        self.rag_pipeline = rag_pipeline
        self.conversation_history = {}
    
    def process_message(self, message: str, session_id: str) -> Dict[str, Any]:
        """Process user message with conversation context and return response"""

        try:
            # Initialize conversation history if needed
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []

            # Enhance message with conversation context
            enhanced_message = self._enhance_message_with_context(message, session_id)

            # Use the robust pipeline to process the enhanced query
            result = self.rag_pipeline.process_query(enhanced_message)

            # Store in conversation history
            self.conversation_history[session_id].append({
                "user_message": message,
                "enhanced_message": enhanced_message,
                "response": result,
                "query_type": result.get("query_type", "unknown")
            })

            # Add conversation context indicators to response
            result["_internal"] = result.get("_internal", {})
            result["_internal"]["conversation_context"] = {
                "has_context": len(self.conversation_history[session_id]) > 1,
                "message_count": len(self.conversation_history[session_id]),
                "context_used": enhanced_message != message
            }

            # Format response for UI
            formatted_response = {
                "answer": result.get("answer", "I apologize, but I couldn't generate a response."),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "query_type": result.get("query_type", "unknown"),
                "session_id": session_id,
                "_internal": result.get("_internal", {})
            }

            return formatted_response
            
        except Exception as e:
            # Fallback response
            return {
                "answer": f"I apologize, but I encountered an error processing your request: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "query_type": "error",
                "session_id": session_id
            }

    def _enhance_message_with_context(self, message: str, session_id: str) -> str:
        """Enhance message with conversation context for better understanding"""

        conversation = self.conversation_history.get(session_id, [])

        # If no previous conversation, return original message
        if not conversation:
            return message

        # Get recent context (last 2 exchanges)
        recent_context = conversation[-2:] if len(conversation) >= 2 else conversation

        # Check for pronouns and references that need context
        message_lower = message.lower()
        needs_context = any(ref in message_lower for ref in [
            'it', 'that', 'this', 'they', 'them', 'their', 'its',
            'the company', 'the previous', 'compared to', 'versus',
            'same period', 'last year', 'this year', 'follow up'
        ])

        if not needs_context:
            return message

        # Build context from recent conversation
        context_parts = []

        for exchange in recent_context:
            user_msg = exchange.get("user_message", "")
            response = exchange.get("response", {})

            # Extract companies mentioned
            companies = self._extract_companies_from_text(user_msg)
            if companies:
                context_parts.append(f"Previously discussed companies: {', '.join(companies)}")

            # Extract time periods mentioned
            years = self._extract_years_from_text(user_msg)
            if years:
                context_parts.append(f"Previously discussed years: {', '.join(years)}")

            # Extract query type for context
            query_type = response.get("query_type", "")
            if query_type and query_type != "unknown":
                context_parts.append(f"Previous query was about: {query_type}")

        # Enhance message with context
        if context_parts:
            context_str = " | ".join(context_parts)
            enhanced_message = f"[Context: {context_str}] {message}"
            return enhanced_message

        return message

    def _extract_companies_from_text(self, text: str) -> list:
        """Extract company names/tickers from text"""
        companies = []
        text_lower = text.lower()

        company_map = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'amazon': 'AMZN',
            'google': 'GOOGL', 'alphabet': 'GOOGL', 'meta': 'META',
            'facebook': 'META', 'nvidia': 'NVDA', 'tesla': 'TSLA',
            'aapl': 'AAPL', 'msft': 'MSFT', 'amzn': 'AMZN',
            'googl': 'GOOGL', 'nvda': 'NVDA', 'tsla': 'TSLA'
        }

        for name, ticker in company_map.items():
            if name in text_lower and ticker not in companies:
                companies.append(ticker)

        return companies

    def _extract_years_from_text(self, text: str) -> list:
        """Extract years from text"""
        import re
        years = re.findall(r'\b(20[0-9]{2})\b', text)
        return list(set(years))
    
    def get_conversation_history(self, session_id: str) -> list:
        """Get conversation history for a session"""
        return self.conversation_history.get(session_id, [])
    
    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]

    def get_session_stats(self) -> dict:
        """Get statistics about the current session"""
        total_sessions = len(self.conversation_history)
        total_messages = sum(len(history) for history in self.conversation_history.values())

        # Calculate query type distribution
        query_types = {}
        for history in self.conversation_history.values():
            for entry in history:
                query_type = entry.get("query_type", "unknown")
                query_types[query_type] = query_types.get(query_type, 0) + 1

        return {
            "active_sessions": total_sessions,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "query_type_distribution": query_types,
            "pipeline_type": "RobustRAGPipeline",
            "status": "active"
        }

    def get_pipeline_stats(self) -> dict:
        """Get statistics about the RAG pipeline"""
        try:
            # Get stats from the underlying robust pipeline
            if hasattr(self.rag_pipeline, 'get_pipeline_stats'):
                return self.rag_pipeline.get_pipeline_stats()
            else:
                # Fallback stats
                return {
                    "pipeline_type": "RobustRAGPipeline",
                    "vector_store_initialized": hasattr(self.rag_pipeline, 'vector_store') and self.rag_pipeline.vector_store is not None,
                    "llm_initialized": hasattr(self.rag_pipeline, 'llm') and self.rag_pipeline.llm is not None,
                    "query_classifier": "QueryClassifier",
                    "company_extractor": "CompanyExtractor",
                    "time_extractor": "TimeExtractor",
                    "supported_query_types": ["basic", "comparative", "complex", "trend_analysis", "cross_company"]
                }
        except Exception as e:
            return {
                "pipeline_type": "RobustRAGPipeline",
                "status": "error",
                "error": str(e)
            }

    def get_conversation_summary(self, session_id: str) -> dict:
        """Get summary of conversation for a session"""
        conversation = self.conversation_history.get(session_id, [])

        if not conversation:
            return {
                "total_messages": 0,
                "query_types": {},
                "companies_mentioned": [],
                "recent_topics": [],
                "session_id": session_id
            }

        # Analyze conversation
        query_types = {}
        companies_mentioned = set()
        recent_topics = []

        for entry in conversation:
            # Count query types
            query_type = entry.get("query_type", "unknown")
            query_types[query_type] = query_types.get(query_type, 0) + 1

            # Extract companies from sources
            sources = entry.get("response", {}).get("sources", [])
            for source in sources:
                if "company" in source:
                    companies_mentioned.add(source["company"])

            # Add recent topics (last 3 messages)
            if len(recent_topics) < 3:
                user_message = entry.get("user_message", "")
                if user_message:
                    recent_topics.append(user_message[:100])

        return {
            "total_messages": len(conversation),
            "query_types": query_types,
            "companies_mentioned": list(companies_mentioned),
            "recent_topics": recent_topics,
            "session_id": session_id
        }

    def clear_history(self, session_id: str = None):
        """Clear conversation history for specific session or all sessions"""
        if session_id:
            if session_id in self.conversation_history:
                del self.conversation_history[session_id]
        else:
            # Clear all sessions
            self.conversation_history.clear()

def create_robust_agent():
    """Factory function to create a robust conversational agent"""

    # Initialize vector store
    from vector_store.faiss_store import FAISSVectorStore
    from config.settings import get_secure_api_key

    vector_store = FAISSVectorStore(
        db_path="data/mag7_complete_final_fixed",
        collection_name="mag7_all_data_properly_indexed"
    )

    if not vector_store.initialize():
        raise Exception("Failed to initialize vector store")

    # Get API key securely from environment
    try:
        api_key = get_secure_api_key()
    except ValueError as e:
        raise Exception(f"API key configuration error: {str(e)}")

    llm_config = {
        "model_name": "gemini-pro",
        "temperature": 0.1,
        "max_tokens": 3000,
        "api_key": api_key
    }

    retrieval_config = {
        "top_k": 15,
        "similarity_threshold": 0.3,
        "max_context_length": 4000
    }

    robust_pipeline = RobustRAGPipeline(vector_store, llm_config, retrieval_config)

    # Create and return robust agent
    return RobustConversationalAgent(robust_pipeline)
