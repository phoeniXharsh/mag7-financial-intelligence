"""
Conversational Agent for multi-step reasoning and context management
Handles complex financial queries with conversation history
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import json
from datetime import datetime
from collections import defaultdict


# Import robust RAG pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from implement_robust_rag_pipeline import RobustRAGPipeline

class ConversationalAgent:
    """Agent for handling conversational financial Q&A with multi-step reasoning"""

    def __init__(self, rag_pipeline: Any):
        self.rag_pipeline = rag_pipeline
        self.conversation_sessions = {}  # session_id -> conversation history
        self.logger = logging.getLogger(__name__)

        # Context tracking
        self.current_context = {}  # Track current discussion context
        self.entity_tracker = {}   # Track mentioned companies, metrics, etc.

    def process_message(self, message: str, session_id: str = "default") -> Dict:
        """
        Process a user message with conversation context and multi-step reasoning

        Args:
            message: User message/question
            session_id: Session identifier for conversation tracking

        Returns:
            Structured response with context awareness
        """
        try:
            self.logger.info(f"Processing message: {message} (Session: {session_id})")

            # Initialize session if needed
            if session_id not in self.conversation_sessions:
                self.conversation_sessions[session_id] = []

            # Add user message to history
            user_message = {
                "timestamp": datetime.now().isoformat(),
                "type": "user",
                "content": message,
                "session_id": session_id
            }
            self.conversation_sessions[session_id].append(user_message)

            # Analyze message for complexity and context
            query_analysis = self._analyze_query(message, session_id)

            # Process based on query complexity
            if query_analysis["is_complex"]:
                response = self._handle_complex_query(message, query_analysis, session_id)
            else:
                response = self._handle_simple_query(message, session_id)

            # Enhance response with conversation context
            response = self._enhance_response_with_context(response, session_id)

            # Add assistant response to history
            assistant_message = {
                "timestamp": datetime.now().isoformat(),
                "type": "assistant",
                "content": response,
                "session_id": session_id,
                "query_analysis": query_analysis
            }
            self.conversation_sessions[session_id].append(assistant_message)

            # Update context tracking
            self._update_context_tracking(message, response, session_id)

            return response

        except Exception as e:
            self.logger.error(f"Message processing failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your message. Please try rephrasing your question.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e),
                "query_type": "error"
            }

    def _analyze_query(self, message: str, session_id: str) -> Dict:
        """Analyze query complexity and type"""
        analysis = {
            "is_complex": False,
            "query_type": "basic",
            "entities": [],
            "requires_decomposition": False,
            "follow_up": False,
            "comparison_requested": False,
            "time_series_requested": False
        }

        message_lower = message.lower()

        # Check for comparison queries
        comparison_indicators = ['compare', 'vs', 'versus', 'between', 'difference', 'better', 'worse']
        if any(indicator in message_lower for indicator in comparison_indicators):
            analysis["is_complex"] = True
            analysis["query_type"] = "comparative"
            analysis["comparison_requested"] = True
            analysis["requires_decomposition"] = True

        # Check for trend/time series queries
        trend_indicators = ['trend', 'growth', 'change', 'over time', 'yoy', 'year over year', 'quarterly', 'annual']
        if any(indicator in message_lower for indicator in trend_indicators):
            analysis["is_complex"] = True
            analysis["query_type"] = "trend_analysis"
            analysis["time_series_requested"] = True

        # Check for follow-up queries
        follow_up_indicators = ['what about', 'how about', 'and for', 'also', 'additionally', 'furthermore']
        if any(indicator in message_lower for indicator in follow_up_indicators):
            analysis["follow_up"] = True

        # Extract company entities
        companies = ['apple', 'aapl', 'microsoft', 'msft', 'amazon', 'amzn', 'google', 'googl', 'alphabet', 'meta', 'facebook', 'nvidia', 'nvda', 'tesla', 'tsla']
        mentioned_companies = [company for company in companies if company in message_lower]
        analysis["entities"] = mentioned_companies

        # Check if multiple companies mentioned
        if len(mentioned_companies) > 1:
            analysis["is_complex"] = True
            analysis["requires_decomposition"] = True

        return analysis

    def _handle_complex_query(self, message: str, analysis: Dict, session_id: str) -> Dict:
        """Handle complex queries requiring multi-step reasoning"""

        if analysis["comparison_requested"]:
            return self._handle_comparison_query(message, analysis, session_id)
        elif analysis["time_series_requested"]:
            return self._handle_trend_query(message, analysis, session_id)
        elif analysis["requires_decomposition"]:
            return self._handle_decomposed_query(message, analysis, session_id)
        else:
            # Fallback to simple query
            return self._handle_simple_query(message, session_id)

    def _handle_comparison_query(self, message: str, analysis: Dict, session_id: str) -> Dict:
        """Handle comparison queries between companies"""

        self.logger.info(f"Processing comparison query: {message}")

        # Extract companies to compare
        companies = analysis.get("entities", [])

        if len(companies) < 2:
            # Try to infer from context or ask for clarification
            context_companies = self._get_context_companies(session_id)
            if len(context_companies) >= 2:
                companies = context_companies[:2]
            else:
                return {
                    "answer": "I'd be happy to help you compare companies. Could you please specify which companies you'd like me to compare? For example: 'Compare Apple and Microsoft revenue'",
                    "sources": [],
                    "confidence": 0.0,
                    "query_type": "clarification_needed",
                    "suggested_companies": ["Apple", "Microsoft", "Amazon", "Google", "Meta", "NVIDIA", "Tesla"]
                }

        # Decompose into individual queries
        base_query = self._extract_base_query(message)
        sub_queries = []

        for company in companies[:2]:  # Limit to 2 companies for now
            company_query = f"{base_query} for {company}"
            sub_queries.append(company_query)

        # Execute sub-queries
        sub_results = []
        for query in sub_queries:
            result = self.rag_pipeline.query(query, self.conversation_sessions[session_id])
            sub_results.append(result)

        # Synthesize comparison response
        comparison_response = self._synthesize_comparison(sub_results, companies, base_query)

        return comparison_response

    def _handle_trend_query(self, message: str, analysis: Dict, session_id: str) -> Dict:
        """Handle trend analysis queries"""

        self.logger.info(f"Processing trend query: {message}")

        # For trend queries, we need to search for time-series data
        enhanced_query = f"{message} quarterly annual growth change trend"

        result = self.rag_pipeline.query(enhanced_query, self.conversation_sessions[session_id])

        # Enhance with trend-specific context
        result["query_type"] = "trend_analysis"
        result["analysis_type"] = "time_series"

        return result

    def _handle_decomposed_query(self, message: str, analysis: Dict, session_id: str) -> Dict:
        """Handle queries that need to be broken down into sub-questions"""

        self.logger.info(f"Processing decomposed query: {message}")

        # Use RAG pipeline's decomposition
        sub_queries = self.rag_pipeline.decompose_query(message)

        if len(sub_queries) <= 1:
            # No decomposition needed, handle as simple query
            return self._handle_simple_query(message, session_id)

        # Process each sub-query
        sub_results = []
        for sub_query in sub_queries:
            result = self.rag_pipeline.query(sub_query, self.conversation_sessions[session_id])
            sub_results.append(result)

        # Combine results
        combined_response = self._combine_sub_results(sub_results, message)

        return combined_response

    def _handle_simple_query(self, message: str, session_id: str) -> Dict:
        """Handle simple, direct queries"""

        # Check for follow-up context
        conversation_history = self.conversation_sessions.get(session_id, [])

        # Process with RAG pipeline
        result = self.rag_pipeline.query(message, conversation_history)

        return result

    def _extract_base_query(self, message: str) -> str:
        """Extract the base query from a comparison message"""
        # Instead of trying to remove company names, extract the core topic
        message_lower = message.lower()

        # Look for key financial topics
        if 'revenue' in message_lower:
            base_topic = 'revenue'
        elif 'profit' in message_lower or 'earnings' in message_lower:
            base_topic = 'earnings'
        elif 'growth' in message_lower:
            base_topic = 'growth'
        elif 'risk' in message_lower:
            base_topic = 'risk factors'
        elif 'performance' in message_lower:
            base_topic = 'financial performance'
        elif 'segment' in message_lower or 'business' in message_lower:
            base_topic = 'business segments'
        else:
            base_topic = 'financial performance'

        # Extract year if present
        import re
        year_match = re.search(r'\b(20\d{2})\b', message)
        if year_match:
            year = year_match.group(1)
            return f"{base_topic} in {year}"
        else:
            return base_topic

    def _synthesize_comparison(self, sub_results: List[Dict], companies: List[str], base_query: str) -> Dict:
        """Synthesize comparison response from individual company results"""

        if len(sub_results) < 2:
            return {
                "answer": "I couldn't gather enough information to make a meaningful comparison.",
                "sources": [],
                "confidence": 0.0,
                "query_type": "comparison_failed"
            }

        # Extract key information from each result
        company_data = {}
        all_sources = []

        for i, result in enumerate(sub_results):
            company = companies[i] if i < len(companies) else f"Company {i+1}"
            company_data[company] = {
                "answer": result.get("answer", "No information available"),
                "confidence": result.get("confidence", 0.0),
                "sources": result.get("sources", [])
            }
            all_sources.extend(result.get("sources", []))

        # Create comparison summary
        comparison_text = f"Comparison of {base_query}:\n\n"

        for company, data in company_data.items():
            # Don't truncate the answer - show full response
            answer = data['answer']
            comparison_text += f"**{company.upper()}**: {answer}\n\n"

        # Calculate overall confidence
        confidences = [data["confidence"] for data in company_data.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "answer": comparison_text,
            "sources": all_sources[:10],  # Limit sources
            "confidence": round(avg_confidence, 3),
            "query_type": "comparative",
            "comparison_data": company_data,
            "companies_compared": list(companies)
        }

    def _combine_sub_results(self, sub_results: List[Dict], original_query: str) -> Dict:
        """Combine results from decomposed sub-queries"""

        if not sub_results:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "confidence": 0.0,
                "query_type": "no_results"
            }

        # Combine answers
        combined_answer = f"Based on your question '{original_query}', here's what I found:\n\n"

        all_sources = []
        confidences = []

        for i, result in enumerate(sub_results, 1):
            answer = result.get("answer", "No information available")
            combined_answer += f"{i}. {answer}\n\n"

            all_sources.extend(result.get("sources", []))
            confidences.append(result.get("confidence", 0.0))

        # Calculate overall confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "answer": combined_answer,
            "sources": all_sources[:10],  # Limit sources
            "confidence": round(avg_confidence, 3),
            "query_type": "multi_step",
            "sub_results_count": len(sub_results)
        }

    def _get_context_companies(self, session_id: str) -> List[str]:
        """Get companies mentioned in recent conversation context"""
        conversation = self.conversation_sessions.get(session_id, [])

        companies = []
        # Look at last 5 messages
        for message in conversation[-5:]:
            content = message.get("content", "")
            if isinstance(content, str):
                content_lower = content.lower()
                company_mentions = ['apple', 'aapl', 'microsoft', 'msft', 'amazon', 'amzn', 'google', 'googl', 'alphabet', 'meta', 'facebook', 'nvidia', 'nvda', 'tesla', 'tsla']
                for company in company_mentions:
                    if company in content_lower and company not in companies:
                        companies.append(company)

        return companies

    def _enhance_response_with_context(self, response: Dict, session_id: str) -> Dict:
        """Enhance response with conversation context"""

        conversation = self.conversation_sessions.get(session_id, [])

        # Add conversation context indicators
        if len(conversation) > 2:  # Has previous conversation
            response["conversation_context"] = {
                "has_context": True,
                "message_count": len(conversation),
                "recent_topics": self._extract_recent_topics(session_id)
            }
        else:
            response["conversation_context"] = {
                "has_context": False,
                "message_count": len(conversation)
            }

        # Add follow-up suggestions
        response["follow_up_suggestions"] = self._generate_follow_up_suggestions(response, session_id)

        return response

    def _update_context_tracking(self, message: str, response: Dict, session_id: str):
        """Update context tracking for the session"""

        # Track entities mentioned
        if session_id not in self.entity_tracker:
            self.entity_tracker[session_id] = defaultdict(int)

        # Count company mentions
        companies = ['apple', 'aapl', 'microsoft', 'msft', 'amazon', 'amzn', 'google', 'googl', 'alphabet', 'meta', 'facebook', 'nvidia', 'nvda', 'tesla', 'tsla']
        for company in companies:
            if company in message.lower():
                self.entity_tracker[session_id][company] += 1

        # Track query types
        query_type = response.get("query_type", "basic")
        self.entity_tracker[session_id][f"query_type_{query_type}"] += 1

    def _extract_recent_topics(self, session_id: str) -> List[str]:
        """Extract recent topics from conversation"""
        conversation = self.conversation_sessions.get(session_id, [])

        topics = []
        # Look at last 3 user messages
        user_messages = [msg for msg in conversation if msg.get("type") == "user"][-3:]

        for message in user_messages:
            content = message.get("content", "").lower()

            # Extract financial topics
            if "revenue" in content:
                topics.append("revenue")
            if "profit" in content or "earnings" in content:
                topics.append("earnings")
            if "risk" in content:
                topics.append("risk_factors")
            if "growth" in content:
                topics.append("growth")
            if "compare" in content or "vs" in content:
                topics.append("comparison")

        return list(set(topics))  # Remove duplicates

    def _generate_follow_up_suggestions(self, response: Dict, session_id: str) -> List[str]:
        """Generate follow-up question suggestions"""

        suggestions = []
        query_type = response.get("query_type", "basic")

        # Based on query type
        if query_type == "comparative":
            suggestions.extend([
                "How do their profit margins compare?",
                "What about their growth rates?",
                "Which company has better risk management?"
            ])
        elif query_type == "financial_metrics":
            suggestions.extend([
                "What are the main risk factors?",
                "How has this changed over time?",
                "How does this compare to competitors?"
            ])
        elif query_type == "basic":
            suggestions.extend([
                "Tell me more about their financial performance",
                "What are their main business segments?",
                "How do they compare to competitors?"
            ])

        # Based on companies mentioned
        companies_in_context = self._get_context_companies(session_id)
        if companies_in_context:
            company = companies_in_context[0].upper()
            suggestions.append(f"What are {company}'s main risk factors?")
            suggestions.append(f"How is {company} performing this quarter?")

        return suggestions[:3]  # Limit to 3 suggestions

    def clear_history(self, session_id: Optional[str] = None):
        """Clear conversation history"""
        if session_id:
            if session_id in self.conversation_sessions:
                del self.conversation_sessions[session_id]
            if session_id in self.entity_tracker:
                del self.entity_tracker[session_id]
        else:
            self.conversation_sessions.clear()
            self.entity_tracker.clear()

        self.logger.info(f"Cleared conversation history for session: {session_id or 'all'}")

    def get_conversation_summary(self, session_id: str) -> Dict:
        """Get summary of conversation"""

        conversation = self.conversation_sessions.get(session_id, [])
        entities = self.entity_tracker.get(session_id, {})

        # Count message types
        user_messages = [msg for msg in conversation if msg.get("type") == "user"]
        assistant_messages = [msg for msg in conversation if msg.get("type") == "assistant"]

        # Extract topics discussed
        topics = self._extract_recent_topics(session_id)

        # Get most mentioned companies
        company_mentions = {k: v for k, v in entities.items() if not k.startswith("query_type_")}
        top_companies = sorted(company_mentions.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "session_id": session_id,
            "total_messages": len(conversation),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "topics_discussed": topics,
            "top_companies_mentioned": [company for company, count in top_companies],
            "conversation_start": conversation[0].get("timestamp") if conversation else None,
            "last_activity": conversation[-1].get("timestamp") if conversation else None
        }

    def get_session_stats(self) -> Dict:
        """Get statistics across all sessions"""

        total_sessions = len(self.conversation_sessions)
        total_messages = sum(len(conv) for conv in self.conversation_sessions.values())

        # Active sessions (with messages in last hour)
        active_sessions = 0
        current_time = datetime.now()

        for session_id, conversation in self.conversation_sessions.items():
            if conversation:
                last_message_time = datetime.fromisoformat(conversation[-1].get("timestamp", ""))
                if (current_time - last_message_time).seconds < 3600:  # 1 hour
                    active_sessions += 1

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "average_messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0
        }

    def export_conversation(self, session_id: str) -> Dict:
        """Export conversation for analysis or backup"""

        conversation = self.conversation_sessions.get(session_id, [])
        summary = self.get_conversation_summary(session_id)

        return {
            "session_id": session_id,
            "conversation": conversation,
            "summary": summary,
            "export_timestamp": datetime.now().isoformat()
        }
