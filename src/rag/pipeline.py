"""
RAG Pipeline implementation using LangChain
Connects retrieval with Gemini Flash 2.5 for question answering
"""

from typing import Dict, List, Optional, Any
import logging
import json
import time
import re
import requests
from datetime import datetime

try:
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    import google.generativeai as genai

    # Try to import LangChain Google GenAI, fallback to custom implementation
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        ChatGoogleGenerativeAI = None

except ImportError as e:
    logging.warning(f"LangChain dependencies not installed: {e}")
    PromptTemplate = None
    Document = None
    genai = None
    ChatGoogleGenerativeAI = None

class CustomGeminiLLM:
    """Custom Gemini LLM wrapper for LangChain compatibility"""

    def __init__(self, model_name="gemini-pro", temperature=0.1, max_tokens=2048, api_key=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = None
        self.api_method = "mock"  # Fallback to mock if API fails
        self.api_key = api_key

        # Try direct HTTP API first (bypasses library version issues)
        if api_key:
            if self._test_direct_api(api_key):
                self.api_method = "direct_http"
                self.model = "direct_http_client"
                logging.info("Gemini API configured successfully - using DIRECT HTTP API")
                return

        if api_key and genai:
            try:
                genai.configure(api_key=api_key)

                # Test API availability by checking models
                models = list(genai.list_models())
                text_models = [m for m in models if 'generateText' in m.supported_generation_methods or 'generate' in str(m.supported_generation_methods)]

                if text_models:
                    # Found text generation models
                    self.model = genai
                    self.api_method = "generate_text"
                    logging.info(f"Gemini API configured successfully - using REAL LLM with {len(text_models)} models")
                elif hasattr(genai, 'GenerativeModel'):
                    # Try newer API
                    self.model = genai.GenerativeModel(model_name)
                    self.api_method = "real_gemini"
                    logging.info(f"Gemini API configured successfully - using REAL LLM (GenerativeModel)")
                else:
                    # API key works but no text generation available
                    self.model = None
                    self.api_method = "intelligent_mock"
                    logging.warning("Gemini API configured but no text generation models available - using intelligent responses")

            except Exception as e:
                logging.error(f"Gemini API initialization failed: {e}")
                self.model = None
                self.api_method = "intelligent_mock"
                logging.warning("Falling back to intelligent mock responses")

    def _test_direct_api(self, api_key):
        """Test direct HTTP API to bypass library version issues"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{"text": "Test"}]
                }]
            }

            response = requests.post(url, headers=headers, json=data, timeout=10)

            if response.status_code == 200:
                logging.info("Direct HTTP API test successful")
                return True
            else:
                logging.warning(f"Direct HTTP API test failed: {response.status_code}")
                return False

        except Exception as e:
            logging.warning(f"Direct HTTP API test error: {e}")
            return False

    def _generate_direct_http(self, prompt):
        """Generate response using direct HTTP API"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_key}"

            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens
                }
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()

                if 'candidates' in result and len(result['candidates']) > 0:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    logging.info("Successfully generated response using direct HTTP API")
                    return text.strip()
                else:
                    logging.warning("Empty response from direct HTTP API")
                    return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            else:
                logging.error(f"Direct HTTP API failed: {response.status_code} - {response.text}")
                return "I'm having trouble connecting to the AI service. Please check your API configuration and try again."

        except Exception as e:
            logging.error(f"Direct HTTP API error: {e}")
            return "I encountered an error while generating a response. Please check your API configuration and try again."

    def __call__(self, prompt):
        """Make the class callable for LangChain compatibility"""
        return self.generate(prompt)

    def invoke(self, prompt):
        """Modern LangChain compatibility method"""
        return self.generate(prompt)

    def generate(self, prompt):
        """Generate response from Gemini (with intelligent fallback)"""
        if not self.model:
            return "Gemini model not initialized - please check your API key"

        try:
            if self.api_method == "direct_http":
                # Use direct HTTP API (bypasses library version issues)
                logging.info("Using REAL Gemini LLM via direct HTTP API")
                return self._generate_direct_http(prompt)

            elif self.api_method == "real_gemini":
                # Use REAL Gemini API
                logging.info("Using REAL Gemini LLM for response generation")

                response = self.model.generate_content(prompt)

                if response and response.text:
                    logging.info("Successfully generated response using Gemini LLM")
                    return response.text.strip()
                else:
                    logging.warning("Empty response from Gemini API")
                    return "I apologize, but I couldn't generate a response. Please try rephrasing your question."

            elif self.api_method == "mock":
                # Fallback to mock only if API failed
                logging.warning("Using mock responses - Gemini API not available")
                return self._generate_mock_response(prompt)

            elif self.api_method == "generate_text":
                # Use older generate_text API
                logging.info("Using REAL Gemini LLM (generate_text API)")

                response = self.model.generate_text(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )

                if response and hasattr(response, 'result') and response.result:
                    logging.info("Successfully generated response using Gemini generate_text")
                    return response.result.strip()
                elif response:
                    logging.info("Successfully generated response using Gemini generate_text (direct)")
                    return str(response).strip()
                else:
                    logging.warning("Empty response from Gemini generate_text API")
                    return "I apologize, but I couldn't generate a response. Please try rephrasing your question."

            elif self.api_method == "GenerativeModel":
                # Newer API version (fallback)
                model = self.model.GenerativeModel(self.model_name)

                # Configure generation
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )

                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                return response.text

            else:
                return "Unknown Gemini API method"

        except Exception as e:
            logging.error(f"Gemini generation error: {e}")
            return "I encountered an error while generating a response. Please check your API configuration and try again."



class RAGPipeline:
    """LangChain-based RAG pipeline for financial Q&A"""

    def __init__(self,
                 vector_store: Any,
                 llm_config: Dict,
                 retrieval_config: Dict):
        self.vector_store = vector_store
        self.llm_config = llm_config
        self.retrieval_config = retrieval_config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.llm = None
        self.qa_chain = None
        self.citation_chain = None

        # Check dependencies
        if not genai:
            self.logger.error("Google GenerativeAI not available")
            return

        # Initialize
        self._initialize_llm()
        self._initialize_chains()

    def _initialize_llm(self):
        """Initialize Gemini LLM"""
        try:
            api_key = self.llm_config.get("api_key")
            if not api_key:
                self.logger.error("Gemini API key not provided")
                return

            # Use custom implementation (more reliable)
            self.llm = CustomGeminiLLM(
                model_name=self.llm_config.get("model_name", "gemini-1.5-flash"),
                temperature=self.llm_config.get("temperature", 0.1),
                max_tokens=self.llm_config.get("max_tokens", 2048),
                api_key=api_key
            )

            if self.llm.model:
                self.logger.info(f"Initialized Custom Gemini LLM: {self.llm_config.get('model_name')}")
            else:
                self.logger.error("Failed to initialize Gemini LLM - check API key")
                self.llm = None

        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini LLM: {e}")
            self.llm = None

    def _initialize_chains(self):
        """Initialize LangChain chains for Q&A and citation"""
        # Always initialize prompts, even if LLM is None
        # This allows the pipeline to initialize and show better error messages

        try:
            # Q&A Chain Prompt
            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a financial analyst AI assistant specializing in SEC filings analysis for the Magnificent 7 (MAG7) technology companies: Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), Alphabet/Google (GOOGL), Meta (META), NVIDIA (NVDA), and Tesla (TSLA).

Based on the provided context from SEC filings, answer the user's question accurately and comprehensively.

Context from SEC filings:
{context}

Question: {question}

Instructions:
1. Provide a clear, accurate answer based ONLY on the provided context
2. Include specific financial figures, dates, and metrics when available
3. If comparing companies, be explicit about the comparison
4. If the context doesn't contain enough information, say so clearly
5. Use professional financial language
6. Be concise but comprehensive

Answer:"""
            )

            # Citation Chain Prompt
            citation_prompt = PromptTemplate(
                input_variables=["answer", "sources"],
                template="""Given the answer and source documents, create a structured response with proper citations.

Answer: {answer}

Source Documents: {sources}

Create a JSON response with:
1. "answer": The main answer
2. "sources": Array of source citations with company, filing_type, period, snippet, and confidence
3. "confidence": Overall confidence score (0.0-1.0)
4. "query_type": Type of query (basic, comparative, trend_analysis, etc.)

Ensure all claims in the answer are supported by the sources."""
            )

            # Store prompts for manual processing (modern approach)
            self.qa_prompt = qa_prompt
            self.citation_prompt = citation_prompt
            self.qa_chain = "modern"
            self.citation_chain = "modern"

            self.logger.info("Initialized LangChain Q&A and citation chains")

        except Exception as e:
            self.logger.error(f"Failed to initialize chains: {e}")
            self.qa_chain = None
            self.citation_chain = None

    def initialize(self) -> bool:
        """Check if RAG pipeline is initialized"""
        # Check vector store first
        if not self.vector_store or not self.vector_store.initialize():
            return False

        # Initialize chains (this sets prompts even if LLM fails)
        self._initialize_chains()

        # Check if we have prompts (required) and LLM (preferred but not required for testing)
        return (hasattr(self, 'qa_prompt') and
                self.qa_prompt is not None and
                self.vector_store is not None)

    def query(self, question: str, conversation_history: Optional[List] = None) -> Dict:
        """
        Process a question and return structured response

        Args:
            question: User question
            conversation_history: Previous conversation context

        Returns:
            Structured response with answer, sources, and confidence
        """
        start_time = time.time()

        try:
            if not self.initialize():
                return {
                    "answer": "RAG pipeline not properly initialized. Please check API key and vector store.",
                    "sources": [],
                    "confidence": 0.0,
                    "_internal": {
                        "error": "initialization_failed",
                        "processing_time": time.time() - start_time,
                        "query_type": "error"
                    }
                }

            self.logger.info(f"Processing query: {question}")

            # Step 1: Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(question, conversation_history)

            if not retrieved_docs:
                return {
                    "answer": "I couldn't find relevant information in the SEC filings to answer your question. Please try rephrasing or asking about specific companies or financial metrics.",
                    "sources": [],
                    "confidence": 0.0,
                    "_internal": {
                        "query_type": "no_results",
                        "processing_time": time.time() - start_time,
                        "retrieved_docs_count": 0
                    }
                }

            # Step 2: Generate answer using retrieved context
            answer = self._generate_answer(question, retrieved_docs)

            # Step 3: Format sources and create structured response
            formatted_sources = self.format_sources(retrieved_docs)

            # Step 4: Determine query type and confidence
            query_type = self._classify_query(question)
            confidence = self._calculate_confidence(retrieved_docs, answer)

            # Format response to match target JSON structure exactly
            response = {
                "answer": answer,
                "sources": formatted_sources,
                "confidence": confidence
            }

            # Add internal metadata for UI (not part of target JSON)
            response["_internal"] = {
                "query_type": query_type,
                "processing_time": time.time() - start_time,
                "retrieved_docs_count": len(retrieved_docs)
            }

            self.logger.info(f"Query processed successfully in {response['_internal']['processing_time']:.2f}s")
            return response

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Query processing failed: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again or rephrase your question.",
                "sources": [],
                "confidence": 0.0,
                "_internal": {
                    "error": str(e),
                    "processing_time": processing_time,
                    "query_type": "error"
                }
            }

    def _retrieve_documents(self, question: str, conversation_history: Optional[List] = None) -> List[Dict]:
        """Retrieve relevant documents from vector store"""
        try:
            # Enhance query with conversation context if available
            enhanced_query = self._enhance_query_with_context(question, conversation_history)

            # Check if this is a yearly query that should prioritize annual reports
            is_yearly_query = self._is_yearly_query(question)

            # Retrieve documents
            top_k = self.retrieval_config.get("top_k", 5)
            similarity_threshold = self.retrieval_config.get("similarity_threshold", 0.3)

            # Get initial results
            results = self.vector_store.search(enhanced_query, top_k=top_k * 3)  # Get more results for filtering

            # Filter by similarity threshold
            filtered_results = [
                result for result in results
                if result.get('score', 0) >= similarity_threshold
            ]

            # If this is a yearly query, prioritize annual reports (10-K) over quarterly (10-Q)
            if is_yearly_query:
                final_results = self._prioritize_annual_reports(filtered_results, top_k)
            else:
                final_results = filtered_results[:top_k]

            self.logger.info(f"Retrieved {len(final_results)} documents (from {len(results)} initial results, yearly_query: {is_yearly_query})")
            return final_results

        except Exception as e:
            self.logger.error(f"Document retrieval failed: {e}")
            return []

    def _is_yearly_query(self, question: str) -> bool:
        """Check if the query is asking for yearly/annual data"""
        question_lower = question.lower()

        # Look for year indicators
        yearly_indicators = [
            'year', 'annual', 'fiscal year', 'fy', 'yearly',
            'for 2015', 'for 2016', 'for 2017', 'for 2018', 'for 2019',
            'for 2020', 'for 2021', 'for 2022', 'for 2023', 'for 2024', 'for 2025',
            'in 2015', 'in 2016', 'in 2017', 'in 2018', 'in 2019',
            'in 2020', 'in 2021', 'in 2022', 'in 2023', 'in 2024', 'in 2025'
        ]

        # Check for quarterly indicators (should NOT prioritize annual)
        quarterly_indicators = ['quarter', 'q1', 'q2', 'q3', 'q4', 'quarterly']

        has_yearly = any(indicator in question_lower for indicator in yearly_indicators)
        has_quarterly = any(indicator in question_lower for indicator in quarterly_indicators)

        # Prioritize annual reports if yearly indicators present and no quarterly indicators
        return has_yearly and not has_quarterly

    def _prioritize_annual_reports(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Prioritize annual reports (10-K) over quarterly reports (10-Q) for yearly queries"""
        if not results:
            return results

        # Separate annual and quarterly reports
        annual_reports = []
        quarterly_reports = []
        other_reports = []

        for result in results:
            metadata = result.get('metadata', {})
            filing_type = metadata.get('filing_type', '').upper()

            if filing_type == '10-K':
                annual_reports.append(result)
            elif filing_type == '10-Q':
                quarterly_reports.append(result)
            else:
                other_reports.append(result)

        # Prioritize annual reports, then quarterly, then others
        prioritized_results = annual_reports + quarterly_reports + other_reports

        # Return top_k results
        final_results = prioritized_results[:top_k]

        self.logger.info(f"Prioritized annual reports: {len(annual_reports)} annual, {len(quarterly_reports)} quarterly, returning {len(final_results)} total")

        return final_results

    def _enhance_query_with_context(self, question: str, conversation_history: Optional[List] = None) -> str:
        """Enhance query with conversation context"""
        if not conversation_history:
            return question

        # Simple context enhancement - can be improved
        recent_context = []
        for msg in conversation_history[-3:]:  # Last 3 messages
            if msg.get("type") == "user":
                recent_context.append(msg.get("content", ""))

        if recent_context:
            context_str = " ".join(recent_context)
            return f"{context_str} {question}"

        return question

    def _generate_answer(self, question: str, retrieved_docs: List[Dict]) -> str:
        """Generate answer using LLM and retrieved documents"""
        try:
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                metadata = doc.get('metadata', {})
                company = metadata.get('company', 'Unknown')
                filing_type = metadata.get('filing_type', 'Unknown')
                section = metadata.get('section', 'Unknown')
                text = doc.get('text', '')

                context_part = f"[Source {i}] Company: {company}, Filing: {filing_type}, Section: {section}\n{text}\n"
                context_parts.append(context_part)

            context = "\n".join(context_parts)

            # Limit context length to avoid token limits
            max_context_length = self.retrieval_config.get("max_context_length", 4000)
            if len(context) > max_context_length:
                context = context[:max_context_length] + "...[truncated]"

            # Generate answer using modern approach
            if hasattr(self.qa_prompt, 'format'):
                prompt = self.qa_prompt.format(context=context, question=question)
            else:
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

            # Check if LLM is available
            if not self.llm:
                response = "I apologize, but the AI model is not properly configured. Please check your API key and try again."
            elif hasattr(self.llm, 'invoke'):
                # Modern LangChain LLM
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    response = response.content
            else:
                # Custom LLM
                response = self.llm.generate(prompt)

            return response.strip()

        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return "I encountered an error while generating the answer. Please try again."

    def format_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """Format retrieved documents as citation sources in target JSON format"""
        sources = []

        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})

            # Format according to target structure: company, filing, period, snippet, url
            source = {
                "company": metadata.get('ticker', metadata.get('company', 'AAPL')),  # Use ticker format (AAPL, MSFT)
                "filing": metadata.get('filing_type', '10-K'),
                "period": f"FY{metadata.get('year', 2023)}",  # Format as FY2023
                "snippet": doc.get('text', '')[:150] + "..." if len(doc.get('text', '')) > 150 else doc.get('text', ''),
                "url": self._generate_sec_url(metadata)
            }

            sources.append(source)

        return sources

    def _generate_sec_url(self, metadata: Dict) -> str:
        """Generate SEC EDGAR URL for the filing"""
        # Generate realistic SEC EDGAR URLs
        ticker = metadata.get('ticker', metadata.get('company', 'AAPL'))
        filing_type = metadata.get('filing_type', '10-K')
        year = metadata.get('year', 2023)

        if ticker and filing_type:
            # Format: https://www.sec.gov/edgar/browse/?CIK=TICKER&owner=exclude&action=getcompany&type=FILING_TYPE
            return f"https://www.sec.gov/edgar/browse/?CIK={ticker}&owner=exclude&action=getcompany&type={filing_type}&dateb={year}1231"

        return "https://www.sec.gov/edgar"

    def _classify_query(self, question: str) -> str:
        """Classify the type of query"""
        question_lower = question.lower()

        # Comparative queries
        if any(word in question_lower for word in ['compare', 'vs', 'versus', 'between', 'difference']):
            return "comparative"

        # Trend analysis
        if any(word in question_lower for word in ['trend', 'growth', 'change', 'over time', 'yoy', 'year over year']):
            return "trend_analysis"

        # Financial metrics
        if any(word in question_lower for word in ['revenue', 'profit', 'earnings', 'margin', 'cash flow']):
            return "financial_metrics"

        # Risk analysis
        if any(word in question_lower for word in ['risk', 'challenge', 'threat', 'concern']):
            return "risk_analysis"

        return "basic"

    def _calculate_confidence(self, retrieved_docs: List[Dict], answer: str) -> float:
        """Calculate confidence score for the answer"""
        if not retrieved_docs:
            return 0.0

        # Base confidence on average similarity scores
        scores = [doc.get('score', 0.0) for doc in retrieved_docs]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Adjust based on number of sources
        source_bonus = min(0.1 * len(retrieved_docs), 0.3)

        # Adjust based on answer length (longer answers might be more comprehensive)
        length_factor = min(len(answer) / 500, 1.0) * 0.1

        confidence = min(avg_score + source_bonus + length_factor, 1.0)
        return round(confidence, 3)

    def decompose_query(self, question: str) -> List[str]:
        """Decompose complex queries into sub-questions"""
        # Simple query decomposition - can be enhanced with LLM
        question_lower = question.lower()

        # Check for comparative queries
        if 'compare' in question_lower or 'vs' in question_lower:
            # Try to extract companies being compared
            companies = []
            for ticker in ['aapl', 'msft', 'amzn', 'googl', 'meta', 'nvda', 'tsla']:
                if ticker in question_lower:
                    companies.append(ticker.upper())

            if len(companies) >= 2:
                base_question = re.sub(r'\b(compare|vs|versus)\b.*', '', question, flags=re.IGNORECASE).strip()
                sub_questions = [f"{base_question} for {company}" for company in companies]
                return sub_questions

        # For now, return the original question
        return [question]

    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the RAG pipeline"""
        stats = {
            "initialized": self.initialize(),
            "llm_model": self.llm_config.get("model_name", "unknown"),
            "vector_store_stats": {},
            "retrieval_config": self.retrieval_config.copy()
        }

        if self.vector_store:
            try:
                stats["vector_store_stats"] = self.vector_store.get_collection_stats()
            except Exception as e:
                stats["vector_store_stats"] = {"error": str(e)}

        return stats

    def test_pipeline(self, test_question: str = "What was Apple's revenue in 2023?") -> Dict:
        """Test the RAG pipeline with a simple question"""
        try:
            self.logger.info(f"Testing pipeline with question: {test_question}")
            result = self.query(test_question)

            test_result = {
                "test_question": test_question,
                "success": "error" not in result.get("_internal", {}),
                "response_time": result.get("_internal", {}).get("processing_time", 0),
                "sources_found": len(result.get("sources", [])),
                "confidence": result.get("confidence", 0),
                "answer_preview": result.get("answer", "")[:100] + "..." if len(result.get("answer", "")) > 100 else result.get("answer", "")
            }

            return test_result

        except Exception as e:
            return {
                "test_question": test_question,
                "success": False,
                "error": str(e)
            }

    def batch_query(self, questions: List[str]) -> List[Dict]:
        """Process multiple questions in batch"""
        results = []

        for question in questions:
            self.logger.info(f"Processing batch question: {question}")
            result = self.query(question)
            result["question"] = question
            results.append(result)

        return results

    def clear_cache(self):
        """Clear any cached data (placeholder for future caching implementation)"""
        self.logger.info("Cache cleared (no caching implemented yet)")
        pass
