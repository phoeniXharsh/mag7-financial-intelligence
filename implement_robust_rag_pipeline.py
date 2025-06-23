"""
Robust RAG Pipeline Implementation
Sophisticated RAG pipeline with specialized query handling for financial data analysis.
"""

import sys
import os
import re
from typing import List, Dict, Any, Optional
from enum import Enum

# Add src to path
sys.path.append('src')
sys.path.append(os.path.dirname(__file__))

class QueryType(Enum):
    BASIC = "basic"
    COMPARATIVE = "comparative"
    COMPLEX = "complex"
    TREND_ANALYSIS = "trend_analysis"
    CROSS_COMPANY = "cross_company"

class QueryClassifier:
    """Classify queries into different types for specialized handling"""

    def __init__(self):
        self.patterns = {
            QueryType.CROSS_COMPANY: [
                r'compare.*across.*all.*mag7',
                r'which.*mag7.*company',
                r'rank.*mag7.*companies',
                r'all.*mag7.*companies',
                r'among.*mag7',
                r'across.*companies'
            ],
            QueryType.COMPARATIVE: [
                r'compare.*vs.*',
                r'.*vs.*vs.*',
                r'.*between.*and.*',
                r'yoy.*growth',
                r'year.*over.*year',
                r'.*in.*vs.*in.*'
            ],
            QueryType.TREND_ANALYSIS: [
                r'trend.*over.*time',
                r'most.*consistent',
                r'growth.*trajectory',
                r'evolved.*during',
                r'pattern.*in.*',
                r'from.*\d{4}.*to.*\d{4}'
            ],
            QueryType.COMPLEX: [
                r'how.*did.*impact',
                r'what.*factors.*drove',
                r'key.*drivers',
                r'affect.*profitability',
                r'impact.*on.*',
                r'factors.*behind'
            ],
            QueryType.BASIC: [
                r'what.*was.*revenue',
                r'what.*was.*income',
                r'how.*much.*spend',
                r'what.*was.*margin'
            ]
        }

    def classify(self, query: str) -> QueryType:
        """Classify query into appropriate type"""
        query_lower = query.lower()

        # Check patterns in order of specificity
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type

        # Default to basic if no pattern matches
        return QueryType.BASIC

class CompanyExtractor:
    """Extract company names and tickers from queries"""

    def __init__(self):
        self.company_mappings = {
            'apple': 'AAPL',
            'aapl': 'AAPL',
            'amazon': 'AMZN',
            'amzn': 'AMZN',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'googl': 'GOOGL',
            'meta': 'META',
            'facebook': 'META',
            'microsoft': 'MSFT',
            'msft': 'MSFT',
            'nvidia': 'NVDA',
            'nvda': 'NVDA',
            'tesla': 'TSLA',
            'tsla': 'TSLA'
        }

        self.mag7_companies = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']

    def extract_companies(self, query: str) -> List[str]:
        """Extract company tickers from query"""
        query_lower = query.lower()
        found_companies = []

        # Check for MAG7 reference
        if 'mag7' in query_lower or 'all' in query_lower:
            return self.mag7_companies

        # Extract specific companies
        for company_name, ticker in self.company_mappings.items():
            if company_name in query_lower:
                if ticker not in found_companies:
                    found_companies.append(ticker)

        return found_companies if found_companies else self.mag7_companies

class TimeExtractor:
    """Extract time periods from queries"""

    def extract_years(self, query: str) -> List[str]:
        """Extract years from query"""
        # Find 4-digit years
        years = re.findall(r'\b(20[0-9]{2})\b', query)
        return list(set(years))

    def extract_quarters(self, query: str) -> List[str]:
        """Extract quarters from query"""
        quarters = re.findall(r'\b(Q[1-4])\b', query, re.IGNORECASE)
        return [q.upper() for q in quarters]

class RobustRAGPipeline:
    """Robust RAG pipeline with specialized handling for different query types"""

    def __init__(self, vector_store, llm_config, retrieval_config):
        self.vector_store = vector_store
        self.llm_config = llm_config
        self.retrieval_config = retrieval_config

        # Initialize components
        self.query_classifier = QueryClassifier()
        self.company_extractor = CompanyExtractor()
        self.time_extractor = TimeExtractor()

        # Initialize LLM using existing CustomGeminiLLM
        try:
            from rag.pipeline import CustomGeminiLLM
        except ImportError:
            # Fallback import path
            sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
            from rag.pipeline import CustomGeminiLLM
        self.llm = CustomGeminiLLM(
            model_name=llm_config.get("model_name", "gemini-pro"),
            temperature=llm_config.get("temperature", 0.1),
            max_tokens=llm_config.get("max_tokens", 3000),
            api_key=llm_config.get("api_key")
        )

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with specialized handling based on type"""

        try:
            # Input validation
            if not query or not query.strip():
                return {
                    "answer": "Please provide a valid question about MAG7 companies.",
                    "sources": [],
                    "confidence": 0.0,
                    "query_type": "error"
                }

            query = query.strip()

            # Step 1: Classify query
            query_type = self.query_classifier.classify(query)

            # Step 2: Extract entities
            companies = self.company_extractor.extract_companies(query)
            years = self.time_extractor.extract_years(query)
            quarters = self.time_extractor.extract_quarters(query)

            print(f"🔍 Query Analysis:")
            print(f"   Type: {query_type.value}")
            print(f"   Companies: {companies}")
            print(f"   Years: {years}")
            print(f"   Quarters: {quarters}")

            # Step 3: Route to specialized handler with error handling
            try:
                if query_type == QueryType.CROSS_COMPANY:
                    return self._handle_cross_company_query(query, companies, years)
                elif query_type == QueryType.COMPARATIVE:
                    return self._handle_comparative_query(query, companies, years)
                elif query_type == QueryType.COMPLEX:
                    return self._handle_complex_query(query, companies, years)
                elif query_type == QueryType.TREND_ANALYSIS:
                    return self._handle_trend_analysis_query(query, companies, years)
                else:  # BASIC
                    return self._handle_basic_query(query, companies, years, quarters)
            except Exception as handler_error:
                print(f"❌ Handler error: {handler_error}")
                return {
                    "answer": f"I encountered an issue processing your {query_type.value} query. Please try rephrasing your question or contact support if the issue persists.",
                    "sources": [],
                    "confidence": 0.0,
                    "query_type": query_type.value,
                    "_internal": {"error": str(handler_error)}
                }

        except Exception as e:
            print(f"❌ Query processing error: {e}")
            return {
                "answer": "I encountered an unexpected error while processing your question. Please try again with a different question.",
                "sources": [],
                "confidence": 0.0,
                "query_type": "error",
                "_internal": {"error": str(e)}
            }

    def _handle_basic_query(self, query: str, companies: List[str], years: List[str], quarters: List[str]) -> Dict[str, Any]:
        """Handle basic factual queries"""
        print(f"📊 Handling BASIC query")

        # For basic queries, focus on specific company and time period
        target_companies = companies[:1] if companies else ['AAPL']  # Default to Apple if none specified
        target_years = years if years else ['2023']  # Default to 2023 if none specified

        # Enhanced search strategy for basic queries
        all_documents = []

        # Strategy 1: Direct search with multiple variations
        search_variations = []
        for company in target_companies:
            for year in target_years:
                if quarters:
                    for quarter in quarters:
                        # Try multiple search patterns for quarterly data
                        search_variations.extend([
                            f"{company} {quarter} {year} revenue quarterly earnings",
                            f"{company} quarterly revenue {quarter} {year}",
                            f"{company} {year} {quarter} financial results",
                            f"{company} three months ended {year}"
                        ])
                else:
                    # Try multiple search patterns for annual data
                    search_variations.extend([
                        f"{company} {year} revenue annual earnings",
                        f"{company} fiscal year {year} revenue",
                        f"{company} {year} total revenue net sales",
                        f"{company} {year} financial performance results"
                    ])

        # Strategy 2: Search by filing type
        for company in target_companies:
            for year in target_years:
                if quarters:
                    search_variations.append(f"{company} 10-Q {year} revenue")
                else:
                    search_variations.append(f"{company} 10-K {year} revenue")

        # Execute searches with deduplication
        seen_docs = set()
        for search_query in search_variations[:8]:  # Limit to avoid too many searches
            docs = self.vector_store.search(search_query, top_k=3)
            for doc in docs:
                doc_id = f"{doc.get('metadata', {}).get('ticker', '')}-{doc.get('metadata', {}).get('year', '')}-{hash(doc.get('text', '')[:100])}"
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    all_documents.append(doc)

        # Filter for relevant documents
        relevant_docs = self._filter_documents_by_entities(all_documents, target_companies, target_years)

        # If no relevant docs found, try broader search
        if not relevant_docs:
            print(f"   🔍 No specific docs found, trying broader search...")
            broader_search = f"{' '.join(target_companies)} {' '.join(target_years)} revenue financial"
            broader_docs = self.vector_store.search(broader_search, top_k=10)
            relevant_docs = self._filter_documents_by_entities(broader_docs, target_companies, target_years)

        print(f"   📚 Found {len(relevant_docs)} relevant documents")

        # Generate response
        context = self._format_context(relevant_docs[:10])
        response = self._generate_response(query, context, "basic")

        return {
            "answer": response,
            "sources": self._format_sources(relevant_docs[:10]),
            "confidence": self._calculate_confidence(response, relevant_docs),
            "query_type": "basic"
        }

    def _handle_cross_company_query(self, query: str, companies: List[str], years: List[str]) -> Dict[str, Any]:
        """Handle cross-company comparison queries"""
        print(f"🏢 Handling CROSS-COMPANY query")

        # For cross-company queries, we need data from ALL MAG7 companies
        target_companies = companies if len(companies) > 1 else self.company_extractor.mag7_companies
        target_years = years if years else ['2023']  # Default to 2023

        # Build comprehensive search strategy
        all_documents = []

        # Search for each company individually
        for company in target_companies:
            for year in target_years:
                search_query = f"{company} {year} revenue margin profit financial"
                docs = self.vector_store.search(search_query, top_k=3)
                company_docs = self._filter_documents_by_entities(docs, [company], [year])
                all_documents.extend(company_docs)

        # Also search for general comparison terms
        comparison_queries = [
            f"MAG7 companies {' '.join(target_years)} comparison",
            f"technology companies {' '.join(target_years)} financial performance",
            f"operating margin profit {' '.join(target_years)}"
        ]

        for comp_query in comparison_queries:
            docs = self.vector_store.search(comp_query, top_k=5)
            all_documents.extend(docs)

        # Ensure we have representation from multiple companies
        company_representation = {}
        final_docs = []

        for doc in all_documents:
            ticker = doc.get('metadata', {}).get('ticker', 'UNKNOWN')
            if ticker in target_companies:
                if ticker not in company_representation:
                    company_representation[ticker] = []
                if len(company_representation[ticker]) < 3:  # Max 3 docs per company
                    company_representation[ticker].append(doc)
                    final_docs.append(doc)

        print(f"   📊 Company representation: {list(company_representation.keys())}")

        # Generate response with cross-company context
        context = self._format_cross_company_context(final_docs, target_companies)
        response = self._generate_response(query, context, "cross_company")

        return {
            "answer": response,
            "sources": self._format_sources(final_docs[:15]),
            "confidence": self._calculate_confidence(response, final_docs),
            "query_type": "cross_company"
        }

    def _handle_comparative_query(self, query: str, companies: List[str], years: List[str]) -> Dict[str, Any]:
        """Handle comparative analysis queries"""
        print(f"📊 Handling COMPARATIVE query")

        # Extract comparison entities
        target_companies = companies if companies else ['AAPL', 'MSFT']  # Default comparison
        target_years = years if years else ['2022', '2023']  # Default years

        # Build search strategy for comparisons
        all_documents = []

        # Search for each company-year combination
        for company in target_companies:
            for year in target_years:
                search_query = f"{company} {year} revenue growth financial performance"
                docs = self.vector_store.search(search_query, top_k=3)
                filtered_docs = self._filter_documents_by_entities(docs, [company], [year])
                all_documents.extend(filtered_docs)

        # Generate comparative response
        context = self._format_comparative_context(all_documents, target_companies, target_years)
        response = self._generate_response(query, context, "comparative")

        return {
            "answer": response,
            "sources": self._format_sources(all_documents[:10]),
            "confidence": self._calculate_confidence(response, all_documents),
            "query_type": "comparative"
        }

    def _handle_complex_query(self, query: str, companies: List[str], years: List[str]) -> Dict[str, Any]:
        """Handle complex analytical queries"""
        print(f"🧠 Handling COMPLEX query")

        # For complex queries, cast a wider net for context
        target_companies = companies if companies else self.company_extractor.mag7_companies
        target_years = years if years else ['2020', '2021', '2022', '2023']  # Wider time range

        # Extract key topics from query
        topics = self._extract_topics(query)

        # Build comprehensive search
        all_documents = []

        # Search with topic-specific terms
        for company in target_companies[:3]:  # Limit to avoid too much data
            for topic in topics:
                search_query = f"{company} {topic} {' '.join(target_years)} impact analysis"
                docs = self.vector_store.search(search_query, top_k=5)
                all_documents.extend(docs)

        # Generate analytical response
        context = self._format_analytical_context(all_documents, topics)
        response = self._generate_response(query, context, "complex")

        return {
            "answer": response,
            "sources": self._format_sources(all_documents[:15]),
            "confidence": self._calculate_confidence(response, all_documents),
            "query_type": "complex"
        }

    def _handle_trend_analysis_query(self, query: str, companies: List[str], years: List[str]) -> Dict[str, Any]:
        """Handle trend analysis queries"""
        print(f"📈 Handling TREND ANALYSIS query")

        # For trend analysis, we need multi-year data
        target_companies = companies if companies else ['AAPL']  # Default to Apple
        target_years = years if len(years) > 1 else ['2020', '2021', '2022', '2023', '2024']  # Multi-year range

        # Build trend-focused search
        all_documents = []

        for company in target_companies:
            # Search for each year to build trend
            for year in target_years:
                search_query = f"{company} {year} revenue growth R&D investment trend"
                docs = self.vector_store.search(search_query, top_k=2)
                filtered_docs = self._filter_documents_by_entities(docs, [company], [year])
                all_documents.extend(filtered_docs)

        # Generate trend analysis response
        context = self._format_trend_context(all_documents, target_companies, target_years)
        response = self._generate_response(query, context, "trend_analysis")

        return {
            "answer": response,
            "sources": self._format_sources(all_documents[:12]),
            "confidence": self._calculate_confidence(response, all_documents),
            "query_type": "trend_analysis"
        }

    def _filter_documents_by_entities(self, documents: List[Dict], companies: List[str], years: List[str]) -> List[Dict]:
        """Filter documents to match specified companies and years"""
        filtered = []

        for doc in documents:
            metadata = doc.get('metadata', {})
            doc_ticker = metadata.get('ticker', '')
            doc_year = metadata.get('year', '')

            # Check if document matches criteria
            company_match = not companies or doc_ticker in companies
            year_match = not years or doc_year in years

            if company_match and year_match:
                filtered.append(doc)

        return filtered

    def _extract_topics(self, query: str) -> List[str]:
        """Extract key topics from complex queries"""
        topic_keywords = {
            'covid': ['covid', 'pandemic', 'coronavirus'],
            'cloud': ['cloud', 'aws', 'azure'],
            'ai': ['ai', 'artificial intelligence', 'machine learning'],
            'revenue': ['revenue', 'sales', 'income'],
            'investment': ['investment', 'r&d', 'research'],
            'growth': ['growth', 'expansion', 'increase'],
            'margin': ['margin', 'profitability', 'profit']
        }

        query_lower = query.lower()
        found_topics = []

        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                found_topics.append(topic)

        return found_topics if found_topics else ['revenue', 'growth']

    def _format_context(self, documents: List[Dict]) -> str:
        """Format documents into context for LLM"""
        if not documents:
            return "No relevant financial data found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')

            company = metadata.get('company', 'Unknown')
            year = metadata.get('year', 'Unknown')
            filing = metadata.get('filing_type', 'Unknown')

            context_parts.append(f"Document {i}: {company} ({year} {filing})\n{text}\n")

        return "\n".join(context_parts)

    def _format_cross_company_context(self, documents: List[Dict], companies: List[str]) -> str:
        """Format context specifically for cross-company analysis"""
        if not documents:
            return "No relevant cross-company financial data found."

        # Group documents by company
        company_docs = {}
        for doc in documents:
            ticker = doc.get('metadata', {}).get('ticker', 'UNKNOWN')
            if ticker not in company_docs:
                company_docs[ticker] = []
            company_docs[ticker].append(doc)

        context_parts = [f"Cross-company analysis for: {', '.join(companies)}\n"]

        for ticker, docs in company_docs.items():
            if docs:
                company_name = docs[0].get('metadata', {}).get('company', ticker)
                context_parts.append(f"\n{company_name} ({ticker}):")
                for doc in docs[:2]:  # Limit to 2 docs per company
                    text = doc.get('text', '')
                    context_parts.append(f"  {text}")

        return "\n".join(context_parts)

    def _format_comparative_context(self, documents: List[Dict], companies: List[str], years: List[str]) -> str:
        """Format context for comparative analysis"""
        if not documents:
            return "No relevant comparative financial data found."

        context_parts = [f"Comparative analysis for {', '.join(companies)} across {', '.join(years)}:\n"]

        # Group by company and year
        for company in companies:
            for year in years:
                relevant_docs = [doc for doc in documents
                               if doc.get('metadata', {}).get('ticker') == company
                               and doc.get('metadata', {}).get('year') == year]

                if relevant_docs:
                    company_name = relevant_docs[0].get('metadata', {}).get('company', company)
                    context_parts.append(f"\n{company_name} {year}:")
                    context_parts.append(relevant_docs[0].get('text', ''))

        return "\n".join(context_parts)

    def _format_analytical_context(self, documents: List[Dict], topics: List[str]) -> str:
        """Format context for complex analytical queries"""
        if not documents:
            return "No relevant analytical data found."

        context_parts = [f"Analytical context for topics: {', '.join(topics)}\n"]

        for i, doc in enumerate(documents[:10], 1):
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')

            company = metadata.get('company', 'Unknown')
            year = metadata.get('year', 'Unknown')

            context_parts.append(f"Analysis {i} - {company} ({year}):\n{text}\n")

        return "\n".join(context_parts)

    def _format_trend_context(self, documents: List[Dict], companies: List[str], years: List[str]) -> str:
        """Format context for trend analysis"""
        if not documents:
            return "No relevant trend data found."

        context_parts = [f"Trend analysis for {', '.join(companies)} from {min(years)} to {max(years)}:\n"]

        # Sort documents by year for trend analysis
        sorted_docs = sorted(documents, key=lambda x: x.get('metadata', {}).get('year', '0'))

        for doc in sorted_docs:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')

            company = metadata.get('company', 'Unknown')
            year = metadata.get('year', 'Unknown')

            context_parts.append(f"{company} {year}: {text}")

        return "\n".join(context_parts)

    def _generate_response(self, query: str, context: str, query_type: str) -> str:
        """Generate response using LLM with specialized prompts"""

        # Build specialized prompt based on query type
        if query_type == "cross_company":
            prompt = f"""You are a financial analyst specializing in cross-company comparisons.

Analyze the following financial data and answer the user's question with a comprehensive comparison across all mentioned companies.

Context:
{context}

Question: {query}

Provide a detailed comparison with specific financial figures for each company. Structure your response to clearly compare the companies side by side."""

        elif query_type == "comparative":
            prompt = f"""You are a financial analyst specializing in comparative analysis.

Analyze the following financial data and provide a detailed comparison as requested.

Context:
{context}

Question: {query}

Focus on the specific comparison requested. Include percentage changes, growth rates, and specific financial figures."""

        elif query_type == "complex":
            prompt = f"""You are a senior financial analyst specializing in complex business analysis.

Analyze the following financial data and provide insights into the complex factors and relationships mentioned in the question.

Context:
{context}

Question: {query}

Provide a comprehensive analysis that explains the underlying factors, causes, and impacts. Use specific data points to support your analysis."""

        elif query_type == "trend_analysis":
            prompt = f"""You are a financial analyst specializing in trend analysis and time-series data.

Analyze the following financial data to identify and explain trends over time.

Context:
{context}

Question: {query}

Focus on identifying patterns, trends, and trajectories. Include specific data points and explain the trend direction and consistency."""

        else:  # basic
            prompt = f"""You are a financial analyst specializing in SEC filings analysis. Answer the following question using ONLY the provided financial data.

Context from SEC filings:
{context}

Question: {query}

Instructions:
1. If the exact data requested is available in the context, provide it with specific figures
2. If the context contains related but not exact data (e.g., annual instead of quarterly), explain what data is available
3. If the context contains data for the company but different time period, mention what periods are available
4. Always cite the specific filing type (10-K for annual, 10-Q for quarterly) and year
5. Be precise about what data is and isn't available
6. Do not make assumptions or extrapolate data not explicitly stated

Answer:"""

        try:
            # Check if LLM is available
            if not self.llm:
                return "I apologize, but the AI model is not properly configured. Please check your API key and try again."

            # Use the initialized LLM directly
            response = self.llm.generate(prompt)

            if not response or len(response.strip()) < 10:
                return "I apologize, but I couldn't generate a comprehensive response. Please try rephrasing your question."

            return response.strip()

        except Exception as e:
            error_msg = str(e).lower()

            # Provide specific error messages for common issues
            if "api key" in error_msg or "authentication" in error_msg:
                return "I'm having trouble connecting to the AI service due to an API key issue. Please check your configuration."
            elif "quota" in error_msg or "limit" in error_msg:
                return "I've reached the API usage limit. Please try again later or check your API quota."
            elif "timeout" in error_msg or "connection" in error_msg:
                return "I'm experiencing connection issues. Please try again in a moment."
            else:
                return f"I encountered an error while generating the response. Please try rephrasing your question."

    def _format_sources(self, documents: List[Dict]) -> List[Dict]:
        """Format sources for response in assignment specification format"""
        sources = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')

            # Extract company ticker (prefer ticker over company name)
            company = metadata.get('ticker', metadata.get('company', 'Unknown'))

            # Extract filing type
            filing_type = metadata.get('filing_type', 'Unknown')

            # Extract year and create period
            year = metadata.get('year', 'Unknown')
            period = f"FY{year}" if filing_type == '10-K' else f"Q{metadata.get('quarter', '1')} FY{year}"
            if year == 'Unknown':
                period = 'Unknown'

            # Create snippet (first 200 characters of text)
            snippet = text[:200] + "..." if len(text) > 200 else text

            # Create SEC URL (placeholder - would need actual filing URLs)
            url = f"https://www.sec.gov/edgar/browse/?CIK={company}&owner=exclude"

            sources.append({
                'company': company,
                'filing': filing_type,
                'period': period,
                'snippet': snippet,
                'url': url
            })
        return sources

    def _calculate_confidence(self, response: str, documents: List[Dict]) -> float:
        """Calculate confidence score for response"""
        if not response or not documents:
            return 0.0

        # Base confidence from document count
        doc_confidence = min(len(documents) / 10, 1.0)

        # Response quality indicators
        has_numbers = any(char.isdigit() for char in response)
        has_financial_terms = any(term in response.lower() for term in ['billion', 'million', 'revenue', 'profit'])
        response_length = min(len(response) / 200, 1.0)

        # Combine factors
        quality_score = (
            doc_confidence * 0.4 +
            (0.2 if has_numbers else 0) +
            (0.2 if has_financial_terms else 0) +
            response_length * 0.2
        )

        return min(quality_score, 1.0)

    def get_pipeline_stats(self) -> dict:
        """Get statistics about the RAG pipeline"""
        try:
            # Check vector store status
            vector_store_status = "initialized" if hasattr(self, 'vector_store') and self.vector_store else "not_initialized"

            # Check LLM status
            llm_status = "initialized" if hasattr(self, 'llm') and self.llm else "not_initialized"

            # Get vector store collection info if available
            collection_info = {}
            if hasattr(self, 'vector_store') and self.vector_store:
                try:
                    # Try to get basic info about the vector store
                    collection_info = {
                        "db_path": getattr(self.vector_store, 'db_path', 'unknown'),
                        "collection_name": getattr(self.vector_store, 'collection_name', 'unknown')
                    }
                except:
                    collection_info = {"status": "info_unavailable"}

            return {
                "pipeline_type": "RobustRAGPipeline",
                "vector_store_status": vector_store_status,
                "llm_status": llm_status,
                "collection_info": collection_info,
                "query_classifier": "QueryClassifier",
                "company_extractor": "CompanyExtractor",
                "time_extractor": "TimeExtractor",
                "supported_query_types": ["basic", "comparative", "complex", "trend_analysis", "cross_company"],
                "retrieval_config": {
                    "top_k": getattr(self.retrieval_config, 'get', lambda x, y: y)('top_k', 15),
                    "similarity_threshold": getattr(self.retrieval_config, 'get', lambda x, y: y)('similarity_threshold', 0.3),
                    "max_context_length": getattr(self.retrieval_config, 'get', lambda x, y: y)('max_context_length', 4000)
                } if hasattr(self, 'retrieval_config') else {},
                "llm_config": {
                    "model_name": getattr(self.llm_config, 'get', lambda x, y: y)('model_name', 'gemini-pro'),
                    "temperature": getattr(self.llm_config, 'get', lambda x, y: y)('temperature', 0.1),
                    "max_tokens": getattr(self.llm_config, 'get', lambda x, y: y)('max_tokens', 3000)
                } if hasattr(self, 'llm_config') else {}
            }
        except Exception as e:
            return {
                "pipeline_type": "RobustRAGPipeline",
                "status": "error",
                "error": str(e),
                "vector_store_status": "unknown",
                "llm_status": "unknown"
            }

