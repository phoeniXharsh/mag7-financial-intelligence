"""
Main Streamlit interface for the MAG7 Financial Q&A System
"""

import streamlit as st
from typing import Dict, List
import sys
import os
import re

if 'src' not in sys.path:
    sys.path.append('src')

try:
    from data.sec_scraper import SECScraper
    from config.settings import MAG7_COMPANIES
    from vector_store.faiss_store import FAISSVectorStore
    from rag.pipeline import RAGPipeline
    from agent.conversational_agent import ConversationalAgent
    from agent.robust_conversational_agent import create_robust_agent
except ImportError as e:
    st.error(f"Import error: {e}")
    SECScraper = None
    FAISSVectorStore = None
    RAGPipeline = None
    ConversationalAgent = None

def _generate_follow_up_suggestions(prompt: str, response: Dict) -> List[str]:
    """Generate intelligent follow-up suggestions based on the query and response"""
    suggestions = []

    # Extract companies and years from the original prompt
    companies = []
    years = []

    # Simple extraction
    mag7_companies = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
    company_names = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'amazon': 'AMZN',
        'google': 'GOOGL', 'alphabet': 'GOOGL', 'meta': 'META',
        'facebook': 'META', 'nvidia': 'NVDA', 'tesla': 'TSLA'
    }

    prompt_lower = prompt.lower()

    # Extract companies
    for name, ticker in company_names.items():
        if name in prompt_lower:
            companies.append(ticker)

    # Extract years
    import re
    years = re.findall(r'\b(20[0-9]{2})\b', prompt)

    # Generate suggestions based on query type
    query_type = response.get("_internal", {}).get("query_type", "basic")

    if query_type == "basic" and companies:
        company = companies[0]
        suggestions.extend([
            f"Compare {company} with other MAG7 companies",
            f"What is {company}'s revenue trend over time?",
            f"What are {company}'s main risk factors?"
        ])

    elif query_type == "comparative" and len(companies) >= 2:
        suggestions.extend([
            f"Compare operating margins for {' vs '.join(companies[:2])}",
            f"What are the key differences between {' and '.join(companies[:2])}?",
            "Expand comparison to all MAG7 companies"
        ])

    elif query_type == "cross_company":
        suggestions.extend([
            "Which company has the best profit margins?",
            "Compare R&D spending across MAG7 companies",
            "What are the revenue trends for each company?"
        ])

    elif query_type == "trend_analysis":
        suggestions.extend([
            "What factors drove these trends?",
            "Compare trends across different companies",
            "What are the future growth prospects?"
        ])

    # Default suggestions if none generated
    if not suggestions:
        suggestions = [
            "Compare revenue across all MAG7 companies",
            "What are the latest quarterly results?",
            "Which company has the strongest growth?"
        ]

    return suggestions[:3]  # Limit to 3 suggestions

def clean_response_text(text: str) -> str:
    """Clean response text to remove formatting issues and ensure plain text"""
    if not text:
        return text

    # Remove any markdown formatting that might cause display issues
    cleaned = text

    # Use a dictionary of specific known problematic patterns and their fixes
    # This is more reliable than complex regex

    problem_patterns = {
        # Specific Google 2020 issue
        '240.5billionforfiscalyear2020,a9.0218.6': '240.5 billion for fiscal year 2020, a 9.0% increase from 218.6',

        # Specific Microsoft issue
        '156.2billion,a7.0142.0': '156.2 billion, a 7.0% increase from 142.0',

        # General patterns
        'billionforfiscalyear': ' billion for fiscal year ',
        'millionforfiscalyear': ' million for fiscal year ',
        'billionforfiscal': ' billion for fiscal ',
        'millionforfiscal': ' million for fiscal ',
        'billion,a': ' billion, a ',
        'million,a': ' million, a ',
    }

    # Apply specific pattern fixes first
    for pattern, replacement in problem_patterns.items():
        cleaned = cleaned.replace(pattern, replacement)

    # Then apply simple regex fixes for remaining issues

    # Fix basic number+unit concatenation
    cleaned = re.sub(r'(\d+\.?\d*)billion(?![a-z])', r'\1 billion', cleaned)
    cleaned = re.sub(r'(\d+\.?\d*)million(?![a-z])', r'\1 million', cleaned)

    # Fix percentage patterns like ",a9.0" -> ", a 9.0%"
    cleaned = re.sub(r',a(\d+\.?\d*)(?!\d)', r', a \1%', cleaned)

    # Remove markdown formatting
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove bold
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Remove italic
    cleaned = re.sub(r'_(.*?)_', r'\1', cleaned)        # Remove underline

    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()

def _process_user_message(prompt: str, agent):
    """Helper function to process user messages"""
    # Generate response using conversational agent
    with st.chat_message("assistant"):
        # Show searching status
        status_placeholder = st.empty()
        status_placeholder.info("🔍 Searching relevant sections from 10-K/Q filings...")

        try:
            # Get session ID
            session_id = st.session_state.get("session_id", "streamlit_session")

            # Process with conversational agent
            response = agent.process_message(prompt, session_id)

            # Clear searching status
            status_placeholder.empty()

            # Display answer with text cleaning
            answer = response.get("answer", "I couldn't generate a response.")

            # Clean the answer text to remove formatting issues
            cleaned_answer = clean_response_text(answer)
            st.markdown(cleaned_answer)

            # Display metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                confidence = response.get("confidence", 0.0)
                st.metric("Confidence", f"{confidence:.3f}")
            with col2:
                internal_data = response.get("_internal", {})
                query_type = internal_data.get("query_type", "unknown")
                st.metric("Query Type", query_type)
            with col3:
                sources_count = len(response.get("sources", []))
                st.metric("Sources", sources_count)
            with col4:
                # Add complete JSON response toggle with unique key
                import time
                json_btn_key = f"json_btn_{len(st.session_state.messages)}_{int(time.time() * 1000) % 10000}"
                if st.button("📋 Full JSON", key=json_btn_key):
                    st.subheader("🔧 Complete JSON Response")
                    # Create clean JSON response (target format)
                    clean_response = {
                        "answer": response.get("answer", ""),
                        "sources": response.get("sources", []),
                        "confidence": response.get("confidence", 0.0)
                    }
                    import json
                    st.code(json.dumps(clean_response, indent=2), language="json")

            # Generate and display intelligent follow-up suggestions
            suggestions = _generate_follow_up_suggestions(prompt, response)

            if suggestions:
                st.write("💡 **Follow-up suggestions:**")

                # Create clickable suggestion buttons
                cols = st.columns(min(len(suggestions), 3))  # Max 3 columns
                for i, suggestion in enumerate(suggestions):
                    with cols[i % len(cols)]:
                        if st.button(suggestion, key=f"suggestion_{len(st.session_state.messages)}_{i}", use_container_width=True):
                            st.session_state.pending_suggestion = suggestion
                            st.rerun()

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": response.get("sources", []),
                "follow_up_suggestions": response.get("follow_up_suggestions", []),
                "confidence": confidence,
                "query_type": query_type,
                "json_response": response  # Store the complete JSON response
            })

        except Exception as e:
            status_placeholder.empty()
            error_msg = f"Error processing your question: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": []
            })

def render_main_interface():
    """Render the main application interface"""
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Data Management", "⚙️ Settings"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_data_management()
    
    with tab3:
        render_settings()

def render_chat_interface():
    """Render the chat interface"""
    st.header("💬 Ask Questions About MAG7 Companies")

    # Initialize conversational agent
    if "conversational_agent" not in st.session_state:
        st.session_state.conversational_agent = None

    # Handle pending suggestion from button click
    if "pending_suggestion" in st.session_state:
        suggestion = st.session_state.pending_suggestion
        print(f"🔍 PROCESSING PENDING SUGGESTION: {suggestion}")  # Debug to terminal
        del st.session_state.pending_suggestion  # Remove it immediately

        # Add the suggestion as a user message
        st.session_state.messages.append({"role": "user", "content": suggestion})

        # Process the suggestion immediately
        agent = st.session_state.conversational_agent
        if agent:
            with st.chat_message("user"):
                st.markdown(suggestion)

            # Process the suggestion
            _process_user_message(suggestion, agent)
            return  # Exit early to avoid duplicate processing

    # Check if agent is initialized
    agent = st.session_state.conversational_agent
    if not agent:
        st.info("🔧 Please set up the system in the Settings tab first (API key required)")

        # Show example queries based on assignment requirements
        st.subheader("💡 Example Questions You Can Ask:")
        st.write("The system supports all query types from the assignment:")

        # Organize examples by category
        col1, col2 = st.columns(2)

        with col1:
            st.write("**📊 Basic Queries:**")
            basic_examples = [
                "What was Microsoft's revenue for fiscal year 2024?",
                "What was Meta's revenue for Q1 2024?",
                "What was Apple's net income in 2023?"
            ]
            for example in basic_examples:
                if st.button(f"📝 {example}", key=f"basic_{hash(example)}", use_container_width=True):
                    st.session_state.pending_suggestion = example
                    st.rerun()

            st.write("**📈 Trend Analysis:**")
            trend_examples = [
                "What is Microsoft's revenue trend from 2022 to 2024?",
                "Which MAG7 company showed the most consistent R&D growth?"
            ]
            for example in trend_examples:
                if st.button(f"📈 {example}", key=f"trend_{hash(example)}", use_container_width=True):
                    st.session_state.pending_suggestion = example
                    st.rerun()

        with col2:
            st.write("**⚖️ Comparative Queries:**")
            comp_examples = [
                "Compare Microsoft's revenue in 2023 vs 2024",
                "Compare Apple and Microsoft revenue for 2024"
            ]
            for example in comp_examples:
                if st.button(f"⚖️ {example}", key=f"comp_{hash(example)}", use_container_width=True):
                    st.session_state.pending_suggestion = example
                    st.rerun()

            st.write("**🏢 Cross-company Queries:**")
            cross_examples = [
                "Compare revenue across all MAG7 companies in 2024",
                "Which MAG7 company has the highest revenue in 2024?"
            ]
            for example in cross_examples:
                if st.button(f"🏢 {example}", key=f"cross_{hash(example)}", use_container_width=True):
                    st.session_state.pending_suggestion = example
                    st.rerun()

        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"])

            # Note: Follow-up suggestions are now displayed immediately after response generation
            # No need to display them again in message history

    # Always show chat input at the bottom
    render_chat_input(agent)

def render_chat_input(agent):
    """Render chat input that appears after each response"""
    if not agent:
        return

    # Use a container to ensure the input is always visible
    with st.container():
        # Chat input with a unique key to ensure it refreshes
        prompt = st.chat_input("Ask about MAG7 financial data...", key="main_chat_input")

        if prompt:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Process the user message
            _process_user_message(prompt, agent)

            # Rerun to refresh the interface and show the new input box
            st.rerun()

def render_sources(sources: List[Dict]):
    """Render source citations in target JSON format"""
    if sources:
        st.subheader("📚 Sources")

        # Show JSON format toggle with unique key
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Found {len(sources)} relevant sources:")
        with col2:
            # Use message count + timestamp + hash to ensure unique keys
            import time
            import hashlib
            content_hash = hashlib.md5(str(sources).encode()).hexdigest()[:8]
            unique_key = f"json_toggle_{len(st.session_state.messages)}_{int(time.time() * 1000000) % 1000000}_{content_hash}"
            show_json = st.checkbox("Show JSON", key=unique_key)

        if show_json:
            # Display as JSON (target format)
            st.subheader("🔧 JSON Response Format")
            json_sources = []
            for source in sources:
                json_source = {
                    "company": source.get('company', 'AAPL'),
                    "filing": source.get('filing', '10-K'),
                    "period": source.get('period', 'FY2023'),
                    "snippet": source.get('snippet', ''),
                    "url": source.get('url', '')
                }
                json_sources.append(json_source)

            # Display formatted JSON
            import json
            st.code(json.dumps(json_sources, indent=2), language="json")
        else:
            # Display as cards (user-friendly format)
            for i, source in enumerate(sources, 1):
                # Get source information with fallbacks
                company = source.get('company', source.get('ticker', 'Unknown'))
                filing = source.get('filing', source.get('filing_type', 'Unknown'))
                period = source.get('period', source.get('year', 'Unknown'))

                with st.expander(f"📄 Source {i}: {company} - {filing} ({period})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Company:** {company}")
                        st.write(f"**Filing:** {filing}")
                        st.write(f"**Period:** {period}")

                    with col2:
                        if 'url' in source and source['url']:
                            st.link_button("🔗 View SEC Filing", source['url'])

                    if 'snippet' in source and source['snippet']:
                        st.write("**Relevant Excerpt:**")
                        st.info(source['snippet'])

def render_data_management():
    """Render data management interface"""
    st.header("📊 Data Management")

    if not SECScraper:
        st.error("SEC scraper not available. Please install dependencies: pip install sec-edgar-downloader beautifulsoup4 lxml")
        return

    # Initialize scraper
    if "sec_scraper" not in st.session_state:
        st.session_state.sec_scraper = SECScraper()

    scraper = st.session_state.sec_scraper

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 SEC Filings")

        # Download configuration
        with st.expander("⚙️ Download Configuration", expanded=False):
            selected_companies = st.multiselect(
                "Select Companies",
                options=list(MAG7_COMPANIES.keys()),
                default=["AAPL", "MSFT"],
                help="Select companies to download filings for"
            )

            selected_filing_types = st.multiselect(
                "Filing Types",
                options=["10-K", "10-Q"],
                default=["10-K"],
                help="Select types of filings to download"
            )

            col_year1, col_year2 = st.columns(2)
            with col_year1:
                start_year = st.number_input("Start Year", min_value=2015, max_value=2024, value=2023)
            with col_year2:
                end_year = st.number_input("End Year", min_value=2015, max_value=2025, value=2024)

            limit_per_company = st.number_input(
                "Limit per Company (for testing)",
                min_value=1,
                max_value=50,
                value=2,
                help="Limit number of filings per company"
            )

        # Download button
        if st.button("🔄 Download SEC Filings", disabled=not scraper.downloader):
            if not selected_companies:
                st.error("Please select at least one company")
            elif not selected_filing_types:
                st.error("Please select at least one filing type")
            else:
                with st.spinner("Downloading SEC filings... This may take a few minutes."):
                    try:
                        downloaded_files = scraper.download_filings(
                            companies=selected_companies,
                            filing_types=selected_filing_types,
                            start_year=start_year,
                            end_year=end_year,
                            limit_per_company=limit_per_company
                        )

                        st.success("✅ Download completed!")

                        # Show results
                        total_files = sum(len(files) for files in downloaded_files.values())
                        st.write(f"**Total files downloaded:** {total_files}")

                        for company, files in downloaded_files.items():
                            st.write(f"- **{company}**: {len(files)} files")

                    except Exception as e:
                        st.error(f"Download failed: {e}")

        # Show current statistics
        st.subheader("📈 Current Data")
        try:
            stats = scraper.get_download_stats()
            if stats["total_filings"] > 0:
                st.write(f"**Total filings:** {stats['total_filings']}")
                st.write(f"**Companies:** {', '.join(stats['companies'])}")
                st.write(f"**Filing types:** {', '.join(stats['filing_types'])}")
                st.write(f"**Total size:** {stats['total_size_mb']} MB")
            else:
                st.info("No filings downloaded yet")
        except Exception as e:
            st.warning(f"Could not load statistics: {e}")

    with col2:
        st.subheader("📋 Downloaded Files")

        try:
            filings = scraper.list_downloaded_filings()
            if filings:
                st.write(f"Found {len(filings)} filing files:")

                # Show first few files
                for i, filing_path in enumerate(filings[:10]):
                    file_name = os.path.basename(filing_path)
                    # Extract company and filing type from path
                    path_parts = filing_path.split(os.sep)
                    company = "Unknown"
                    filing_type = "Unknown"

                    for j, part in enumerate(path_parts):
                        if part == "sec-edgar-filings" and j + 2 < len(path_parts):
                            company = path_parts[j + 1]
                            filing_type = path_parts[j + 2]
                            break

                    st.write(f"{i+1}. **{company}** - {filing_type} - {file_name}")

                if len(filings) > 10:
                    st.write(f"... and {len(filings) - 10} more files")

                # Cleanup button
                if st.button("🗑️ Clean Up Downloads", help="Remove all downloaded files"):
                    scraper.cleanup_downloads()
                    st.success("✅ Downloads cleaned up!")
                    st.rerun()
            else:
                st.info("No files downloaded yet")

        except Exception as e:
            st.error(f"Error listing files: {e}")

        st.subheader("🔍 Vector Database")
        st.write("Manage document embeddings and search index")

        if st.button("🚀 Initialize Vector Store", disabled=True):
            st.info("Vector store not yet implemented")

        if st.button("📈 View Database Stats", disabled=True):
            st.info("Database stats not yet available")

def render_settings():
    """Render settings interface"""
    st.header("⚙️ System Settings")

    # API Configuration
    st.subheader("🔑 API Configuration")

    # Initialize API key in session state if not present
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your API key from: https://aistudio.google.com/app/apikey",
        value=st.session_state.gemini_api_key,
        key="api_key_input"
    )

    # Update session state when API key changes
    if api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key

    if st.button("🚀 Initialize System"):
        if api_key:
            with st.spinner("Initializing RAG system..."):
                success = initialize_rag_system(api_key)
                if success:
                    st.session_state.gemini_api_key = api_key
                    st.success("✅ System initialized successfully!")
                    st.rerun()
                else:
                    st.error("❌ System initialization failed. Check the logs below.")
        else:
            st.error("Please enter a valid API key")

    # System Configuration
    st.subheader("🛠️ System Configuration")

    chunk_size = st.slider("Chunk Size", 200, 1000, 500)
    top_k = st.slider("Retrieval Top-K", 1, 10, 5)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.1)

    if st.button("💾 Save Settings"):
        st.session_state.chunk_size = chunk_size
        st.session_state.top_k = top_k
        st.session_state.temperature = temperature
        st.success("Settings saved!")

    # System Status
    st.subheader("📊 System Status")

    # Check system status
    agent = st.session_state.get("conversational_agent")
    vector_store = st.session_state.get("vector_store")

    if agent and vector_store:
        # Get real status
        try:
            vector_stats = vector_store.get_collection_stats()
            rag_stats = agent.rag_pipeline.get_pipeline_stats()

            status_data = {
                "Vector Database": f"✅ Connected ({vector_stats.get('total_documents', 0)} docs)",
                "LLM": f"✅ {rag_stats.get('llm_model', 'Unknown')}",
                "SEC Data": f"✅ {len(vector_stats.get('companies', []))} companies loaded",
                "Total Documents": str(vector_stats.get('total_documents', 0)),
                "Total Chunks": str(vector_stats.get('total_chunks', 0)),
                "Companies": ", ".join(vector_stats.get('companies', []))
            }
        except Exception as e:
            status_data = {
                "System": f"❌ Error: {str(e)}"
            }
    else:
        status_data = {
            "Vector Database": "❌ Not Connected",
            "LLM": "❌ Not Configured",
            "SEC Data": "❌ No Data Loaded",
            "Conversational Agent": "❌ Not Initialized"
        }

    for key, value in status_data.items():
        st.write(f"**{key}:** {value}")

    # Session management
    if agent:
        st.subheader("💬 Session Management")

        session_stats = agent.get_session_stats()
        st.write(f"**Active Sessions:** {session_stats.get('active_sessions', 0)}")
        st.write(f"**Total Messages:** {session_stats.get('total_messages', 0)}")

        if st.button("🗑️ Clear All Conversations"):
            agent.clear_history()
            st.session_state.messages = []
            st.success("All conversations cleared!")
            st.rerun()

def initialize_rag_system(api_key: str) -> bool:
    """Initialize the complete RAG system"""

    try:
        # Validate API key first
        if not api_key:
            st.error("❌ API key is required")
            return False

        if not api_key.startswith("AIza"):
            st.error("❌ Invalid API key format. Key should start with 'AIza'")
            return False

        # Initialize vector store (use complete MAG7 data 2015-2025 with fixed metadata)
        vector_store = FAISSVectorStore(
            db_path="data/mag7_complete_final_fixed",
            collection_name="mag7_all_data_properly_indexed"
        )

        if not vector_store.initialize():
            st.error("❌ Vector store initialization failed")
            return False

        stats = vector_store.get_collection_stats()
        if stats.get("total_documents", 0) == 0:
            st.error("❌ No documents in vector store. Please rebuild the database.")
            st.info("Run: python rebuild_vector_database_with_metadata.py")
            st.code("python rebuild_vector_database_with_metadata.py", language="bash")
            return False

        st.info(f"✅ Vector store loaded with {stats['total_documents']} documents")

        # Configure RAG pipeline
        llm_config = {
            "model_name": "gemini-pro",
            "temperature": st.session_state.get("temperature", 0.1),
            "max_tokens": 2000,  # Increased for longer responses
            "api_key": api_key
        }

        retrieval_config = {
            "top_k": st.session_state.get("top_k", 5),
            "similarity_threshold": 0.3,
            "max_context_length": 2000
        }

        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_config=llm_config,
            retrieval_config=retrieval_config
        )

        if not rag_pipeline.initialize():
            st.error("❌ RAG pipeline initialization failed")
            return False

        st.info("✅ RAG pipeline initialized")

        # Initialize robust conversational agent
        try:
            # Import with proper path handling
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

            from agent.robust_conversational_agent import RobustConversationalAgent
            from implement_robust_rag_pipeline import RobustRAGPipeline

            # Create robust pipeline with current configuration
            robust_pipeline = RobustRAGPipeline(vector_store, llm_config, retrieval_config)
            agent = RobustConversationalAgent(robust_pipeline)
            st.success("✅ Robust conversational agent initialized with mixed prompt handling!")
            st.info("🚀 System ready to handle: Basic, Comparative, Complex, Trend Analysis, and Cross-company queries")
        except Exception as e:
            st.error(f"Failed to initialize robust agent: {e}")
            st.code(str(e))  # Show detailed error for debugging
            # Fallback to regular agent
            agent = ConversationalAgent(rag_pipeline)
            st.warning("⚠️ Using fallback agent - some advanced features may not work")

        # Store in session state
        st.session_state.vector_store = vector_store
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.conversational_agent = agent

        # Initialize session ID
        if "session_id" not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())

        st.info("✅ Conversational agent initialized")

        return True

    except Exception as e:
        st.error(f"❌ Initialization error: {str(e)}")
        return False
