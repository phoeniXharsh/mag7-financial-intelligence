"""
MAG7 Financial Intelligence Q&A System
AI-powered analysis of SEC filings for Magnificent 7 tech stocks
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from src.ui.main_interface import render_main_interface
    from src.config.settings import APP_CONFIG
except ImportError:
    APP_CONFIG = {"app_name": "MAG7 Financial Q&A", "version": "1.0.0"}

def render_dynamic_sidebar():
    """Render dynamic sidebar with real-time system status"""

    with st.sidebar:
        st.header("🔧 System Status")

        # Check if system is initialized
        agent = st.session_state.get("conversational_agent")
        vector_store = st.session_state.get("vector_store")

        # Application Status
        st.success("✅ Application Running")

        # Vector Database Status
        if vector_store:
            try:
                stats = vector_store.get_collection_stats()
                total_docs = stats.get('total_documents', 0)
                companies = stats.get('companies', [])
                filing_types = stats.get('filing_types', [])

                st.success(f"✅ Vector Database: {total_docs:,} documents")

                # Show data breakdown
                with st.expander("📊 Data Details", expanded=False):
                    st.write(f"**Companies**: {len(companies)}")
                    st.write(f"**Filing Types**: {', '.join(filing_types)}")
                    st.write(f"**Total Documents**: {total_docs:,}")

                    # Show company breakdown
                    if companies:
                        st.write("**Company Coverage**:")
                        for company in companies[:7]:  # Show first 7
                            st.write(f"  • {company}")
                        if len(companies) > 7:
                            st.write(f"  • ... and {len(companies) - 7} more")

            except Exception as e:
                st.error(f"❌ Vector Database: Error ({str(e)[:30]}...)")
        else:
            st.warning("⚠️ Vector Database: Not Connected")

        # LLM Status
        if agent:
            try:
                # Check if agent has pipeline stats
                if hasattr(agent, 'rag_pipeline') and hasattr(agent.rag_pipeline, 'get_pipeline_stats'):
                    pipeline_stats = agent.rag_pipeline.get_pipeline_stats()
                    model_name = pipeline_stats.get('llm_config', {}).get('model_name', 'Unknown')
                    st.success(f"✅ LLM: {model_name}")
                else:
                    st.success("✅ LLM: Configured")

                # Show conversation stats
                if hasattr(agent, 'conversation_history'):
                    total_conversations = len(agent.conversation_history)
                    if total_conversations > 0:
                        with st.expander("💬 Conversation Stats", expanded=False):
                            st.write(f"**Active Sessions**: {total_conversations}")
                            total_messages = sum(len(conv) for conv in agent.conversation_history.values())
                            st.write(f"**Total Messages**: {total_messages}")

            except Exception as e:
                st.warning(f"⚠️ LLM: Partially configured")
        else:
            st.warning("⚠️ LLM: Not Configured")

        # Data Management Status
        try:
            import sys
            sys.path.append('src')
            from data.sec_scraper import SECScraper

            scraper = SECScraper()
            download_stats = scraper.get_download_stats()

            if download_stats['total_filings'] > 0:
                st.success(f"✅ SEC Data: {download_stats['total_filings']} filings")

                with st.expander("📁 Downloaded Data", expanded=False):
                    st.write(f"**Total Size**: {download_stats['total_size_mb']:.1f} MB")
                    st.write(f"**Companies**: {', '.join(download_stats['companies'])}")
                    st.write(f"**Filing Types**: {', '.join(download_stats['filing_types'])}")
            else:
                st.info("ℹ️ SEC Data: No additional downloads")

        except Exception as e:
            st.info("ℹ️ SEC Data: Scraper not available")

        # Quick Actions
        st.header("⚡ Quick Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", help="Refresh system status"):
                st.rerun()
        with col2:
            if st.button("⚙️ Settings", help="Go to settings"):
                st.session_state.active_tab = "Settings"
                st.rerun()

        # MAG7 Companies with Status
        st.header("📋 MAG7 Companies")

        companies_info = {
            "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
            "MSFT": {"name": "Microsoft Corp.", "sector": "Technology"},
            "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
            "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
            "META": {"name": "Meta Platforms Inc.", "sector": "Technology"},
            "NVDA": {"name": "NVIDIA Corp.", "sector": "Technology"},
            "TSLA": {"name": "Tesla Inc.", "sector": "Consumer Discretionary"}
        }

        # Check which companies have data
        available_companies = []
        if vector_store:
            try:
                stats = vector_store.get_collection_stats()
                available_companies = stats.get('companies', [])
            except:
                pass

        for ticker, info in companies_info.items():
            status_icon = "✅" if ticker in available_companies else "⚪"
            with st.expander(f"{status_icon} {ticker} - {info['name']}", expanded=False):
                st.write(f"**Sector**: {info['sector']}")
                if ticker in available_companies:
                    st.write("**Status**: Data Available")
                else:
                    st.write("**Status**: No Data")

        # System Info
        st.header("ℹ️ System Info")

        with st.expander("📊 Performance", expanded=False):
            # Show memory usage if available
            try:
                import psutil
                memory = psutil.virtual_memory()
                st.write(f"**Memory Usage**: {memory.percent}%")
                st.write(f"**Available Memory**: {memory.available / (1024**3):.1f} GB")
            except:
                st.write("Memory info not available")

            # Show session info
            if 'messages' in st.session_state:
                st.write(f"**Session Messages**: {len(st.session_state.messages)}")

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title=APP_CONFIG.get("app_name", "MAG7 Financial Q&A"),
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">📊 MAG7 Financial Intelligence Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered analysis of SEC filings for Magnificent 7 tech stocks</div>', unsafe_allow_html=True)
    
    # Dynamic Sidebar
    render_dynamic_sidebar()
    
    # Main content area
    try:
        render_main_interface()
    except NameError:
        # Placeholder interface for initial setup
        st.header("🚀 Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Data Pipeline")
            st.write("1. Download SEC filings")
            st.write("2. Process and chunk documents")
            st.write("3. Create vector embeddings")
            st.write("4. Store in ChromaDB")
            
            if st.button("🔄 Initialize Data Pipeline", disabled=True):
                st.info("Data pipeline not yet implemented")
        
        with col2:
            st.subheader("💬 Query Interface")
            st.write("Ask questions about MAG7 companies:")
            
            query = st.text_area(
                "Enter your question:",
                placeholder="e.g., What was Microsoft's revenue for Q1 2024?",
                height=100
            )
            
            if st.button("🔍 Ask Question", disabled=True):
                st.info("RAG system not yet implemented")
        
        # Example queries
        st.header("💡 Example Queries")
        examples = [
            "What was Microsoft's revenue for Q1 2024?",
            "Compare YoY revenue growth for Google between Q1 2024 and Q1 2025",
            "How did COVID-19 impact Amazon's cloud vs retail revenue?",
            "Which MAG7 company showed the most consistent R&D investment growth?",
            "Compare operating margins across all MAG7 companies in 2023"
        ]
        
        for i, example in enumerate(examples, 1):
            st.write(f"{i}. {example}")

if __name__ == "__main__":
    main()
