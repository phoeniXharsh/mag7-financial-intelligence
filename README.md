# 📊 MAG7 Financial Intelligence Q&A System

An AI-powered conversational system that analyzes SEC filings for the Magnificent 7 tech stocks (Apple, Microsoft, Amazon, Google, Meta, NVIDIA, Tesla) using advanced RAG (Retrieval-Augmented Generation) technology.

## ✨ Features

- **🤖 Conversational AI Interface** - Natural language Q&A with multi-step reasoning
- **📊 Real-time SEC Data Analysis** - Processes 10-K and 10-Q filings from 2015-2025
- **🔍 Smart Query Classification** - Handles Basic, Comparative, Complex, Trend, and Cross-company queries
- **📚 Source Attribution** - Every answer includes citations to specific SEC filings
- **💬 Context Awareness** - Maintains conversation history and resolves references
- **⚡ High Performance** - Fast responses with comprehensive financial data coverage

## 🏢 Supported Companies

| Ticker | Company | Sector |
|--------|---------|--------|
| **AAPL** | Apple Inc. | Technology |
| **MSFT** | Microsoft Corporation | Technology |
| **AMZN** | Amazon.com Inc. | Consumer Discretionary |
| **GOOGL** | Alphabet Inc. | Technology |
| **META** | Meta Platforms Inc. | Technology |
| **NVDA** | NVIDIA Corporation | Technology |
| **TSLA** | Tesla Inc. | Consumer Discretionary |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Gemini API key](https://aistudio.google.com/app/apikey) (free tier available)
- Stable internet connection (for downloading SEC filings on first run)
- ~2GB free disk space (for storing processed documents)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/mag7-financial-qa.git
cd mag7-financial-qa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env

# 4. Run the application
streamlit run app.py
```

### Setup & First Run

1. **Start the application**
   ```bash
   streamlit run app.py
   ```
   Open http://localhost:8501

2. **Initialize the system** (First time only)
   - Go to **Settings** tab
   - Enter your Gemini API key
   - Click **"Initialize System"**
   - **Wait for data download**: The system will automatically download and process SEC filings (this takes 5-10 minutes on first run)
   - You'll see progress updates as it downloads and processes documents

3. **Start using the system**
   - Once initialization is complete, switch to **Chat** tab
   - Start asking financial questions about MAG7 companies!

> **⚠️ Important**: The system needs to download ~1,600 SEC documents on first run. Make sure you have a stable internet connection and allow 5-10 minutes for initial setup.

## 💡 Example Queries

### Basic Queries
```
"What was Microsoft's revenue for fiscal year 2024?"
"What was Meta's revenue for Q1 2024?"
"What was Apple's net income in 2023?"
```

### Comparative Queries
```
"Compare Microsoft's revenue in 2023 vs 2024"
"Compare Apple and Microsoft revenue for 2024"
"How does Tesla's profit margin compare to NVIDIA's?"
```

### Complex Queries
```
"How did COVID-19 impact Amazon's cloud vs retail revenue?"
"What are Tesla's main risk factors mentioned in recent filings?"
"Analyze the impact of AI on Microsoft's business segments"
```

### Trend Analysis
```
"What is Microsoft's revenue trend from 2022 to 2024?"
"Which MAG7 company showed the most consistent R&D growth?"
"How has Apple's services revenue grown over time?"
```

### Cross-company Queries
```
"Compare revenue across all MAG7 companies in 2024"
"Which MAG7 company has the highest revenue in 2024?"
"Rank MAG7 companies by profit margins"
```

## 🏗️ Architecture

The system uses a modular RAG (Retrieval-Augmented Generation) architecture:

- **Data Ingestion**: Automated SEC filing scraper with intelligent document chunking
- **Vector Store**: FAISS-based semantic search with 1,600+ indexed documents
- **Conversational Agent**: Multi-step reasoning with context management
- **RAG Pipeline**: Robust query processing with specialized handlers
- **Web Interface**: Streamlit-based chat interface with real-time updates

## 📋 Response Format

```json
{
  "answer": "Microsoft reported revenue of $61.9B in Q1 FY2024...",
  "sources": [
    {
      "company": "MSFT",
      "filing": "10-Q",
      "period": "Q1 FY2024",
      "snippet": "Total revenue was $61.9 billion...",
      "url": "https://www.sec.gov/..."
    }
  ],
  "confidence": 0.95
}
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: FAISS
- **Embeddings**: sentence-transformers
- **LLM**: Google Gemini Flash 2.5
- **Data Source**: SEC EDGAR API
- **Language**: Python 3.8+

## 🔧 Key Features

### Smart Query Understanding
- Financial terminology recognition ("cloud revenue" → "AWS" for Amazon)
- Fiscal year handling across different companies
- Automatic entity extraction (companies, years, quarters)

### Conversation Context
- Maintains conversation history across sessions
- Resolves pronouns and references ("it", "that company", "the previous result")
- Provides intelligent follow-up suggestions

### Data Management
- Download additional SEC filings for any date range (e.g., 2010-2015)
- Real-time progress tracking and storage management
- Automatic data processing and indexing

## 📁 Project Structure

```
mag7-financial-qa/
├── app.py                              # Main Streamlit application
├── implement_robust_rag_pipeline.py    # Core RAG implementation
├── requirements.txt                     # Python dependencies
├── validate_setup.py                   # Setup validation script
├── .env.example                        # Environment template
├── src/
│   ├── agent/          # Conversational agents
│   ├── config/         # Configuration management
│   ├── data/           # Data ingestion pipeline
│   ├── rag/            # RAG pipeline components
│   ├── ui/             # Streamlit interface
│   └── vector_store/   # Vector database management
├── data/               # SEC filing data & vector stores
└── sec-edgar-filings/  # Downloaded SEC filings
```

## 🧪 Usage

1. **Start the application**: `streamlit run app.py`
2. **Initialize system**: Go to Settings tab, enter API key, click "Initialize System"
3. **Start chatting**: Switch to Chat tab and ask financial questions
4. **Explore data**: Use Data Management tab to download additional time periods

## 🔧 Troubleshooting

**Common Issues:**

**"Vector Database: Not Connected"**
- Solution: Go to Settings tab and click "Initialize System" with a valid Gemini API key
- Wait for the data download to complete (5-10 minutes on first run)

**"No data found for your query"**
- Cause: System hasn't downloaded data yet or initialization failed
- Solution: Check Data Management tab to see if filings are available
- Re-run initialization if needed

**Slow or failed initialization**
- Check your internet connection (needs to download ~1GB of SEC filings)
- Verify your Gemini API key is valid and has quota remaining
- Try restarting the application and re-initializing

**Slow responses during chat**
- Normal for complex cross-company queries (may take 5-10 seconds)
- Simple queries should respond in 3-5 seconds
- If consistently slow, check system resources

**Setup Validation:**
```bash
python validate_setup.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � Acknowledgments

- SEC EDGAR for providing public financial data
- Google Gemini for LLM capabilities
- Streamlit for the web framework
- FAISS for vector search

---

**Built for financial analysis and AI-powered insights** �
