"""
Configuration settings for the MAG7 Financial Q&A System
"""

import os
from typing import Dict, List

# Application configuration
APP_CONFIG = {
    "app_name": "MAG7 Financial Intelligence Q&A",
    "version": "0.1.0",
    "description": "AI-powered analysis of SEC filings for Magnificent 7 tech stocks"
}

# MAG7 Companies
MAG7_COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation", 
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla Inc."
}

# SEC Filing configuration
SEC_CONFIG = {
    "filing_types": ["10-K", "10-Q"],
    "start_year": 2015,
    "end_year": 2025,
    "data_dir": "data/sec_filings",
    "processed_dir": "data/processed"
}

# Vector database configuration
VECTOR_CONFIG = {
    "db_path": "data/vector_db",
    "collection_name": "mag7_filings",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
    "chunk_overlap": 50
}

# LLM configuration
LLM_CONFIG = {
    "model_name": "gemini-1.5-flash",
    "temperature": 0.1,
    "max_tokens": 2048,
    "api_key_env": "GEMINI_API_KEY"
}

# Retrieval configuration
RETRIEVAL_CONFIG = {
    "top_k": 5,
    "similarity_threshold": 0.7,
    "max_context_length": 4000
}

def get_api_key() -> str:
    """Get Gemini API key from environment"""
    api_key = os.getenv(LLM_CONFIG["api_key_env"], "")
    if not api_key:
        # Try alternative environment variable names
        api_key = os.getenv("GOOGLE_API_KEY", "")
    return api_key

def get_secure_api_key() -> str:
    """Get API key with validation and security checks"""
    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Please set the GEMINI_API_KEY environment variable.\n"
            "You can get an API key from: https://aistudio.google.com/app/apikey"
        )

    # Basic validation
    if not api_key.startswith("AIza"):
        raise ValueError("Invalid Gemini API key format. Key should start with 'AIza'")

    if len(api_key) < 30:
        raise ValueError("API key appears to be too short. Please check your key.")

    return api_key

def validate_config() -> bool:
    """Validate configuration settings"""
    # Check if API key is set
    if not get_api_key():
        return False
    
    # Check if required directories exist or can be created
    for dir_path in [SEC_CONFIG["data_dir"], SEC_CONFIG["processed_dir"], VECTOR_CONFIG["db_path"]]:
        os.makedirs(dir_path, exist_ok=True)
    
    return True
