#!/usr/bin/env python3
"""
MAG7 Financial Intelligence Q&A System - Setup Validation
Validates that the system is properly configured and ready to run.
"""

import sys
import os
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required, found {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'faiss-cpu',
        'sentence-transformers',
        'google-generativeai',
        'python-dotenv',
        'sec-edgar-downloader',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'faiss-cpu':
                import faiss
            elif package == 'google-generativeai':
                import google.generativeai
            elif package == 'sentence-transformers':
                import sentence_transformers
            elif package == 'sec-edgar-downloader':
                import sec_edgar_downloader
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_project_structure():
    """Check if project structure is correct"""
    print("\n📁 Checking project structure...")
    
    required_files = [
        'app.py',
        'implement_robust_rag_pipeline.py',
        'requirements.txt',
        'README.md',
        '.env.example'
    ]
    
    required_dirs = [
        'src',
        'src/agent',
        'src/config', 
        'src/data',
        'src/rag',
        'src/ui',
        'src/vector_store',
        'data'
    ]
    
    missing_items = []
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Missing file: {file}")
            missing_items.append(file)
        else:
            print(f"✅ {file}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ Missing directory: {dir_path}")
            missing_items.append(dir_path)
        else:
            print(f"✅ {dir_path}/")
    
    if missing_items:
        print(f"\n❌ Missing items: {', '.join(missing_items)}")
        return False
    
    print("✅ Project structure is correct")
    return True

def check_environment():
    """Check environment configuration"""
    print("\n🔧 Checking environment configuration...")
    
    # Check if .env file exists
    if Path('.env').exists():
        print("✅ .env file found")
        
        # Check if API key is set
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here':
            print("✅ GEMINI_API_KEY is configured")
            return True
        else:
            print("⚠️ GEMINI_API_KEY not configured in .env")
            print("   Add your API key to .env file")
            return False
    else:
        print("⚠️ .env file not found")
        print("   Copy .env.example to .env and add your API key")
        return False

def check_imports():
    """Check if core modules can be imported"""
    print("\n🔍 Checking core module imports...")
    
    sys.path.append('src')
    
    modules_to_test = [
        ('config.settings', 'Configuration'),
        ('vector_store.faiss_store', 'Vector Store'),
        ('rag.pipeline', 'RAG Pipeline'),
        ('agent.conversational_agent', 'Conversational Agent'),
        ('ui.main_interface', 'UI Interface')
    ]
    
    import_errors = []
    
    for module_name, description in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {description} ({module_name})")
        except ImportError as e:
            print(f"❌ {description} ({module_name}) - {e}")
            import_errors.append(module_name)
    
    if import_errors:
        print(f"\n❌ Import errors in: {', '.join(import_errors)}")
        return False
    
    print("✅ All core modules can be imported")
    return True

def check_data_availability():
    """Check if data is available"""
    print("\n📊 Checking data availability...")
    
    vector_db_path = Path('data/mag7_complete_final_fixed')
    
    if vector_db_path.exists():
        print("✅ Vector database directory found")
        
        # Check if there are any files in the directory
        files = list(vector_db_path.glob('*'))
        if files:
            print(f"✅ Vector database contains {len(files)} files")
            return True
        else:
            print("⚠️ Vector database directory is empty")
            print("   Run the system initialization to download data")
            return False
    else:
        print("⚠️ Vector database not found")
        print("   Run the system initialization to create vector database")
        return False

def main():
    """Run all validation checks"""
    print("🎯 MAG7 Financial Intelligence Q&A System - Setup Validation")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Environment", check_environment),
        ("Module Imports", check_imports),
        ("Data Availability", check_data_availability)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 VALIDATION RESULTS: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 SYSTEM READY! You can run: streamlit run app.py")
        return True
    elif passed_checks >= total_checks - 2:
        print("⚠️ MOSTLY READY - Minor issues found, but system should work")
        print("🚀 Try running: streamlit run app.py")
        return True
    else:
        print("❌ SETUP INCOMPLETE - Please fix the issues above")
        print("\n📋 Quick fixes:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Copy .env.example to .env and add your Gemini API key")
        print("   3. Run the application: streamlit run app.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
