# 📄 Arxiv PDF Info Extracter

A lightweight tool to **download Arxiv PDFs, extract their text, and summarize them** using an LLM (via [Ollama](https://ollama.ai/)).  
Built with **FastAPI** for serving requests asynchronously.

---

## 🚀 Features
- Fetch and process Arxiv PDFs via URL
- Extract text using PyMuPDF (`fitz`)
- Summarize content using Ollama (default: Gemma-3 model)
- Async execution for speed
- REST API endpoint for easy integration

---

## 🛠️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/sans1012/arxiv_pdf_info_extracter.git
   cd arxiv_pdf_info_extracter
   
2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   source venv/bin/activate # Linux / Mac

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4. **Install and run Ollama**
   ```bash
   ollama pull gemma2:2b

5. **To run the backend.**
   ```bash
   unicorn main:app --host 0.0.0.0 --port 8000 --reload

6. **To run frontend**
    ```bash
   streamlit run frontend.py 
