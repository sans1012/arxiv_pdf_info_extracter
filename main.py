import os
import logging
import requests
import fitz
import asyncio
import json
import httpx
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()


class URLRequest(BaseModel):
    url: str


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI backend is running!"}


@app.post("/summarize_arxiv/")
async def summarize_arxiv(request: URLRequest):
    """Downloads an Arxiv PDF, extracts text, and summarizes it using Ollama (Gemma 3) in parallel."""
    try:
        url = request.url
        logger.info("---------------------------------------------------------")
        logger.info(f"Downloading PDF from: {url}")

        pdf_path = download_pdf(url)
        if not pdf_path:
            return {"error": "Failed to download PDF. Check the URL."}

        logger.info(f"PDF saved at: {pdf_path}")

        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "No text extracted from PDF"}

        logger.info(f"Extracted text length: {len(text)} characters")
        logger.info("---------------------------------------------------------")

        # Summarize extracted text in parallel
        summary = await summarize_text_parallel(text)
        logger.info("Summarization complete")

        return {"summary": summary}

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {"error": "Failed to process PDF"}


def download_pdf(url):
    """Downloads a PDF from a given URL and saves it locally."""
    try:
        if not url.startswith("https://arxiv.org/pdf/"):
            logger.error(f"Invalid URL: {url}")
            return None  # Prevents downloading non-Arxiv PDFs

        response = requests.get(url, timeout=30)  # Set timeout to prevent long waits
        if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
            pdf_filename = "arxiv_paper.pdf"
            with open(pdf_filename, "wb") as f:
                f.write(response.content)
            return pdf_filename
        else:
            logger.error(f"Failed to download PDF: {response.status_code} (Not a valid PDF)")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {e}")
        return None


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF (faster than Unstructured PDFLoader)."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

async def summarize_chunk_with_retry(chunk, chunk_id, total_chunks, max_retries=2):
    """Retry mechanism wrapper for summarize_chunk_wrapper."""
    retries = 0
    while retries <= max_retries:
        try:
            if retries > 0:
                logger.info(f"🔄 Retry attempt {retries}/{max_retries} for chunk {chunk_id}/{total_chunks}")
            
            result = await summarize_chunk_wrapper(chunk, chunk_id, total_chunks)
            
            # If the result starts with "Error", it means there was an error but no exception was thrown
            if isinstance(result, str) and result.startswith("Error"):
                logger.warning(f"⚠️ Soft error on attempt {retries+1}/{max_retries+1} for chunk {chunk_id}: {result}")
                retries += 1
                if retries <= max_retries:
                    # Exponential backoff: 5s, 10s, 20s, etc.
                    wait_time = 5 * (2 ** (retries - 1))
                    logger.info(f"⏳ Waiting {wait_time}s before retry for chunk {chunk_id}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ All retry attempts failed for chunk {chunk_id}")
                    return result
            else:
                # Success
                if retries > 0:
                    logger.info(f"✅ Successfully processed chunk {chunk_id} after {retries} retries")
                return result
                
        except Exception as e:
            retries += 1
            logger.error(f"❌ Exception on attempt {retries}/{max_retries+1} for chunk {chunk_id}: {str(e)}")
            
            if retries <= max_retries:
                # Exponential backoff
                wait_time = 5 * (2 ** (retries - 1))
                logger.info(f"⏳ Waiting {wait_time}s before retry for chunk {chunk_id}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"❌ All retry attempts exhausted for chunk {chunk_id}")
                return f"Error processing chunk {chunk_id} after {max_retries+1} attempts: {str(e)}"
    
    # This should never be reached, but just in case
    return f"Error: Unexpected end of retry loop for chunk {chunk_id}"



async def summarize_text_parallel(text):
    """Splits text into smaller chunks, processes them in parallel, then merges with original text grounding."""
    token_estimate = len(text) // 4
    logger.info(f"📝 Token Estimate: {token_estimate}")

    # Smaller chunks with overlap to preserve context
    chunk_size = 2500 * 4   # ~8K tokens
    chunk_overlap = 500

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    logger.info(f"📚 Split into {len(chunks)} chunks")

    tasks = [summarize_chunk_with_retry(chunk, i + 1, len(chunks), max_retries=2)
             for i, chunk in enumerate(chunks)]
    summaries = await asyncio.gather(*tasks, return_exceptions=True)
    summaries = [s if isinstance(s, str) else f"Error: {s}" for s in summaries]

    successful_summaries = [s for s in summaries if not s.startswith("Error")]
    if not successful_summaries:
        return "No meaningful summary could be generated."

    combined_chunk_summaries = "\n\n".join(
        f"Chunk {i+1}:\n{s}" for i, s in enumerate(summaries)
    )

    # Grounding: Use a limited slice of the original text to avoid RAM blow-up
    grounding_text = text[:6000]

    logger.info("🔄 Generating grounded final summary...")

    final_messages = [
        {
            "role": "system",
            "content": (
                "You are a technical documentation writer.\n"
                "Cross-check every fact against the ORIGINAL TEXT provided.\n"
                "Rules:\n"
                "- Use only information present in the original text.\n"
                "- If something in chunk summaries is not supported by the original text, remove it.\n"
                "- If information is missing, write exactly 'Not mentioned'.\n"
                "Output format:\n"
                "1. System Architecture\n"
                "2. Technical Implementation\n"
                "3. Infrastructure & Setup\n"
                "4. Performance Analysis\n"
                "5. Optimization Techniques"
            )
        },
        {
            "role": "user",
            "content": f"Original text excerpt:\n{grounding_text}\n\nChunk summaries:\n{combined_chunk_summaries}"
        }
    ]

    payload = {
        "model": "gemma2:2b",
        "messages": final_messages,
        "stream": False
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=httpx.Timeout(connect=60, read=3600, write=60, pool=60)
            )
            if response.status_code != 200:
                return f"Error generating final summary: {response.status_code}"
            return response.json().get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"❌ Final summary generation failed: {e}")
        return "Final summary generation failed."
    

async def summarize_chunk_wrapper(chunk, chunk_id, total_chunks):
    """Asynchronous wrapper for summarizing a single chunk using Ollama via async httpx."""
    logger.info("---------------------------------------------------------")
    logger.info(f"🎯 Starting processing of chunk {chunk_id}/{total_chunks}")
    try:
        # Add system message to better control output
        messages = [
            {"role": "system", "content": """"Extract all technical details in maximum depth.\n"
    "- Provide detailed explanations for methods.\n"
    "- Expand abbreviations.\n"
    "- Include equations, pseudocode, and parameter values if present.\n"
    "- Use the exact terminology from the text.\n"
    "- No speculation."""},
            {"role": "user", "content": f"Extract technical content: {chunk}"}
        ]
        
        # Use httpx for truly parallel API calls
        payload = {
            "model": "gemma2:2b",
            "messages": messages,
            "stream": False
        }
        
        # Add better timeout and error handling
        try:
            # Make async HTTP request directly to Ollama API
            async with httpx.AsyncClient(timeout=3600) as client:  # Increased timeout to 10 minutes
                logger.info(f"📤 Sending request for chunk {chunk_id}/{total_chunks} to Ollama API - Gemm23 ")
                response = await client.post(
                    "http://localhost:11434/api/chat",  # Default Ollama API endpoint
                    json=payload,
                    # Adding connection timeout and timeout parameters
                    timeout=httpx.Timeout(connect=60, read=3600, write=60, pool=60)
                )
                logger.info("---------------------------------------------------------")
                logger.info(f"📥 Received response for chunk {chunk_id}/{total_chunks}, status code: {response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return f"Error processing chunk {chunk_id}: API returned status code {response.status_code}"
                    
                response_data = response.json()
                summary = response_data['message']['content']
            
            logger.info(f"✅ Completed chunk {chunk_id}/{total_chunks}")
            logger.info(f"📑 Summary length: {len(summary)} characters")
            logger.info("---------------------------------------------------------")
            return summary
            
        except httpx.TimeoutException as te:
            error_msg = f"Timeout error for chunk {chunk_id}: {str(te)}"
            logger.error(error_msg)
            return f"Error in chunk {chunk_id}: Request timed out after 30 minutes. Consider increasing the timeout or reducing chunk size."
            
        except httpx.ConnectionError as ce:
            error_msg = f"Connection error for chunk {chunk_id}: {str(ce)}"
            logger.error(error_msg)
            return f"Error in chunk {chunk_id}: Could not connect to Ollama API. Check if Ollama is running correctly."
            
    except Exception as e:
        # Capture and log the full exception details
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"❌ Error processing chunk {chunk_id}: {str(e)}")
        logger.error(f"Traceback: {error_details}")
        return f"Error processing chunk {chunk_id}: {str(e)}"


# Keep this function as a reference or remove it as it's been replaced by summarize_chunk_wrapper
def summarize_chunk(chunk, chunk_id):
    """Summarizes a single chunk using Ollama (Gemma 3 LLM)."""
    logger.info(f"\n{'=' * 40} Processing Chunk {chunk_id} {'=' * 40}")
    logger.info(f"📄 Input chunk length: {len(chunk)} characters")

    prompt = f"""
    You are a technical content extractor. Extract and explain ONLY the technical details from this section.

    Focus on:
    1. **System Architecture** – Design, component interactions, algorithms, configurations.
    2. **Implementation** – Code/pseudocode, data structures, formulas (with explanations), parameter values.
    3. **Experiments** – Hardware (GPUs, RAM), software versions, dataset size, training hyperparameters.
    4. **Results** – Performance metrics (accuracy, latency, memory usage), comparisons.

    **Rules:**
    - NO citations, references, or related work.
    - NO mention of authors or external papers.
    - ONLY technical details, numbers, and implementations.

    Text to analyze:
    {chunk}
    """
    try:
        logger.info(f"🤖 Sending chunk {chunk_id} to Ollama...")
        response = ollama.chat(model="gemma2:2b", messages=[{"role": "user", "content": prompt}])
        summary = response['message']['content']
        logger.info(f"✅ Successfully processed chunk {chunk_id}")
        logger.info(f"📊 Summary length: {len(summary)} characters")
        print(summary)
        return summary
    except Exception as e:
        logger.error(f"❌ Error summarizing chunk {chunk_id}: {e}")
        return f"Error summarizing chunk {chunk_id}"


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")