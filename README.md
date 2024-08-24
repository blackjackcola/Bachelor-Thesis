# Bachelor Thesis Repository

## Overview

This repository contains materials related to my Bachelor thesis, including:

1. **Original Question and Answers:**
   - `40Questions_for_LLM.pdf`: Contains the original question provided by the author.
   - `40Answers_by_Author.pdf`: Contains the answers provided by the author.
   - `Answers_by_StableLLM12B.pdf`: Includes answers generated by the Language Model (LLM).

3. **Code:**
   - The code for downloading the LLM, ingesting data, and querying the LLM is provided in three separate Python files:
     - `LLM_download.py`: Script to download LLM to cache.
     - `ingestion_test_huggingface.py`: Script to embedd and ingest data into the vectore storage.
     - `query_test_huggingface.py`: Script to query the LLM with RAG and obtain results.

## Instructions

1. **Download the PDFs:**
   - To access the original question and answers, and the LLM-generated answers, download the corresponding PDF files from this repository.

2. **Run the Code:**
   - Ensure you have Python installed on your system.
   - Install the required libraries listed below.
   - Run the Python scripts in the following order:
     - `LLM_download.py`: Downloads the LLM.
     - `ingestion_test_huggingface.py`: Ingests the necessary data into the vectore store.
     - `query_test_huggingface.py`: Executes queries using the LLM with RAG.

## Dependencies

To run the provided scripts, install the following Python packages:

```bash
pip install requests beautifulsoup4 llama-index pydantic==1.10.11 llama-index-embeddings-huggingface black python-dotenv llama-hub unstructured callbacks llama-index-readers-web transformers sentence-transformers torch llama-index-llms-huggingface llama-index-readers-json huggingface-hub langchain-community llama-index-llms-langchain bitsandbytes accelerate
