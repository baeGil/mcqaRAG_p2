# RAG Multimodal System

A sophisticated Retrieval-Augmented Generation (RAG) system with multimodal support, hybrid search capabilities, and agentic behavior.

## Features

- 🔍 Hybrid Vector Search
  - HNSW index for fast approximate nearest neighbor search
  - BM25 sparse retrieval with term frequency support
  - Weighted combination of dense and sparse embeddings

- 🤖 Agentic RAG
  - Dynamic retrieval workflow with document grading
  - Automated source hint extraction
  - Iterative retrieval until finding relevant documents
  - Structured MCQ answer generation

- 💾 Vector Store Integration
  - Milvus as primary vector database
  - Multiple vector store support
  - Optimized HNSW parameters for better recall
  - Strong consistency guarantee

- 📚 Document Processing
  - Source normalization and filtering
  - Metadata extraction and tracking
  - Page-level granularity support
  - Hybrid chunking strategies

## Architecture

```
src/
├── agent.py      # Main RAG agent implementation
├── index.py      # Indexing and document processing
├── milvus_store.py   # Vector store management
├── config.py     # Configuration management
└── ui.py         # User interface components
```

## Prerequisites

- Python 3.8+
- Milvus 2.0+
- Docker & Docker Compose

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-multimodal.git
cd rag-multimodal
```

2. Start Milvus and related services:
```bash
docker-compose up -d
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
```

## Usage

### Index Documents

```python
from src.index import index_documents

# Index documents with hybrid chunking
index_documents("path/to/documents", chunk_size=500)
```

### Run RAG Agent

```python
from src.agent import AgenticRAG

# Initialize agent
agent = AgenticRAG()

# Run question answering
response = agent.run("What is the main theme in the document?")

# Run MCQ answering
response = agent.run_mcq(
    question="Which of these is correct?",
    options={"A": "Option 1", "B": "Option 2"}
)
```

### Batch Processing

```python
# Process multiple MCQs from CSV
results = agent.run_mcq_csv("questions.csv")
```

## Configuration

Key configuration options in `config.yaml`:

```yaml
model:
  text_generation: "qwen2.5:latest"  # LLM model
  embeddings: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

database:
  uri: "http://localhost:19530"
  name: "gil"
  collection_name: "multimodal_rag"

retrieval:
  k: 3  # Number of documents to retrieve
  weights: [0.6, 0.4]  # Dense vs sparse weights
```

## Vector Store Parameters

The system uses optimized HNSW parameters:

- `M`: 48 (Graph connectivity)
- `efConstruction`: 512 (Index build quality)
- `ef`: max(k*4, 128) (Search quality)
- BM25 with term frequency enabled

## Acknowledgments

- Milvus for vector database
- LangChain for LLM framework
- LangGraph for workflow management
- Sentence Transformers for embeddings
