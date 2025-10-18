"""
Simplified indexing module with HNSW support for Milvus.
This module provides functionality to process files in a directory and index them 
to a vector store using HNSW indexing for optimal performance.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Union, Optional

from src.config import config
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling.chunking import HybridChunker
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer, MarkdownParams
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document
from transformers import AutoTokenizer

from src.milvus_store import MilvusStore

# Configure logging
logger = logging.getLogger(__name__)

# Default log file path
DEFAULT_LOG_FILE = os.path.join('logs', 'index.log')

def setup_logging(level=logging.INFO, log_file=None):
    """Configure logging for the index module."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:  
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file}: {e}")

# Configure logging with default settings
setup_logging(log_file=DEFAULT_LOG_FILE)

def get_document_converter(pdf_pipeline_options: Optional[PdfPipelineOptions] = None) -> DocumentConverter:
    """Create and configure a document converter."""
    pdf_pipeline_options = pdf_pipeline_options or config.get_pdf_pipeline_options()
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
        }
    )

def get_chunker(
    tokenizer_model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    image_mode: Optional[ImageRefMode] = None,
    image_placeholder: Optional[str] = None,
    mark_annotations: Optional[bool] = None,
    include_annotations: Optional[bool] = None
) -> HybridChunker:
    """Create and configure a document chunker."""
    # Use config defaults
    model_id = tokenizer_model_id or config.get('model', 'tokenizer', default='sentence-transformers/all-MiniLM-L6-v2')
    tokens = max_tokens or config.get('document', 'max_tokens', default=512)
    img_mode = image_mode if image_mode is not None else ImageRefMode.PLACEHOLDER
    img_placeholder = image_placeholder if image_placeholder is not None else ""
    mark_annot = mark_annotations if mark_annotations is not None else True
    include_annot = include_annotations if include_annotations is not None else True
    
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        max_tokens=tokens,
    )
    
    class CustomMDSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(
                    image_mode=img_mode,
                    image_placeholder=img_placeholder,
                    mark_annotations=mark_annot,
                    include_annotations=include_annot
                )
            )
    
    return HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=CustomMDSerializerProvider(),
    )

def process_file(
    file_path: Union[str, Path], 
    converter: DocumentConverter, 
    chunker: HybridChunker
) -> List[Document]:
    """Process a single file and prepare documents for indexing."""
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Create document loader
    loader = DoclingLoader(
        file_path=file_path,
        converter=converter,
        chunker=chunker,
        export_type=ExportType.DOC_CHUNKS
    )
    
    # Load and process documents
    docs = loader.load()
    
    # Prepare documents for indexing (simplified - removed namespace)
    processed_docs = []
    for doc in docs:
        metadata = doc.metadata
        _metadata = dict()
        
        # Original source path string
        _metadata["source"] = str(metadata["source"])
        
        # Normalized source to improve matching (e.g., Public001.pdf vs Public_001)
        try:
            src_name = Path(_metadata["source"]).stem
        except Exception:
            src_name = str(_metadata["source"]).rsplit('.', 1)[0]
        
        src_norm = re.sub(r"[^a-z0-9]", "", src_name.lower())
        _metadata["source_normalized"] = src_norm
        
        # Page number
        _metadata["page_no"] = metadata["dl_meta"]['doc_items'][0]['prov'][0]['page_no']
        
        processed_doc = Document(
            page_content=doc.page_content,
            metadata=_metadata
        )
        processed_docs.append(processed_doc)
    
    return processed_docs

def process_and_index_directory(
    directory_path: Union[str, Path],
    drop_existing: bool = False,
    file_extensions: List[str] = None,
    uri: str = None,
    db_name: str = None,
    collection_name: str = None,
    embed_model: str = None,
    config: Optional[object] = None,
) -> None:
    """
    Process all files in a directory and index them to a vector store with HNSW indexing.
    
    Args:
        directory_path: Path to the directory containing files to process
        drop_existing: Whether to drop the existing collection if it exists
        file_extensions: List of file extensions to process (defaults to [".pdf"])
        uri: Milvus URI (defaults to config)
        db_name: Database name (defaults to config)
        collection_name: Collection name (defaults to config)
        embed_model: Embedding model (defaults to config)
    """
    directory_path = Path(directory_path) if isinstance(directory_path, str) else directory_path
    # Use provided config or fall back to module-level config
    config_to_use = config or globals().get('config')

    file_extensions = file_extensions or config_to_use.get("document", "supported_file_types", default=[".pdf"]) if config_to_use else (file_extensions or [".pdf"])
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")
    
    # Get all files in the directory with specified extensions
    files = []
    for ext in file_extensions:
        files.extend(directory_path.glob(f"*{ext}"))
    
    if not files:
        logger.warning(f"No files with extensions {file_extensions} found in {directory_path}")
        return
    
    logger.info(f"Found {len(files)} files to process with HNSW indexing")
    
    # Create document converter and chunker using provided or global config
    pdf_pipeline_options = config_to_use.get_pdf_pipeline_options() if config_to_use else None
    converter = get_document_converter(pdf_pipeline_options=pdf_pipeline_options)
    chunker = get_chunker()
    
    # Create MilvusStore with HNSW indexing
    # Resolve MilvusStore params from config if not provided
    uri = uri or (config_to_use.get("database", "uri") if config_to_use else uri)
    db_name = db_name or (config_to_use.get("database", "name") if config_to_use else db_name)
    collection_name = collection_name or (config_to_use.get("database", "collection_name") if config_to_use else collection_name)
    embed_model = embed_model or (config_to_use.get("model", "embeddings") if config_to_use else embed_model)

    milvus_store = MilvusStore(
        uri=uri,
        db_name=db_name,
        collection_name=collection_name,
        embed_model=embed_model,
        drop_old=drop_existing
    )
    
    # Process and index each file
    all_docs = []
    for file in files:
        if file.name == '.DS_Store':
            continue
        
        logger.info(f"Processing {file}...")
        try:
            docs = process_file(file, converter, chunker)
            all_docs.extend(docs)
            logger.info(f"Processed {file}: {len(docs)} chunks")
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
    
    # Index all documents with HNSW
    if all_docs:
        logger.info(f"Indexing {len(all_docs)} documents with HNSW...")
        ids = milvus_store.add_documents(documents=all_docs)
        if ids:
            logger.info(f"Successfully indexed {len(ids)} documents with HNSW indexing")
        else:
            logger.error("Failed to index documents")
    else:
        logger.warning("No documents to index")

def set_log_level(level=logging.INFO, log_file=None):
    """Set the logging level for the index module."""
    setup_logging(level=level, log_file=log_file)
    logger.info(f"Log level set to: {logging.getLevelName(level)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and index files with HNSW")
    parser.add_argument("directory", help="Directory containing files to process")
    parser.add_argument("--drop", action="store_true", help="Drop existing collection")
    parser.add_argument("--extensions", nargs="+", default=[".pdf"], help="File extensions to process")
    
    args = parser.parse_args()
    
    process_and_index_directory(
        directory_path=args.directory,
        drop_existing=args.drop,
        file_extensions=args.extensions
    )