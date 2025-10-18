"""
Agentic RAG with HNSW index support for Milvus.
Compatible with the provided index.py and milvus_store.py modules.
"""

import logging
import os
import time
import uuid
from typing import Dict, List, Literal, Any, Optional, Union
import re
import csv
import json

# Import configuration
from src.config import config

# Import LangGraph components
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

# Import LangChain components
from langchain_core.messages import convert_to_messages, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field

# Import Milvus store
from src.milvus_store import MilvusStore

# Configure logging
logger = logging.getLogger(__name__)
DEFAULT_LOG_FILE = 'logs/agent.log'

def configure_logging(level=logging.INFO, log_file=None):
    """Configure logging for the agent module."""
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
configure_logging(log_file=DEFAULT_LOG_FILE)

def generate_thread_id() -> str:
    """Generate a unique thread ID using timestamp and UUID."""
    timestamp = str(int(time.time() * 1000))
    random_uuid = str(uuid.uuid4()).replace('-', '')
    return f"{timestamp}_{random_uuid}"

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

class ExtractSourceHint(BaseModel):
    """Structured extraction of a potential document/file hint from the question."""
    source_hint: Optional[str] = Field(default=None, description="Document/file name if mentioned; otherwise null")

class MCQAnswer(BaseModel):
    """Structured answer for multiple-choice questions."""
    thinking: str = Field(description="Quá trình suy nghĩ chi tiết bằng tiếng Việt")
    rationale: str = Field(description="Giải thích ngắn gọn bằng tiếng Việt")
    final_answer: str = Field(description="Các đáp án đúng dạng chữ cái, ví dụ: 'A' hoặc 'A,B'")

class AgenticRAG:
    """
    Optimized Agentic RAG implementation with HNSW index support.
    
    This class implements an optimized agentic RAG system that:
    1. Uses HNSW indexing for fast vector search
    2. Supports hybrid search with BM25 and vector similarity
    3. Implements document relevance grading
    4. Handles multiple-choice questions efficiently
    5. Provides source filtering capabilities
    """
    
    def __init__(
        self,
        vector_stores: Optional[List[Dict[str, Any]]] = None,
        vector_store: Optional[MilvusStore] = None,
        model_name: str = None,
        temperature: float = 0,
        thread_id: Optional[str] = None,
        checkpointer: Optional[InMemorySaver] = None,
    ):
        """
        Initialize the OptimizedAgenticRAG system.
        
        Args:
            vector_stores: List of vector store configurations
            vector_store: Single MilvusStore instance for backward compatibility
            model_name: Name of the model to use
            temperature: Temperature for the model
            thread_id: Optional thread ID for the conversation
            checkpointer: Optional checkpointer for saving conversation state
        """
        self.model_name = model_name or config.get("model", "text_generation", default="qwen2.5:latest")
        self.temperature = temperature
        self.checkpointer = checkpointer or InMemorySaver()
        
        # Generate unique thread_id if not provided
        self.thread_id = thread_id or generate_thread_id()
        
        # Initialize vector stores and create retriever tools
        self.vector_stores = []
        self.retriever_tools = []
        
        self._initialize_vector_stores(vector_stores, vector_store)
        self._initialize_models()
        
        # Create the graph
        self.graph = self._build_graph()
        
        logger.info(f"OptimizedAgenticRAG initialized with model: {self.model_name} and {len(self.vector_stores)} vector store(s)")
    
    def _initialize_vector_stores(self, vector_stores, vector_store):
        """Initialize vector stores and create retriever tools."""
        if vector_stores:
            # Use the new multiple vector stores approach
            for vs_config in vector_stores:
                self._validate_and_add_vector_store(vs_config)
        elif vector_store:
            # Backward compatibility: single vector store
            self._add_single_vector_store(vector_store)
        else:
            # Default: create a single MilvusStore with HNSW
            default_store = MilvusStore()
            self._add_single_vector_store(default_store)
        
        # Maintain backward compatibility attributes
        self.vector_store = self.vector_stores[0]['store'] if self.vector_stores else None
        self.retriever = self.vector_stores[0]['retriever'] if self.vector_stores else None
        self.retriever_tool = self.retriever_tools[0] if self.retriever_tools else None
    
    def _validate_and_add_vector_store(self, vs_config):
        """Validate and add a vector store configuration."""
        if not isinstance(vs_config, dict):
            raise ValueError("Each vector store configuration must be a dictionary")
        
        required_keys = ['store', 'name', 'description']
        missing_keys = [key for key in required_keys if key not in vs_config]
        if missing_keys:
            raise ValueError(f"Vector store configuration missing required keys: {missing_keys}")
        
        store = vs_config['store']
        name = vs_config['name']
        description = vs_config['description']
        k = vs_config.get('k', config.get("retrieval", "k", default=3))
        ranker_weights = vs_config.get('ranker_weights', config.get("retrieval", "weights", default=[0.6, 0.4]))
        
        # Create retriever with HNSW optimized parameters
        retriever = store.as_retriever(
            k=k, 
            ranker_weights=ranker_weights,
            mmr=True,  # Use MMR for diversity
            fetch_k=k * 4  # Fetch more candidates for better results
        )
        
        # Create retriever tool
        retriever_tool = create_retriever_tool(retriever, name, description)
        
        # Store the configuration
        vs_data = {
            'store': store,
            'name': name,
            'description': description,
            'retriever': retriever,
            'tool': retriever_tool,
            'k': k,
            'ranker_weights': ranker_weights
        }
        
        self.vector_stores.append(vs_data)
        self.retriever_tools.append(retriever_tool)
    
    def _add_single_vector_store(self, vector_store):
        """Add a single vector store for backward compatibility."""
        k = config.get("retrieval", "k", default=3)
        ranker_weights = config.get("retrieval", "weights", default=[0.6, 0.4])
        
        retriever = vector_store.as_retriever(
            k=k,
            ranker_weights=ranker_weights,
            mmr=True,
            fetch_k=k * 4
        )
        
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_documents",
            "Search and retrieve information from the document collection using HNSW index."
        )
        
        vs_data = {
            'store': vector_store,
            'name': 'retrieve_documents',
            'description': 'Search and retrieve information from the document collection using HNSW index.',
            'retriever': retriever,
            'tool': retriever_tool,
            'k': k,
            'ranker_weights': ranker_weights
        }
        
        self.vector_stores.append(vs_data)
        self.retriever_tools.append(retriever_tool)
    
    def _initialize_models(self):
        """Initialize the LLM models."""
        base_url = config.get("model", "url", default="http://localhost:11434/v1")
        
        self.response_model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            base_url=base_url,
            api_key="ollama",
            top_p=0.1
        )
        
        self.grader_model = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            base_url=base_url,
            api_key="ollama",
            top_p=0.1
        )
    
    def _route_tools(self, state: MessagesState) -> str:
        """Custom routing function to route to the appropriate tool node or END."""
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            tool_name = ai_message.tool_calls[0]["name"]
            valid_tool_names = [vs['name'] for vs in self.vector_stores]
            
            if tool_name in valid_tool_names:
                return tool_name
            else:
                logger.warning(f"Unknown tool name: {tool_name}. Available tools: {valid_tool_names}")
                return END
        return END
    
    def _build_graph(self) -> StateGraph:
        """Build the optimized LangGraph for the agentic RAG system."""
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)
        
        # Add each retriever tool as an individual node
        retriever_node_names = []
        for vs_config in self.vector_stores:
            node_name = vs_config['name']
            retriever_node_names.append(node_name)
            workflow.add_node(node_name, ToolNode([vs_config['tool']]))
        
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")
        
        # Create tools mapping
        tools_mapping = {}
        for vs_config in self.vector_stores:
            tool_name = vs_config['name']
            tools_mapping[tool_name] = tool_name
        tools_mapping[END] = END
        
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            self._route_tools,
            tools_mapping,
        )
        
        # Grade documents after retrieval
        for node_name in retriever_node_names:
            workflow.add_conditional_edges(
                node_name,
                self._grade_documents,
            )
        
        workflow.add_edge("generate_answer", END)
        
        # Compile the graph
        graph = workflow.compile(checkpointer=self.checkpointer)
        
        # Save graph visualization
        try:
            output_file = "optimized_graph.png"
            graph.get_graph().draw_mermaid_png(output_file_path=output_file)
            logger.info(f"Graph visualization saved to {output_file}")
        except Exception as e:
            logger.warning(f"Could not save graph visualization: {e}")
        
        return graph
    
    def _extract_source_hint(self, question: str) -> tuple[Optional[str], Optional[str]]:
        """Extract source hint from question using both LLM and regex patterns."""
        source_hint = None
        source_hint_norm = None
        
        try:
            # Try LLM extraction first
            hint_prompt = (
                "Bạn là một bộ phân tích câu hỏi. Nếu trong câu hỏi có đề cập đến tên tài liệu hoặc tệp cụ thể "
                "(ví dụ: Public001, Public_001, Public001.pdf, bài báo Public003, tài liệu Public002), "
                "hãy trích xuất và trả về chính xác tên đó. Nếu không có, hãy trả về null.\n"
                f"Câu hỏi: {question}\n"
                "Tên tài liệu:"
            )
            
            hint_resp = (
                self.response_model
                .with_structured_output(ExtractSourceHint)
                .invoke([{"role": "user", "content": hint_prompt}])
            )
            
            if hint_resp and getattr(hint_resp, 'source_hint', None):
                raw_hint = str(hint_resp.source_hint).strip().strip('"\'')
                # Validate: should be short document name
                if len(raw_hint) < 50 and any(char.isdigit() for char in raw_hint):
                    source_hint = raw_hint
                    logger.info(f"Source hint extracted by LLM: {source_hint}")
        
        except Exception as e:
            logger.warning(f"LLM failed to extract source hint: {e}")
        
        # Fallback: regex patterns
        if not source_hint:
            patterns = [
                r"dựa vào tài liệu\s+([\w\-_. ]+)",
                r"theo\s+tài liệu\s+([\w\-_. ]+)",
                r"tài liệu\s+([\w\-_. ]+)",
                r"bài báo\s+([\w\-_. ]+)",
                r"trong\s+tài liệu\s+([\w\-_. ]+)",
                r"([A-Za-z]+\d+)",  # Catch patterns like Public002, Public003, etc.
                r"([A-Za-z]+_\d+)",  # Catch patterns like Public_002, etc.
            ]
            
            for pattern in patterns:
                match = re.search(pattern, question, flags=re.IGNORECASE)
                if match:
                    source_hint = match.group(1).strip().strip('"\'')
                    logger.info(f"Source hint extracted by regex: {source_hint}")
                    break
        
        # Normalize source hint
        if source_hint:
            base = source_hint.lower().replace('.pdf', '')
            source_hint_norm = re.sub(r"[^a-z0-9]", "", base)
            logger.info(f"Normalized source hint: {source_hint_norm}")
        
        return source_hint, source_hint_norm
    
    def _update_retriever_filters(self, source_hint_norm: Optional[str]):
        """Update retriever filters based on source hint."""
        if not self.vector_stores:
            return
        
        for vs in self.vector_stores:
            retriever = vs.get('retriever')
            if retriever and hasattr(retriever, 'search_kwargs'):
                if source_hint_norm:
                    # Apply source filter using HNSW optimized search
                    retriever.search_kwargs['expr'] = f'source_normalized == "{source_hint_norm}"'
                    # Optimize HNSW search parameters for filtered search
                    if 'param' not in retriever.search_kwargs:
                        retriever.search_kwargs['param'] = {}
                    retriever.search_kwargs['param']['ef'] = max(vs['k'], 64)  # HNSW ef parameter[(1)](https://milvus.io/docs/hnsw.md#Index-params)
                    logger.info(f"Applied source filter: {source_hint_norm} with HNSW ef={retriever.search_kwargs['param']['ef']}")
                else:
                    # Reset filters
                    retriever.search_kwargs.pop('expr', None)
                    if 'param' in retriever.search_kwargs:
                        retriever.search_kwargs['param']['ef'] = max(vs['k'], 32)  # Default HNSW ef[(1)](https://milvus.io/docs/hnsw.md#Index-params)
                    logger.info("Reset retriever filters")
    
    def _extract_mcq_options(self, content: str) -> Dict[str, str]:
        """Extract multiple choice options from content."""
        options = {}
        for key in ["A", "B", "C", "D"]:
            pattern = rf"\b{key}\s*[:\-]\s*(.+)"
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match:
                options[key] = match.group(1).strip()
        return options
    
    def _generate_query_or_respond(self, state: MessagesState) -> Dict:
        """Generate a response or call retrieval tools with HNSW optimization."""
        logger.debug("Generating query or response with HNSW optimization")
        logger.info("[trace] _generate_query_or_respond called")
        
        # Extract source hint from original question
        source_hint, source_hint_norm = None, None
        first_user_msg = None
        
        messages = state.get("messages", [])
        for msg in messages:
            role = getattr(msg, 'role', getattr(msg, 'type', None))
            if role in ["user", "human"]:
                first_user_msg = msg
                break
        
        if first_user_msg:
            content = getattr(first_user_msg, 'content', '')
            if content:
                source_hint, source_hint_norm = self._extract_source_hint(content)
        
        # Update retriever filters with HNSW optimization
        self._update_retriever_filters(source_hint_norm)
        
        # Extract MCQ options and create augmented messages
        augmented_messages = state["messages"]
        if first_user_msg:
            content = getattr(first_user_msg, 'content', '')
            options = self._extract_mcq_options(content)
            
            if options:
                logger.info(f"MCQ options detected: {list(options.keys())}")
                # Create combined query for better HNSW retrieval
                combined_parts = [f"Question: {content}"]
                opt_str = "; ".join([f"{k}) {v}" for k, v in options.items()])
                combined_parts.append(f"Options: {opt_str}")
                hint = "\n".join(combined_parts)
                
                system_msg = {
                    "role": "system", 
                    "content": "Use the combined query including question and options for optimal HNSW retrieval matching."
                }
                hint_msg = {
                    "role": "system", 
                    "content": f"Combined retrieval query:\n{hint}"
                }
                augmented_messages = [system_msg] + augmented_messages + [hint_msg]
        
        # Generate response with tool binding
        response = (
            self.response_model
            .bind_tools(self.retriever_tools)
            .invoke(augmented_messages)
        )
        
        # Force tool calling for MCQ RAG to ensure document retrieval
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            logger.info("No tool calls detected, forcing retrieval for MCQ...")
            if self.retriever_tools:
                tool_name = self.retriever_tools[0].name
                query_content = augmented_messages[-1].get("content", "retrieve documents") if augmented_messages else "retrieve documents"
                
                response.tool_calls = [{
                    "name": tool_name,
                    "args": {"query": query_content},
                    "id": "forced_mcq_call"
                }]
                logger.info(f"Forced tool call to: {tool_name}")
        
        return {"messages": [response]}
    
    def _grade_documents(self, state: MessagesState) -> Literal["generate_answer", "generate_query_or_respond"]:
        """Grade document relevance with optimized context extraction."""
        logger.debug("Grading retrieved documents")
        
        question = state["messages"][0].content
        context = self._extract_tool_context(state["messages"])
        
        if not context:
            logger.warning("No context found for grading") 
            return "generate_query_or_respond"
        
        grade_prompt = (
            "You are a grader assessing relevance of retrieved documents to a user question.\n"
            "Here is the retrieved document:\n\n{context}\n\n"
            "Here is the user question: {question}\n"
            "If the document contains keywords or semantic meaning related to the question, grade it as relevant.\n"
            "Give a binary score 'yes' or 'no' to indicate document relevance."
        )
        
        prompt = grade_prompt.format(question=question, context=context)
        
        try:
            response = (
                self.grader_model
                .with_structured_output(GradeDocuments)
                .invoke([{"role": "user", "content": prompt}])
            )
            score = response.binary_score
            logger.info(f"Document grade score: {score}")
            
            return "generate_answer" if score == "yes" else "generate_query_or_respond"
            
        except Exception as e:
            logger.error(f"Error grading documents: {e}")
            return "generate_answer"  # Default to generate answer
    
    def _extract_tool_context(self, messages: List[Any]) -> str:
        """Extract context from tool messages with detailed logging."""
        texts = []
        num_docs = 0
        num_tools = 0
        meta_summaries = []
        
        for msg in messages:
            # Detect tool messages
            role = getattr(msg, 'role', getattr(msg, 'type', None))
            is_tool = (role == "tool") or (getattr(msg, '__class__', type('X',(object,),{})).__name__ == "ToolMessage")
            
            if not is_tool:
                continue
            
            num_tools += 1
            content = getattr(msg, 'content', None)
            
            if isinstance(content, str):
                # Try to parse JSON content
                try:
                    parsed = json.loads(content.strip()) if content.strip().startswith('{') or content.strip().startswith('[') else None
                    if parsed:
                        if isinstance(parsed, list):
                            for item in parsed:
                                self._process_document_item(item, texts, meta_summaries)
                                num_docs += 1
                        elif isinstance(parsed, dict):
                            self._process_document_item(parsed, texts, meta_summaries)
                            num_docs += 1
                    else:
                        texts.append(content)
                except json.JSONDecodeError:
                    texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    self._process_document_item(item, texts, meta_summaries)
                    num_docs += 1
            elif isinstance(content, dict):
                self._process_document_item(content, texts, meta_summaries)
                num_docs += 1
        
        merged = "\n\n".join(texts).strip()
        
        # Detailed logging
        logger.info(f"Retrieved docs count: {num_docs}")
        if meta_summaries:
            short_list = ", ".join(meta_summaries[:10])
            more = f" (+{len(meta_summaries)-10} more)" if len(meta_summaries) > 10 else ""
            logger.info(f"Retrieved docs meta: {short_list}{more}")
        logger.info(f"Context length: {len(merged)}")
        
        return merged
    
    def _process_document_item(self, item: Any, texts: List[str], meta_summaries: List[str]):
        """Process a single document item and extract content and metadata."""
        if hasattr(item, "page_content"):
            texts.append(str(item.page_content))
            metadata = getattr(item, "metadata", {}) or {}
        elif isinstance(item, dict):
            if "page_content" in item:
                texts.append(str(item["page_content"]))
                metadata = item.get("metadata", {}) or {}
            elif "content" in item:
                texts.append(str(item["content"]))
                metadata = item.get("metadata", {}) or {}
            else:
                texts.append(str(item))
                metadata = {}
        else:
            texts.append(str(item))
            metadata = {}
        
        # Extract metadata summary
        if metadata:
            source_norm = metadata.get("source_normalized", metadata.get("source", ""))
            page_no = metadata.get("page_no", metadata.get("page", ""))
            if source_norm or page_no:
                meta_summaries.append(f"{source_norm}:{page_no}")
    
    def _rewrite_question(self, state: MessagesState) -> Dict:
        """Rewrite question to improve HNSW retrieval."""
        logger.debug("Rewriting question for better retrieval")
        
        # Find the last human message
        last_human_message = None
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                last_human_message = message
                break
        
        if not last_human_message:
            return {"messages": state["messages"]}
        
        question = last_human_message.content
        
        rewrite_prompt = (
            "Analyze the input question and reformulate it to improve document retrieval.\n"
            "Focus on key terms and concepts that would match relevant documents.\n"
            "Original question:\n{question}\n\n"
            "Reformulated question (in Vietnamese):"
        )
        
        prompt = rewrite_prompt.format(question=question)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        
        logger.info(f"Question rewritten from: {question[:100]}...")
        logger.info(f"To: {response.content[:100]}...")
        
        return {"messages": [{"role": "user", "content": response.content}]}
    
    def _generate_answer(self, state: MessagesState) -> Dict:
        """Generate MCQ answer using retrieved context with HNSW optimization."""
        logger.debug("Generating MCQ answer")
        logger.info("[trace] _generate_answer called")
        
        question = state["messages"][0].content
        context = self._extract_tool_context(state["messages"])
        
        if not context:
            context = "No relevant documents found."
            logger.warning("No context available for answer generation")
        
        # Extract MCQ options from original question
        options_detected = self._extract_mcq_options(question)
        
        if not options_detected:
            # Create dummy options for fallback
            options_detected = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}
            logger.info("No MCQ options detected, using dummy options")
        
        options_text = "\n".join([f"{k}) {v}" for k, v in options_detected.items()])
        
        # Generate structured MCQ answer
        mcq_prompt = (
            "Bạn là trợ lý trả lời trắc nghiệm. Sử dụng ngữ cảnh để phân tích từng lựa chọn.\n"
            "Yêu cầu: thinking (quá trình suy nghĩ chi tiết), rationale (giải thích ngắn gọn), "
            "final_answer (chỉ các chữ cái đáp án đúng, ví dụ: 'A' hoặc 'A,B').\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Lựa chọn:\n{options_text}\n\n"
            f"Ngữ cảnh từ tài liệu:\n{context}"
        )
        
        try:
            # Use structured output for consistent formatting
            structured_response = (
                self.response_model
                .with_structured_output(MCQAnswer)
                .invoke([{"role": "user", "content": mcq_prompt}])
            )
            
            content = (
                f"Thinking: {structured_response.thinking}\n"
                f"Rationale: {structured_response.rationale}\n"
                f"Final Answer: {structured_response.final_answer}"
            )
            
            logger.info("Structured MCQ answer generated successfully")
            return {"messages": [{"role": "assistant", "content": content}]}
            
        except Exception as e:
            logger.warning(f"Structured output failed, using fallback: {e}")
            
            # Fallback to regular generation
            fallback_prompt = (
                "Trả lời theo định dạng:\nThinking: [quá trình suy nghĩ]\n"
                "Rationale: [giải thích ngắn gọn]\nFinal Answer: [A,B hoặc A]\n\n"
                f"Câu hỏi: {question}\n\n"
                f"Lựa chọn:\n{options_text}\n\n"
                f"Ngữ cảnh:\n{context}"
            )
            
            response = self.response_model.invoke([{"role": "user", "content": fallback_prompt}])
            return {"messages": [response]}
    
    # Public methods for interaction
    
    def add_vector_store(self, store: MilvusStore, name: str, description: str, 
                        k: Optional[int] = None, ranker_weights: Optional[List[float]] = None) -> None:
        """Add a new vector store to the agent with HNSW optimization."""
        k = k or config.get("retrieval", "k", default=3)
        ranker_weights = ranker_weights or config.get("retrieval", "weights", default=[0.6, 0.4])
        
        # Create retriever with HNSW optimization
        retriever = store.as_retriever(
            k=k, 
            ranker_weights=ranker_weights,
            mmr=True,
            fetch_k=k * 4
        )
        
        # Create retriever tool
        retriever_tool = create_retriever_tool(retriever, name, description)
        
        # Store the configuration
        vs_config = {
            'store': store,
            'name': name,
            'description': description,
            'retriever': retriever,
            'tool': retriever_tool,
            'k': k,
            'ranker_weights': ranker_weights
        }
        
        self.vector_stores.append(vs_config)
        self.retriever_tools.append(retriever_tool)
        
        # Rebuild the graph to include the new tool
        self.graph = self._build_graph()
        logger.info(f"Added vector store '{name}' to OptimizedAgenticRAG")
    
    def remove_vector_store(self, name: str) -> bool:
        """Remove a vector store by name."""
        for i, vs_config in enumerate(self.vector_stores):
            if vs_config['name'] == name:
                # Remove from both lists
                removed_config = self.vector_stores.pop(i)
                self.retriever_tools.pop(i)
                
                # Update backward compatibility attributes if needed
                if self.vector_store == removed_config['store']:
                    self.vector_store = self.vector_stores[0]['store'] if self.vector_stores else None
                    self.retriever = self.vector_stores[0]['retriever'] if self.vector_stores else None
                    self.retriever_tool = self.retriever_tools[0] if self.retriever_tools else None
                
                # Rebuild the graph
                self.graph = self._build_graph()
                logger.info(f"Removed vector store '{name}' from OptimizedAgenticRAG")
                return True
        
        logger.warning(f"Vector store '{name}' not found")
        return False
    
    def get_vector_store_info(self) -> List[Dict[str, Any]]:
        """Get information about all configured vector stores."""
        return [{
            'name': vs['name'],
            'description': vs['description'],
            'k': vs['k'],
            'ranker_weights': vs['ranker_weights']
        } for vs in self.vector_stores]
    
    def update_thread_id(self, new_thread_id: Optional[str] = None) -> str:
        """Update the thread_id for the conversation."""
        if new_thread_id is None:
            self.thread_id = generate_thread_id()
        else:
            self.thread_id = new_thread_id
        
        logger.info(f"Thread ID updated to: {self.thread_id}")
        return self.thread_id
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for the agent."""
        return {"configurable": {"thread_id": self.thread_id}}
    
    def run_mcq(self, question: str, options: Dict[str, str]) -> str:
        """
        Run the agent on a multiple-choice question with HNSW optimized retrieval.
        
        Args:
            question: The question string (may contain document hints)
            options: Dict with keys A/B/C/D and their corresponding option text
            
        Returns:
            str: Generated response with thinking, rationale, and final answer
        """
        logger.info(f"Running MCQ with HNSW optimization: {question[:100]}...")
        logger.info(f"MCQ options: {list(options.keys())}")
        
        # Build user message with question and labeled options
        opts_text = []
        for key in ["A", "B", "C", "D"]:
            if key in options and options[key]:
                opts_text.append(f"{key}: {options[key]}")
        
        options_block = "\n".join(opts_text)
        content = f"{question}\n{options_block}" if options_block else question
        
        # Reset thread per question to avoid context leakage
        self.update_thread_id()
        
        # Create message and config
        message = {"messages": [{"role": "user", "content": content}]}
        config = self.get_config()
        
        # Run the optimized graph
        try:
            result = self.graph.invoke(message, config)
            final_message = result["messages"][-1]
            response = final_message.content
            
            logger.info(f"MCQ response generated: {response[:200]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error running MCQ: {e}")
            return f"Error: {e}"
    
    def run_mcq_csv(self, csv_path: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
        """
        Batch process MCQ questions from CSV with HNSW optimization.
        
        Args:
            csv_path: Path to CSV file with columns: Question, A, B, C, D
            
        Returns:
            List[Dict]: Results with question, options, and response
        """
        results = []
        
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    question = row.get("Question", "")
                    options = {k: row.get(k, "") for k in ["A", "B", "C", "D"]}
                    
                    logger.info(f"Processing CSV row {i+1}: {question[:50]}...")
                    
                    try:
                        # Reset thread for each question
                        self.update_thread_id()
                        response = self.run_mcq(question, options)
                    except Exception as e:
                        logger.error(f"Error processing row {i+1}: {e}")
                        response = f"Error: {e}"
                    
                    results.append({
                        "question": question,
                        **options,
                        "response": response
                    })
                    
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
        
        logger.info(f"Processed {len(results)} MCQ questions from CSV")
        return results
    
    def run(self, query: str) -> str:
        """
        Run the agentic RAG system with HNSW optimization.
        
        Args:
            query: User query
            
        Returns:
            str: Generated response
        """
        logger.info(f"Running optimized agentic RAG: {query[:100]}...")
        
        # Create initial state
        message = {"messages": [{"role": "user", "content": query}]}
        config = self.get_config()
        
        try:
            # Run the graph with HNSW optimization
            result = self.graph.invoke(message, config)
            final_message = result["messages"][-1]
            response = final_message.content
            
            logger.info("Optimized agentic RAG execution completed")
            return response
            
        except Exception as e:
            logger.error(f"Error running agentic RAG: {e}")
            return f"Error: {e}"
    
    def optimize_retrieval_params(self, k: int = None, ranker_weights: List[float] = None, 
                                 ef: int = None) -> None:
        """
        Optimize retrieval parameters for HNSW search.
        
        Args:
            k: Number of documents to retrieve
            ranker_weights: Weights for hybrid ranking
            ef: HNSW search parameter for exploration factor
        """
        for vs_config in self.vector_stores:
            retriever = vs_config['retriever']
            
            if k is not None:
                vs_config['k'] = k
                # Update retriever k parameter
                if hasattr(retriever, 'search_kwargs'):
                    retriever.search_kwargs['k'] = k
            
            if ranker_weights is not None:
                vs_config['ranker_weights'] = ranker_weights
                # Update ranker parameters
                if hasattr(retriever, 'search_kwargs'):
                    retriever.search_kwargs.setdefault('ranker_params', {})['weights'] = ranker_weights
            
            if ef is not None:
                # Update HNSW ef parameter for search optimization
                if hasattr(retriever, 'search_kwargs'):
                    retriever.search_kwargs.setdefault('param', {})['ef'] = ef
        
        logger.info(f"Optimized retrieval parameters: k={k}, weights={ranker_weights}, ef={ef}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval configuration."""
        stats = {
            'num_vector_stores': len(self.vector_stores),
            'vector_stores': []
        }
        
        for vs_config in self.vector_stores:
            vs_stats = {
                'name': vs_config['name'],
                'k': vs_config['k'],
                'ranker_weights': vs_config['ranker_weights']
            }
            
            # Get HNSW parameters if available
            retriever = vs_config['retriever']
            if hasattr(retriever, 'search_kwargs') and 'param' in retriever.search_kwargs:
                vs_stats['hnsw_ef'] = retriever.search_kwargs['param'].get('ef', 'default')
            
            stats['vector_stores'].append(vs_stats)
        
        return stats