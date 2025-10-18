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
# Removed tools_condition import as we use custom routing
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
# Default log file path
DEFAULT_LOG_FILE = 'logs/agent.log'

def configure_logging(level=logging.INFO, log_file=None):
    """
    Configure logging for the agent module.
    Args:
        level: Logging level (default: logging.INFO)
        log_file: Path to log file (default: None, logs to console only)
    """
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Configure the logger
    logger.setLevel(level)
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:  
        logger.removeHandler(handler)
    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        try:
            # Create directory for log file if it doesn't exist
            import os
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Create file handler and set level
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file}: {e}")
        # Trace file logging disabled per user request; only logging to agent.log and console
    logger.debug("Logging configured for agent module")

# Configure logging with default settings
configure_logging(log_file=DEFAULT_LOG_FILE)

def generate_thread_id() -> str:
    """
    Generate a unique thread ID using timestamp and UUID.
    Returns:
        str: Unique thread ID in format "{timestamp}_{random_uuid}"
    """
    timestamp = str(int(time.time() * 1000))  # milliseconds
    random_uuid = str(uuid.uuid4()).replace('-', '')  # 8 chars from UUID
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
    Agentic RAG implementation using LangGraph.
    
    This class implements an agentic RAG system that uses a graph-based approach to:
    1. Generate a query or respond directly
    2. Retrieve relevant documents from multiple vector stores
    3. Grade document relevance
    4. Rewrite the question if needed
    5. Generate a final answer
    
    The system supports multiple vector stores, each converted to a retriever tool that
    the agent can access. Uses LangGraph for orchestrating the RAG workflow.
    """
    def __init__(
            self,
            vector_stores: Optional[List[Dict[str, Any]]] = None,
            vector_store: Optional[MilvusStore] = None,  # Backward compatibility
            model_name: str = None,
            temperature: float = 0,
            thread_id: Optional[str] = None,
            checkpointer: Optional[InMemorySaver] = None,
        ):
        """
        Initialize the AgenticRAG system.
        
        Args:
            vector_stores: List of vector store configurations. Each dict should contain:
                          - 'store': MilvusStore instance
                          - 'name': Tool name (string)
                          - 'description': Tool description (string)
                          - 'k': Optional retrieval count (defaults to config)
                          - 'ranker_weights': Optional ranker weights (defaults to config)
            vector_store: Single MilvusStore instance for backward compatibility
            model_name: Name of the model to use (defaults to config.get("model", "text_generation"))
            temperature: Temperature for the model (default: 0)
            thread_id: Optional thread ID for the conversation (default: None)
            checkpointer: Optional checkpointer for saving the conversation state (default: None)
        """
        self.model_name = model_name or config.get("model", "text_generation", default="qwen2.5:latest")
        self.temperature = temperature
        self.checkpointer = checkpointer or InMemorySaver()
        # Generate unique thread_id if not provided
        if thread_id is None:
            self.thread_id = generate_thread_id()
        else:
            self.thread_id = thread_id
        # Initialize vector stores and create retriever tools
        self.vector_stores = []
        self.retriever_tools = []
        
        if vector_stores:
            # Use the new multiple vector stores approach
            for vs_config in vector_stores:
                if not isinstance(vs_config, dict):
                    raise ValueError("Each vector store configuration must be a dictionary")
                
                required_keys = ['store', 'name', 'description']
                missing_keys = [key for key in required_keys if key not in vs_config]
                if missing_keys:
                    raise ValueError(f"Vector store configuration missing required keys: {missing_keys}")
                
                store = vs_config['store']
                name = vs_config['name']
                description = vs_config['description']
                k = vs_config.get('k', config.get("retrieval", "k", default=2))
                ranker_weights = vs_config.get('ranker_weights', config.get("retrieval", "weights", default=[0.6, 0.4]))
                # Create retriever
                retriever = store.as_retriever(k=k, ranker_weights=ranker_weights)
                # Create retriever tool
                retriever_tool = create_retriever_tool(retriever, name, description)
                # Store the configuration
                self.vector_stores.append({
                    'store': store,
                    'name': name,
                    'description': description,
                    'retriever': retriever,
                    'tool': retriever_tool,
                    'k': k,
                    'ranker_weights': ranker_weights
                })
                self.retriever_tools.append(retriever_tool)
                
        elif vector_store:
            # Backward compatibility: single vector store
            retriever = vector_store.as_retriever(
                k=config.get("retrieval", "k", default=6),
                ranker_weights=config.get("retrieval", "weights", default=[0.6, 0.4])
            )
            retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_documents",
                "Search and retrieve information from the document collection."
            )
            self.vector_stores.append({
                'store': vector_store,
                'name': 'retrieve_documents',
                'description': 'Search and retrieve information from the document collection.',
                'retriever': retriever,
                'tool': retriever_tool,
                'k': config.get("retrieval", "k", default=2),
                'ranker_weights': config.get("retrieval", "weights", default=[0.6, 0.4])
            })
            self.retriever_tools.append(retriever_tool)
        else:
            # Default: create a single MilvusStore (local Milvus)
            default_store = MilvusStore()
            retriever = default_store.as_retriever(
                k=config.get("retrieval", "k", default=6),
                ranker_weights=config.get("retrieval", "weights", default=[0.6, 0.4])
            )
            retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_documents",
                "Search and retrieve information from the document collection."
            )
            self.vector_stores.append({
                'store': default_store,
                'name': 'retrieve_documents',
                'description': 'Search and retrieve information from the document collection.',
                'retriever': retriever,
                'tool': retriever_tool,
                'k': config.get("retrieval", "k", default=2),
                'ranker_weights': config.get("retrieval", "weights", default=[0.6, 0.4])
            })
            self.retriever_tools.append(retriever_tool)
        # Maintain backward compatibility attributes
        self.vector_store = self.vector_stores[0]['store'] if self.vector_stores else None
        self.retriever = self.vector_stores[0]['retriever'] if self.vector_stores else None
        self.retriever_tool = self.retriever_tools[0] if self.retriever_tools else None
        # Initialize the LLM via Ollama (OpenAI-compatible API)
        base_url = config.get("model", "url", default="http://localhost:11434/v1")
        # ChatOpenAI requires an api_key field; any non-empty string works for Ollama
        self.response_model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            base_url=base_url,
            api_key="ollama",
            top_p= 0.1
        )
        self.grader_model = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            base_url=base_url,
            api_key="ollama",
            top_p= 0.1
        )
        # Create the graph
        self.graph = self._build_graph()

        logger.info(f"AgenticRAG initialized with model: {self.model_name} and {len(self.vector_stores)} vector store(s)")
    
    def add_vector_store(self, store: MilvusStore, name: str, description: str, 
                        k: Optional[int] = None, ranker_weights: Optional[List[float]] = None) -> None:
        """
        Add a new vector store to the agent.
        
        Args:
            store: MilvusStore instance
            name: Tool name for the retriever
            description: Tool description for the retriever
            k: Optional retrieval count (defaults to config)
            ranker_weights: Optional ranker weights (defaults to config)
        """
        k = k or config.get("retrieval", "k", default=2)
        ranker_weights = ranker_weights or config.get("retrieval", "weights", default=[0.6, 0.4])
        # Create retriever
        retriever = store.as_retriever(k=k, ranker_weights=ranker_weights)
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
        logger.info(f"Added vector store '{name}' to AgenticRAG")
    def remove_vector_store(self, name: str) -> bool:
        """
        Remove a vector store by name.
    
        Args:
            name: Name of the vector store to remove
            
        Returns:
            bool: True if removed successfully, False if not found
        """
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
                logger.info(f"Removed vector store '{name}' from AgenticRAG")
                return True
        logger.warning(f"Vector store '{name}' not found")
        return False
    
    def get_vector_store_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all configured vector stores.
        
        Returns:
            List[Dict]: List of vector store information (excluding the actual store and tool objects)
        """
        return [{
            'name': vs['name'],
            'description': vs['description'],
            'k': vs['k'],
            'ranker_weights': vs['ranker_weights']
        } for vs in self.vector_stores]
    def _route_tools(self, state: MessagesState) -> str:
        """
        Custom routing function to route to the appropriate tool node or END.
        
        Use in the conditional_edge to route to the specific ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        
        Args:
            state: Current state containing messages
            
        Returns:
            str: Node name to route to (tool name or END)
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
            
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            # Get the tool name from the first tool call
            tool_name = ai_message.tool_calls[0]["name"]
            
            # Verify that the tool name corresponds to one of our vector store tools
            valid_tool_names = [vs['name'] for vs in self.vector_stores]
            if tool_name in valid_tool_names:
                return tool_name
            else:
                logger.warning(f"Unknown tool name: {tool_name}. Available tools: {valid_tool_names}")
                return END
        return END
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph for the agentic RAG system.
        
        Returns:
            StateGraph: Compiled graph for the RAG workflow
        """
        # Create the workflow
        workflow = StateGraph(MessagesState)
        # Add nodes
        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)
        # Add each retriever tool as an individual node
        retriever_node_names = []
        for vs_config in self.vector_stores:
            node_name = vs_config['name']
            retriever_node_names.append(node_name)
            workflow.add_node(node_name, ToolNode([vs_config['tool']]))
        
        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)
        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")
        # Decide whether to retrieve - map each tool name to its corresponding node
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
        # Grade documents after retrieval from each retriever node
        for node_name in retriever_node_names:
            workflow.add_conditional_edges(
                node_name,
                self._grade_documents,
            )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        # Compile the graph
        graph = workflow.compile(checkpointer=self.checkpointer)
        output_file = "graph.png"
        graph.get_graph().draw_mermaid_png(output_file_path=output_file)
        return graph
    
    def _generate_query_or_respond(self, state: MessagesState) -> Dict:
        """
        Call the model to generate a response based on the current state.
        Given the input messages, it will decide to retrieve using the retriever tool,
        or respond directly to the user.
        
        Args:
            state: Current state containing messages
            
        Returns:
            Dict: Updated state with new messages
        """
        logger.debug("Generating query or response")
        logger.info("[trace] _generate_query_or_respond called")
        # Extract source hint from ORIGINAL question (first user message) to avoid losing it during rewrite
        source_hint = None
        source_hint_norm = None
        try:
            # Get the first user message (original question)
            first_user_msg = None
            messages = state.get("messages", [])
            logger.info(f"[trace] state.messages count: {len(messages)}")
            logger.info(f"[trace] DEBUG: Starting source hint extraction...")
            for i, m in enumerate(messages):
                # Debug message structure
                logger.info(f"[trace] message[{i}] type={type(m)}")
                logger.info(f"[trace] message[{i}] dir={[attr for attr in dir(m) if not attr.startswith('_')]}")
                # Try different ways to get role
                role = None
                if hasattr(m, 'role'):
                    role = m.role
                elif hasattr(m, 'type'):
                    role = m.type
                elif isinstance(m, dict):
                    role = m.get('role', 'unknown')
                logger.info(f"[trace] message[{i}] role={role}")
                if role in ["user", "human"]:
                    first_user_msg = m
                    logger.info(f"[trace] DEBUG: Found user message at index {i} with role={role}")
                    logger.info(f"[trace] DEBUG: first_user_msg type: {type(first_user_msg)}")
                    logger.info(f"[trace] DEBUG: first_user_msg has content: {hasattr(first_user_msg, 'content')}")
                    if hasattr(first_user_msg, 'content'):
                        logger.info(f"[trace] DEBUG: first_user_msg.content type: {type(first_user_msg.content)}")
                        logger.info(f"[trace] DEBUG: first_user_msg.content preview: {str(first_user_msg.content)[:100]}...")
                    break
            if first_user_msg:
                # Try different ways to get content
                content = None
                if hasattr(first_user_msg, 'content'):
                    content = first_user_msg.content
                elif isinstance(first_user_msg, dict):
                    content = first_user_msg.get('content', '')
                if content and isinstance(content, str):
                    original_question = content
                    logger.info(f"[trace] original_question={original_question[:100]}...")
                    # Try LLM extraction first - SIMPLIFIED PROMPT
                    hint_prompt = (
                        "Bạn là một bộ phân tích câu hỏi. Nếu trong câu hỏi có đề cập đến tên tài liệu hoặc tệp cụ thể (ví dụ: Public001, Public_001, Public001.pdf, bài báo Public003, tài liệu Public002), hãy trích xuất và trả về chính xác tên đó (giữ nguyên số, ký tự, hoặc phần mở rộng). Nếu không có, hãy trả về chuỗi rỗng. Chỉ trả về đúng tên tệp hoặc chuỗi rỗng, không thêm bất kỳ giải thích nào.\n"
                        f"Câu hỏi: {original_question}.\n"
                        "Tên tài liệu:"
                    )
                    try:
                        hint_resp = (
                            self.response_model
                            .with_structured_output(ExtractSourceHint)
                            .invoke([{"role": "user", "content": hint_prompt}])
                        )
                        if hint_resp and getattr(hint_resp, 'source_hint', None):
                            raw_hint = str(hint_resp.source_hint).strip().strip('"\'')
                            # Validate: should be short document name, not full question
                            if len(raw_hint) < 50 and any(char.isdigit() for char in raw_hint):
                                source_hint = raw_hint
                                logger.info(f"[trace] source_hint(by-llm)={source_hint}")
                            else:
                                logger.warning(f"[trace] LLM returned invalid source_hint (too long or no digits): {raw_hint[:50]}...")
                                source_hint = None
                    except Exception as e:
                        logger.warning(f"[trace] LLM failed to extract source hint: {e}")
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
                        for p in patterns:
                            m = re.search(p, original_question, flags=re.IGNORECASE)
                            if m:
                                source_hint = m.group(1).strip().strip('"\'')
                                logger.info(f"[trace] source_hint(by-regex)={source_hint}")
                                break
                    # Normalize source hint
                    if source_hint:
                        base = source_hint.lower().replace('.pdf', '')
                        source_hint_norm = re.sub(r"[^a-z0-9]", "", base)
                        logger.info(f"[trace] source_hint_norm={source_hint_norm}")
        except Exception as e:
            logger.warning(f"[trace] Failed to extract source hint: {e}")

        # Update retriever expr dynamically to include source filter if hinted
        if self.retriever_tools and self.vector_stores:
            for vs in self.vector_stores:
                retriever = vs.get('retriever')
                if retriever and hasattr(retriever, 'search_kwargs'):
                    ns = None
                    # Try to keep existing namespace expression if present in expr or config
                    if 'expr' in retriever.search_kwargs and isinstance(retriever.search_kwargs['expr'], str):
                        # Extract namespace if present
                        ns_match = re.search(r'namespace\s*==\s*"([^"]+)"', retriever.search_kwargs['expr'])
                        if ns_match:
                            ns = ns_match.group(1)
                    if ns is None:
                        ns = getattr(self, 'namespace', None)

                    if source_hint or source_hint_norm:
                        # Chỉ sử dụng source_normalized để lọc, bỏ qua namespace và source
                        if source_hint_norm:
                            retriever.search_kwargs['expr'] = f'source_normalized == "{source_hint_norm}"'
                            logger.info(f"[trace] retriever.expr={retriever.search_kwargs['expr']}")
                    else:
                        # Reset to no filter if we previously set source
                        if 'expr' in retriever.search_kwargs:
                            retriever.search_kwargs.pop('expr', None)
                        logger.info(f"[trace] retriever.expr={retriever.search_kwargs.get('expr', 'None')}")
                        
                    # Thêm log để debug
                    logger.info(f"[trace] search_kwargs={retriever.search_kwargs}")
                    logger.info(f"[trace] source_hint_norm={source_hint_norm}")
                    logger.info(f"[trace] using_namespace=False")
                    # Lưu ý: retriever dùng expr cố định, không tự fallback theo query
                    logger.info(f"[trace] retriever_fallback_enabled=False")

        # Extract MCQ options from ORIGINAL question and inject into state
        augmented_messages = state["messages"]
        try:
            # Use the same first_user_msg we found earlier
            if first_user_msg:
                options = {}
                if hasattr(first_user_msg, 'content'):
                    content = first_user_msg.content
                elif isinstance(first_user_msg, dict):
                    content = first_user_msg.get('content', '')
                
                if content and isinstance(content, str):
                    # Extract lines like A:, B:, C:, D:
                    for key in ["A", "B", "C", "D"]:
                        mopt = re.search(rf"\b{key}\s*[:\-]\s*(.+)", content, flags=re.IGNORECASE)
                        if mopt:
                            options[key] = mopt.group(1).strip()
                            
                if options:
                    logger.info(f"[trace] mcq_options_detected={list(options.keys())}")
                    # Build combined query hint
                    combined = [f"Question: {content}"]
                    opt_str = "; ".join([f"{k}) {v}" for k, v in options.items()])
                    combined.append(f"Options: {opt_str}")
                    hint = " \n".join(combined)
                    augmented_messages = [{"role": "system", "content": "When you call the retrieval tool, pass the combined query including both the question and the options to maximize matching."}] + augmented_messages + [{"role": "system", "content": f"Combined retrieval query hint:\n{hint}"}]
                else:
                    logger.info("[trace] mcq_options_detected=[]")
                    
        except Exception as e:
            logger.warning(f"[trace] Failed to extract MCQ options: {e}")

        response = (
            self.response_model
            .bind_tools(self.retriever_tools)
            .invoke(augmented_messages)
        )
        # FORCE tool calling for MCQ RAG - always retrieve documents
        if not (hasattr(response, "tool_calls") and response.tool_calls):
            logger.info("[trace] No tool calls detected, forcing retrieval...")
            # Create a forced tool call to the first retriever
            if self.retriever_tools:
                tool_name = self.retriever_tools[0].name
                logger.info(f"[trace] Forcing tool call to: {tool_name}")
                # Manually create tool call
                response.tool_calls = [{
                    "name": tool_name,
                    "args": {"query": augmented_messages[-1]["content"] if augmented_messages else "retrieve documents"},
                    "id": "forced_call"
                }]
        logger.info(f"[trace] Final response has tool_calls: {bool(hasattr(response, 'tool_calls') and response.tool_calls)}")
        return {"messages": [response]}
    def _grade_documents(
        self, 
        state: MessagesState
    ) -> Literal["generate_answer", "rewrite_question"]:
        """
        Determine whether the retrieved documents are relevant to the question.
        
        Args:
            state: Current state containing messages
            
        Returns:
            str: Next node to execute ("generate_answer" or "rewrite_question")
        """
        logger.debug("Grading retrieved documents")
        question = state["messages"][0].content
        # Reuse the same context assembly as in _generate_answer
        def _extract_tool_context(messages: List[Any]) -> str:
            texts: List[str] = []
            for msg in messages:
                role = None
                if hasattr(msg, 'role'):
                    role = msg.role
                elif hasattr(msg, 'type'):
                    role = msg.type
                elif isinstance(msg, dict):
                    role = msg.get('role', None)
                is_tool = (role == 'tool') or (getattr(msg, '__class__', type('X',(object,),{})).__name__ == 'ToolMessage')
                if not is_tool:
                    continue
                content = None
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get('content', None)
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if hasattr(item, 'page_content'):
                            texts.append(str(item.page_content))
                        elif isinstance(item, dict):
                            if 'page_content' in item:
                                texts.append(str(item['page_content']))
                            elif 'content' in item:
                                texts.append(str(item['content']))
                            else:
                                texts.append(str(item))
                        else:
                            texts.append(str(item))
                elif isinstance(content, dict):
                    if 'page_content' in content:
                        texts.append(str(content['page_content']))
                    elif 'content' in content:
                        texts.append(str(content['content']))
                    else:
                        texts.append(str(content))

            merged = "\n\n".join(texts).strip()
            logger.info(f"[trace] grade_context_length={len(merged)}")
            return merged

        context = _extract_tool_context(state["messages"]) or state["messages"][-1].content
        grade_prompt = (
            "You are a grader assessing relevance of a retrieved document to a user question. \n "
            "Here is the retrieved document: \n\n {context} \n\n"
            "Here is the user question: {question} \n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        )
        prompt = grade_prompt.format(question=question, context=context)
        response = (
            self.grader_model
            .with_structured_output(GradeDocuments)
            .invoke([{"role": "user", "content": prompt}])
        )
        score = response.binary_score
        logger.info(f"[trace] grade_score={score}")
        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"
        
    def _rewrite_question(self, state: MessagesState) -> Dict:
        """
        Rewrite the original user question to improve retrieval.
        
        Args:
            state: Current state containing messages
            
        Returns:
            Dict: Updated state with rewritten question
        """
        logger.debug("Rewriting question")
        last_human_message = None
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                last_human_message = message
                break
        question = last_human_message.content
        rewrite_prompt = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:"
            "\n ------- \n"
            "{question}"
            "\n ------- \n"
            "Formulate an improved question then response in Vietnamese:"
        )
        prompt = rewrite_prompt.format(question=question)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])
        logger.debug(f"Original question: {question}")
        logger.debug(f"Rewritten question: {response.content}")
        return {"messages": [{"role": "user", "content": response.content}]}
    
    def _generate_answer(self, state: MessagesState) -> Dict:
        """
        Generate an answer based on the retrieved documents.
        Always use MCQ mode since this project is for multiple-choice questions.
        
        Args:
            state: Current state containing messages
            
        Returns:
            Dict: Updated state with generated answer
        """
        logger.debug("Generating answer")
        logger.info("[trace] _generate_answer called")
        question = state["messages"][0].content
        # Assemble context from ToolMessages (robust against different formats)
        def _extract_tool_context(messages: List[Any]) -> str:
            texts: List[str] = []
            num_docs = 0
            num_tools = 0
            pre_merge_total_len = 0
            meta_summaries: List[str] = []

            def _try_append(item: Any):
                nonlocal num_docs, pre_merge_total_len, meta_summaries
                # LangChain Document-like object
                if hasattr(item, "page_content"):
                    txt = str(getattr(item, "page_content", ""))
                    texts.append(txt)
                    pre_merge_total_len += len(txt)
                    num_docs += 1
                    # metadata
                    md = getattr(item, "metadata", {}) or {}
                    srcn = md.get("source_normalized", md.get("source", ""))
                    pno = md.get("page_no", md.get("page", md.get("page_number", "")))
                    if srcn or pno != "":
                        meta_summaries.append(f"{srcn}:{pno}")
                    return True
                # Dict-like document
                if isinstance(item, dict):
                    if "page_content" in item:
                        txt = str(item.get("page_content", ""))
                        texts.append(txt)
                        pre_merge_total_len += len(txt)
                        num_docs += 1
                        md = item.get("metadata", {}) or {}
                        srcn = md.get("source_normalized", md.get("source", ""))
                        pno = md.get("page_no", md.get("page", md.get("page_number", "")))
                        if srcn or pno != "":
                            meta_summaries.append(f"{srcn}:{pno}")
                        return True
                    if "content" in item:
                        txt = str(item.get("content", ""))
                        texts.append(txt)
                        pre_merge_total_len += len(txt)
                        num_docs += 1
                        md = item.get("metadata", {}) or {}
                        srcn = md.get("source_normalized", md.get("source", ""))
                        pno = md.get("page_no", md.get("page", md.get("page_number", "")))
                        if srcn or pno != "":
                            meta_summaries.append(f"{srcn}:{pno}")
                        return True
                # Fallback: just stringify
                s = str(item)
                texts.append(s)
                pre_merge_total_len += len(s)
                return False

            for msg in messages:
                # role detection
                role = None
                if hasattr(msg, "role"):
                    role = msg.role
                elif hasattr(msg, "type"):
                    role = msg.type
                elif isinstance(msg, dict):
                    role = msg.get("role", None)

                is_tool = (role == "tool") or (getattr(msg, "__class__", type("X",(object,),{})).__name__ == "ToolMessage")
                if not is_tool:
                    continue
                num_tools += 1

                # content extraction
                content = None
                if hasattr(msg, "content"):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get("content", None)

                if isinstance(content, str):
                    # Thử parse JSON để trích xuất Document nếu tool trả về JSON dưới dạng chuỗi
                    parsed = None
                    try:
                        stripped = content.strip()
                        if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
                            parsed = json.loads(stripped)
                    except Exception:
                        parsed = None
                    if parsed is not None:
                        # parsed có thể là 1 doc (dict) hoặc danh sách docs
                        if isinstance(parsed, list):
                            for item in parsed:
                                _try_append(item)
                        elif isinstance(parsed, dict):
                            # Một số tool trả {"documents": [...]} hay {"result": [...]}
                            if "documents" in parsed and isinstance(parsed["documents"], list):
                                for item in parsed["documents"]:
                                    _try_append(item)
                            elif "result" in parsed and isinstance(parsed["result"], list):
                                for item in parsed["result"]:
                                    _try_append(item)
                            else:
                                _try_append(parsed)
                    else:
                        # Không phải JSON hợp lệ, coi như text thuần
                        texts.append(content)
                        pre_merge_total_len += len(content)
                elif isinstance(content, list):
                    for item in content:
                        _try_append(item)
                elif isinstance(content, dict):
                    _try_append(content)

                # Extra fallbacks from additional_kwargs/response_metadata
                try:
                    addkw = getattr(msg, "additional_kwargs", None)
                    if isinstance(addkw, dict):
                        for k in ["output", "content", "result", "messages"]:
                            if k in addkw and addkw[k]:
                                # can be complex; just stringify
                                s = str(addkw[k])
                                texts.append(s)
                                pre_merge_total_len += len(s)
                except Exception:
                    pass
                try:
                    resp_meta = getattr(msg, "response_metadata", None)
                    if isinstance(resp_meta, dict):
                        for k in ["output", "content", "result"]:
                            if k in resp_meta and resp_meta[k]:
                                s = str(resp_meta[k])
                                texts.append(s)
                                pre_merge_total_len += len(s)
                except Exception:
                    pass

            merged = "\n\n".join(texts).strip()
            # Logging chi tiết theo yêu cầu
            logger.info(f"[trace] retrieved_docs_count={num_docs}")
            if meta_summaries:
                # In ra tối đa 10 mục để gọn log
                short_list = ", ".join(meta_summaries[:10])
                more = "" if len(meta_summaries) <= 10 else f" (+{len(meta_summaries)-10} more)"
                logger.info(f"[trace] retrieved_docs_meta={short_list}{more}")
            else:
                logger.info(f"[trace] retrieved_docs_meta=[]")
            logger.info(f"[trace] pre_merge_total_length={pre_merge_total_len}")
            logger.info(f"[trace] tool_messages_seen={num_tools}")
            logger.info(f"[trace] context_merged_length={len(merged)}")
            return merged

        context = _extract_tool_context(state["messages"]) or state["messages"][-1].content
        
        logger.info(f"[trace] question={question[:100]}...")
        logger.info(f"[trace] context_length={len(context)}")
        
        # Extract multiple-choice options from the original user question
        options_detected = {}
        try:
            # Look for options in the first user message (original question)
            first_user = None
            messages = state["messages"]
            logger.info(f"[trace] _generate_answer messages count: {len(messages)}")
            
            for i, m in enumerate(messages):
                # Try different ways to get role
                role = None
                if hasattr(m, 'role'):
                    role = m.role
                elif hasattr(m, 'type'):
                    role = m.type
                elif isinstance(m, dict):
                    role = m.get('role', 'unknown')
                
                if role in ["user", "human"]:
                    first_user = m
                    break
            
            if first_user:
                # Try different ways to get content
                content = None
                if hasattr(first_user, 'content'):
                    content = first_user.content
                elif isinstance(first_user, dict):
                    content = first_user.get('content', '')
                
                if content and isinstance(content, str):
                    logger.info(f"[trace] first_user.content={content[:200]}...")
                    for key in ["A", "B", "C", "D"]:
                        mopt = re.search(rf"\b{key}\s*[:\-]\s*(.+)", content, flags=re.IGNORECASE)
                        if mopt:
                            options_detected[key] = mopt.group(1).strip()
                            logger.info(f"[trace] found option {key}: {mopt.group(1).strip()[:50]}...")
                        
            logger.info(f"[trace] mcq_options_detected={list(options_detected.keys())}")
        except Exception as e:
            logger.warning(f"[trace] Failed to detect MCQ options: {e}")
        
        # Always use MCQ mode for this project - no need to detect, just use what we found
        if not options_detected:
            # Fallback: create dummy options if none detected
            options_detected = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}
            logger.info("[trace] No options detected, using dummy options for MCQ mode")
        
        options_text = "\n".join([f"{k}) {v}" for k, v in options_detected.items()])
        mcq_prompt = (
            "Bạn là trợ lý trả lời trắc nghiệm. Hãy dùng ngữ cảnh để đánh giá từng lựa chọn và liệt kê tất cả đáp án đúng.\n"
            "Yêu cầu: thinking chi tiết quá trình suy nghĩ; rationale ngắn gọn bằng tiếng Việt; final_answer chỉ là các chữ cái (A,B,...) không thêm ký tự khác.\n"
            f"Câu hỏi: {question}\n"
            f"Lựa chọn:\n{options_text}\n"
            f"Ngữ cảnh:\n{context}"
        )
        try:
            structured = self.response_model.with_structured_output(MCQAnswer).invoke([{"role": "user", "content": mcq_prompt}])
            content = f"Thinking: {structured.thinking}\nRationale: {structured.rationale}\nFinal Answer: {structured.final_answer}"
            logger.info("[trace] structured_output=success (MCQ)")
            return {"messages": [{"role": "assistant", "content": content}]}
        except Exception as e:
            logger.info(f"[trace] structured_output=fallback (MCQ): {e}")
            fallback_prompt = (
                "Bạn là trợ lý trả lời trắc nghiệm. Hãy dùng ngữ cảnh để đánh giá từng lựa chọn và liệt kê tất cả đáp án đúng.\n"
                "Hãy trả lời theo đúng định dạng: Thinking: ...\nRationale: ...\nFinal Answer: A,B\n"
                f"Câu hỏi: {question}\n"
                f"Lựa chọn:\n{options_text}\n"
                f"Ngữ cảnh:\n{context}"
            )
            response = self.response_model.invoke([{"role": "user", "content": fallback_prompt}])
            return {"messages": [response]}
    
    def update_thread_id(self, new_thread_id: Optional[str] = None) -> str:
        """
        Update the thread_id for the conversation.
        
        Args:
            new_thread_id: New thread ID to use. If None, generates a new unique thread_id.
            
        Returns:
            str: The updated thread_id
        """
        if new_thread_id is None:
            self.thread_id = generate_thread_id()
        else:
            self.thread_id = new_thread_id
            
        logger.info(f"Thread ID updated to: {self.thread_id}")
        return self.thread_id
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for the agent.
        
        Returns:
            Dict[str, Any]: Configuration dictionary with thread_id
        """
        return {"configurable": {"thread_id": self.thread_id}}

    def run_mcq(self, question: str, options: Dict[str, str]) -> str:
        """
        Run the agent on a multiple-choice question with options A/B/C/D.
        The question string may also contain a document hint (e.g., "dựa vào tài liệu Public001").
        Options should be a dict like {"A": "...", "B": "...", "C": "...", "D": "..."}.
        """
        logger.info(f"[trace] run_mcq called with question: {question[:100]}...")
        logger.info(f"[trace] run_mcq options: {list(options.keys())}")
        
        # Build a single user message that includes question and labeled options
        opts_text = []
        for key in ["A", "B", "C", "D"]:
            if key in options and options[key]:
                opts_text.append(f"{key}: {options[key]}")
        options_block = "\n".join(opts_text)
        content = f"{question}\n{options_block}" if options_block else question
        
        logger.info(f"[trace] run_mcq content: {content[:200]}...")
        
        # Reset thread per question to avoid cross-row context leakage
        self.update_thread_id()
        msg = {"messages": [{"role": "user", "content": content}]}
        config = self.get_config()
        
        logger.info(f"[trace] run_mcq invoking graph...")
        result = self.graph.invoke(msg, config)
        final_message = result["messages"][-1]
        
        logger.info(f"[trace] run_mcq result: {final_message.content[:200]}...")
        return final_message.content

    def run_mcq_csv(self, csv_path: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
        """
        Batch-run MCQ questions from a CSV file with columns: Question, A, B, C, D.
        Returns a list of dicts: {"question": ..., "A": ..., ..., "response": ...}
        """
        rows: List[Dict[str, Any]] = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("Question", "")
                options = {k: row.get(k, "") for k in ["A", "B", "C", "D"]}
                try:
                    # Reset thread per row
                    self.update_thread_id()
                    response = self.run_mcq(q, options)
                except Exception as e:
                    response = f"Error: {e}"
                rows.append({"question": q, **options, "response": response})
        return rows

    def run(self, query: str) -> str:
        """
        Run the agentic RAG system with a query.
        
        Args:
            query: User query
            
        Returns:
            str: Generated response
        """
        logger.info(f"Running agentic RAG with query: {query}")
        # Create initial state
        message = {"messages": [{"role": "user", "content": query}]}
        # Run the graph
        config = self.get_config()
        result = self.graph.invoke(message, config)
        # Extract the final response
        final_message = result["messages"][-1]
        response = final_message.content
        logger.info("Agentic RAG execution completed")
        return response