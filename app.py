"""
AEC Compliance Agent - Streamlit Web Interface
Main entry point for the Streamlit application.
"""

import streamlit as st
import os
import json
import tempfile
import time
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from aec_agent.utils.ifc_to_json import IFCToJSONConverter
from services.pdf_rag_manager import PDFRAGManager
from services.session_manager import SessionManager
from aec_agent.agent import create_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class StreamlitLogHandler(logging.Handler):
    """Log handler that captures user-friendly progress messages for Streamlit display."""
    
    # Messages to filter out (too technical/debug)
    FILTERED_PATTERNS = [
        r'context after',
        r'updated context',
        r'execution guardrail',
        r'iteration \d+',
        r'progress:',
        r'graph metrics',
        r'task graph',
        r'validation passed',
        r'output validation',
        r'execution time',
        r'completed in \d+',
        r'\.\d+s',
        r'\.\d+ms',
    ]
    
    # Tool name mappings for cleaner display
    TOOL_NAMES = {
        'load_building_data': 'Load building data',
        'get_all_elements': 'Get elements',
        'get_element_properties': 'Get properties',
        'query_elements': 'Query elements',
        'calculate_distances': 'Calculate distances',
        'calculate_areas': 'Calculate areas',
        'find_related_elements': 'Find related',
        'validate_compliance_rule': 'Validate compliance',
        'search_compliance_documents': 'Search documents',
        'document_findings': 'Document findings',
        'simple_response': 'Generate response'
    }
    
    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
        self.messages: List[Dict[str, Any]] = []
        self.last_component = None
    
    def emit(self, record: logging.LogRecord):
        """Capture log message and format it for user display."""
        try:
            message = record.getMessage()
            
            # Skip filtered messages
            if self._should_filter(message):
                return
            
            # Convert to user-friendly message
            user_message = self._convert_to_user_friendly(message, record.name)
            if not user_message:
                return
            
            # Extract component for grouping
            component = self._extract_component(record.name)
            
            # Store formatted message
            formatted_message = {
                "level": record.levelname,
                "component": component,
                "message": user_message,
                "raw_message": message
            }
            
            # Only add if different from last message (avoid duplicates)
            if not self.messages or formatted_message["message"] != self.messages[-1]["message"]:
                self.messages.append(formatted_message)
                self.last_component = component
            
            # Keep only last 20 messages to avoid clutter
            if len(self.messages) > 20:
                self.messages = self.messages[-20:]
                
        except Exception:
            pass  # Ignore errors in logging handler
    
    def _should_filter(self, message: str) -> bool:
        """Check if message should be filtered out."""
        message_lower = message.lower()
        for pattern in self.FILTERED_PATTERNS:
            if re.search(pattern, message_lower):
                return True
        return False
    
    def _convert_to_user_friendly(self, message: str, logger_name: str) -> Optional[str]:
        """Convert technical log message to user-friendly format with task/tool info."""
        message_lower = message.lower()
        
        # Starting reasoning
        if 'starting reasoning' in message_lower:
            return 'Analyzing your question...'
        
        # Goal decomposition - extract task count if available
        if 'decomposing goal' in message_lower:
            return 'Breaking down into steps...'
        
        # Task execution - show task name
        if 'executing task' in message_lower:
            task_match = re.search(r"task\s+['\"]([^'\"]+)['\"]|task:\s*([^,\.]+)", message_lower)
            if task_match:
                task_name = (task_match.group(1) or task_match.group(2)).strip()
                # Clean up task name
                task_name = task_name.replace('_', ' ').title()
                if len(task_name) > 40:
                    task_name = task_name[:37] + '...'
                return f'Task: {task_name}'
        
        # Tool planning - show what's being planned
        if 'planning tools' in message_lower:
            task_match = re.search(r"task:\s*['\"]?([^'\"]+)['\"]?", message_lower)
            if task_match:
                task_name = task_match.group(1).strip()
                task_name = task_name.replace('_', ' ').title()
                if len(task_name) > 35:
                    task_name = task_name[:32] + '...'
                return f'Planning: {task_name}'
            return 'Planning tools...'
        
        # Tool execution - show tool name
        if 'executing tool' in message_lower:
            tool_match = re.search(r"tool\s+['\"]([^'\"]+)['\"]", message_lower)
            if tool_match:
                tool_key = tool_match.group(1)
                tool_display = self.TOOL_NAMES.get(tool_key, tool_key.replace('_', ' ').title())
                
                # Try to extract task name for context
                task_match = re.search(r"task\s+['\"]([^'\"]+)['\"]|for task ['\"]([^'\"]+)['\"]", message_lower)
                if task_match:
                    task_name = (task_match.group(1) or task_match.group(2)).strip()
                    task_name = task_name.replace('_', ' ').title()
                    if len(task_name) > 25:
                        task_name = task_name[:22] + '...'
                    return f'Tool: {tool_display} ({task_name})'
                return f'Tool: {tool_display}'
        
        # Tool completed
        if 'tool' in message_lower and 'executed successfully' in message_lower:
            tool_match = re.search(r"tool\s+['\"]([^'\"]+)['\"]", message_lower)
            if tool_match:
                tool_key = tool_match.group(1)
                tool_display = self.TOOL_NAMES.get(tool_key, tool_key.replace('_', ' ').title())
                return f'Completed: {tool_display}'
        
        # Reasoning completed
        if 'reasoning completed' in message_lower:
            return 'Analysis complete'
        
        # New flexible recovery system messages
        if 'llm recovery decision:' in message_lower:
            strategy = message.split('llm recovery decision:')[-1].strip()
            # Shorten LLM response for display
            if len(strategy) > 80:
                strategy = strategy[:77] + "..."
            return f'🧠 Recovery analysis: {strategy}'
        
        if 'modified task input:' in message_lower:
            modifications = message.split('modified task input:')[-1].strip()
            return f'🔧 Task modified: {modifications}'
        
        if 'switching to alternative tool:' in message_lower:
            tool = message.split('switching to alternative tool:')[-1].strip()
            return f'🔄 Switching to: {tool}'
        
        if 'task requires goal replanning:' in message_lower:
            task = message.split('task requires goal replanning:')[-1].strip()
            return f'📋 Replanning needed for: {task}'
        
        if 'gracefully skipping task:' in message_lower:
            task = message.split('gracefully skipping task:')[-1].strip()
            return f'⏭️ Skipping: {task}'
        
        if 'attempting recovery for planning failure' in message_lower:
            return '🔄 Recovering from planning failure...'
        
        if 'created fallback task:' in message_lower:
            task = message.split('created fallback task:')[-1].strip()
            return f'✨ Created fallback: {task}'
        
        if 'attempting recovery' in message_lower:
            if 'parameter recovery' in message_lower:
                return '🔧 Attempting parameter fix...'
            elif 'tool recovery' in message_lower:
                return '🔄 Trying alternative tool...'
            else:
                return '🔄 Attempting recovery...'
        
        if 'recovery successful' in message_lower:
            return '✅ Recovery successful'
        
        if 'recovery failed' in message_lower:
            return '⚠️ Recovery failed, trying alternatives...'
        
        if 'graceful degradation' in message_lower:
            return '📉 Providing partial results...'
        
        if 'goal replanning' in message_lower or 'replanning' in message_lower:
            return '🎯 Replanning approach...'
        
        if 'skipped non-critical task' in message_lower:
            task_match = re.search(r"task.*?([A-Z][^:]+)", message)
            if task_match:
                task_name = task_match.group(1).strip()
                return f'⏭️ Skipped optional: {task_name}'
            return '⏭️ Skipped optional task'
        
        # Tool execution failures (show before recovery)
        if 'tool execution failed' in message_lower:
            tool_match = re.search(r"for\s+['\"]([^'\"]+)['\"]", message_lower)
            if tool_match:
                tool_name = tool_match.group(1)
                tool_display = self.TOOL_NAMES.get(tool_name, tool_name.replace('_', ' ').title())
                return f'❌ {tool_display} failed'
            return '❌ Tool failed'
        
        # Task validation failures
        if 'task validation failed' in message_lower:
            return '⚠️ Validating results...'
        
        # Skip other technical messages
        return None
    
    def _extract_component(self, logger_name: str) -> str:
        """Extract clean component name from logger name."""
        parts = logger_name.split('.')
        component = parts[-1] if parts else logger_name
        component = component.replace('_', ' ').title()
        
        name_map = {
            'Reasoning Controller': 'Controller',
            'Goal Decomposer': 'Decomposer',
            'Tool Planner': 'Planner',
            'Tool Executor': 'Executor',
            'Result Validator': 'Validator',
            'Reasoning Agent': 'Agent',
            'Recovery System': 'Recovery'
        }
        
        return name_map.get(component, component)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all captured messages."""
        return self.messages.copy()
    
    def clear(self):
        """Clear captured messages."""
        self.messages.clear()


def setup_streamlit_logging():
    """Set up logging to capture messages for Streamlit display."""
    # Initialize log handler in session state if not exists
    if 'log_handler' not in st.session_state:
        st.session_state.log_handler = StreamlitLogHandler()
    
    # Get root logger and add our handler
    root_logger = logging.getLogger()
    
    # Remove existing StreamlitLogHandler if any
    root_logger.handlers = [h for h in root_logger.handlers if not isinstance(h, StreamlitLogHandler)]
    
    # Add our handler
    root_logger.addHandler(st.session_state.log_handler)
    root_logger.setLevel(logging.INFO)
    
    return st.session_state.log_handler

def get_llm_insight(action, context):
    """Get LLM-generated insight about what the agent is doing."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Simple prompt to explain what's happening
        prompt = f"In 3-4 words, what is an AI agent doing when: {action} with context: {context[:50]}"
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Explain what an AI agent is doing in 3-4 simple words. Be direct and clear."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=8,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception:
        # If LLM fails, return minimal fallback
        return f"Processing {action.split()[-1].lower()}"

# Set page config
st.set_page_config(
    page_title="AEC Compliance Agent",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application."""
    # Initialize session manager and current session
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()
    
    if "current_session_id" not in st.session_state:
        # Create first session
        st.session_state.current_session_id = st.session_state.session_manager.create_new_session()
    
    # Initialize session-specific state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_ifc_files" not in st.session_state:
        st.session_state.processed_ifc_files = {}
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = {}
    if "pdf_rag_manager" not in st.session_state:
        st.session_state.pdf_rag_manager = PDFRAGManager()
    if "reasoning_agent" not in st.session_state:
        # Set up logging before creating agent
        setup_streamlit_logging()
        
        # Initialize the actual ReAct agent with 3-layer memory management
        st.session_state.reasoning_agent = create_agent(
            model_name="gpt-5-mini",
            temperature=0.1,
            verbose=True,
            enable_memory=True,
            session_id=st.session_state.current_session_id,
            max_iterations=20
        )
    
    # Show session sidebar
    show_session_sidebar()
    
    # Main content
    st.title("🏗️ AEC Compliance Agent")
    st.markdown("**Architecture, Engineering & Construction Compliance Assistant**")
    
    # Create two columns for upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        show_ifc_upload_section()
    
    with col2:
        show_legal_docs_upload_section()
    
    # Chat interface
    show_chat_interface()
    
    # Auto-save session periodically
    save_current_session()

def load_cached_ifc_files():
    """Load previously processed IFC files from persistent storage."""
    processed_dir = Path("data/processed_ifc")
    if not processed_dir.exists():
        return {}
    
    cached_files = {}
    for json_file in processed_dir.glob("*.json"):
        try:
            # Extract original filename (remove .json extension)
            original_name = json_file.name[:-5]  # Remove ".json"
            
            # Load the JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Get file stats for cache info
            file_stats = json_file.stat()
            
            cached_files[original_name] = {
                "json_data": json_data,
                "file_size": file_stats.st_size,
                "processed_at": str(datetime.fromtimestamp(file_stats.st_mtime)),
                "cached": True
            }
            
        except Exception as e:
            st.warning(f"⚠️ Could not load cached file {json_file.name}: {str(e)}")
            continue
    
    return cached_files

def show_ifc_upload_section():
    """Display IFC file upload section with persistent caching."""
    st.subheader("📊 IFC Building Models")
    
    # Load cached files on first run or if session state is empty
    if not st.session_state.processed_ifc_files:
        st.session_state.processed_ifc_files = load_cached_ifc_files()
    
    # Show cache management controls
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.processed_ifc_files:
            cached_count = sum(1 for f in st.session_state.processed_ifc_files.values() if f.get('cached'))
            if cached_count > 0:
                st.info(f"📁 {cached_count} previously processed file(s) loaded from cache")
    
    with col2:
        if st.session_state.processed_ifc_files:
            if st.button("🗑️ Clear Cache", help="Remove all cached IFC files"):
                clear_ifc_cache()
    
    with st.container():
        st.markdown("""
        <div style="
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background-color: #f9f9f9;
            margin-bottom: 10px;
        ">
            <p style="margin: 0; color: #666;">
                🏗️ Drop IFC files here or click to browse
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_ifc = st.file_uploader(
            "Upload IFC Files",
            type=['ifc'],
            accept_multiple_files=True,
            key="ifc_uploader",
            label_visibility="collapsed"
        )
        
        # Show cached files even if no new files are uploaded
        if st.session_state.processed_ifc_files and not uploaded_ifc:
            st.subheader("📋 Cached IFC Files")
            for filename, data in st.session_state.processed_ifc_files.items():
                show_ifc_file_summary(filename, data)
        
        if uploaded_ifc:
            for file in uploaded_ifc:
                # Check if already processed (by filename and size)
                existing_data = st.session_state.processed_ifc_files.get(file.name)
                if existing_data:
                    # Compare file sizes to detect if file was modified
                    if existing_data.get("original_file_size") == file.size:
                        # Show single consolidated status line for cached files
                        show_ifc_file_status(file.name, existing_data, "cached")
                        continue
                    else:
                        st.warning(f"⚠️ {file.name} has different size - reprocessing...")
                
                # Process new or modified file
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.ifc') as temp_file:
                            temp_file.write(file.read())
                            temp_file_path = temp_file.name
                        
                        # Convert IFC to JSON
                        converter = IFCToJSONConverter(temp_file_path)
                        json_data = converter.extract_to_dict()
                        
                        # Store in session state
                        st.session_state.processed_ifc_files[file.name] = {
                            "json_data": json_data,
                            "file_size": file.size,
                            "original_file_size": file.size,
                            "processed_at": str(datetime.now()),
                            "cached": False
                        }
                        
                        # Create processed files directory if it doesn't exist
                        processed_dir = Path("data/processed_ifc")
                        processed_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save JSON to persistent storage
                        json_file_path = processed_dir / f"{file.name}.json"
                        with open(json_file_path, 'w', encoding='utf-8') as f:
                            json.dump(json_data, f, indent=2, ensure_ascii=False)
                        
                        # Cleanup temporary file
                        os.unlink(temp_file_path)
                        
                        # Show single consolidated status line for new files
                        show_ifc_file_status(file.name, st.session_state.processed_ifc_files[file.name], "processed")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {str(e)}")

def show_ifc_file_status(filename, data, status_type):
    """Display a single consolidated status line for IFC files."""
    json_data = data["json_data"]
    
    # Format file size
    file_size_mb = data.get("original_file_size", data.get("file_size", 0)) / (1024 * 1024)
    size_str = f"{file_size_mb:.1f}MB"
    
    # Get project info for quick display
    project_name = json_data["file_info"]["project_name"]
    total_elements = json_data["file_info"]["total_elements"]
    
    # Status indicators
    if status_type == "cached":
        status_icon = "🔄"
        status_text = "Using cached version"
    elif status_type == "processed":
        status_icon = "✅"
        status_text = "Processed successfully"
    else:
        status_icon = "📄"
        status_text = "Ready"
    
    # Single line with all info and expandable details
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"**{status_icon} {filename}**")
    
    with col2:
        st.caption(f"{size_str}")
    
    with col3:
        st.caption(f"{status_text}")
    
    # Optional expandable details
    with st.expander(f"📊 Project: {project_name} ({total_elements} elements)", expanded=False):
        summary_data = {
            "project": project_name,
            "total_elements": total_elements,
            "spaces": len(json_data.get("spaces", [])),
            "walls": len(json_data.get("walls", [])),
            "doors": len(json_data.get("doors", [])),
            "slabs": len(json_data.get("slabs", [])),
            "stairs": len(json_data.get("stairs", []))
        }
        st.json(summary_data)

def show_ifc_file_summary(filename, data):
    """Display summary information for an IFC file (used for cached file listing)."""
    st.write(f"📄 {filename}")
    
    # Add cache indicator
    cache_status = "🔄 Cached" if data.get('cached') else "🆕 New"
    processed_time = data.get('processed_at', 'Unknown')
    
    with st.expander(f"{cache_status} - Processed: {processed_time}", expanded=False):
        json_data = data["json_data"]
        summary_data = {
            "project": json_data["file_info"]["project_name"],
            "total_elements": json_data["file_info"]["total_elements"],
            "spaces": len(json_data.get("spaces", [])),
            "walls": len(json_data.get("walls", [])),
            "doors": len(json_data.get("doors", [])),
            "slabs": len(json_data.get("slabs", [])),
            "stairs": len(json_data.get("stairs", []))
        }
        st.json(summary_data)

def clear_ifc_cache():
    """Clear all cached IFC files."""
    try:
        processed_dir = Path("data/processed_ifc")
        if processed_dir.exists():
            for json_file in processed_dir.glob("*.json"):
                json_file.unlink()
        
        # Clear session state
        st.session_state.processed_ifc_files = {}
        
        st.success("✅ Cache cleared successfully")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error clearing cache: {str(e)}")

def show_legal_docs_upload_section():
    """Display legal documents upload section."""
    st.subheader("📚 Legal Documentation")
    
    with st.container():
        st.markdown("""
        <div style="
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background-color: #f9f9f9;
            margin-bottom: 10px;
        ">
            <p style="margin: 0; color: #666;">
                📄 Drop PDF documents here or click to browse
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_pdfs = st.file_uploader(
            "Upload Legal Documents",
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_pdfs:
            st.success(f"✅ {len(uploaded_pdfs)} PDF file(s) uploaded")
            
            for file in uploaded_pdfs:
                st.write(f"📄 {file.name} ({file.size} bytes)")
                
                # Check if already processed
                if file.name not in st.session_state.uploaded_pdfs:
                    with st.spinner(f"Processing {file.name} for search..."):
                        try:
                            # Read file content
                            file_content = file.read()
                            
                            # Upload to RAG system
                            rag_manager = st.session_state.pdf_rag_manager
                            upload_result = rag_manager.upload_pdf_from_streamlit(file, file_content)
                            
                            if upload_result["status"] == "success":
                                st.session_state.uploaded_pdfs[file.name] = upload_result["data"]
                                st.success(f"✅ {upload_result['message']}")
                                
                                # Show file info
                                file_info = upload_result["data"]
                                st.json({
                                    "file_name": file_info["file_name"],
                                    "file_size_mb": round(file_info["file_size"] / (1024*1024), 2),
                                    "document_type": file_info["document_type"],
                                    "processed_for_search": file_info["processed_for_search"]
                                })
                                
                            elif upload_result["status"] == "already_exists":
                                st.session_state.uploaded_pdfs[file.name] = upload_result["data"]
                                st.info(f"ℹ️ {upload_result['message']}")
                                
                            else:
                                st.error(f"❌ {upload_result['message']}")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing {file.name}: {str(e)}")
                else:
                    st.info(f"✅ {file.name} already processed for search")
                    
                    # Show summary from session state
                    file_info = st.session_state.uploaded_pdfs[file.name]
                    st.json({
                        "file_name": file_info["file_name"],
                        "file_size_mb": round(file_info["file_size"] / (1024*1024), 2),
                        "document_type": file_info.get("document_type", "legal_document"),
                        "processed_for_search": file_info.get("processed_for_search", True)
                    })
        
        # Show knowledge base status
        if st.session_state.uploaded_pdfs:
            with st.expander("📚 Knowledge Base Status"):
                try:
                    rag_manager = st.session_state.pdf_rag_manager
                    status = rag_manager.get_knowledge_base_summary()
                    
                    if status["status"] == "ready":
                        st.success(f"✅ Knowledge base ready with {status['document_count']} documents")
                    elif status["status"] == "empty":
                        st.warning("⚠️ Knowledge base is empty")
                    else:
                        st.error(f"❌ {status['message']}")
                        
                except Exception as e:
                    st.error(f"Error checking knowledge base: {e}")

def show_chat_interface():
    """Display the streaming chat interface with tool usage visualization."""
    st.divider()
    st.subheader("💬 Chat Assistant")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "thinking_steps" in message:
                    # Show thinking process for past messages
                    with st.expander("Thinking process", expanded=True):
                        for step in message["thinking_steps"]:
                            st.markdown(f"**{step['action']}** - {step['description']}")
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about compliance, regulations, or your uploaded files..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response with streaming
        with st.chat_message("assistant"):
            response, thinking_steps = generate_streaming_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history with thinking steps
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "thinking_steps": thinking_steps
        })
        
        # Save session after each interaction
        save_current_session()

def generate_streaming_response(prompt):
    """Generate streaming response with real-time agent log display."""
    thinking_steps = []
    
    # Set up logging to capture agent messages
    log_handler = setup_streamlit_logging()
    log_handler.clear()  # Clear previous messages
    
    # Create containers for live log display
    status_container = st.empty()
    log_container = st.empty()
    
    # Function to update log display immediately when called
    def update_log_display():
        messages = log_handler.get_messages()
        
        if messages:
            # Update status in real-time
            latest = messages[-1]
            status_text = latest["message"]
            
            # Show current status
            is_complete = "complete" in status_text.lower() or "error" in status_text.lower()
            
            with status_container.container():
                if is_complete:
                    st.info(status_text)
                else:
                    # Show with spinner for active processing
                    with st.spinner(status_text):
                        pass
            
            # Update detailed log view
            with log_container.container():
                if len(messages) > 1:
                    with st.expander("View progress", expanded=False):
                        # Show only unique messages (deduplicate)
                        seen = set()
                        unique_messages = []
                        for msg in reversed(messages[-15:]):  # Last 15, reversed
                            msg_key = msg["message"]
                            if msg_key not in seen:
                                seen.add(msg_key)
                                unique_messages.append(msg)
                        unique_messages.reverse()  # Back to chronological order
                        
                        for msg in unique_messages:
                            # Display all messages as normal text without color styling
                            st.caption(msg['message'])
    
    # Initial update
    update_log_display()
    
    # Hook into the log handler to update display immediately when new messages arrive
    original_emit = log_handler.emit
    def emit_with_update(record):
        original_emit(record)
        # Update display immediately when new log arrives
        update_log_display()
    
    # Replace emit method to auto-update
    log_handler.emit = emit_with_update
    
    try:
        # Generate the actual response (this will generate logs)
        # Each log will trigger an immediate display update
        response = generate_detailed_response(prompt, thinking_steps, None)
    finally:
        # Restore original emit
        log_handler.emit = original_emit
    
    # Final update to show all messages
    update_log_display()
    
    # Convert log messages to thinking steps for history
    messages = log_handler.get_messages()
    for msg in messages:
        if msg["level"] in ["INFO", "DEBUG"] and msg["component"] != "Other":
            thinking_steps.append({
                "action": msg['component'],
                "description": msg['message']
            })
    
    return response, thinking_steps

def _format_tool_output(tool_name, tool_output):
    """Format tool output into user-readable response using generic logic."""
    if not isinstance(tool_output, dict):
        return None
    
    # First check if tool has a direct message
    if 'message' in tool_output and tool_output['message']:
        return tool_output['message']
    
    # Then check for valuable data
    if 'data' not in tool_output:
        return None
        
    data = tool_output['data']
    tool_display_name = tool_name.replace('_', ' ').title()
    
    # Handle different data types with generic logic
    if isinstance(data, (int, float)) and data != 0:
        # Special formatting for measurement tools
        if 'distance' in tool_name:
            result = f"Distance calculation result: {data:.3f} units"
        elif 'area' in tool_name:
            result = f"Area calculation result: {data:.2f} square units"
        else:
            result = f"{tool_display_name} result: {data}"
        
        # Add details from logs if available
        if 'logs' in tool_output and tool_output['logs']:
            for log in tool_output['logs']:
                if isinstance(log, str) and any(keyword in log.lower() for keyword in [
                    'calculated', 'distance between', 'area of', 'result'
                ]):
                    result += f"\n\nDetails: {log}"
                    break
        
        return result
    
    elif isinstance(data, str) and data.strip():
        return f"{tool_display_name}: {data}"
    
    elif isinstance(data, list):
        if not data:
            return f"No items found by {tool_display_name.lower()}."
        
        # Handle element lists with type grouping
        if tool_name == 'get_all_elements' and all(isinstance(item, dict) for item in data[:3]):
            element_types = {}
            for element in data[:10]:
                element_type = element.get('type', 'Unknown')
                element_types[element_type] = element_types.get(element_type, 0) + 1
            
            result = f"Found {len(data)} elements in the building model:\n"
            for elem_type, count in element_types.items():
                result += f"- {elem_type}: {count} element(s)\n"
            
            # Add examples
            if data:
                result += f"\nExample elements:\n"
                for i, element in enumerate(data[:3]):
                    elem_id = element.get('id', 'N/A')
                    elem_type = element.get('type', 'Unknown')
                    elem_name = element.get('name', 'Unnamed')
                    result += f"{i+1}. {elem_type}: {elem_name} (ID: {elem_id})\n"
            
            return result.strip()
        
        # Generic list handling
        elif all(isinstance(item, (str, int, float)) for item in data[:3]):
            # Simple list
            display_items = data[:5]
            more_text = f" (and {len(data)-5} more)" if len(data) > 5 else ""
            return f"{tool_display_name}: {', '.join(map(str, display_items))}{more_text}"
        else:
            return f"{tool_display_name}: Found {len(data)} items"
    
    elif isinstance(data, dict):
        if not data:
            return f"{tool_display_name}: No data available."
        
        # Handle building data summary
        if tool_name == 'load_building_data':
            summary_parts = []
            for category, items in data.items():
                if isinstance(items, list):
                    summary_parts.append(f"{category}: {len(items)} items")
                elif isinstance(items, dict):
                    summary_parts.append(f"{category}: data loaded")
            
            if summary_parts:
                return f"Building data loaded successfully:\n" + "\n".join(f"- {part}" for part in summary_parts)
        
        # Handle calculation dictionaries
        elif any(keyword in tool_name for keyword in ['calculate', 'distance', 'area']):
            formatted_items = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if 'distance' in tool_name:
                        formatted_items.append(f"{key}: {value:.3f} units")
                    elif 'area' in tool_name:
                        formatted_items.append(f"{key}: {value:.2f} square units")
                    else:
                        formatted_items.append(f"{key}: {value}")
                elif isinstance(value, str) and value:
                    formatted_items.append(f"{key}: {value}")
            
            if formatted_items:
                return f"{tool_display_name} results:\n" + "\n".join(formatted_items)
        
        # Generic dict handling
        else:
            formatted_items = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    formatted_items.append(f"  {key}: {value}")
                elif isinstance(value, str) and value:
                    formatted_items.append(f"  {key}: {value}")
            
            if formatted_items:
                return f"{tool_display_name} results:\n" + "\n".join(formatted_items)
    
    # Fallback for any other data type
    return f"{tool_display_name}: {str(data)}"

def _synthesize_goal_response(user_prompt, reasoning_result):
    """Synthesize a goal-focused response based on user intent and tool results."""
    
    # Extract key information from tool outputs
    tool_data = {}
    if 'outputs' in reasoning_result:
        for output in reasoning_result['outputs']:
            tool_name = output.get('tool')
            tool_output = output.get('output', {})
            
            if isinstance(tool_output, dict) and 'data' in tool_output:
                tool_data[tool_name] = tool_output['data']
                
                # Also capture logs for additional context
                if 'logs' in tool_output:
                    tool_data[f"{tool_name}_logs"] = tool_output['logs']
    
    # Analyze user intent and synthesize appropriate response
    prompt_lower = user_prompt.lower()
    
    # Distance/measurement queries
    if any(word in prompt_lower for word in ['distance', 'far', 'how far', 'between']):
        if 'calculate_distances' in tool_data:
            distance = tool_data['calculate_distances']
            if isinstance(distance, (int, float)):
                # Look for context in logs
                context = ""
                if 'calculate_distances_logs' in tool_data:
                    for log in tool_data['calculate_distances_logs']:
                        if 'distance between' in log.lower():
                            context = f" {log.split('distance between')[-1].strip()}"
                            break
                
                return f"The distance between the elements is {distance:.2f} units.{context}"
            else:
                return f"Distance calculation completed with result: {distance}"
        else:
            return "I couldn't calculate the distance. Please ensure the building model is loaded and specify which elements you want to measure between."
    
    # Area calculation queries
    elif any(word in prompt_lower for word in ['area', 'size', 'square']):
        if 'calculate_areas' in tool_data:
            area = tool_data['calculate_areas']
            if isinstance(area, (int, float)):
                return f"The calculated area is {area:.2f} square units."
            else:
                return f"Area calculation completed: {area}"
        else:
            return "I couldn't calculate the area. Please ensure the building model is loaded and specify which areas you want to calculate."
    
    # Element/door/wall listing queries
    elif any(word in prompt_lower for word in ['show', 'list', 'all', 'doors', 'walls', 'elements', 'find']):
        if 'get_all_elements' in tool_data:
            elements = tool_data['get_all_elements']
            if isinstance(elements, list) and elements:
                element_type = "elements"
                if 'door' in prompt_lower:
                    element_type = "doors"
                elif 'wall' in prompt_lower:
                    element_type = "walls"
                
                # Count by type if possible
                type_counts = {}
                for element in elements[:20]:  # Sample first 20
                    elem_type = element.get('type', 'Unknown') if isinstance(element, dict) else 'Unknown'
                    type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
                
                response = f"Found {len(elements)} {element_type} in the building."
                if type_counts and len(type_counts) > 1:
                    type_summary = ", ".join([f"{count} {etype}" for etype, count in list(type_counts.items())[:3]])
                    response += f" Types include: {type_summary}."
                
                # Add a few examples if available
                if elements and isinstance(elements[0], dict):
                    examples = []
                    for element in elements[:3]:
                        name = element.get('name', 'Unnamed')
                        if name != 'Unnamed':
                            examples.append(name)
                    if examples:
                        response += f" Examples: {', '.join(examples)}."
                
                return response
            else:
                return f"No {element_type} found in the building model."
        else:
            return "I couldn't retrieve the building elements. Please ensure the building model is loaded properly."
    
    # Building data/loading queries
    elif any(word in prompt_lower for word in ['load', 'building', 'data', 'model', 'file']):
        if 'load_building_data' in tool_data:
            building_data = tool_data['load_building_data']
            if isinstance(building_data, dict):
                categories = []
                for category, items in building_data.items():
                    if isinstance(items, list) and items:
                        categories.append(f"{len(items)} {category}")
                    elif items:
                        categories.append(category)
                
                if categories:
                    return f"Building model loaded successfully. Available data: {', '.join(categories[:5])}."
                else:
                    return "Building model loaded but appears to be empty."
            else:
                return "Building data has been loaded and is ready for analysis."
        else:
            return "I couldn't load the building data. Please check that a valid IFC file is available."
    
    # Generic technical analysis
    else:
        # Look for any successful tool executions
        successful_tools = [tool for tool, data in tool_data.items() if not tool.endswith('_logs') and data]
        
        if successful_tools:
            if len(successful_tools) == 1:
                tool_name = successful_tools[0].replace('_', ' ')
                return f"I've completed the {tool_name} analysis. The task has been processed successfully."
            else:
                return f"I've completed the analysis using {len(successful_tools)} different tools. All requested tasks have been processed."
        else:
            # Check if we have task completion info
            if 'summary' in reasoning_result:
                summary = reasoning_result['summary']
                completed = summary.get('completed_tasks', 0)
                total = summary.get('total_tasks', 0)
                
                if completed > 0:
                    return f"I've processed your request and completed {completed} out of {total} analysis tasks."
            
            return "I've processed your request, but couldn't generate specific results. Please try asking a more specific question about the building analysis."

def generate_detailed_response(prompt, thinking_steps, log_update_callback=None):
    """Generate response using the ReAct agent with tool reasoning and streaming insights."""
    try:
        # Set up logging if callback provided
        if log_update_callback:
            log_handler = setup_streamlit_logging()
        
        # Use the actual ReasoningAgent instead of hardcoded logic
        agent = st.session_state.reasoning_agent
        
        # Track active files in the agent's memory system
        processed_files_dict = st.session_state.get('processed_ifc_files', {})
        if processed_files_dict:
            # Track each processed IFC file as active in the agent's memory
            for filename, data in processed_files_dict.items():
                # Create the JSON file path that was saved during processing
                json_file_path = f"data/processed_ifc/{filename}.json"
                
                # Track this file as active in the agent's memory system
                if hasattr(agent, 'memory_manager') and agent.memory_manager:
                    agent.memory_manager.track_active_file(json_file_path)
            
            # Set session goal with available building data context
            available_files = list(processed_files_dict.keys())
            agent.set_session_goal(
                goal=f"Available building model: {available_files[0]}",
                context=f"Building data available with {len(available_files)} processed IFC file(s)"
            )
        
        # Create enhanced prompt with available data context
        context_info = []
        if processed_files_dict:
            ifc_files = list(processed_files_dict.keys())
            context_info.append(f"Available building models: {', '.join(ifc_files)}")
            
        uploaded_pdfs_dict = st.session_state.get('uploaded_pdfs', {})
        if uploaded_pdfs_dict:
            pdf_files = list(uploaded_pdfs_dict.keys())
            context_info.append(f"Available documents: {', '.join(pdf_files)}")
        
        if context_info:
            enhanced_prompt = f"{prompt}\n\nNote: {' | '.join(context_info)}"
        else:
            enhanced_prompt = f"{prompt}\n\nNote: No files uploaded yet. Please upload IFC building models or PDF documents first."
        
        # Execute the agent with the enhanced prompt (logs will be captured automatically)
        # The callback will be called periodically during execution if provided
        result = agent.process_goal(enhanced_prompt)
        
        # Update log display after agent execution
        if log_update_callback:
            log_update_callback()
        
        # Extract REAL agent reasoning steps for transparency
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            
            # Log messages are already captured by StreamlitLogHandler
            # No need to manually extract thinking steps here
            
            # Extract final response - The agent now handles intelligent response synthesis internally
            response = reasoning.get('message', 'No response generated')
            
            # Only apply fallback logic if the response is actually generic
            if response in ['Goal achieved', 'Goal partially achieved', 'No response generated']:
                if 'summary' in reasoning and 'final_answer' in reasoning['summary']:
                    response = reasoning['summary']['final_answer']
                elif 'summary' in reasoning and isinstance(reasoning['summary'].get('completed_tasks'), list):
                    completed = reasoning['summary']['completed_tasks']
                    if completed:
                        response = f"Analysis complete! Executed {len(completed)} tasks. Please ask a more specific question about the building analysis results."
                else:
                    response = "I've processed your request but couldn't generate a specific response. Please try asking a more specific question about the building analysis."
        else:
            response = str(result)
            
        return response
        
    except Exception as e:
        # Fallback for any agent errors
        error_insight = get_llm_insight("Error Handling", f"Agent error occurred: {str(e)[:50]}")
        thinking_steps.append({
            "action": "⚠️ Error Recovery", 
            "description": error_insight
        })
        return f"I encountered an issue processing your request. Please try rephrasing your question or check that your files are properly uploaded. Error: {str(e)}"


def show_session_sidebar():
    """Display a simplified session management sidebar."""
    with st.sidebar:
        st.header("Chats")
        
        # New chat button
        if st.button("➕ New chat", use_container_width=True):
            create_new_session()
        
        st.divider()
        
        # Session history with simpler layout
        sessions = st.session_state.session_manager.get_all_sessions()
        
        if not sessions:
            st.caption("No previous chats")
        else:
            # Show recent sessions in a cleaner format
            for session in sessions[:15]:  # Show more sessions but cleaner
                session_id = session["session_id"]
                title = session["title"]
                
                # Highlight current session
                is_current = session_id == st.session_state.current_session_id
                
                # Simple session button with clean styling
                button_style = "🟢 " if is_current else ""
                display_title = title if len(title) <= 30 else title[:27] + "..."
                
                # Use container for each session
                with st.container():
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        if st.button(
                            f"{button_style}{display_title}",
                            key=f"session_{session_id}",
                            use_container_width=True,
                            disabled=is_current,
                            type="secondary" if not is_current else "primary"
                        ):
                            load_session(session_id)
                    
                    with col2:
                        if st.button("×", key=f"delete_{session_id}", help="Delete chat"):
                            delete_session(session_id)
            
            # Show count if there are more sessions
            if len(sessions) > 15:
                st.caption(f"... {len(sessions) - 15} more chats")


def create_new_session():
    """Create a new session and switch to it."""
    # Save current session first
    save_current_session()
    
    # Create new session
    new_session_id = st.session_state.session_manager.create_new_session()
    
    # Clear current state
    st.session_state.messages = []
    st.session_state.processed_ifc_files = {}
    st.session_state.uploaded_pdfs = {}
    st.session_state.current_session_id = new_session_id
    
    # Refresh page
    st.rerun()


def load_session(session_id: str):
    """Load an existing session."""
    # Save current session first
    save_current_session()
    
    # Load the selected session
    session_data = st.session_state.session_manager.load_session(session_id)
    
    if session_data:
        st.session_state.current_session_id = session_id
        st.session_state.messages = session_data.get("messages", [])
        st.session_state.processed_ifc_files = session_data.get("processed_ifc_files", {})
        st.session_state.uploaded_pdfs = session_data.get("uploaded_pdfs", {})
        
        # Refresh page
        st.rerun()
    else:
        st.error("Failed to load session")


def delete_session(session_id: str):
    """Delete a session."""
    if session_id == st.session_state.current_session_id:
        # If deleting current session, create a new one
        new_session_id = st.session_state.session_manager.create_new_session()
        st.session_state.current_session_id = new_session_id
        st.session_state.messages = []
        st.session_state.processed_ifc_files = {}
        st.session_state.uploaded_pdfs = {}
    
    # Delete the session
    st.session_state.session_manager.delete_session(session_id)
    
    # Refresh page
    st.rerun()


def save_current_session():
    """Save the current session state."""
    try:
        if "current_session_id" in st.session_state and hasattr(st.session_state, 'session_manager'):
            session_data = {
                "session_id": st.session_state.current_session_id,
                "title": "Chat Session",  # Will be auto-generated based on first message
                "messages": getattr(st.session_state, 'messages', []),
                "processed_ifc_files": getattr(st.session_state, 'processed_ifc_files', {}),
                "uploaded_pdfs": getattr(st.session_state, 'uploaded_pdfs', {})
            }
            
            # Auto-generate title if this is the first save and we have messages
            if st.session_state.messages:
                try:
                    current_session = st.session_state.session_manager.load_session(st.session_state.current_session_id)
                    if current_session:
                        current_title = current_session.get("title", "")
                        if current_title in ["Chat Session", "New Chat"] or "New Chat" in current_title:
                            st.session_state.session_manager.auto_generate_title(st.session_state.current_session_id)
                except Exception as e:
                    # If title generation fails, just continue without it
                    print(f"Warning: Could not auto-generate title: {e}")
            
            st.session_state.session_manager.save_session(st.session_state.current_session_id, session_data)
    except Exception as e:
        # If session saving fails, log the error but don't crash the app
        print(f"Warning: Could not save session: {e}")


if __name__ == "__main__":
    main()