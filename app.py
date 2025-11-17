"""
AEC Compliance Agent - Streamlit Web Interface
Main entry point for the Streamlit application.
"""

import streamlit as st
import os
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from aec_agent.utils.ifc_to_json import IFCToJSONConverter
from services.pdf_rag_manager import PDFRAGManager
from services.session_manager import SessionManager
from aec_agent.agent import create_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm_insight(action, context):
    """Get LLM-generated insight about what the agent is doing."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Simple prompt to explain what's happening
        prompt = f"In 3-4 words, what is an AI agent doing when: {action} with context: {context[:50]}"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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
        # Initialize the actual ReAct agent with 3-layer memory management
        st.session_state.reasoning_agent = create_agent(
            model_name="gpt-4o-mini",
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

def show_ifc_upload_section():
    """Display IFC file upload section."""
    st.subheader("📊 IFC Building Models")
    
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
        
        if uploaded_ifc:
            st.success(f"✅ {len(uploaded_ifc)} IFC file(s) uploaded")
            
            for file in uploaded_ifc:
                st.write(f"📄 {file.name} ({file.size} bytes)")
                
                # Check if already processed
                if file.name not in st.session_state.processed_ifc_files:
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
                                "processed_at": str(datetime.now())
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
                            
                            st.success(f"✅ Processed {file.name} → {json_file_path}")
                            
                            # Show summary of extracted elements
                            st.json({
                                "project": json_data["file_info"]["project_name"],
                                "total_elements": json_data["file_info"]["total_elements"],
                                "spaces": len(json_data.get("spaces", [])),
                                "walls": len(json_data.get("walls", [])),
                                "doors": len(json_data.get("doors", [])),
                                "slabs": len(json_data.get("slabs", [])),
                                "stairs": len(json_data.get("stairs", []))
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Error processing {file.name}: {str(e)}")
                else:
                    st.info(f"✅ {file.name} already processed")
                    
                    # Show summary from session state
                    data = st.session_state.processed_ifc_files[file.name]["json_data"]
                    st.json({
                        "project": data["file_info"]["project_name"],
                        "total_elements": data["file_info"]["total_elements"],
                        "spaces": len(data.get("spaces", [])),
                        "walls": len(data.get("walls", [])),
                        "doors": len(data.get("doors", [])),
                        "slabs": len(data.get("slabs", [])),
                        "stairs": len(data.get("stairs", []))
                    })

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
                    with st.expander("🧠 Thinking process", expanded=True):
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
    """Generate streaming response with LLM-powered insights."""
    thinking_steps = []
    
    # Create containers for progress indicators
    progress_container = st.empty()
    thinking_container = st.empty()
    response_container = st.empty()
    
    # Step 1: Analyzing user input
    with progress_container:
        st.info("🧠 **Analyzing your question...**")
    
    # Generate dynamic insight about question analysis
    analysis_insight = get_llm_insight(
        "Analyzing Question", 
        prompt[:100], 
    )
    
    thinking_steps.append({
        "action": "🧠 Analyzing Question",
        "description": analysis_insight
    })
    
    with thinking_container:
        st.markdown(f"🧠 **{analysis_insight}**")
    time.sleep(0.6)
    
    # Step 2: Check available data sources
    processed_files = st.session_state.processed_ifc_files
    uploaded_pdfs = st.session_state.uploaded_pdfs
    
    with progress_container:
        st.info("🔍 **Checking available data sources...**")
    
    data_sources = []
    if processed_files:
        data_sources.append(f"{len(processed_files)} IFC building model(s)")
    if uploaded_pdfs:
        data_sources.append(f"{len(uploaded_pdfs)} legal document(s)")
    
    # Generate dynamic insight about data sources
    data_context = f"Found: {', '.join(data_sources) if data_sources else 'No uploaded files'}"
    data_insight = get_llm_insight(
        "Data Source Check",
        data_context,
    )
    
    thinking_steps.append({
        "action": "🔍 Data Source Check",
        "description": data_insight
    })
    
    with thinking_container:
        st.markdown(f"🔍 **{data_insight}**")
    time.sleep(0.5)
    
    # Step 3: Determine which tools to use
    with progress_container:
        st.info("🔧 **Selecting analysis tools...**")
    
    tools_to_use = []
    if any(keyword in prompt.lower() for keyword in ["ifc", "building", "model", "space", "door", "stair", "wall"]):
        tools_to_use.append("IFC Building Data Analyzer")
    if any(keyword in prompt.lower() for keyword in ["regulation", "code", "compliance", "legal", "standard"]):
        tools_to_use.append("Legal Document Search")
    
    # Generate dynamic insight about tool selection
    tool_context = f"For query about {prompt[:50]}, selected: {', '.join(tools_to_use) if tools_to_use else 'general knowledge'}"
    tool_insight = get_llm_insight(
        "Tool Selection",
        tool_context,
    )
    
    thinking_steps.append({
        "action": "🔧 Tool Selection", 
        "description": tool_insight
    })
    
    with thinking_container:
        st.markdown(f"🔧 **{tool_insight}**")
    time.sleep(0.7)
    
    # Step 4: Execute analysis based on prompt type
    with progress_container:
        st.info("⚡ **Analyzing building data...**")
    
    # Generate specific analysis insight
    analysis_context = f"Analyzing {prompt} with available data: {data_context}"
    analysis_insight = get_llm_insight(
        "Building Analysis",
        analysis_context,
    )
    
    with thinking_container:
        st.markdown(f"⚡ **{analysis_insight}**")
    time.sleep(0.9)
    
    # Generate the actual response with dynamic insights
    response = generate_detailed_response(prompt, thinking_steps)
    
    # Step 5: Finalizing response
    with progress_container:
        st.info("📝 **Preparing your analysis...**")
    
    # Generate final insight
    final_insight = get_llm_insight(
        "Response Preparation",
        f"Compiling comprehensive analysis for: {prompt[:50]}",
    )
    
    thinking_steps.append({
        "action": "📝 Response Generation",
        "description": final_insight
    })
    
    with thinking_container:
        st.markdown(f"📝 **{final_insight}**")
    time.sleep(0.5)
    
    # Clear progress indicators and show final response
    progress_container.empty()
    thinking_container.empty()
    
    return response, thinking_steps

def generate_detailed_response(prompt, thinking_steps):
    """Generate response using the ReAct agent with tool reasoning and streaming insights."""
    try:
        # Use the actual ReasoningAgent instead of hardcoded logic
        agent = st.session_state.reasoning_agent
        
        # Add context about available data to the agent and set session context
        if st.session_state.processed_ifc_files:
            # Add building data to agent's session context
            for filename, data in st.session_state.processed_ifc_files.items():
                agent.set_session_goal(
                    goal=f"Available building model: {filename}",
                    context=f"IFC data loaded with {data['json_data']['file_info']['total_elements']} elements"
                )
        
        # Create enhanced prompt with available data context
        context_info = []
        if st.session_state.processed_ifc_files:
            ifc_files = list(st.session_state.processed_ifc_files.keys())
            context_info.append(f"Available building models: {', '.join(ifc_files)}")
            
        if st.session_state.uploaded_pdfs:
            pdf_files = list(st.session_state.uploaded_pdfs.keys())
            context_info.append(f"Available documents: {', '.join(pdf_files)}")
        
        if context_info:
            enhanced_prompt = f"{prompt}\n\nNote: {' | '.join(context_info)}"
        else:
            enhanced_prompt = f"{prompt}\n\nNote: No files uploaded yet. Please upload IFC building models or PDF documents first."
        
        # Add agent thinking step
        agent_insight = get_llm_insight("Agent Reasoning", f"Autonomous analysis of: {prompt[:50]}")
        thinking_steps.append({
            "action": "🤖 ReAct Agent",
            "description": agent_insight
        })
        
        # Execute the agent with the enhanced prompt
        result = agent.process_goal(enhanced_prompt)
        
        # Extract REAL agent reasoning steps for transparency
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            
            # Add actual agent reasoning steps to thinking_steps
            if 'summary' in reasoning:
                summary = reasoning['summary']
                
                # Show goal decomposition
                if 'goal_analysis' in summary:
                    thinking_steps.append({
                        "action": "🎯 Goal Decomposition",
                        "description": f"Broke down query into {len(summary.get('planned_tasks', []))} actionable tasks"
                    })
                
                # Show completed tasks
                if 'completed_tasks' in summary:
                    for task in summary['completed_tasks'][:3]:  # Show first 3 tasks
                        thinking_steps.append({
                            "action": f"✅ {task.get('name', 'Task')}",
                            "description": task.get('result', 'Task completed')[:60] + '...'
                        })
                
                # Show failed tasks
                if 'failed_tasks' in summary:
                    for task in summary['failed_tasks'][:2]:  # Show failed attempts
                        thinking_steps.append({
                            "action": f"⚠️ {task.get('name', 'Task')}",
                            "description": f"Task blocked: {task.get('error', 'Unknown issue')[:50]}"
                        })
                
                # Show tools used
                if 'tools_used' in summary:
                    tools = summary['tools_used']
                    if tools:
                        thinking_steps.append({
                            "action": "🔧 Tools Executed",
                            "description": f"Used {len(tools)} tools: {', '.join(tools[:3])}"
                        })
            
            # Extract final response
            response = reasoning.get('message', 'No response generated')
            if 'summary' in reasoning and 'final_answer' in reasoning['summary']:
                response = reasoning['summary']['final_answer']
            elif 'summary' in reasoning and 'completed_tasks' in reasoning['summary']:
                completed = reasoning['summary']['completed_tasks']
                if completed:
                    response = f"Analysis complete! Executed {len(completed)} tasks.\n\n" + response
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
    if "current_session_id" in st.session_state:
        session_data = {
            "session_id": st.session_state.current_session_id,
            "title": "Chat Session",  # Will be auto-generated based on first message
            "messages": st.session_state.messages,
            "processed_ifc_files": st.session_state.processed_ifc_files,
            "uploaded_pdfs": st.session_state.uploaded_pdfs
        }
        
        # Auto-generate title if this is the first save and we have messages
        if st.session_state.messages:
            current_session = st.session_state.session_manager.load_session(st.session_state.current_session_id)
            if current_session and current_session.get("title") in ["Chat Session", "New Chat"] or "New Chat" in current_session.get("title", ""):
                st.session_state.session_manager.auto_generate_title(st.session_state.current_session_id)
        
        st.session_state.session_manager.save_session(st.session_state.current_session_id, session_data)


if __name__ == "__main__":
    main()