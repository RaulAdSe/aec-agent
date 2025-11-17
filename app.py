"""
AEC Compliance Agent - Streamlit Web Interface
Main entry point for the Streamlit application.
"""

import streamlit as st
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from aec_agent.utils.ifc_to_json import IFCToJSONConverter
from services.pdf_rag_manager import PDFRAGManager
from services.session_manager import SessionManager

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
    """Display the chat interface."""
    st.divider()
    st.subheader("💬 Chat Assistant")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about compliance, regulations, or your uploaded files..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            response = generate_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Save session after each interaction
        save_current_session()

def generate_response(prompt):
    """Generate response to user prompt."""
    # Check if we have processed IFC files and PDFs
    processed_files = st.session_state.processed_ifc_files
    uploaded_pdfs = st.session_state.uploaded_pdfs
    rag_manager = st.session_state.pdf_rag_manager
    
    if "ifc" in prompt.lower() or "building" in prompt.lower() or "model" in prompt.lower():
        if processed_files:
            file_list = list(processed_files.keys())
            response = f"I can analyze your uploaded IFC building models: {', '.join(file_list)}. "
            
            # Provide summary of all processed files
            total_spaces = sum(len(data['json_data'].get('spaces', [])) for data in processed_files.values())
            total_walls = sum(len(data['json_data'].get('walls', [])) for data in processed_files.values())
            total_doors = sum(len(data['json_data'].get('doors', [])) for data in processed_files.values())
            
            response += f"I found {total_spaces} spaces, {total_walls} walls, and {total_doors} doors across your building models. "
            response += "Ask me specific questions about compliance, accessibility, or building regulations!"
            
            return response
        else:
            return "I can help you analyze IFC building models for compliance. Please upload your IFC files using the upload section above, and I'll process them for analysis."
    
    elif "space" in prompt.lower() or "room" in prompt.lower():
        if processed_files:
            all_spaces = []
            for data in processed_files.values():
                all_spaces.extend(data['json_data'].get('spaces', []))
            
            if all_spaces:
                space_names = [space.get('name', 'Unnamed') for space in all_spaces[:5]]
                response = f"I found {len(all_spaces)} spaces in your building models. "
                response += f"Some examples: {', '.join(space_names)}. "
                response += "I can help analyze space compliance, accessibility requirements, or specific room regulations."
                return response
        
        return "Upload your IFC building models first, then I can analyze spaces and rooms for compliance."
    
    elif "door" in prompt.lower():
        if processed_files:
            all_doors = []
            for data in processed_files.values():
                all_doors.extend(data['json_data'].get('doors', []))
            
            if all_doors:
                response = f"I found {len(all_doors)} doors in your building models. "
                response += "I can help check door compliance including width requirements, accessibility standards, and fire safety regulations."
                return response
        
        return "Upload your IFC building models first, then I can analyze doors for compliance requirements."
    
    elif "regulation" in prompt.lower() or "legal" in prompt.lower() or "compliance" in prompt.lower():
        if uploaded_pdfs:
            try:
                # Search the legal documents using RAG
                search_result = rag_manager.search_legal_documents(prompt, max_results=3)
                
                if search_result["status"] == "success":
                    response = search_result.get("answer", "No answer found.")
                    
                    # Add citations if available
                    citations = search_result.get("formatted_citations", [])
                    if citations:
                        response += "\n\n**Sources:**\n"
                        for i, citation in enumerate(citations[:3], 1):
                            source = citation.get('display_name', 'Unknown')
                            response += f"{i}. {source}\n"
                    
                    return response
                    
                elif search_result["status"] == "no_documents":
                    return search_result.get("answer", "No documents in knowledge base. Please upload legal documents first.")
                else:
                    return f"Sorry, I couldn't search the legal documents right now. {search_result.get('message', 'Please try again.')}"
                    
            except Exception as e:
                return f"Sorry, there was an error searching the legal documents: {str(e)}"
        else:
            return "For regulatory compliance questions, I can reference the legal documents you've uploaded. Please add relevant PDF documents to enhance my knowledge base using the upload section above."
    
    elif "hello" in prompt.lower() or "hi" in prompt.lower():
        return "Hello! I'm your AEC Compliance Assistant. Upload your IFC building models and legal documents, then ask me about compliance, regulations, accessibility, or safety requirements."
    
    else:
        # Try to search both IFC data and legal documents for general queries
        if uploaded_pdfs and ("requirement" in prompt.lower() or "standard" in prompt.lower() or "code" in prompt.lower()):
            try:
                # Search legal documents for general compliance questions
                search_result = rag_manager.search_legal_documents(prompt, max_results=2)
                
                if search_result["status"] == "success":
                    response = search_result.get("answer", "No answer found.")
                    
                    # Add building model context if available
                    if processed_files:
                        response += f"\n\nI also have access to {len(processed_files)} building model(s) for detailed analysis."
                    
                    return response
            except:
                pass  # Fall through to default response
        
        # Default response based on available data
        resources = []
        if processed_files:
            resources.append(f"{len(processed_files)} building model(s)")
        if uploaded_pdfs:
            resources.append(f"{len(uploaded_pdfs)} legal document(s)")
            
        if resources:
            return f"I have access to {' and '.join(resources)}. Ask me about compliance, accessibility, building regulations, or specific elements like doors, spaces, walls, or stairs."
        else:
            return "I'm ready to help with AEC compliance questions. Upload your IFC models and legal documents to get started with detailed analysis."

def show_session_sidebar():
    """Display the session management sidebar."""
    with st.sidebar:
        st.header("💬 Chat Sessions")
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True):
            create_new_session()
        
        st.divider()
        
        # Current session info
        current_session = st.session_state.session_manager.load_session(st.session_state.current_session_id)
        if current_session:
            st.write(f"**Current:** {current_session.get('title', 'New Chat')[:20]}...")
            
            # Session stats
            msg_count = len(st.session_state.messages)
            ifc_count = len(st.session_state.processed_ifc_files)
            pdf_count = len(st.session_state.uploaded_pdfs)
            
            st.caption(f"📝 {msg_count} messages | 🏗️ {ifc_count} IFC | 📄 {pdf_count} PDF")
        
        st.divider()
        
        # Session history
        st.subheader("Recent Chats")
        
        sessions = st.session_state.session_manager.get_all_sessions()
        
        if not sessions:
            st.info("No chat history yet")
        else:
            # Show recent sessions
            for session in sessions[:10]:  # Show last 10 sessions
                session_id = session["session_id"]
                title = session["title"]
                message_count = session["message_count"]
                
                # Highlight current session
                is_current = session_id == st.session_state.current_session_id
                
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    if st.button(
                        f"{'🟢' if is_current else '💬'} {title[:25]}...",
                        key=f"session_{session_id}",
                        use_container_width=True,
                        disabled=is_current
                    ):
                        load_session(session_id)
                
                with col2:
                    if st.button("🗑️", key=f"delete_{session_id}", help="Delete session"):
                        delete_session(session_id)
                
                # Show session info
                if message_count > 0:
                    st.caption(f"{message_count} messages")
                else:
                    st.caption("Empty")
                
                st.divider()
            
            # Show overall stats
            if len(sessions) > 10:
                st.caption(f"... and {len(sessions) - 10} more sessions")
            
        # Session statistics
        stats = st.session_state.session_manager.get_session_stats()
        with st.expander("📊 Statistics"):
            st.write(f"**Total Sessions:** {stats['total_sessions']}")
            st.write(f"**Total Messages:** {stats['total_messages']}")
            st.write(f"**IFC Files:** {stats['total_ifc_files']}")
            st.write(f"**PDF Files:** {stats['total_pdf_files']}")


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