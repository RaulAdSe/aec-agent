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

# Set page config
st.set_page_config(
    page_title="AEC Compliance Agent",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    """Main Streamlit application."""
    st.title("🏗️ AEC Compliance Agent")
    st.markdown("**Architecture, Engineering & Construction Compliance Assistant**")
    
    # Initialize session state for chat and processed files
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_ifc_files" not in st.session_state:
        st.session_state.processed_ifc_files = {}
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = {}
    
    # Create two columns for upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        show_ifc_upload_section()
    
    with col2:
        show_legal_docs_upload_section()
    
    # Chat interface
    show_chat_interface()

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
                # Placeholder for processing logic
                st.info("*Processing logic will be implemented in pdf-upload-rag branch*")

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
        
        # Generate assistant response (placeholder)
        with st.chat_message("assistant"):
            response = generate_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def generate_response(prompt):
    """Generate response to user prompt."""
    # Check if we have processed IFC files
    processed_files = st.session_state.processed_ifc_files
    
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
    
    elif "regulation" in prompt.lower() or "legal" in prompt.lower():
        return "For regulatory compliance questions, I can reference the legal documents you've uploaded. Please add relevant PDF documents to enhance my knowledge base using the upload section above."
    
    elif "hello" in prompt.lower() or "hi" in prompt.lower():
        return "Hello! I'm your AEC Compliance Assistant. Upload your IFC building models and legal documents, then ask me about compliance, regulations, accessibility, or safety requirements."
    
    else:
        if processed_files:
            return f"I have access to {len(processed_files)} building model(s). Ask me about compliance, accessibility, building regulations, or specific elements like doors, spaces, walls, or stairs."
        else:
            return "I'm ready to help with AEC compliance questions. Upload your IFC models and legal documents to get started with detailed analysis."

if __name__ == "__main__":
    main()