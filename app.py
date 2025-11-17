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
                    with st.expander("🧠 Thinking process", expanded=False):
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
    """Generate detailed response with dynamic LLM-powered tool insights."""
    # Check if we have processed IFC files and PDFs
    processed_files = st.session_state.processed_ifc_files
    uploaded_pdfs = st.session_state.uploaded_pdfs
    rag_manager = st.session_state.pdf_rag_manager
    
    if "ifc" in prompt.lower() or "building" in prompt.lower() or "model" in prompt.lower():
        if processed_files:
            # Generate dynamic insight for building overview
            building_context = f"Found {len(processed_files)} IFC files: {', '.join(processed_files.keys())}"
            building_insight = get_llm_insight(
                "Building Overview",
                building_context,
            )
            thinking_steps.append({
                "action": "📋 Building Overview",
                "description": building_insight
            })
            
            file_list = list(processed_files.keys())
            response = f"I can analyze your uploaded IFC building models: {', '.join(file_list)}. "
            
            # Provide summary of all processed files
            total_spaces = sum(len(data['json_data'].get('spaces', [])) for data in processed_files.values())
            total_walls = sum(len(data['json_data'].get('walls', [])) for data in processed_files.values())
            total_doors = sum(len(data['json_data'].get('doors', [])) for data in processed_files.values())
            total_stairs = sum(len(data['json_data'].get('stairs', [])) for data in processed_files.values())
            
            # Generate dynamic insight for element counting
            element_context = f"Building has {total_spaces} spaces, {total_walls} walls, {total_doors} doors, {total_stairs} stairs"
            element_insight = get_llm_insight(
                "Element Count",
                element_context,
            )
            thinking_steps.append({
                "action": "🔢 Element Count",
                "description": element_insight
            })
            
            response += f"I found {total_spaces} spaces, {total_walls} walls, {total_doors} doors, and {total_stairs} stairs across your building models. "
            response += "Ask me specific questions about compliance, accessibility, or building regulations!"
            
            return response
        else:
            return "I can help you analyze IFC building models for compliance. Please upload your IFC files using the upload section above, and I'll process them for analysis."
    
    elif "space" in prompt.lower() or "room" in prompt.lower():
        if processed_files:
            # Generate dynamic insight for space analysis
            space_context = f"Analyzing spaces in {len(processed_files)} building models for compliance and accessibility"
            space_insight = get_llm_insight(
                "Space Analyzer",
                space_context,
            )
            thinking_steps.append({
                "action": "🏠 IFC Space Analyzer",
                "description": space_insight
            })
            
            all_spaces = []
            for data in processed_files.values():
                all_spaces.extend(data['json_data'].get('spaces', []))
            
            # Dynamic insight for space discovery
            discovery_context = f"Discovered {len(all_spaces)} spaces with various types and properties"
            discovery_insight = get_llm_insight(
                "Space Discovery",
                discovery_context,
            )
            thinking_steps.append({
                "action": "📊 Space Discovery",
                "description": discovery_insight
            })
            
            if all_spaces:
                response = f"I found {len(all_spaces)} spaces in your building models:\n\n"
                
                # Show detailed space information
                for i, space in enumerate(all_spaces[:8], 1):  # Show first 8 spaces
                    space_name = space.get('name', f'Space {i}')
                    space_id = space.get('id', 'Unknown ID')
                    space_type = space.get('type', 'Unknown type')
                    
                    response += f"**{space_name}** (ID: {space_id})\n"
                    response += f"  - Type: {space_type}\n"
                    
                    # Add area if available
                    if 'area' in space:
                        response += f"  - Area: {space['area']} m²\n"
                    
                    # Add height if available
                    if 'height' in space:
                        response += f"  - Height: {space['height']} m\n"
                    
                    # Add position if available
                    position = space.get('position', {})
                    if position:
                        x = position.get('x', 'N/A')
                        y = position.get('y', 'N/A')
                        response += f"  - Position: X={x}, Y={y}\n"
                    
                    response += "\n"
                
                if len(all_spaces) > 8:
                    response += f"...and {len(all_spaces) - 8} more spaces.\n\n"
                
                response += "I can help analyze space compliance, accessibility requirements, or specific room regulations."
                return response
        
        return "Upload your IFC building models first, then I can analyze spaces and rooms for compliance."
    
    elif "door" in prompt.lower():
        if processed_files:
            # Generate dynamic insight for door analysis
            door_context = f"Extracting door specifications from {len(processed_files)} building models to check compliance"
            door_insight = get_llm_insight(
                "Door Analyzer",
                door_context,
            )
            thinking_steps.append({
                "action": "🚪 IFC Door Analyzer",
                "description": door_insight
            })
            
            all_doors = []
            for data in processed_files.values():
                all_doors.extend(data['json_data'].get('doors', []))
            
            # Dynamic insight for compliance checking
            compliance_context = f"Checking {len(all_doors)} doors against ADA and safety standards"
            compliance_insight = get_llm_insight(
                "Door Compliance Check",
                compliance_context,
            )
            thinking_steps.append({
                "action": "📏 Compliance Check",
                "description": compliance_insight
            })
            
            if all_doors:
                response = f"I found {len(all_doors)} doors in your building models:\n\n"
                
                # Show detailed door information
                for i, door in enumerate(all_doors[:5], 1):  # Show first 5 doors
                    door_name = door.get('name', f'Door {i}')
                    door_id = door.get('id', 'Unknown ID')
                    door_width = door.get('width', 'Unknown width')
                    door_height = door.get('height', 'Unknown height')
                    
                    response += f"**{door_name}** (ID: {door_id})\n"
                    if door_width != 'Unknown width':
                        response += f"  - Width: {door_width}\n"
                    if door_height != 'Unknown height':
                        response += f"  - Height: {door_height}\n"
                    response += "\n"
                
                if len(all_doors) > 5:
                    response += f"...and {len(all_doors) - 5} more doors.\n\n"
                
                response += "I can help check door compliance including width requirements, accessibility standards, and fire safety regulations."
                return response
        
        return "Upload your IFC building models first, then I can analyze doors for compliance requirements."
    
    elif "stair" in prompt.lower():
        if processed_files:
            # Generate dynamic insight for stair analysis
            stair_context = f"Scanning building models for stair safety and code compliance"
            stair_insight = get_llm_insight(
                "Stair Analyzer",
                stair_context,
            )
            thinking_steps.append({
                "action": "🪜 IFC Stair Analyzer",
                "description": stair_insight
            })
            
            all_stairs = []
            for data in processed_files.values():
                all_stairs.extend(data['json_data'].get('stairs', []))
            
            if "distance" in prompt.lower():
                distance_context = f"Computing spatial relationships between {len(all_stairs)} stairs for egress analysis"
                distance_insight = get_llm_insight(
                    "Distance Calculator",
                    distance_context,
                )
                thinking_steps.append({
                    "action": "📐 Distance Calculator",
                    "description": distance_insight
                })
            else:
                analysis_context = f"Examining {len(all_stairs)} stairs for code compliance and safety standards"
                analysis_insight = get_llm_insight(
                    "Stair Analysis",
                    analysis_context,
                )
                thinking_steps.append({
                    "action": "📊 Stair Analysis",
                    "description": analysis_insight
                })
            
            if all_stairs:
                response = f"I found {len(all_stairs)} stairs in your building models:\n\n"
                
                # Show detailed stair information and analyze distances
                stair_positions = []
                for i, stair in enumerate(all_stairs, 1):
                    stair_name = stair.get('name', f'Stair {i}')
                    stair_id = stair.get('id', 'Unknown ID')
                    
                    # Get position if available
                    position = stair.get('position', {})
                    if position:
                        x = position.get('x', 'N/A')
                        y = position.get('y', 'N/A')
                        z = position.get('z', 'N/A')
                        stair_positions.append((stair_name, x, y, z))
                        response += f"**{stair_name}** (ID: {stair_id})\n"
                        response += f"  - Position: X={x}, Y={y}, Z={z}\n"
                    else:
                        response += f"**{stair_name}** (ID: {stair_id})\n"
                        response += "  - Position: Not available\n"
                    
                    # Add other stair properties
                    if 'width' in stair:
                        response += f"  - Width: {stair['width']}\n"
                    if 'height' in stair:
                        response += f"  - Height: {stair['height']}\n"
                    response += "\n"
                
                # Calculate distances between stairs if positions are available
                if len(stair_positions) > 1 and "distance" in prompt.lower():
                    response += "**Distances between stairs:**\n"
                    import math
                    
                    for i in range(len(stair_positions)):
                        for j in range(i + 1, len(stair_positions)):
                            name1, x1, y1, z1 = stair_positions[i]
                            name2, x2, y2, z2 = stair_positions[j]
                            
                            try:
                                # Calculate 3D distance
                                distance = math.sqrt((float(x2) - float(x1))**2 + 
                                                   (float(y2) - float(y1))**2 + 
                                                   (float(z2) - float(z1))**2)
                                response += f"- {name1} ↔ {name2}: {distance:.2f} meters\n"
                            except (ValueError, TypeError):
                                response += f"- {name1} ↔ {name2}: Cannot calculate (position data incomplete)\n"
                
                response += "\nI can help analyze stair compliance including width, rise/run ratios, and accessibility requirements."
                return response
            else:
                return "No stairs found in your building models. Upload IFC files that contain stair elements."
        
        return "Upload your IFC building models first, then I can analyze stairs and their positions."
    
    elif "regulation" in prompt.lower() or "legal" in prompt.lower() or "compliance" in prompt.lower():
        if uploaded_pdfs:
            try:
                # Generate dynamic insight for legal search
                legal_context = f"Searching {len(uploaded_pdfs)} legal documents for regulations related to: {prompt[:50]}"
                legal_insight = get_llm_insight(
                    "Legal Document Search",
                    legal_context,
                )
                thinking_steps.append({
                    "action": "📚 Legal Document Search",
                    "description": legal_insight
                })
                
                # Search the legal documents using RAG
                search_result = rag_manager.search_legal_documents(prompt, max_results=3)
                
                # Dynamic insight for RAG processing
                rag_context = f"Processing search results to find specific compliance requirements"
                rag_insight = get_llm_insight(
                    "RAG Processing",
                    rag_context,
                )
                thinking_steps.append({
                    "action": "🔍 RAG Processing",
                    "description": rag_insight
                })
                
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
    
    elif "wall" in prompt.lower():
        if processed_files:
            # Track tool usage
            thinking_steps.append({
                "action": "🧱 IFC Wall Analyzer",
                "description": f"Extracting wall data from {len(processed_files)} building model(s)"
            })
            
            all_walls = []
            for data in processed_files.values():
                all_walls.extend(data['json_data'].get('walls', []))
            
            thinking_steps.append({
                "action": "📊 Wall Classification",
                "description": f"Categorizing {len(all_walls)} walls by type and properties"
            })
            
            if all_walls:
                response = f"I found {len(all_walls)} walls in your building models:\n\n"
                
                # Group walls by type or show first few with details
                wall_types = {}
                for wall in all_walls:
                    wall_type = wall.get('type', 'Unknown type')
                    if wall_type not in wall_types:
                        wall_types[wall_type] = []
                    wall_types[wall_type].append(wall)
                
                # Show wall types summary
                for wall_type, walls in wall_types.items():
                    response += f"**{wall_type}:** {len(walls)} walls\n"
                    
                    # Show first wall of each type with details
                    if walls:
                        wall = walls[0]
                        wall_name = wall.get('name', 'Unnamed wall')
                        if 'thickness' in wall:
                            response += f"  - Example: {wall_name} (Thickness: {wall['thickness']})\n"
                        else:
                            response += f"  - Example: {wall_name}\n"
                    response += "\n"
                
                response += "I can help analyze wall compliance including fire ratings, structural requirements, and thermal properties."
                return response
        
        return "Upload your IFC building models first, then I can analyze walls and their properties."
    
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
        
        # Check for specific building element queries even if not exact matches
        if processed_files:
            # Look for any building element mentioned
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ["distance", "location", "position", "where", "far"]):
                # Try to provide general building analysis
                response = "Based on your building models, I can analyze:\n\n"
                
                # Count all elements
                total_spaces = sum(len(data['json_data'].get('spaces', [])) for data in processed_files.values())
                total_walls = sum(len(data['json_data'].get('walls', [])) for data in processed_files.values())
                total_doors = sum(len(data['json_data'].get('doors', [])) for data in processed_files.values())
                total_stairs = sum(len(data['json_data'].get('stairs', [])) for data in processed_files.values())
                
                if total_spaces > 0:
                    response += f"- **{total_spaces} Spaces** - Ask about specific rooms or space layouts\n"
                if total_doors > 0:
                    response += f"- **{total_doors} Doors** - Ask about door widths, positions, or compliance\n"
                if total_walls > 0:
                    response += f"- **{total_walls} Walls** - Ask about wall types, thicknesses, or properties\n"
                if total_stairs > 0:
                    response += f"- **{total_stairs} Stairs** - Ask about stair positions, distances, or compliance\n"
                
                response += "\nTry asking: 'Tell me about the stairs' or 'Show me the doors' for detailed information."
                return response
        
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