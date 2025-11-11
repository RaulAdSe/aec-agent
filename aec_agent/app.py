"""Streamlit web application for AEC Compliance Agent.

Main deployment entry point for the compliance verification system.
"""

import streamlit as st
import json
import io
from typing import Dict, Any, Optional
from pathlib import Path

from src.core.config import config
from src.core.logger import get_logger
from src.agents.compliance_agent import ComplianceAgent, ComplianceAgentConfig

# Configure page
st.set_page_config(
    page_title="AEC Compliance Agent",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = get_logger(__name__)

@st.cache_resource
def initialize_agent() -> ComplianceAgent:
    """Initialize the compliance agent (cached)."""
    try:
        agent_config = ComplianceAgentConfig(
            use_toon=config.use_toon,
            verbose=config.debug
        )
        agent = ComplianceAgent(agent_config)
        logger.info("Streamlit agent initialized successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        st.error(f"Failed to initialize agent: {e}")
        st.stop()

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🏗️ AEC Compliance Agent")
    st.markdown("**Building Code Compliance Verification System**")
    st.markdown("Verify compliance with Spanish building codes (CTE DB-SI, CTE DB-SUA)")
    
    # Sidebar - Configuration & Status
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Agent status
        agent = initialize_agent()
        status = agent.get_status()
        
        if status.get("status") == "ready":
            st.success("✅ Agent Ready")
        else:
            st.error("❌ Agent Error")
        
        st.info(f"**Model:** {config.default_model}")
        st.info(f"**TOON Format:** {'✅' if config.use_toon else '❌'}")
        
        # API Key Status
        st.subheader("🔑 API Configuration")
        openai_status = "✅ Configured" if config.openai_api_key else "❌ Missing"
        st.text(f"OpenAI API Key: {openai_status}")
        
        if not config.openai_api_key:
            st.warning("⚠️ OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        # Memory summary
        if st.button("🔄 Clear Memory"):
            agent.clear_memory()
            st.success("Memory cleared!")
            st.rerun()
        
        memory_summary = agent.get_memory_summary()
        st.subheader("🧠 Memory")
        st.json(memory_summary)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🏗️ Building Analysis", "📊 Analysis History", "⚙️ Settings"])
    
    with tab1:
        st.header("Building Data Analysis")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Analysis Type",
            ["general", "fire_safety", "accessibility"],
            help="Select the type of compliance analysis to perform"
        )
        
        # Data input methods
        input_method = st.radio(
            "Data Input Method",
            ["JSON Text", "File Upload"],
            horizontal=True
        )
        
        building_data = None
        
        if input_method == "JSON Text":
            # JSON text input
            st.subheader("Building Data (JSON)")
            
            # Sample data button
            if st.button("📋 Load Sample Data"):
                sample_data = {
                    "metadata": {
                        "project_name": "Sample Building",
                        "total_area": 500.0,
                        "levels": 2,
                        "location": "Madrid, Spain"
                    },
                    "rooms": [
                        {
                            "id": "R001",
                            "name": "Office 1",
                            "area": 25.0,
                            "use": "office",
                            "occupancy_load": 3
                        },
                        {
                            "id": "R002", 
                            "name": "Conference Room",
                            "area": 50.0,
                            "use": "assembly",
                            "occupancy_load": 20
                        }
                    ],
                    "doors": [
                        {
                            "id": "D001",
                            "width": 0.9,
                            "height": 2.1,
                            "fire_rated": True
                        },
                        {
                            "id": "D002",
                            "width": 1.2,
                            "height": 2.1,
                            "fire_rated": False
                        }
                    ]
                }
                st.session_state.json_input = json.dumps(sample_data, indent=2)
            
            # JSON input area
            json_text = st.text_area(
                "Enter building data as JSON:",
                value=st.session_state.get('json_input', '{}'),
                height=300,
                help="Paste your building data in JSON format"
            )
            
            try:
                if json_text.strip():
                    building_data = json.loads(json_text)
                    st.success("✅ Valid JSON format")
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {e}")
        
        else:
            # File upload
            st.subheader("Upload Building Data File")
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['json'],
                help="Upload a JSON file containing building data"
            )
            
            if uploaded_file is not None:
                try:
                    building_data = json.load(uploaded_file)
                    st.success(f"✅ File loaded: {uploaded_file.name}")
                    
                    # Show preview
                    with st.expander("📋 Data Preview"):
                        st.json(building_data)
                        
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON file: {e}")
        
        # Analysis button
        if st.button("🔍 Analyze Building", type="primary", disabled=building_data is None):
            if building_data:
                with st.spinner("Analyzing building data..."):
                    try:
                        # Perform analysis
                        if analysis_type == "general":
                            results = agent.process(building_data)
                        else:
                            results = agent.analyze_specific_compliance(building_data, analysis_type)
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        # Results in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Analysis Results")
                            st.json(results)
                        
                        with col2:
                            st.subheader("💾 Download Results")
                            
                            # JSON download
                            json_str = json.dumps(results, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_str,
                                file_name=f"compliance_analysis_{analysis_type}.json",
                                mime="application/json"
                            )
                            
                            # Show project info if available
                            if "metadata" in building_data:
                                st.info(f"**Project:** {building_data['metadata'].get('project_name', 'Unknown')}")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
                        logger.error(f"Analysis failed: {e}")
    
    with tab2:
        st.header("📊 Analysis History")
        
        # Memory summary
        memory_summary = agent.get_memory_summary()
        
        if memory_summary.get("total_entries", 0) > 0:
            st.subheader("Recent Analyses")
            
            # Get history from agent
            history = agent.memory.get_history(limit=10)
            
            if history:
                for i, entry in enumerate(reversed(history)):
                    with st.expander(f"Analysis {i+1} - {entry['timestamp'][:19]}"):
                        st.json(entry['data'])
            else:
                st.info("No analysis history available")
        else:
            st.info("No analyses performed yet")
    
    with tab3:
        st.header("⚙️ Settings")
        
        st.subheader("Environment Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text(f"Environment: {config.environment}")
            st.text(f"Model: {config.default_model}")
            st.text(f"Temperature: {config.temperature}")
            st.text(f"Max Tokens: {config.max_tokens}")
        
        with col2:
            st.text(f"TOON Format: {config.use_toon}")
            st.text(f"Debug Mode: {config.debug}")
            st.text(f"LangSmith Project: {config.langchain_project}")
        
        st.subheader("LangSmith Integration")
        st.info("LangSmith tracing enables detailed monitoring of agent operations in LangChain Studio")
        
        if config.langchain_tracing_v2:
            st.success("✅ LangSmith tracing enabled")
            if config.langchain_api_key:
                st.success("✅ LangSmith API key configured")
            else:
                st.warning("⚠️ LangSmith API key not configured")
        else:
            st.info("ℹ️ LangSmith tracing disabled")

if __name__ == "__main__":
    main()