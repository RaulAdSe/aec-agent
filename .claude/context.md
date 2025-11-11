# context.md - Agentic AI for Building Code Compliance Verification

## Project Overview

This project is a Proof of Concept (POC) demonstrating how autonomous AI agents can verify building projects against Spanish building codes, specifically the **Código Técnico de la Edificación (CTE)** regulations. The system represents a practical application of agentic AI in the Architecture, Engineering, and Construction (AEC) sector.

## Core Problem & Solution

### The Problem
Manual verification of building code compliance is:
- Time-consuming and error-prone
- Requires expert knowledge of extensive regulations
- Difficult to trace back to specific regulatory articles
- Expensive for technical offices

### Our Solution
An autonomous AI agent system that:
- Automatically extracts data from architectural plans (IFC files)
- Performs geometric calculations and spatial analysis
- Queries building regulations using natural language
- Verifies compliance autonomously
- Generates reports with full traceability to specific regulatory articles

## Project Philosophy

**Practical Over Impressive**: We prioritize tools and approaches that can be immediately implemented in real technical office environments, rather than showcasing cutting-edge technology that remains impractical.

**Educational First**: This is designed as a learning tool. Every component is built to be understood, explained, and extended by students and practitioners.

**Transparency**: The system shows its reasoning process - why it made certain decisions, which regulations it consulted, and how it reached conclusions.

## Technical Architecture

### Four-Pillar Design

#### 1. **Data Extraction**
- Extract structured data from architectural plans (DWG/RVT formats)
- Convert geometric entities into queryable data structures
- Validate extracted data against schemas

#### 2. **Geometric Calculations**
- Calculate areas, perimeters, distances
- Analyze circulation paths and evacuation routes
- Perform spatial relationship analysis
- Graph-based modeling of building topology

#### 3. **RAG (Retrieval Augmented Generation)**
- Vector database of Spanish building codes (CTE documents)
- Natural language querying of regulations
- Context-aware responses with source citations
- Multilingual support (Spanish regulatory documents)

#### 4. **ReAct Agent**
- Autonomous reasoning and acting cycle
- Tool selection and orchestration
- Iterative problem-solving
- Compliance verification workflow

## Project Context from Experience

**Prior Achievement**: Successfully applied AI to detect problematic concrete types with high accuracy, demonstrating practical AI applications in construction diagnostics.

**Target Audience Insights**: Students are learning generative AI, data processing, and neural networks - they need bridges between theory and practice.

**Industry Alignment**: The Spanish AEC sector is actively exploring AI applications but needs practical, implementable solutions rather than research prototypes.

## Success Metrics for POC

1. **Functional Demonstration**
   - Successfully verify compliance for sample building project
   - Show agent reasoning process in real-time
   - Generate compliance report with regulatory citations

2. **Educational Value**
   - Students can understand each component
   - Code is readable and well-documented
   - Tutorials enable independent exploration

3. **Technical Soundness**
   - Accurate geometric calculations
   - Relevant regulatory retrieval
   - Correct compliance assessments

4. **Extensibility**
   - Clear pathways to add new regulations
   - Modular design for new capabilities
   - Well-documented APIs and interfaces

## Known Scope Limitations

This is a **POC**, not a production system:
- Limited to basic geometric elements from DWG files
- Focuses on CTE DB-SI (fire safety) regulations
- Single-agent architecture (no multi-agent collaboration yet)
- No computer vision for plan interpretation
- Manual data validation required
- Not certified for official compliance submissions

## Version 2.0 Vision

After presenting the initial version and gathering insights, this rebuild incorporates:
- **Streamlit** for better user experience and interactivity
- **LangSmith** for observability and debugging
- **Token-oriented data structures** for efficiency
- **Cleaner separation of concerns** between pillars
- **Better prompt engineering** based on live demo learnings
- **More robust error handling** and validation
- **Enhanced educational documentation** based on student feedback

## Related Documents

- `IMPLEMENTATION_ROADMAP.md` - Day-by-day development plan
- `guia_docent.pdf` - Course syllabus and learning objectives
- `Mails` - Communication about presentation logistics
- Tutorial notebooks (to be created) - Interactive learning materials

---

**Last Updated**: November 2025  
**Status**: Rebuilding from scratch with accumulated knowledge  
**Next Milestone**: UPC-EPSEB presentation on October 29, 2025