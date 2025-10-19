#!/usr/bin/env python3
"""
Full pipeline script for AEC compliance verification.

This script demonstrates the complete workflow:
1. Load project data from extracted JSON
2. Setup RAG system for building code queries
3. Create and run the ReAct compliance agent
4. Generate comprehensive compliance reports

Usage:
    python scripts/run_full_pipeline.py [options]

Options:
    --project-file PATH    Path to project JSON file (default: data/extracted/tech_tower_p2.json)
    --query TEXT          Custom verification query
    --mode MODE           Agent mode: verification, door_widths, egress_distances, comprehensive
    --output-dir PATH     Output directory for reports (default: outputs/reports)
    --verbose             Enable verbose output
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agent.tools import load_project_data, set_vectorstore_manager
from agent.graph import create_compliance_agent, comprehensive_verification
from rag.vectorstore_manager import VectorstoreManager
from rag.qa_chain import create_qa_chain


class CompliancePipeline:
    """
    Complete pipeline for AEC compliance verification.
    
    This class orchestrates the entire process from data loading
    to compliance verification and report generation.
    """
    
    def __init__(
        self,
        project_file: Path,
        vectorstore_dir: Path,
        normativa_dir: Path,
        output_dir: Path,
        verbose: bool = False
    ):
        """
        Initialize the compliance pipeline.
        
        Args:
            project_file: Path to project JSON file
            vectorstore_dir: Directory for RAG vectorstore
            normativa_dir: Directory containing building code PDFs
            output_dir: Directory for output reports
            verbose: Enable verbose output
        """
        self.project_file = project_file
        self.vectorstore_dir = vectorstore_dir
        self.normativa_dir = normativa_dir
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Initialize components
        self.project_data = None
        self.rag_manager = None
        self.agent = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_pipeline(self) -> bool:
        """
        Setup all pipeline components.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self._log("Setting up compliance verification pipeline...")
            
            # 1. Load project data
            self._log("1. Loading project data...")
            self.project_data = load_project_data(self.project_file)
            self._log(f"   ✅ Loaded project: {self.project_data.metadata.project_name}")
            
            # 2. Setup RAG system
            self._log("2. Setting up RAG system...")
            self.rag_manager = VectorstoreManager(self.vectorstore_dir)
            
            if self.vectorstore_dir.exists():
                self._log("   Loading existing vectorstore...")
                self.rag_manager.load_existing()
            else:
                self._log("   Creating vectorstore from PDFs...")
                self.rag_manager.create_from_pdfs(self.normativa_dir)
            
            # Create QA chain wrapper
            retriever = self.rag_manager.get_retriever(k=3)
            qa_chain = create_qa_chain(retriever)
            
            class RAGWrapper:
                def __init__(self, qa_chain):
                    self.qa_chain = qa_chain
                
                def query(self, question: str) -> Dict[str, Any]:
                    result = self.qa_chain({"query": question})
                    return {
                        "answer": result.get("result", ""),
                        "sources": [doc.metadata for doc in result.get("source_documents", [])],
                        "confidence": 0.8,  # Placeholder
                        "regulation_references": []
                    }
            
            set_vectorstore_manager(RAGWrapper(qa_chain))
            self._log("   ✅ RAG system ready")
            
            # 3. Create compliance agent
            self._log("3. Creating compliance agent...")
            self.agent = create_compliance_agent(
                model_name="gemini-pro",
                temperature=0.1,
                max_iterations=15
            )
            self._log("   ✅ Agent ready")
            
            self._log("✅ Pipeline setup complete!")
            return True
            
        except Exception as e:
            self._log(f"❌ Pipeline setup failed: {e}")
            return False
    
    def run_verification(
        self,
        query: Optional[str] = None,
        mode: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Run compliance verification.
        
        Args:
            query: Custom verification query
            mode: Agent mode (verification, door_widths, egress_distances, comprehensive)
            
        Returns:
            Verification results
        """
        if not self.agent:
            raise RuntimeError("Pipeline not setup. Call setup_pipeline() first.")
        
        # Default query if none provided
        if not query:
            query = self._get_default_query(mode)
        
        self._log(f"Running {mode} verification...")
        self._log(f"Query: {query}")
        
        # Run verification
        results = self.agent.verify_compliance(query, agent_mode=mode)
        
        self._log(f"✅ Verification complete in {results['iterations']} iterations")
        return results
    
    def generate_report(
        self,
        results: Dict[str, Any],
        mode: str = "comprehensive"
    ) -> Path:
        """
        Generate a compliance verification report.
        
        Args:
            results: Verification results
            mode: Agent mode used
            
        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"compliance_report_{mode}_{timestamp}.json"
        
        # Prepare report data
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "project_file": str(self.project_file),
                "agent_mode": mode,
                "iterations": results["iterations"],
                "execution_successful": results["execution_successful"]
            },
            "project_info": {
                "name": self.project_data.metadata.project_name if self.project_data else "Unknown",
                "building_type": self.project_data.metadata.building_type if self.project_data else "Unknown",
                "levels": len(self.project_data.levels) if self.project_data else 0
            },
            "verification_results": results
        }
        
        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self._log(f"📄 Report saved: {report_file}")
        return report_file
    
    def _get_default_query(self, mode: str) -> str:
        """Get default query based on mode."""
        queries = {
            "verification": "Please verify building code compliance for this project according to Spanish CTE regulations.",
            "door_widths": "Please verify that all doors in the project meet minimum width requirements according to CTE DB-SI regulations.",
            "egress_distances": "Please verify that all rooms have adequate egress distances according to CTE DB-SI regulations.",
            "comprehensive": """Please perform a comprehensive building code compliance verification according to Spanish CTE regulations.
            
            This should include:
            1. Door width compliance verification for all doors
            2. Egress distance compliance verification for all rooms
            3. Accessibility compliance where applicable
            4. Fire safety compliance for fire-rated elements
            
            Provide a detailed report with specific measurements, regulation references, and recommendations."""
        }
        return queries.get(mode, queries["comprehensive"])
    
    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)


def main():
    """Main function for the compliance pipeline."""
    parser = argparse.ArgumentParser(
        description="Run AEC compliance verification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--project-file",
        type=Path,
        default=Path("data/extracted/tech_tower_p2.json"),
        help="Path to project JSON file"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Custom verification query"
    )
    parser.add_argument(
        "--mode",
        choices=["verification", "door_widths", "egress_distances", "comprehensive"],
        default="comprehensive",
        help="Agent mode"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Output directory for reports"
    )
    parser.add_argument(
        "--vectorstore-dir",
        type=Path,
        default=Path("vectorstore/normativa_db"),
        help="Vectorstore directory"
    )
    parser.add_argument(
        "--normativa-dir",
        type=Path,
        default=Path("data/normativa"),
        help="Directory containing building code PDFs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("AEC COMPLIANCE VERIFICATION PIPELINE")
    print("=" * 80)
    print(f"Project file: {args.project_file}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Create pipeline
    pipeline = CompliancePipeline(
        project_file=args.project_file,
        vectorstore_dir=args.vectorstore_dir,
        normativa_dir=args.normativa_dir,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    try:
        # Setup pipeline
        if not pipeline.setup_pipeline():
            print("❌ Pipeline setup failed. Exiting.")
            sys.exit(1)
        
        # Run verification
        print(f"\n🔍 Running {args.mode} verification...")
        results = pipeline.run_verification(
            query=args.query,
            mode=args.mode
        )
        
        # Generate report
        print("\n📄 Generating report...")
        report_file = pipeline.generate_report(results, args.mode)
        
        # Display summary
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        
        summary = results["compliance_summary"]
        print(f"Total checks performed: {summary['total_checks']}")
        print(f"Compliant elements: {summary['compliant']}")
        print(f"Non-compliant elements: {summary['non_compliant']}")
        print(f"Compliance rate: {summary['compliance_rate']:.1%}")
        print(f"Overall status: {summary['overall_status']}")
        print(f"Report saved to: {report_file}")
        
        # Show agent response
        if args.verbose and results["agent_response"]:
            print(f"\nAgent Response:")
            print("-" * 80)
            print(results["agent_response"])
        
        print("=" * 80)
        print("✅ Pipeline execution complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
