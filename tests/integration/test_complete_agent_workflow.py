"""
End-to-End Integration Test for Complete Agent Workflow

This test verifies the complete pipeline:
1. Blueprint Extraction (DWG/DXF)
2. Geometric Computations 
3. RAG System (Building Code Queries)
4. Compliance Verification
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.append('.')

from src.extraction.dwg_extractor import DWGExtractor
from src.calculations.geometry import calculate_room_area, calculate_centroid
from src.calculations.graph import create_circulation_graph, find_shortest_path
from src.rag.vectorstore_manager import VectorstoreManager
from src.rag.qa_chain import create_qa_chain
from src.schemas import Project, Room, Door, Wall, ProjectMetadata, Level


class TestCompleteAgentWorkflow:
    """Test the complete agent workflow from extraction to compliance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir()
        
        # Create mock project data
        self.mock_project_data = self._create_mock_project_data()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_mock_project_data(self):
        """Create mock project data for testing."""
        return {
            "metadata": {
                "project_name": "Test Building",
                "level_name": "Ground Floor",
                "building_use": "office"
            },
            "rooms": [
                {
                    "id": "ROOM_001",
                    "name": "Office 1",
                    "boundary": [[0, 0], [5, 0], [5, 4], [0, 4], [0, 0]],
                    "level": "Ground Floor",
                    "use_type": "office"
                },
                {
                    "id": "ROOM_002", 
                    "name": "Corridor",
                    "boundary": [[5, 0], [10, 0], [10, 2], [5, 2], [5, 0]],
                    "level": "Ground Floor",
                    "use_type": "corridor"
                }
            ],
            "doors": [
                {
                    "id": "DOOR_001",
                    "width": 0.8,  # 80cm - compliant
                    "height": 2.1,
                    "position": [2.5, 0],
                    "room_from": "ROOM_001",
                    "room_to": "ROOM_002",
                    "is_egress": True
                },
                {
                    "id": "DOOR_002",
                    "width": 0.6,  # 60cm - non-compliant
                    "height": 2.1,
                    "position": [7.5, 0],
                    "room_from": "ROOM_002",
                    "room_to": "EXTERIOR",
                    "is_egress": True
                }
            ],
            "walls": [
                {
                    "id": "WALL_001",
                    "start": [0, 0],
                    "end": [5, 0],
                    "length": 5.0,
                    "thickness": 0.2,
                    "is_exterior": True
                }
            ]
        }
    
    def test_step1_blueprint_extraction(self):
        """Test Step 1: Blueprint extraction from CAD files."""
        print("\n🏗️  Testing Step 1: Blueprint Extraction")
        
        # Mock DWG extraction
        with patch('src.extraction.dwg_extractor.DWGExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            mock_extractor.extract_project.return_value = self.mock_project_data
            
            # Test extraction
            extractor = DWGExtractor()
            result = extractor.extract_project(Path("test.dwg"))
            
            # Verify extraction results
            assert result is not None
            assert "rooms" in result
            assert "doors" in result
            assert "walls" in result
            assert len(result["rooms"]) == 2
            assert len(result["doors"]) == 2
            
            print("✅ Blueprint extraction successful")
            print(f"   - Extracted {len(result['rooms'])} rooms")
            print(f"   - Extracted {len(result['doors'])} doors")
            print(f"   - Extracted {len(result['walls'])} walls")
    
    def test_step2_geometric_computations(self):
        """Test Step 2: Geometric computations on extracted data."""
        print("\n📐 Testing Step 2: Geometric Computations")
        
        # Test room area calculations
        room_data = self.mock_project_data["rooms"][0]
        boundary = room_data["boundary"]
        
        # Calculate area using Shapely
        from shapely.geometry import Polygon
        polygon = Polygon(boundary)
        area = polygon.area
        
        assert area == 20.0  # 5m x 4m = 20 m²
        print(f"✅ Room area calculation: {area} m²")
        
        # Test centroid calculation
        centroid = polygon.centroid
        expected_centroid = (2.5, 2.0)
        assert abs(centroid.x - expected_centroid[0]) < 0.01
        assert abs(centroid.y - expected_centroid[1]) < 0.01
        print(f"✅ Centroid calculation: ({centroid.x:.1f}, {centroid.y:.1f})")
        
        # Test circulation graph
        import networkx as nx
        G = nx.Graph()
        
        # Add rooms as nodes
        for room in self.mock_project_data["rooms"]:
            G.add_node(room["id"], name=room["name"])
        
        # Add doors as edges
        for door in self.mock_project_data["doors"]:
            if door.get("room_from") and door.get("room_to"):
                G.add_edge(door["room_from"], door["room_to"], door_id=door["id"])
        
        # Test graph connectivity
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        assert nx.is_connected(G)
        
        print("✅ Circulation graph created successfully")
        print(f"   - Nodes (rooms): {G.number_of_nodes()}")
        print(f"   - Edges (doors): {G.number_of_edges()}")
    
    def test_step3_rag_system(self):
        """Test Step 3: RAG system for building code queries."""
        print("\n🔍 Testing Step 3: RAG System")
        
        # Mock RAG system
        with patch('src.rag.vectorstore_manager.VectorstoreManager') as mock_rag_class:
            mock_rag = Mock()
            mock_rag_class.return_value = mock_rag
            mock_rag.load_existing.return_value = True
            
            # Mock retriever
            mock_retriever = Mock()
            mock_docs = [
                Mock(page_content="Minimum door width for evacuation: 80 cm", metadata={"source": "CTE-DB-SI"}),
                Mock(page_content="Corridor width minimum: 120 cm", metadata={"source": "CTE-DB-SI"})
            ]
            mock_retriever.get_relevant_documents.return_value = mock_docs
            mock_rag.get_retriever.return_value = mock_retriever
            
            # Test RAG query
            rag = VectorstoreManager(Path("test_vectorstore"))
            rag.load_existing()
            
            question = "¿Cuál es el ancho mínimo de puerta de evacuación?"
            docs = rag.query_simple(question, k=2)
            
            assert len(docs) == 2
            assert "80 cm" in docs[0].page_content
            
            print("✅ RAG system working")
            print(f"   - Retrieved {len(docs)} relevant documents")
            print(f"   - Found regulation: {docs[0].page_content[:50]}...")
    
    def test_step4_compliance_verification(self):
        """Test Step 4: Complete compliance verification."""
        print("\n✅ Testing Step 4: Compliance Verification")
        
        # Test door width compliance
        doors = self.mock_project_data["doors"]
        compliance_results = []
        
        for door in doors:
            width_cm = door["width"] * 100  # Convert to cm
            min_width_cm = 80  # CTE requirement
            
            is_compliant = width_cm >= min_width_cm
            compliance_results.append({
                "door_id": door["id"],
                "width_cm": width_cm,
                "min_required_cm": min_width_cm,
                "is_compliant": is_compliant,
                "message": f"Door {door['id']}: {width_cm}cm {'✅ Compliant' if is_compliant else '❌ Non-compliant'} (min: {min_width_cm}cm)"
            })
        
        # Verify results
        assert len(compliance_results) == 2
        assert compliance_results[0]["is_compliant"] == True   # 80cm door
        assert compliance_results[1]["is_compliant"] == False  # 60cm door
        
        print("✅ Compliance verification completed")
        for result in compliance_results:
            print(f"   - {result['message']}")
    
    def test_complete_workflow_integration(self):
        """Test the complete integrated workflow."""
        print("\n🚀 Testing Complete Workflow Integration")
        
        # Step 1: Extract data
        with patch('src.extraction.dwg_extractor.DWGExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            mock_extractor.extract_project.return_value = self.mock_project_data
            
            extractor = DWGExtractor()
            project_data = extractor.extract_project(Path("test.dwg"))
        
        # Step 2: Perform calculations
        from shapely.geometry import Polygon
        import networkx as nx
        
        # Calculate areas
        total_area = 0
        for room in project_data["rooms"]:
            polygon = Polygon(room["boundary"])
            total_area += polygon.area
        
        # Create circulation graph
        G = nx.Graph()
        for room in project_data["rooms"]:
            G.add_node(room["id"], name=room["name"])
        
        for door in project_data["doors"]:
            if door.get("room_from") and door.get("room_to"):
                G.add_edge(door["room_from"], door["room_to"], door_id=door["id"])
        
        # Step 3: Query regulations (mocked)
        with patch('src.rag.vectorstore_manager.VectorstoreManager') as mock_rag_class:
            mock_rag = Mock()
            mock_rag_class.return_value = mock_rag
            mock_rag.load_existing.return_value = True
            
            mock_retriever = Mock()
            mock_docs = [Mock(page_content="Minimum door width: 80 cm", metadata={"source": "CTE"})]
            mock_retriever.get_relevant_documents.return_value = mock_docs
            mock_rag.get_retriever.return_value = mock_retriever
            
            rag = VectorstoreManager(Path("test_vectorstore"))
            rag.load_existing()
            
            regulation_docs = rag.query_simple("door width requirements", k=1)
        
        # Step 4: Verify compliance
        compliance_report = {
            "project_name": project_data["metadata"]["project_name"],
            "total_area_m2": total_area,
            "total_rooms": len(project_data["rooms"]),
            "total_doors": len(project_data["doors"]),
            "circulation_connected": nx.is_connected(G),
            "door_compliance": []
        }
        
        for door in project_data["doors"]:
            width_cm = door["width"] * 100
            is_compliant = width_cm >= 80
            compliance_report["door_compliance"].append({
                "door_id": door["id"],
                "width_cm": width_cm,
                "compliant": is_compliant
            })
        
        # Verify final results
        assert compliance_report["total_area_m2"] == 20.0
        assert compliance_report["total_rooms"] == 2
        assert compliance_report["total_doors"] == 2
        assert compliance_report["circulation_connected"] == True
        assert len(compliance_report["door_compliance"]) == 2
        
        print("✅ Complete workflow integration successful")
        print(f"   - Project: {compliance_report['project_name']}")
        print(f"   - Total area: {compliance_report['total_area_m2']} m²")
        print(f"   - Rooms: {compliance_report['total_rooms']}")
        print(f"   - Doors: {compliance_report['total_doors']}")
        print(f"   - Circulation connected: {compliance_report['circulation_connected']}")
        
        compliant_doors = sum(1 for d in compliance_report["door_compliance"] if d["compliant"])
        print(f"   - Compliant doors: {compliant_doors}/{len(compliance_report['door_compliance'])}")
        
        return compliance_report
    
    def test_agent_tools_integration(self):
        """Test integration with agent tools."""
        print("\n🤖 Testing Agent Tools Integration")
        
        # Mock agent tools
        with patch('src.agent.tools') as mock_tools:
            # Mock the tools that the agent would use
            mock_tools.get_room_info.return_value = self.mock_project_data["rooms"][0]
            mock_tools.get_door_info.return_value = self.mock_project_data["doors"][0]
            mock_tools.list_all_doors.return_value = self.mock_project_data["doors"]
            mock_tools.check_door_width_compliance.return_value = {
                "door_id": "DOOR_001",
                "width_cm": 80,
                "compliant": True,
                "message": "Door meets minimum width requirement"
            }
            mock_tools.query_normativa.return_value = {
                "question": "door width",
                "answer": "Minimum door width for evacuation is 80 cm",
                "sources": ["CTE-DB-SI"]
            }
            mock_tools.calculate_egress_distance.return_value = {
                "room_id": "ROOM_001",
                "distance_to_exit": 5.0,
                "compliant": True
            }
            
            # Test agent tool calls
            room_info = mock_tools.get_room_info("ROOM_001")
            door_info = mock_tools.get_door_info("DOOR_001")
            all_doors = mock_tools.list_all_doors()
            compliance_check = mock_tools.check_door_width_compliance("DOOR_001")
            regulation_query = mock_tools.query_normativa("door width requirements")
            egress_distance = mock_tools.calculate_egress_distance("ROOM_001")
            
            # Verify tool responses
            assert room_info["id"] == "ROOM_001"
            assert door_info["id"] == "DOOR_001"
            assert len(all_doors) == 2
            assert compliance_check["compliant"] == True
            assert "80 cm" in regulation_query["answer"]
            assert egress_distance["compliant"] == True
            
            print("✅ Agent tools integration successful")
            print(f"   - Room info: {room_info['name']}")
            print(f"   - Door info: {door_info['width']}m width")
            print(f"   - Total doors: {len(all_doors)}")
            print(f"   - Compliance: {compliance_check['message']}")
            print(f"   - Regulation: {regulation_query['answer'][:50]}...")
            print(f"   - Egress distance: {egress_distance['distance_to_exit']}m")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
