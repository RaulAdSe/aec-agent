"""IFC to Dictionary Converter

Extracts geometrical building elements from IFC files into structured dictionaries.
Focuses on architectural elements: spaces, walls, doors, slabs, stairs.
"""

import ifcopenshell
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging


class IFCToDictConverter:
    """Convert IFC building models to structured dictionaries."""
    
    def __init__(self, ifc_path: str):
        """Initialize converter with IFC file."""
        self.ifc_path = Path(ifc_path)
        self.model = ifcopenshell.open(str(self.ifc_path))
        self.logger = logging.getLogger(__name__)
        
    def extract_geometrical_elements(self) -> Dict[str, Any]:
        """Extract all geometrical building elements."""
        return {
            "file_info": self._get_file_info(),
            "spaces": self._extract_spaces(),
            "walls": self._extract_walls(),
            "doors": self._extract_doors(),
            "slabs": self._extract_slabs(),
            "stairs": self._extract_stairs()
        }
    
    def _get_file_info(self) -> Dict[str, Any]:
        """Extract basic file and project information."""
        project = self.model.by_type("IfcProject")[0] if self.model.by_type("IfcProject") else None
        
        return {
            "filename": self.ifc_path.name,
            "project_name": project.Name if project and project.Name else "Unknown",
            "description": project.Description if project and project.Description else "",
            "schema": self.model.schema,
            "total_elements": len(self.model.by_type("IfcProduct"))
        }