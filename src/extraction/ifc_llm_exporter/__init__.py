"""
IFC LLM Exporter Package

A sophisticated IFC extraction system that produces LLM-optimized JSON output
with comprehensive coordinate system handling, property extraction, and 
spatial relationship mapping.

This package implements a two-layer export strategy:
- Layer A: Authoritative data (ifcJSON or original IFC)
- Layer B: LLM-optimized JSON with chunked elements and indices

Key Features:
- Full coordinate system and georeferencing support
- Comprehensive property set (Pset) and quantity (QTO) extraction
- Spatial relationship mapping and containment hierarchies
- Chunked JSONL output optimized for LLM consumption
- Geometry processing with external mesh references
- Multi-material and layered material support
"""

from .exporter import IFCLLMExporter, export_to_llm_json, ExtractionConfig
from .crs import CoordinateSystemHandler, UnitInfo
from .spatial import SpatialHierarchyExtractor
from .props import PropertyExtractor
from .geom import GeometryProcessor
from .writers import LLMJSONWriter, OutputConfig, ChunkingConfig

__version__ = "1.0.0"
__author__ = "AEC Compliance Agent"

__all__ = [
    "IFCLLMExporter",
    "export_to_llm_json",
    "ExtractionConfig",
    "CoordinateSystemHandler", 
    "UnitInfo",
    "SpatialHierarchyExtractor",
    "PropertyExtractor",
    "GeometryProcessor",
    "LLMJSONWriter",
    "OutputConfig",
    "ChunkingConfig"
]