"""
LLM-Optimized JSON Writers

This module handles:
- Chunked JSONL output for efficient LLM consumption
- Domain-based chunking (architecture, MEP, structural)
- Size-based chunking to control token usage
- Project metadata JSON generation
- Element validation and sanitization
- Output directory management
"""

import json
import logging
from typing import Dict, List, Optional, Any, Iterator, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for element chunking."""
    max_elements_per_chunk: int = 1000
    max_chunk_size_kb: int = 5000  # 5MB max per chunk
    chunk_by_domain: bool = True
    chunk_by_storey: bool = True
    min_chunk_size: int = 10  # Minimum elements per chunk


@dataclass
class OutputConfig:
    """Configuration for output generation."""
    output_dir: Path
    project_filename: str = "project.json"
    elements_dir: str = "elements"
    meshes_dir: str = "meshes"
    chunk_prefix: str = "elements"
    validate_json: bool = True
    pretty_print: bool = False


@dataclass
class ChunkInfo:
    """Information about a generated chunk."""
    filename: str
    element_count: int
    size_bytes: int
    domains: List[str]
    storeys: List[str]
    checksum: str


class LLMJSONWriter:
    """
    Writes IFC data in LLM-optimized JSON format.
    
    Produces chunked JSONL files for elements and a project metadata JSON.
    Optimizes for LLM consumption with proper chunking and indexing.
    """
    
    def __init__(self, output_config: OutputConfig, chunking_config: ChunkingConfig = None):
        """
        Initialize the LLM JSON writer.
        
        Args:
            output_config: Output configuration
            chunking_config: Chunking configuration (uses defaults if None)
        """
        self.output_config = output_config
        self.chunking_config = chunking_config or ChunkingConfig()
        
        # Create output directories
        self.output_dir = output_config.output_dir
        self.elements_dir = self.output_dir / output_config.elements_dir
        self.meshes_dir = self.output_dir / output_config.meshes_dir
        
        self._create_directories()
        
        # Track generated chunks
        self.chunks_info: List[ChunkInfo] = []
        
        logger.info(f"Initialized LLM JSON writer: {self.output_dir}")
    
    def write_project_data(self, project_metadata: Dict[str, Any], 
                          spatial_hierarchy: Dict[str, Any],
                          indices: Dict[str, Any]) -> Path:
        """
        Write project metadata and spatial hierarchy.
        
        Args:
            project_metadata: Project metadata from CRS handler
            spatial_hierarchy: Spatial tree from spatial extractor
            indices: Element indices for lookup
            
        Returns:
            Path to written project.json file
        """
        project_file = self.output_dir / self.output_config.project_filename
        
        project_data = {
            "project": project_metadata.get("project", {}),
            "units": project_metadata.get("units", {}),
            "coordinate_system": project_metadata.get("coordinate_system", {}),
            "spatial_tree": spatial_hierarchy.get("spatial_tree", []),
            "indices": indices
        }
        
        # Validate and write
        if self.output_config.validate_json:
            self._validate_project_json(project_data)
        
        self._write_json_file(project_file, project_data)
        
        logger.info(f"Wrote project data: {project_file}")
        return project_file
    
    def write_elements_chunked(self, elements: List[Dict[str, Any]]) -> List[ChunkInfo]:
        """
        Write elements in chunked JSONL format.
        
        Args:
            elements: List of element dictionaries
            
        Returns:
            List of chunk information
        """
        logger.info(f"Writing {len(elements)} elements in chunks...")
        
        # Clear previous chunks
        self.chunks_info = []
        
        # Organize elements for chunking
        organized_elements = self._organize_elements_for_chunking(elements)
        
        # Generate chunks
        for chunk_key, chunk_elements in organized_elements.items():
            if len(chunk_elements) < self.chunking_config.min_chunk_size:
                # Merge small chunks
                logger.debug(f"Merging small chunk {chunk_key} with {len(chunk_elements)} elements")
                continue
            
            chunk_info = self._write_element_chunk(chunk_key, chunk_elements)
            if chunk_info:
                self.chunks_info.append(chunk_info)
        
        # Handle remaining small chunks
        self._handle_remaining_small_chunks(organized_elements)
        
        logger.info(f"Wrote {len(self.chunks_info)} element chunks")
        return self.chunks_info
    
    def write_chunk_manifest(self) -> Path:
        """
        Write a manifest file describing all chunks.
        
        Returns:
            Path to manifest file
        """
        manifest_file = self.output_dir / "chunks_manifest.json"
        
        manifest_data = {
            "total_chunks": len(self.chunks_info),
            "total_elements": sum(chunk.element_count for chunk in self.chunks_info),
            "total_size_bytes": sum(chunk.size_bytes for chunk in self.chunks_info),
            "chunks": [asdict(chunk) for chunk in self.chunks_info],
            "chunking_config": asdict(self.chunking_config),
            "generated_at": self._get_timestamp()
        }
        
        self._write_json_file(manifest_file, manifest_data)
        
        logger.info(f"Wrote chunk manifest: {manifest_file}")
        return manifest_file
    
    def _create_directories(self):
        """Create necessary output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.elements_dir.mkdir(parents=True, exist_ok=True)
        
        if self.meshes_dir:
            self.meshes_dir.mkdir(parents=True, exist_ok=True)
    
    def _organize_elements_for_chunking(self, elements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize elements into logical chunks."""
        organized = defaultdict(list)
        
        for element in elements:
            chunk_key = self._determine_chunk_key(element)
            organized[chunk_key].append(element)
        
        # Split large chunks
        final_organized = {}
        
        for chunk_key, chunk_elements in organized.items():
            if len(chunk_elements) <= self.chunking_config.max_elements_per_chunk:
                final_organized[chunk_key] = chunk_elements
            else:
                # Split large chunks
                sub_chunks = self._split_large_chunk(chunk_key, chunk_elements)
                final_organized.update(sub_chunks)
        
        return final_organized
    
    def _determine_chunk_key(self, element: Dict[str, Any]) -> str:
        """Determine which chunk an element belongs to."""
        parts = []
        
        # Domain-based chunking
        if self.chunking_config.chunk_by_domain:
            domain = self._classify_element_domain(element)
            parts.append(domain)
        
        # Storey-based chunking
        if self.chunking_config.chunk_by_storey:
            spatial = element.get('spatial', {})
            storey = spatial.get('storey')
            # Handle None storey values
            if storey is None:
                storey = 'unknown_storey'
            # Sanitize storey name for filename
            storey = self._sanitize_filename_part(storey)
            parts.append(storey)
        
        # If no chunking criteria, use default
        if not parts:
            parts.append('all')
        
        return '_'.join(parts)
    
    def _classify_element_domain(self, element: Dict[str, Any]) -> str:
        """Classify element into domain (arch, mep, structural, etc.)."""
        ifc_class = element.get('ifc_class', '').lower()
        
        # Architecture
        if any(cls in ifc_class for cls in ['wall', 'slab', 'door', 'window', 'roof', 'stair', 'space']):
            return 'arch'
        
        # MEP (Mechanical, Electrical, Plumbing)
        elif any(cls in ifc_class for cls in ['flow', 'distribution', 'duct', 'pipe', 'cable', 'equipment']):
            return 'mep'
        
        # Structural
        elif any(cls in ifc_class for cls in ['beam', 'column', 'footing', 'pile', 'reinforc']):
            return 'structural'
        
        # Site/Civil
        elif any(cls in ifc_class for cls in ['site', 'road', 'earthwork', 'pavement']):
            return 'civil'
        
        # Other/Unknown
        else:
            return 'other'
    
    def _sanitize_filename_part(self, name: str) -> str:
        """Sanitize a string for use in filenames."""
        # Handle None values
        if name is None:
            return 'unknown'
        
        # Convert to string if not already
        sanitized = str(name)
        
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Replace spaces and limit length
        sanitized = sanitized.replace(' ', '_').lower()
        sanitized = sanitized[:50]  # Limit length
        
        # Remove consecutive underscores
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        
        return sanitized.strip('_')
    
    def _split_large_chunk(self, chunk_key: str, elements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Split a large chunk into smaller ones."""
        sub_chunks = {}
        max_elements = self.chunking_config.max_elements_per_chunk
        
        for i in range(0, len(elements), max_elements):
            sub_chunk_key = f"{chunk_key}_part{i // max_elements + 1}"
            sub_chunk_elements = elements[i:i + max_elements]
            sub_chunks[sub_chunk_key] = sub_chunk_elements
        
        logger.debug(f"Split chunk {chunk_key} into {len(sub_chunks)} sub-chunks")
        return sub_chunks
    
    def _write_element_chunk(self, chunk_key: str, elements: List[Dict[str, Any]]) -> Optional[ChunkInfo]:
        """Write a single element chunk as JSONL."""
        filename = f"{self.output_config.chunk_prefix}_{chunk_key}.jsonl"
        chunk_file = self.elements_dir / filename
        
        try:
            # Validate elements
            if self.output_config.validate_json:
                elements = [self._validate_element_json(elem) for elem in elements]
                elements = [elem for elem in elements if elem is not None]
            
            if not elements:
                logger.warning(f"No valid elements for chunk {chunk_key}")
                return None
            
            # Write JSONL
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for element in elements:
                    json_line = json.dumps(element, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_line + '\n')
            
            # Calculate file size and checksum
            file_size = chunk_file.stat().st_size
            checksum = self._calculate_file_checksum(chunk_file)
            
            # Extract metadata
            domains = list(set(self._classify_element_domain(elem) for elem in elements))
            storeys = list(set((elem.get('spatial', {}).get('storey') or 'unknown') for elem in elements))
            storeys = [s for s in storeys if s != 'unknown']
            
            chunk_info = ChunkInfo(
                filename=filename,
                element_count=len(elements),
                size_bytes=file_size,
                domains=domains,
                storeys=storeys,
                checksum=checksum
            )
            
            logger.debug(f"Wrote chunk {filename}: {len(elements)} elements, {file_size} bytes")
            return chunk_info
            
        except Exception as e:
            logger.error(f"Error writing chunk {chunk_key}: {e}")
            return None
    
    def _handle_remaining_small_chunks(self, organized_elements: Dict[str, List[Dict[str, Any]]]):
        """Merge small chunks into larger ones."""
        small_chunks = {
            key: elements for key, elements in organized_elements.items()
            if len(elements) < self.chunking_config.min_chunk_size
        }
        
        if not small_chunks:
            return
        
        # Merge small chunks by domain
        merged_by_domain = defaultdict(list)
        
        for chunk_key, elements in small_chunks.items():
            domain = self._classify_element_domain(elements[0]) if elements else 'other'
            merged_by_domain[domain].extend(elements)
        
        # Write merged chunks
        for domain, merged_elements in merged_by_domain.items():
            if merged_elements:
                chunk_key = f"merged_{domain}"
                chunk_info = self._write_element_chunk(chunk_key, merged_elements)
                if chunk_info:
                    self.chunks_info.append(chunk_info)
    
    def _validate_project_json(self, project_data: Dict[str, Any]) -> bool:
        """Validate project JSON structure."""
        required_keys = ['project', 'units', 'coordinate_system', 'spatial_tree', 'indices']
        
        for key in required_keys:
            if key not in project_data:
                logger.warning(f"Missing required key in project data: {key}")
                project_data[key] = {}
        
        return True
    
    def _validate_element_json(self, element: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and sanitize element JSON."""
        if not element.get('guid'):
            logger.warning("Element missing GUID, skipping")
            return None
        
        # Required fields
        required_fields = {
            'guid': 'unknown',
            'ifc_class': 'IfcElement',
            'name': 'Unnamed Element'
        }
        
        for field, default in required_fields.items():
            if field not in element:
                element[field] = default
        
        # Ensure proper nesting
        if 'spatial' not in element:
            element['spatial'] = {}
        
        if 'coordinates' not in element:
            element['coordinates'] = {}
        
        if 'geometry_summary' not in element:
            element['geometry_summary'] = {}
        
        if 'properties' not in element:
            element['properties'] = {}
        
        if 'relations' not in element:
            element['relations'] = {}
        
        if 'source' not in element:
            element['source'] = {}
        
        return element
    
    def _write_json_file(self, file_path: Path, data: Dict[str, Any]):
        """Write JSON data to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            if self.output_config.pretty_print:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_output_summary(self) -> Dict[str, Any]:
        """Get summary of generated output."""
        total_elements = sum(chunk.element_count for chunk in self.chunks_info)
        total_size = sum(chunk.size_bytes for chunk in self.chunks_info)
        
        domains = set()
        storeys = set()
        
        for chunk in self.chunks_info:
            domains.update(chunk.domains)
            storeys.update(chunk.storeys)
        
        # Filter out None values before sorting
        clean_storeys = [s for s in storeys if s is not None]
        
        return {
            'output_directory': str(self.output_dir),
            'total_chunks': len(self.chunks_info),
            'total_elements': total_elements,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'domains': sorted(list(domains)),
            'storeys': sorted(clean_storeys),
            'chunk_files': [chunk.filename for chunk in self.chunks_info]
        }


class StreamingJSONLWriter:
    """
    Streaming JSONL writer for very large IFC files.
    
    Writes elements as they are processed to avoid memory issues.
    """
    
    def __init__(self, output_file: Path, max_chunk_size_mb: int = 100):
        """
        Initialize streaming writer.
        
        Args:
            output_file: Base output file path
            max_chunk_size_mb: Maximum size per chunk file in MB
        """
        self.base_output_file = output_file
        self.max_chunk_size_bytes = max_chunk_size_mb * 1024 * 1024
        self.current_chunk = 1
        self.current_file = None
        self.current_size = 0
        self.element_count = 0
        
        self._open_new_chunk()
    
    def write_element(self, element: Dict[str, Any]):
        """Write a single element to the current chunk."""
        json_line = json.dumps(element, ensure_ascii=False, separators=(',', ':'))
        line_bytes = json_line.encode('utf-8') + b'\n'
        
        # Check if we need a new chunk
        if self.current_size + len(line_bytes) > self.max_chunk_size_bytes:
            self._close_current_chunk()
            self._open_new_chunk()
        
        # Write to current chunk
        self.current_file.write(json_line + '\n')
        self.current_size += len(line_bytes)
        self.element_count += 1
    
    def close(self):
        """Close the writer and finalize all chunks."""
        if self.current_file:
            self._close_current_chunk()
        
        logger.info(f"Streaming writer completed: {self.element_count} elements in {self.current_chunk} chunks")
    
    def _open_new_chunk(self):
        """Open a new chunk file."""
        if self.current_chunk == 1:
            chunk_file = self.base_output_file
        else:
            stem = self.base_output_file.stem
            suffix = self.base_output_file.suffix
            chunk_file = self.base_output_file.parent / f"{stem}_part{self.current_chunk}{suffix}"
        
        self.current_file = open(chunk_file, 'w', encoding='utf-8')
        self.current_size = 0
        
        logger.debug(f"Opened new chunk: {chunk_file}")
    
    def _close_current_chunk(self):
        """Close the current chunk file."""
        if self.current_file:
            self.current_file.close()
            self.current_file = None
            self.current_chunk += 1