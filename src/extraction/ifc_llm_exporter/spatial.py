"""
Spatial Hierarchy and Relationship Extraction

This module handles:
- Spatial tree extraction (Project -> Site -> Building -> Storey -> Space)
- Element containment relationships (which elements are on which level)
- Connectivity relationships (doors connecting spaces, walls adjacent to spaces)
- Void relationships (openings in walls)
- Zone and classification relationships
- Index generation for efficient LLM lookup
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import ifcopenshell
    import ifcopenshell.util.element
except ImportError:
    ifcopenshell = None

logger = logging.getLogger(__name__)


@dataclass
class SpatialNode:
    """Node in the spatial hierarchy tree."""
    guid: str
    ifc_class: str
    name: str
    children: List['SpatialNode'] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ElementIndices:
    """Indices for efficient element lookup."""
    by_class: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    by_storey: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    by_zone: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    by_space: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class ElementRelationships:
    """Comprehensive relationship information for an element."""
    guid: str
    spatial_container: Optional[str] = None  # Storey/Space GUID
    connected_elements: List[str] = field(default_factory=list)  # Connected element GUIDs
    voids_elements: List[str] = field(default_factory=list)  # Elements this voids
    voided_by: List[str] = field(default_factory=list)  # Elements that void this
    adjacent_spaces: List[str] = field(default_factory=list)  # Adjacent space GUIDs
    type_element: Optional[str] = None  # Type definition GUID
    material_associations: List[str] = field(default_factory=list)  # Material GUIDs


class SpatialHierarchyExtractor:
    """
    Extracts spatial hierarchies and relationships from IFC files.
    
    Builds comprehensive spatial tree and relationship mappings for
    efficient LLM consumption and RAG queries.
    """
    
    def __init__(self, ifc_file):
        """Initialize with IFC file."""
        if not ifcopenshell:
            raise ImportError("ifcopenshell is required for spatial extraction")
        
        self.ifc_file = ifc_file
        self.spatial_tree = None
        self.element_relationships = {}
        self.indices = ElementIndices()
        
        logger.info("Initialized spatial hierarchy extractor")
    
    def extract_spatial_hierarchy(self) -> SpatialNode:
        """
        Extract the complete spatial hierarchy from IFC file.
        
        Returns:
            Root spatial node (typically IfcProject)
        """
        logger.info("Extracting spatial hierarchy...")
        
        # Start from project
        projects = self.ifc_file.by_type('IfcProject')
        if not projects:
            raise ValueError("No IfcProject found in file")
        
        project = projects[0]
        self.spatial_tree = self._build_spatial_node(project)
        
        logger.info(f"Extracted spatial hierarchy with {self._count_nodes(self.spatial_tree)} nodes")
        return self.spatial_tree
    
    def extract_element_relationships(self, element_guids: List[str]) -> Dict[str, ElementRelationships]:
        """
        Extract comprehensive relationships for specified elements.
        
        Args:
            element_guids: List of element GUIDs to analyze
            
        Returns:
            Dictionary mapping GUID to relationships
        """
        logger.info(f"Extracting relationships for {len(element_guids)} elements...")
        
        self.element_relationships = {}
        
        for guid in element_guids:
            try:
                element = self.ifc_file.by_guid(guid)
                if element:
                    relationships = self._extract_element_relationships(element)
                    self.element_relationships[guid] = relationships
            except Exception as e:
                logger.warning(f"Error extracting relationships for {guid}: {e}")
                continue
        
        logger.info(f"Extracted relationships for {len(self.element_relationships)} elements")
        return self.element_relationships
    
    def build_indices(self, elements: List[Dict[str, Any]]) -> ElementIndices:
        """
        Build indices for efficient element lookup.
        
        Args:
            elements: List of element dictionaries with 'guid', 'ifc_class', etc.
            
        Returns:
            Element indices structure
        """
        logger.info(f"Building indices for {len(elements)} elements...")
        
        self.indices = ElementIndices()
        
        for element in elements:
            guid = element.get('guid')
            if not guid:
                continue
            
            # Index by class
            ifc_class = element.get('ifc_class', 'Unknown')
            self.indices.by_class[ifc_class].append(guid)
            
            # Index by spatial container
            spatial = element.get('spatial', {})
            
            storey = spatial.get('storey')
            if storey:
                self.indices.by_storey[storey].append(guid)
            
            zone = spatial.get('zone')
            if zone:
                self.indices.by_zone[zone].append(guid)
            
            space = spatial.get('space')
            if space:
                self.indices.by_space[space].append(guid)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        self.indices.by_class = dict(self.indices.by_class)
        self.indices.by_storey = dict(self.indices.by_storey)
        self.indices.by_zone = dict(self.indices.by_zone)
        self.indices.by_space = dict(self.indices.by_space)
        
        logger.info(f"Built indices: {len(self.indices.by_class)} classes, "
                   f"{len(self.indices.by_storey)} storeys, "
                   f"{len(self.indices.by_zone)} zones")
        
        return self.indices
    
    def find_element_spatial_container(self, element) -> Dict[str, Optional[str]]:
        """
        Find spatial containers (site, building, storey, space) for an element.
        
        Returns:
            Dictionary with site, building, storey, zone, space names
        """
        spatial_info = {
            'site': None,
            'building': None,
            'storey': None,
            'zone': None,
            'space': None
        }
        
        try:
            # Check containment relationships
            container = self._find_spatial_container(element)
            
            if container:
                # Traverse up the spatial hierarchy
                current = container
                while current:
                    if current.is_a('IfcSpace'):
                        spatial_info['space'] = getattr(current, 'Name', None) or f"Space_{current.id()}"
                    elif current.is_a('IfcBuildingStorey'):
                        spatial_info['storey'] = getattr(current, 'Name', None) or f"Level_{current.id()}"
                    elif current.is_a('IfcBuilding'):
                        spatial_info['building'] = getattr(current, 'Name', None) or f"Building_{current.id()}"
                    elif current.is_a('IfcSite'):
                        spatial_info['site'] = getattr(current, 'Name', None) or f"Site_{current.id()}"
                    
                    # Move up hierarchy
                    current = self._get_spatial_parent(current)
            
            # Check zone relationships
            zones = self._find_element_zones(element)
            if zones:
                spatial_info['zone'] = zones[0]  # Take first zone
            
            return spatial_info
            
        except Exception as e:
            logger.warning(f"Error finding spatial container for element {element.id()}: {e}")
            return spatial_info
    
    def _build_spatial_node(self, spatial_element) -> SpatialNode:
        """Recursively build spatial hierarchy node."""
        guid = spatial_element.GlobalId
        ifc_class = spatial_element.is_a()
        name = getattr(spatial_element, 'Name', None) or f"{ifc_class}_{spatial_element.id()}"
        
        # Extract properties
        properties = {}
        if hasattr(spatial_element, 'Description') and spatial_element.Description:
            properties['description'] = spatial_element.Description
        
        if spatial_element.is_a('IfcBuildingStorey'):
            if hasattr(spatial_element, 'Elevation') and spatial_element.Elevation:
                properties['elevation_m'] = float(spatial_element.Elevation)
        
        # Create node
        node = SpatialNode(
            guid=guid,
            ifc_class=ifc_class,
            name=name,
            properties=properties
        )
        
        # Find children through decomposition relationships
        children = self._find_spatial_children(spatial_element)
        for child in children:
            child_node = self._build_spatial_node(child)
            node.children.append(child_node)
        
        return node
    
    def _find_spatial_children(self, spatial_element) -> List:
        """Find spatial children of an element through IfcRelAggregates."""
        children = []
        
        try:
            # Check IsDecomposedBy relationships
            if hasattr(spatial_element, 'IsDecomposedBy'):
                for rel in spatial_element.IsDecomposedBy:
                    if rel.is_a('IfcRelAggregates'):
                        for related_object in rel.RelatedObjects:
                            # Only include spatial elements
                            if self._is_spatial_element(related_object):
                                children.append(related_object)
            
            return children
            
        except Exception as e:
            logger.warning(f"Error finding spatial children for {spatial_element.id()}: {e}")
            return []
    
    def _is_spatial_element(self, element) -> bool:
        """Check if an element is a spatial structure element."""
        spatial_types = [
            'IfcSite', 'IfcBuilding', 'IfcBuildingStorey', 'IfcSpace', 
            'IfcZone', 'IfcSpatialZone'
        ]
        return any(element.is_a(spatial_type) for spatial_type in spatial_types)
    
    def _extract_element_relationships(self, element) -> ElementRelationships:
        """Extract comprehensive relationships for a single element."""
        relationships = ElementRelationships(guid=element.GlobalId)
        
        # Spatial containment
        spatial_container = self._find_spatial_container(element)
        if spatial_container:
            relationships.spatial_container = spatial_container.GlobalId
        
        # Type relationships
        type_element = self._find_type_element(element)
        if type_element:
            relationships.type_element = type_element.GlobalId
        
        # Connection relationships
        relationships.connected_elements = self._find_connected_elements(element)
        
        # Void relationships
        relationships.voids_elements = self._find_voided_elements(element)
        relationships.voided_by = self._find_voiding_elements(element)
        
        # Adjacent spaces (for walls, doors)
        relationships.adjacent_spaces = self._find_adjacent_spaces(element)
        
        # Material associations
        relationships.material_associations = self._find_material_associations(element)
        
        return relationships
    
    def _find_spatial_container(self, element):
        """Find the spatial container of an element."""
        try:
            # Check ContainedInStructure relationships
            if hasattr(element, 'ContainedInStructure'):
                for rel in element.ContainedInStructure:
                    if rel.is_a('IfcRelContainedInSpatialStructure'):
                        return rel.RelatingStructure
            
            # Check Decomposes relationships (for nested elements)
            if hasattr(element, 'Decomposes'):
                for rel in element.Decomposes:
                    if rel.is_a('IfcRelAggregates'):
                        relating_object = rel.RelatingObject
                        if self._is_spatial_element(relating_object):
                            return relating_object
            
            return None
            
        except Exception:
            return None
    
    def _get_spatial_parent(self, spatial_element):
        """Get the parent spatial element in the hierarchy."""
        try:
            if hasattr(spatial_element, 'Decomposes'):
                for rel in spatial_element.Decomposes:
                    if rel.is_a('IfcRelAggregates'):
                        return rel.RelatingObject
            return None
        except Exception:
            return None
    
    def _find_element_zones(self, element) -> List[str]:
        """Find zones that contain the element."""
        zones = []
        
        try:
            # Check IfcRelAssignsToGroup relationships for zones
            if hasattr(element, 'HasAssignments'):
                for assignment in element.HasAssignments:
                    if assignment.is_a('IfcRelAssignsToGroup'):
                        group = assignment.RelatingGroup
                        if group and group.is_a('IfcZone'):
                            zone_name = getattr(group, 'Name', None) or f"Zone_{group.id()}"
                            zones.append(zone_name)
            
            return zones
            
        except Exception:
            return []
    
    def _find_type_element(self, element):
        """Find the type element that defines this instance."""
        try:
            if hasattr(element, 'IsDefinedBy'):
                for rel in element.IsDefinedBy:
                    if rel.is_a('IfcRelDefinesByType'):
                        return rel.RelatingType
            return None
        except Exception:
            return None
    
    def _find_connected_elements(self, element) -> List[str]:
        """Find elements connected to this element."""
        connected = []
        
        try:
            # Check IfcRelConnectsElements
            if hasattr(element, 'ConnectedTo'):
                for rel in element.ConnectedTo:
                    if rel.is_a('IfcRelConnectsElements'):
                        related_element = rel.RelatedElement
                        if related_element:
                            connected.append(related_element.GlobalId)
            
            if hasattr(element, 'ConnectedFrom'):
                for rel in element.ConnectedFrom:
                    if rel.is_a('IfcRelConnectsElements'):
                        relating_element = rel.RelatingElement
                        if relating_element:
                            connected.append(relating_element.GlobalId)
            
            # Check IfcRelConnectsPortToElement for MEP
            if hasattr(element, 'HasPorts'):
                for port in element.HasPorts:
                    if hasattr(port, 'ConnectedTo'):
                        for port_connection in port.ConnectedTo:
                            if hasattr(port_connection, 'RelatedPort'):
                                related_port = port_connection.RelatedPort
                                if (hasattr(related_port, 'ContainedIn') and 
                                    related_port.ContainedIn):
                                    connected_element = related_port.ContainedIn
                                    connected.append(connected_element.GlobalId)
            
            return connected
            
        except Exception:
            return []
    
    def _find_voided_elements(self, element) -> List[str]:
        """Find elements that this element voids (e.g., openings in walls)."""
        voided = []
        
        try:
            if hasattr(element, 'VoidsElements'):
                for rel in element.VoidsElements:
                    if rel.is_a('IfcRelVoidsElement'):
                        voided_element = rel.RelatingBuildingElement
                        if voided_element:
                            voided.append(voided_element.GlobalId)
            
            return voided
            
        except Exception:
            return []
    
    def _find_voiding_elements(self, element) -> List[str]:
        """Find elements that void this element."""
        voiding = []
        
        try:
            if hasattr(element, 'HasOpenings'):
                for rel in element.HasOpenings:
                    if rel.is_a('IfcRelVoidsElement'):
                        voiding_element = rel.RelatedOpeningElement
                        if voiding_element:
                            voiding.append(voiding_element.GlobalId)
            
            return voiding
            
        except Exception:
            return []
    
    def _find_adjacent_spaces(self, element) -> List[str]:
        """Find spaces adjacent to this element (for walls, doors)."""
        adjacent = []
        
        try:
            # For doors, check space boundaries
            if element.is_a('IfcDoor'):
                # Find spaces connected by this door
                spaces = self._find_door_connected_spaces(element)
                adjacent.extend([space.GlobalId for space in spaces])
            
            # For walls, check space boundaries
            elif element.is_a('IfcWall'):
                spaces = self._find_wall_adjacent_spaces(element)
                adjacent.extend([space.GlobalId for space in spaces])
            
            return adjacent
            
        except Exception:
            return []
    
    def _find_door_connected_spaces(self, door) -> List:
        """Find spaces connected by a door."""
        spaces = []
        
        try:
            # Look for space boundaries that reference this door
            space_boundaries = self.ifc_file.by_type('IfcRelSpaceBoundary')
            
            for boundary in space_boundaries:
                if (hasattr(boundary, 'RelatedBuildingElement') and 
                    boundary.RelatedBuildingElement == door):
                    relating_space = boundary.RelatingSpace
                    if relating_space and relating_space not in spaces:
                        spaces.append(relating_space)
            
            return spaces
            
        except Exception:
            return []
    
    def _find_wall_adjacent_spaces(self, wall) -> List:
        """Find spaces adjacent to a wall."""
        spaces = []
        
        try:
            # Look for space boundaries that reference this wall
            space_boundaries = self.ifc_file.by_type('IfcRelSpaceBoundary')
            
            for boundary in space_boundaries:
                if (hasattr(boundary, 'RelatedBuildingElement') and 
                    boundary.RelatedBuildingElement == wall):
                    relating_space = boundary.RelatingSpace
                    if relating_space and relating_space not in spaces:
                        spaces.append(relating_space)
            
            return spaces
            
        except Exception:
            return []
    
    def _find_material_associations(self, element) -> List[str]:
        """Find materials associated with this element."""
        materials = []
        
        try:
            if hasattr(element, 'HasAssociations'):
                for association in element.HasAssociations:
                    if association.is_a('IfcRelAssociatesMaterial'):
                        material = association.RelatingMaterial
                        if material:
                            # Handle different material types
                            if material.is_a('IfcMaterial'):
                                materials.append(material.GlobalId)
                            elif material.is_a('IfcMaterialLayerSet'):
                                for layer in material.MaterialLayers:
                                    if layer.Material:
                                        materials.append(layer.Material.GlobalId)
                            elif material.is_a('IfcMaterialConstituentSet'):
                                for constituent in material.MaterialConstituents:
                                    if constituent.Material:
                                        materials.append(constituent.Material.GlobalId)
            
            return materials
            
        except Exception:
            return []
    
    def _count_nodes(self, node: SpatialNode) -> int:
        """Count total nodes in spatial tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spatial hierarchy and indices to dictionary."""
        result = {}
        
        if self.spatial_tree:
            result['spatial_tree'] = self._spatial_node_to_dict(self.spatial_tree)
        
        result['indices'] = {
            'by_class': self.indices.by_class,
            'by_storey': self.indices.by_storey,
            'by_zone': self.indices.by_zone,
            'by_space': self.indices.by_space
        }
        
        return result
    
    def _spatial_node_to_dict(self, node: SpatialNode) -> Dict[str, Any]:
        """Convert spatial node to dictionary."""
        result = {
            'guid': node.guid,
            'class': node.ifc_class,
            'name': node.name
        }
        
        if node.properties:
            result.update(node.properties)
        
        if node.children:
            result['children'] = [self._spatial_node_to_dict(child) for child in node.children]
        
        return result