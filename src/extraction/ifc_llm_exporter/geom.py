"""
Geometry Processing and Analysis

This module handles:
- Bounding box calculation for elements
- Geometry summary extraction (length, width, height, area, volume)
- Reference point calculation (centroids)
- Mesh extraction and external reference generation
- Coordinate transformation for geometry
- Shape type classification
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.shape
    import ifcopenshell.util.placement
except ImportError:
    ifcopenshell = None

try:
    import shapely.geometry as sg
    from shapely.ops import transform
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ShapeType(Enum):
    """Types of geometric shapes."""
    EXTRUDED = "extruded"
    MESH = "mesh" 
    CURVE = "curve"
    SURFACE = "surface"
    SOLID = "solid"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """3D bounding box representation."""
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    
    @property
    def width(self) -> float:
        """Bounding box width (X dimension)."""
        return self.max_x - self.min_x
    
    @property
    def length(self) -> float:
        """Bounding box length (Y dimension)."""
        return self.max_y - self.min_y
    
    @property
    def height(self) -> float:
        """Bounding box height (Z dimension)."""
        return self.max_z - self.min_z
    
    @property
    def center(self) -> Tuple[float, float, float]:
        """Bounding box center point."""
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2
        )
    
    def to_list(self) -> List[float]:
        """Convert to list format [min_x, min_y, min_z, max_x, max_y, max_z]."""
        return [self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z]


@dataclass
class GeometrySummary:
    """Summary of element geometry properties."""
    shape: ShapeType
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    thickness_m: Optional[float] = None
    area_m2: Optional[float] = None
    volume_m3: Optional[float] = None
    perimeter_m: Optional[float] = None
    mesh_ref: Optional[str] = None  # Reference to external mesh file


@dataclass
class CoordinateInfo:
    """Complete coordinate information for an element."""
    local_placement: Dict[str, Any]
    transform_local_to_model: List[float]  # 4x4 matrix as 16 floats
    transform_model_to_world: List[float]  # 4x4 matrix as 16 floats
    reference_point_model_m: List[float]  # [x, y, z] in model coordinates
    reference_point_world_m: List[float]  # [x, y, z] in world coordinates
    bbox_model_m: List[float]  # [min_x, min_y, min_z, max_x, max_y, max_z]
    bbox_world_m: List[float]  # [min_x, min_y, min_z, max_x, max_y, max_z]


class GeometryProcessor:
    """
    Processes element geometry for LLM consumption.
    
    Extracts geometric summaries, calculates bounding boxes,
    and optionally exports detailed meshes to external files.
    """
    
    def __init__(self, ifc_file, crs_handler, enable_mesh_export: bool = False, 
                 mesh_output_dir: Optional[Path] = None):
        """
        Initialize geometry processor.
        
        Args:
            ifc_file: IFC file object
            crs_handler: CoordinateSystemHandler instance
            enable_mesh_export: Whether to export detailed meshes
            mesh_output_dir: Directory for mesh files
        """
        if not ifcopenshell:
            raise ImportError("ifcopenshell is required for geometry processing")
        
        self.ifc_file = ifc_file
        self.crs_handler = crs_handler
        self.enable_mesh_export = enable_mesh_export
        self.mesh_output_dir = mesh_output_dir
        
        # Initialize geometry settings
        self.settings = ifcopenshell.geom.settings()
        try:
            self.settings.set(self.settings.USE_WORLD_COORDS, False)  # Use local coordinates
        except AttributeError:
            pass  # Setting not available in this version
        
        try:
            self.settings.set(self.settings.WELD_VERTICES, True)
        except AttributeError:
            pass  # Setting not available in this version
        
        try:
            self.settings.set(self.settings.SEW_SHELLS, True)
        except AttributeError:
            pass  # Setting not available in this version
        
        if self.enable_mesh_export and self.mesh_output_dir:
            self.mesh_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized geometry processor (mesh export: {enable_mesh_export})")
    
    def process_element_geometry(self, element) -> Tuple[CoordinateInfo, GeometrySummary]:
        """
        Process complete geometry information for an element.
        
        Returns:
            Tuple of (coordinate_info, geometry_summary)
        """
        # Get coordinate transformations
        coordinate_info = self._extract_coordinate_info(element)
        
        # Get geometry summary
        geometry_summary = self._extract_geometry_summary(element, coordinate_info)
        
        return coordinate_info, geometry_summary
    
    def _extract_coordinate_info(self, element) -> CoordinateInfo:
        """Extract complete coordinate information for an element."""
        # Get transformation matrices
        local_to_model, model_to_world = self.crs_handler.compute_element_transforms(element)
        
        # Get local placement information
        local_placement = self._extract_local_placement(element)
        
        # Calculate reference point (centroid) in local coordinates
        reference_local = self._calculate_reference_point(element)
        
        # Transform reference point to model and world coordinates
        reference_model = self.crs_handler.transform_point(reference_local, local_to_model)
        reference_world = self.crs_handler.transform_point(reference_model, model_to_world)
        
        # Calculate bounding box in local coordinates
        bbox_local = self._calculate_bounding_box(element)
        
        # Transform bounding box to model and world coordinates
        if bbox_local:
            bbox_model_list, bbox_world_list = self.crs_handler.compute_bounding_box_transforms(
                bbox_local.to_list()
            )
        else:
            # Use reference point as fallback
            bbox_model_list = [reference_model[0]] * 3 + [reference_model[0]] * 3
            bbox_world_list = [reference_world[0]] * 3 + [reference_world[0]] * 3
        
        return CoordinateInfo(
            local_placement=local_placement,
            transform_local_to_model=local_to_model.flatten().tolist(),
            transform_model_to_world=model_to_world.flatten().tolist(),
            reference_point_model_m=reference_model.tolist(),
            reference_point_world_m=reference_world.tolist(),
            bbox_model_m=bbox_model_list,
            bbox_world_m=bbox_world_list
        )
    
    def _extract_local_placement(self, element) -> Dict[str, Any]:
        """Extract IFC local placement information."""
        placement_info = {}
        
        try:
            if hasattr(element, 'ObjectPlacement') and element.ObjectPlacement:
                placement = element.ObjectPlacement
                
                if placement.is_a('IfcLocalPlacement'):
                    if placement.RelativePlacement:
                        rel_placement = placement.RelativePlacement
                        
                        if rel_placement.is_a('IfcAxis2Placement3D'):
                            placement_info['type'] = 'IfcAxis2Placement3D'
                            
                            # Location
                            if rel_placement.Location:
                                location = rel_placement.Location.Coordinates
                                placement_info['location'] = list(location)
                            
                            # Axis (Z direction)
                            if rel_placement.Axis:
                                axis = rel_placement.Axis.DirectionRatios
                                placement_info['axis'] = list(axis)
                            
                            # RefDirection (X direction)
                            if rel_placement.RefDirection:
                                ref_direction = rel_placement.RefDirection.DirectionRatios
                                placement_info['ref_direction'] = list(ref_direction)
            
            return placement_info
            
        except Exception as e:
            logger.warning(f"Error extracting local placement for element {element.id()}: {e}")
            return {}
    
    def _calculate_reference_point(self, element) -> np.ndarray:
        """Calculate reference point (centroid) for an element."""
        try:
            # Try to get centroid from geometry
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            if shape:
                # Get vertices and calculate centroid
                geometry = shape.geometry
                if hasattr(geometry, 'verts'):
                    verts = np.array(geometry.verts).reshape(-1, 3)
                    if len(verts) > 0:
                        centroid = np.mean(verts, axis=0)
                        return centroid
            
            # Fallback to placement location
            if hasattr(element, 'ObjectPlacement') and element.ObjectPlacement:
                placement_matrix = ifcopenshell.util.placement.get_local_placement(
                    element.ObjectPlacement
                )
                if placement_matrix is not None:
                    return np.array([placement_matrix[0][3], placement_matrix[1][3], placement_matrix[2][3]])
            
            # Ultimate fallback
            return np.array([0.0, 0.0, 0.0])
            
        except Exception as e:
            logger.warning(f"Error calculating reference point for element {element.id()}: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def _calculate_bounding_box(self, element) -> Optional[BoundingBox]:
        """Calculate bounding box for an element."""
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            if shape and shape.geometry:
                geometry = shape.geometry
                
                if hasattr(geometry, 'verts'):
                    verts = np.array(geometry.verts).reshape(-1, 3)
                    if len(verts) > 0:
                        min_coords = np.min(verts, axis=0)
                        max_coords = np.max(verts, axis=0)
                        
                        return BoundingBox(
                            min_x=float(min_coords[0]),
                            min_y=float(min_coords[1]),
                            min_z=float(min_coords[2]),
                            max_x=float(max_coords[0]),
                            max_y=float(max_coords[1]),
                            max_z=float(max_coords[2])
                        )
            
            return None
            
        except Exception as e:
            logger.warning(f"Error calculating bounding box for element {element.id()}: {e}")
            return None
    
    def _extract_geometry_summary(self, element, coordinate_info: CoordinateInfo) -> GeometrySummary:
        """Extract geometry summary with dimensions and properties."""
        summary = GeometrySummary(shape=ShapeType.UNKNOWN)
        
        try:
            # Determine shape type
            summary.shape = self._classify_shape_type(element)
            
            # Extract dimensions from bounding box
            bbox = coordinate_info.bbox_model_m
            if bbox and len(bbox) == 6:
                summary.length_m = bbox[3] - bbox[0]  # X dimension
                summary.width_m = bbox[4] - bbox[1]   # Y dimension  
                summary.height_m = bbox[5] - bbox[2]  # Z dimension
            
            # Try to get more precise dimensions from quantities
            precise_dims = self._extract_precise_dimensions(element)
            if precise_dims:
                summary.length_m = precise_dims.get('length', summary.length_m)
                summary.width_m = precise_dims.get('width', summary.width_m)
                summary.height_m = precise_dims.get('height', summary.height_m)
                summary.thickness_m = precise_dims.get('thickness', summary.thickness_m)
                summary.area_m2 = precise_dims.get('area', summary.area_m2)
                summary.volume_m3 = precise_dims.get('volume', summary.volume_m3)
                summary.perimeter_m = precise_dims.get('perimeter', summary.perimeter_m)
            
            # Calculate missing dimensions from geometry if possible
            if not summary.area_m2 or not summary.volume_m3:
                calculated_props = self._calculate_geometric_properties(element)
                if calculated_props:
                    summary.area_m2 = summary.area_m2 or calculated_props.get('area')
                    summary.volume_m3 = summary.volume_m3 or calculated_props.get('volume')
            
            # Export mesh if enabled
            if self.enable_mesh_export:
                mesh_ref = self._export_element_mesh(element)
                summary.mesh_ref = mesh_ref
            
            return summary
            
        except Exception as e:
            logger.warning(f"Error extracting geometry summary for element {element.id()}: {e}")
            return summary
    
    def _classify_shape_type(self, element) -> ShapeType:
        """Classify the shape type of an element."""
        try:
            # Check representation items to classify shape
            if hasattr(element, 'Representation') and element.Representation:
                for representation in element.Representation.Representations:
                    for item in representation.Items:
                        item_type = item.is_a()
                        
                        if 'Extrusion' in item_type:
                            return ShapeType.EXTRUDED
                        elif 'Mesh' in item_type or 'Triangulated' in item_type:
                            return ShapeType.MESH
                        elif 'Surface' in item_type:
                            return ShapeType.SURFACE
                        elif 'Curve' in item_type:
                            return ShapeType.CURVE
                        elif 'Solid' in item_type:
                            return ShapeType.SOLID
            
            # Fallback classification based on element type
            element_type = element.is_a()
            
            if element_type in ['IfcWall', 'IfcSlab', 'IfcColumn', 'IfcBeam']:
                return ShapeType.EXTRUDED
            elif element_type in ['IfcSpace']:
                return ShapeType.SOLID
            elif element_type in ['IfcCurve', 'IfcPolyline']:
                return ShapeType.CURVE
            else:
                return ShapeType.UNKNOWN
                
        except Exception:
            return ShapeType.UNKNOWN
    
    def _extract_precise_dimensions(self, element) -> Optional[Dict[str, float]]:
        """Extract precise dimensions from IFC quantity sets."""
        try:
            # Use the property extractor to get quantities
            from .props import PropertyExtractor
            prop_extractor = PropertyExtractor(self.ifc_file, self.crs_handler.units)
            
            property_sets = prop_extractor.extract_element_properties(element)
            
            dimensions = {}
            
            # Look for quantity sets
            for pset_name, prop_set in property_sets.items():
                if prop_set.pset_type == 'quantity':
                    for prop_name, prop_value in prop_set.properties.items():
                        prop_name_lower = prop_name.lower()
                        
                        if 'length' in prop_name_lower:
                            dimensions['length'] = prop_value.value
                        elif 'width' in prop_name_lower:
                            dimensions['width'] = prop_value.value
                        elif 'height' in prop_name_lower:
                            dimensions['height'] = prop_value.value
                        elif 'thickness' in prop_name_lower:
                            dimensions['thickness'] = prop_value.value
                        elif 'area' in prop_name_lower and 'gross' not in prop_name_lower:
                            dimensions['area'] = prop_value.value
                        elif 'volume' in prop_name_lower and 'gross' not in prop_name_lower:
                            dimensions['volume'] = prop_value.value
                        elif 'perimeter' in prop_name_lower:
                            dimensions['perimeter'] = prop_value.value
            
            return dimensions if dimensions else None
            
        except Exception as e:
            logger.warning(f"Error extracting precise dimensions: {e}")
            return None
    
    def _calculate_geometric_properties(self, element) -> Optional[Dict[str, float]]:
        """Calculate geometric properties (area, volume) from shape."""
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            if not shape or not shape.geometry:
                return None
            
            geometry = shape.geometry
            properties = {}
            
            # Try to calculate area and volume
            if hasattr(geometry, 'verts') and hasattr(geometry, 'faces'):
                verts = np.array(geometry.verts).reshape(-1, 3)
                faces = np.array(geometry.faces).reshape(-1, 3)
                
                if SHAPELY_AVAILABLE and len(verts) > 0:
                    # For 2D elements, calculate area
                    if element.is_a() in ['IfcSlab', 'IfcSpace']:
                        try:
                            # Project to XY plane and calculate area
                            xy_points = verts[:, :2]
                            if len(xy_points) >= 3:
                                polygon = sg.Polygon(xy_points)
                                if polygon.is_valid:
                                    properties['area'] = polygon.area
                        except:
                            pass
                    
                    # For 3D elements, calculate volume using mesh
                    if len(faces) > 0:
                        try:
                            # Simple volume calculation using mesh
                            volume = self._calculate_mesh_volume(verts, faces)
                            if volume and volume > 0:
                                properties['volume'] = volume
                        except:
                            pass
            
            return properties if properties else None
            
        except Exception as e:
            logger.warning(f"Error calculating geometric properties: {e}")
            return None
    
    def _calculate_mesh_volume(self, vertices: np.ndarray, faces: np.ndarray) -> Optional[float]:
        """Calculate volume of a triangular mesh using divergence theorem."""
        try:
            if len(faces) == 0:
                return None
            
            volume = 0.0
            
            for face in faces:
                if len(face) != 3:
                    continue
                    
                # Get triangle vertices
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                
                # Calculate contribution to volume
                # V = (1/6) * sum(dot(v0, cross(v1, v2)))
                cross_product = np.cross(v1, v2)
                contribution = np.dot(v0, cross_product) / 6.0
                volume += contribution
            
            return abs(volume)
            
        except Exception:
            return None
    
    def _export_element_mesh(self, element) -> Optional[str]:
        """Export element mesh to external file."""
        if not self.enable_mesh_export or not self.mesh_output_dir:
            return None
        
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, element)
            if not shape or not shape.geometry:
                return None
            
            # Generate mesh filename
            element_guid = element.GlobalId
            mesh_filename = f"{element_guid}.glb"
            mesh_path = self.mesh_output_dir / mesh_filename
            
            # For now, just return the reference - actual mesh export would
            # require additional libraries like trimesh or pygltflib
            # This is a placeholder for the mesh export functionality
            
            logger.debug(f"Mesh reference created for element {element_guid}: {mesh_filename}")
            return f"meshes/{mesh_filename}"
            
        except Exception as e:
            logger.warning(f"Error exporting mesh for element {element.id()}: {e}")
            return None
    
    def get_element_geometric_class(self, element) -> str:
        """Get geometric classification for element (point, curve, surface, solid)."""
        try:
            if hasattr(element, 'Representation') and element.Representation:
                for representation in element.Representation.Representations:
                    for item in representation.Items:
                        item_type = item.is_a()
                        
                        if any(geom_type in item_type for geom_type in ['Point', 'CartesianPoint']):
                            return 'point'
                        elif any(geom_type in item_type for geom_type in ['Curve', 'Line', 'Polyline']):
                            return 'curve'
                        elif any(geom_type in item_type for geom_type in ['Surface', 'Plane', 'Face']):
                            return 'surface'
                        elif any(geom_type in item_type for geom_type in ['Solid', 'Extrusion', 'Volume']):
                            return 'solid'
            
            # Fallback based on element type
            element_type = element.is_a()
            
            if element_type in ['IfcWall', 'IfcSlab', 'IfcColumn', 'IfcBeam', 'IfcSpace']:
                return 'solid'
            elif element_type in ['IfcWindow', 'IfcDoor']:
                return 'surface'
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'
    
    def to_dict(self, coordinate_info: CoordinateInfo, geometry_summary: GeometrySummary) -> Dict[str, Any]:
        """Convert geometry information to dictionary format."""
        result = {
            'coordinates': {
                'local_placement': coordinate_info.local_placement,
                'transform_local_to_model': coordinate_info.transform_local_to_model,
                'transform_model_to_world': coordinate_info.transform_model_to_world,
                'reference_point_model_m': coordinate_info.reference_point_model_m,
                'reference_point_world_m': coordinate_info.reference_point_world_m,
                'bbox_model_m': coordinate_info.bbox_model_m,
                'bbox_world_m': coordinate_info.bbox_world_m
            },
            'geometry_summary': {
                'shape': geometry_summary.shape.value
            }
        }
        
        # Add dimensions if available
        if geometry_summary.length_m is not None:
            result['geometry_summary']['length_m'] = geometry_summary.length_m
        if geometry_summary.width_m is not None:
            result['geometry_summary']['width_m'] = geometry_summary.width_m
        if geometry_summary.height_m is not None:
            result['geometry_summary']['height_m'] = geometry_summary.height_m
        if geometry_summary.thickness_m is not None:
            result['geometry_summary']['thickness_m'] = geometry_summary.thickness_m
        if geometry_summary.area_m2 is not None:
            result['geometry_summary']['area_m2'] = geometry_summary.area_m2
        if geometry_summary.volume_m3 is not None:
            result['geometry_summary']['volume_m3'] = geometry_summary.volume_m3
        if geometry_summary.perimeter_m is not None:
            result['geometry_summary']['perimeter_m'] = geometry_summary.perimeter_m
        if geometry_summary.mesh_ref is not None:
            result['geometry_summary']['mesh_ref'] = geometry_summary.mesh_ref
        
        return result