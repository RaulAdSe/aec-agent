"""
Coordinate Reference System (CRS) and Units Handling

This module handles:
- IFC units parsing and conversion to meters
- Coordinate system extraction (local, model, world)
- Georeferencing with IFC4 MapConversion and ProjectedCRS
- Transform matrix computation for coordinate conversions
- True North orientation handling
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    import ifcopenshell
    import ifcopenshell.util.unit
    import ifcopenshell.util.placement
except ImportError:
    ifcopenshell = None

logger = logging.getLogger(__name__)


class IFCSchema(Enum):
    """IFC Schema versions with different georeferencing capabilities."""
    IFC2X3 = "IFC2X3"
    IFC4 = "IFC4"
    IFC4X1 = "IFC4X1"
    IFC4X3 = "IFC4X3"


@dataclass
class UnitInfo:
    """Unit information extracted from IFC file."""
    length: str = "m"
    area: str = "m2"
    volume: str = "m3"
    angle: str = "deg"
    scale_to_meters: float = 1.0


@dataclass
class CoordinateSystem:
    """Complete coordinate system information."""
    model_origin_note: str
    true_north_deg: float = 0.0
    world: Optional['WorldCRS'] = None


@dataclass
class WorldCRS:
    """World coordinate reference system information."""
    epsg: Optional[int] = None
    name: str = "Unknown CRS"
    origin_llh: Optional[Dict[str, float]] = None  # lat, lon, elev_m
    to_world_transform: List[float] = None  # 4x4 matrix as 16 floats
    notes: str = ""


class CoordinateSystemHandler:
    """
    Handles coordinate system extraction and transformation from IFC files.
    
    Supports:
    - IFC2x3: Basic georeferencing via IfcSite coordinates
    - IFC4+: Full MapConversion and ProjectedCRS support
    - Unit parsing and conversion factors
    - Transform matrix computation
    """
    
    def __init__(self, ifc_file):
        """Initialize with IFC file."""
        if not ifcopenshell:
            raise ImportError("ifcopenshell is required for coordinate system handling")
        
        self.ifc_file = ifc_file
        self.schema = self._detect_schema()
        self.units = self._extract_units()
        self.coordinate_system = self._extract_coordinate_system()
        
        logger.info(f"Initialized coordinate system handler for {self.schema.value}")
    
    def _detect_schema(self) -> IFCSchema:
        """Detect IFC schema version."""
        schema_str = self.ifc_file.schema.upper()
        
        if "IFC4X3" in schema_str:
            return IFCSchema.IFC4X3
        elif "IFC4X1" in schema_str:
            return IFCSchema.IFC4X1
        elif "IFC4" in schema_str:
            return IFCSchema.IFC4
        elif "IFC2X3" in schema_str:
            return IFCSchema.IFC2X3
        else:
            logger.warning(f"Unknown IFC schema: {schema_str}, defaulting to IFC4")
            return IFCSchema.IFC4
    
    def _extract_units(self) -> UnitInfo:
        """Extract unit information from IFC file."""
        try:
            # Get project units
            project = self.ifc_file.by_type('IfcProject')[0]
            unit_assignment = project.UnitsInContext
            
            units = UnitInfo()
            
            if unit_assignment:
                for unit in unit_assignment.Units:
                    if unit.is_a('IfcSIUnit'):
                        unit_type = unit.UnitType
                        unit_name = unit.Name
                        
                        # Handle SI prefixes
                        prefix_multiplier = self._get_si_prefix_multiplier(getattr(unit, 'Prefix', None))
                        
                        if unit_type == 'LENGTHUNIT':
                            if unit_name == 'METRE':
                                units.length = "m"
                                units.scale_to_meters = prefix_multiplier
                            elif unit_name == 'MILLIMETRE':
                                units.length = "mm"
                                units.scale_to_meters = 0.001 * prefix_multiplier
                        
                        elif unit_type == 'AREAUNIT':
                            units.area = f"{units.length}2"
                        
                        elif unit_type == 'VOLUMEUNIT':
                            units.volume = f"{units.length}3"
                        
                        elif unit_type == 'PLANEANGLEUNIT':
                            if unit_name == 'RADIAN':
                                units.angle = "rad"
                            elif unit_name == 'DEGREE':
                                units.angle = "deg"
                    
                    elif unit.is_a('IfcConversionBasedUnit'):
                        # Handle non-SI units
                        unit_type = unit.UnitType
                        try:
                            # Handle both direct values and entity instances
                            value_component = unit.ConversionFactor.ValueComponent
                            if hasattr(value_component, 'wrappedValue'):
                                conversion_factor = float(value_component.wrappedValue)
                            else:
                                conversion_factor = float(value_component)
                        except (ValueError, AttributeError, TypeError):
                            logger.warning(f"Could not extract conversion factor from {value_component}, using 1.0")
                            conversion_factor = 1.0
                        
                        if unit_type == 'LENGTHUNIT':
                            units.scale_to_meters = conversion_factor
            
            logger.info(f"Extracted units: length={units.length}, scale={units.scale_to_meters}")
            return units
            
        except Exception as e:
            logger.warning(f"Error extracting units, using defaults: {e}")
            return UnitInfo()
    
    def _get_si_prefix_multiplier(self, prefix) -> float:
        """Get multiplier for SI unit prefixes."""
        if not prefix:
            return 1.0
        
        prefix_map = {
            'MILLI': 0.001,
            'CENTI': 0.01,
            'DECI': 0.1,
            'KILO': 1000.0,
            'MEGA': 1000000.0
        }
        
        return prefix_map.get(prefix, 1.0)
    
    def _extract_coordinate_system(self) -> CoordinateSystem:
        """Extract coordinate system information from IFC file."""
        # Get true north
        true_north = self._extract_true_north()
        
        # Get world CRS based on schema
        world_crs = None
        if self.schema in [IFCSchema.IFC4, IFCSchema.IFC4X1, IFCSchema.IFC4X3]:
            world_crs = self._extract_ifc4_world_crs()
        else:
            world_crs = self._extract_ifc2x3_world_crs()
        
        return CoordinateSystem(
            model_origin_note="IFC local origin (0,0,0) in site or project coordinate system",
            true_north_deg=true_north,
            world=world_crs
        )
    
    def _extract_true_north(self) -> float:
        """Extract true north direction from geometric representation context."""
        try:
            contexts = self.ifc_file.by_type('IfcGeometricRepresentationContext')
            
            for context in contexts:
                if hasattr(context, 'TrueNorth') and context.TrueNorth:
                    # TrueNorth is a direction vector
                    direction = context.TrueNorth.DirectionRatios
                    if len(direction) >= 2:
                        # Calculate angle from +Y axis (assuming Y is north in model)
                        x, y = direction[0], direction[1]
                        angle_rad = np.arctan2(x, y)
                        angle_deg = np.degrees(angle_rad)
                        
                        logger.info(f"True north extracted: {angle_deg:.2f} degrees")
                        return float(angle_deg)
            
            logger.info("No true north found, using default 0.0 degrees")
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error extracting true north: {e}")
            return 0.0
    
    def _extract_ifc4_world_crs(self) -> Optional[WorldCRS]:
        """Extract world CRS from IFC4+ files using MapConversion and ProjectedCRS."""
        try:
            # Look for coordinate reference systems
            crs_entities = self.ifc_file.by_type('IfcCoordinateReferenceSystem')
            map_conversions = self.ifc_file.by_type('IfcMapConversion')
            
            if not map_conversions:
                logger.info("No MapConversion found in IFC4 file")
                return None
            
            map_conversion = map_conversions[0]
            
            # Extract CRS information
            epsg = None
            crs_name = "Unknown CRS"
            
            if crs_entities:
                crs = crs_entities[0]
                crs_name = getattr(crs, 'Name', 'Unknown CRS')
                
                # Try to extract EPSG code
                if hasattr(crs, 'MapProjection') and crs.MapProjection:
                    epsg_str = str(crs.MapProjection)
                    if 'EPSG:' in epsg_str:
                        try:
                            epsg = int(epsg_str.split('EPSG:')[1].split()[0])
                        except:
                            pass
            
            # Extract map conversion parameters
            eastings = getattr(map_conversion, 'Eastings', 0.0)
            northings = getattr(map_conversion, 'Northings', 0.0)
            orthogonal_height = getattr(map_conversion, 'OrthogonalHeight', 0.0)
            x_axis_abscissa = getattr(map_conversion, 'XAxisAbscissa', 1.0)
            x_axis_ordinate = getattr(map_conversion, 'XAxisOrdinate', 0.0)
            scale = getattr(map_conversion, 'Scale', 1.0)
            
            # Build transformation matrix (model to world)
            transform = self._build_map_conversion_transform(
                eastings, northings, orthogonal_height,
                x_axis_abscissa, x_axis_ordinate, scale
            )
            
            world_crs = WorldCRS(
                epsg=epsg,
                name=crs_name,
                origin_llh=None,  # Will be set if available
                to_world_transform=transform.flatten().tolist(),
                notes=f"IFC4 MapConversion: E={eastings}, N={northings}, Scale={scale}"
            )
            
            logger.info(f"Extracted IFC4 world CRS: {crs_name} (EPSG:{epsg})")
            return world_crs
            
        except Exception as e:
            logger.warning(f"Error extracting IFC4 world CRS: {e}")
            return None
    
    def _extract_ifc2x3_world_crs(self) -> Optional[WorldCRS]:
        """Extract world CRS from IFC2x3 files using IfcSite coordinates."""
        try:
            sites = self.ifc_file.by_type('IfcSite')
            
            if not sites:
                logger.info("No IfcSite found for georeferencing")
                return None
            
            site = sites[0]
            
            # Extract latitude, longitude, elevation
            lat = getattr(site, 'RefLatitude', None)
            lon = getattr(site, 'RefLongitude', None)
            elev = getattr(site, 'RefElevation', None)
            
            if not (lat and lon):
                logger.info("No georeferencing coordinates in IfcSite")
                return None
            
            # Convert from IFC compound format if needed
            latitude = self._convert_ifc_angle(lat) if lat else 0.0
            longitude = self._convert_ifc_angle(lon) if lon else 0.0
            elevation = float(elev) if elev else 0.0
            
            # Create identity transform (no precise conversion without CRS)
            transform = np.eye(4).flatten().tolist()
            
            world_crs = WorldCRS(
                epsg=None,  # Unknown in IFC2x3
                name="IFC2x3 Site Coordinates",
                origin_llh={
                    "lat": latitude,
                    "lon": longitude,
                    "elev_m": elevation
                },
                to_world_transform=transform,
                notes="IFC2x3 site coordinates - approximate georeferencing only"
            )
            
            logger.info(f"Extracted IFC2x3 site coordinates: {latitude:.6f}, {longitude:.6f}")
            return world_crs
            
        except Exception as e:
            logger.warning(f"Error extracting IFC2x3 world CRS: {e}")
            return None
    
    def _convert_ifc_angle(self, angle_value) -> float:
        """Convert IFC angle format to decimal degrees."""
        if isinstance(angle_value, (int, float)):
            return float(angle_value)
        
        # Handle compound angle format (degrees, minutes, seconds)
        if hasattr(angle_value, 'wrappedValue'):
            components = angle_value.wrappedValue
            if len(components) >= 3:
                degrees = float(components[0])
                minutes = float(components[1]) if components[1] else 0.0
                seconds = float(components[2]) if components[2] else 0.0
                
                return degrees + minutes/60.0 + seconds/3600.0
        
        return 0.0
    
    def _build_map_conversion_transform(self, eastings: float, northings: float, 
                                      orthogonal_height: float, x_axis_abscissa: float,
                                      x_axis_ordinate: float, scale: float) -> np.ndarray:
        """Build 4x4 transformation matrix from IFC MapConversion parameters."""
        # Create rotation matrix from X-axis direction
        cos_rot = x_axis_abscissa
        sin_rot = x_axis_ordinate
        
        # Normalize rotation vector
        length = np.sqrt(cos_rot**2 + sin_rot**2)
        if length > 0:
            cos_rot /= length
            sin_rot /= length
        
        # Build transformation matrix
        transform = np.array([
            [scale * cos_rot, -scale * sin_rot, 0, eastings],
            [scale * sin_rot,  scale * cos_rot, 0, northings],
            [0,                0,               1, orthogonal_height],
            [0,                0,               0, 1]
        ])
        
        return transform
    
    def compute_element_transforms(self, element) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute transformation matrices for an element.
        
        Returns:
            Tuple of (local_to_model, model_to_world) transform matrices
        """
        # Get local to model transform
        local_to_model = self._get_local_to_model_transform(element)
        
        # Get model to world transform
        model_to_world = self._get_model_to_world_transform()
        
        return local_to_model, model_to_world
    
    def _get_local_to_model_transform(self, element) -> np.ndarray:
        """Get transformation from element local coordinates to model coordinates."""
        try:
            if hasattr(element, 'ObjectPlacement') and element.ObjectPlacement:
                # Use ifcopenshell utility to get placement matrix
                placement_matrix = ifcopenshell.util.placement.get_local_placement(
                    element.ObjectPlacement
                )
                
                if placement_matrix is not None:
                    return np.array(placement_matrix)
            
            # Return identity if no placement found
            return np.eye(4)
            
        except Exception as e:
            logger.warning(f"Error getting local placement for element {element.id()}: {e}")
            return np.eye(4)
    
    def _get_model_to_world_transform(self) -> np.ndarray:
        """Get transformation from model coordinates to world coordinates."""
        if (self.coordinate_system.world and 
            self.coordinate_system.world.to_world_transform):
            
            transform_list = self.coordinate_system.world.to_world_transform
            return np.array(transform_list).reshape(4, 4)
        else:
            # Return identity if no world transform available
            return np.eye(4)
    
    def transform_point(self, point: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Transform a 3D point using a 4x4 transformation matrix."""
        if point.shape == (3,):
            # Convert to homogeneous coordinates
            point_homo = np.append(point, 1.0)
        elif point.shape == (4,):
            point_homo = point
        else:
            raise ValueError(f"Point must be 3D or 4D, got shape {point.shape}")
        
        # Apply transformation
        transformed = transform @ point_homo
        
        # Return 3D point
        return transformed[:3]
    
    def compute_bounding_box_transforms(self, bbox_local: List[float]) -> Tuple[List[float], List[float]]:
        """
        Transform a bounding box from local to model and world coordinates.
        
        Args:
            bbox_local: [min_x, min_y, min_z, max_x, max_y, max_z] in local coordinates
            
        Returns:
            Tuple of (bbox_model, bbox_world) in same format
        """
        if len(bbox_local) != 6:
            raise ValueError("Bounding box must have 6 values [min_x, min_y, min_z, max_x, max_y, max_z]")
        
        # Define corners of bounding box
        min_pt = np.array(bbox_local[:3])
        max_pt = np.array(bbox_local[3:])
        
        # Get identity transforms (these would come from element-specific transforms)
        local_to_model = np.eye(4)
        model_to_world = self._get_model_to_world_transform()
        
        # Transform min and max points to model coordinates
        min_model = self.transform_point(min_pt, local_to_model)
        max_model = self.transform_point(max_pt, local_to_model)
        
        # Transform to world coordinates
        min_world = self.transform_point(min_model, model_to_world)
        max_world = self.transform_point(max_model, model_to_world)
        
        # Ensure min/max ordering (transformation might flip them)
        bbox_model = [
            min(min_model[0], max_model[0]), min(min_model[1], max_model[1]), min(min_model[2], max_model[2]),
            max(min_model[0], max_model[0]), max(min_model[1], max_model[1]), max(min_model[2], max_model[2])
        ]
        
        bbox_world = [
            min(min_world[0], max_world[0]), min(min_world[1], max_world[1]), min(min_world[2], max_world[2]),
            max(min_world[0], max_world[0]), max(min_world[1], max_world[1]), max(min_world[2], max_world[2])
        ]
        
        return bbox_model, bbox_world
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert coordinate system info to dictionary for JSON export."""
        result = {
            "units": {
                "length": self.units.length,
                "area": self.units.area,
                "volume": self.units.volume,
                "angle": self.units.angle
            },
            "coordinate_system": {
                "model_origin_note": self.coordinate_system.model_origin_note,
                "true_north_deg": self.coordinate_system.true_north_deg
            }
        }
        
        if self.coordinate_system.world:
            world = self.coordinate_system.world
            result["coordinate_system"]["world"] = {
                "epsg": world.epsg,
                "name": world.name,
                "origin_llh": world.origin_llh,
                "to_world_transform": world.to_world_transform,
                "notes": world.notes
            }
        
        return result