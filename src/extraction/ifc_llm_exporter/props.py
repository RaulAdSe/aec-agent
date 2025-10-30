"""
Property Set and Quantity Extraction

This module handles:
- Property set (Pset) extraction and normalization
- Quantity take-off (QTO) extraction with units
- Custom property set handling
- Property value parsing and type conversion
- Unit normalization and conversion
- Material property extraction
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import ifcopenshell
    import ifcopenshell.util.element
except ImportError:
    ifcopenshell = None

logger = logging.getLogger(__name__)


class PropertyType(Enum):
    """Types of IFC properties."""
    SINGLE_VALUE = "single_value"
    ENUMERATED_VALUE = "enumerated_value"
    LIST_VALUE = "list_value"
    BOUNDED_VALUE = "bounded_value"
    TABLE_VALUE = "table_value"
    REFERENCE_VALUE = "reference_value"


@dataclass
class PropertyValue:
    """Normalized property value with type and unit information."""
    value: Any
    property_type: PropertyType
    unit: Optional[str] = None
    data_type: Optional[str] = None  # Text, Real, Integer, Boolean, etc.
    source_value: Any = None  # Original IFC value for reference


@dataclass
class PropertySet:
    """Property set with all properties."""
    name: str
    properties: Dict[str, PropertyValue] = field(default_factory=dict)
    pset_type: str = "property"  # "property" or "quantity"
    source: str = "ifc"  # "ifc", "type", "material"


@dataclass
class MaterialInfo:
    """Material information extracted from IFC."""
    name: str
    category: Optional[str] = None
    properties: Dict[str, PropertyValue] = field(default_factory=dict)
    constituents: List['MaterialInfo'] = field(default_factory=list)
    layers: List['MaterialLayerInfo'] = field(default_factory=list)


@dataclass
class MaterialLayerInfo:
    """Material layer information for layered materials."""
    material: MaterialInfo
    thickness: Optional[float] = None
    is_ventilated: Optional[bool] = None
    priority: Optional[int] = None


class PropertyExtractor:
    """
    Extracts and normalizes property sets and quantities from IFC elements.
    
    Handles all standard IFC property types and provides normalized output
    with proper unit handling for LLM consumption.
    """
    
    def __init__(self, ifc_file, units_info):
        """
        Initialize with IFC file and units information.
        
        Args:
            ifc_file: IFC file object
            units_info: UnitInfo from coordinate system handler
        """
        if not ifcopenshell:
            raise ImportError("ifcopenshell is required for property extraction")
        
        self.ifc_file = ifc_file
        self.units_info = units_info
        self.known_psets = self._load_known_psets()
        
        logger.info("Initialized property extractor")
    
    def extract_element_properties(self, element) -> Dict[str, PropertySet]:
        """
        Extract all property sets for an element.
        
        Returns:
            Dictionary mapping property set name to PropertySet object
        """
        property_sets = {}
        
        try:
            # Get all property sets using ifcopenshell utility
            psets = ifcopenshell.util.element.get_psets(element)
            
            for pset_name, pset_data in psets.items():
                # Determine if this is a quantity set
                is_quantity_set = pset_name.startswith('Qto_') or 'quantity' in pset_name.lower()
                
                # Create property set
                prop_set = PropertySet(
                    name=pset_name,
                    pset_type="quantity" if is_quantity_set else "property"
                )
                
                # Extract properties
                for prop_name, prop_value in pset_data.items():
                    try:
                        normalized_prop = self._normalize_property_value(
                            prop_name, prop_value, pset_name, is_quantity_set
                        )
                        prop_set.properties[prop_name] = normalized_prop
                    except Exception as e:
                        logger.warning(f"Error normalizing property {prop_name} in {pset_name}: {e}")
                        continue
                
                property_sets[pset_name] = prop_set
            
            # Also extract type properties if element has a type
            type_properties = self._extract_type_properties(element)
            for pset_name, prop_set in type_properties.items():
                # Prefix with "Type_" to distinguish from instance properties
                property_sets[f"Type_{pset_name}"] = prop_set
            
            return property_sets
            
        except Exception as e:
            logger.error(f"Error extracting properties for element {element.id()}: {e}")
            return {}
    
    def extract_material_properties(self, element) -> List[MaterialInfo]:
        """
        Extract material information for an element.
        
        Returns:
            List of MaterialInfo objects
        """
        materials = []
        
        try:
            if hasattr(element, 'HasAssociations'):
                for association in element.HasAssociations:
                    if association.is_a('IfcRelAssociatesMaterial'):
                        material_def = association.RelatingMaterial
                        
                        if material_def.is_a('IfcMaterial'):
                            material_info = self._extract_single_material(material_def)
                            materials.append(material_info)
                        
                        elif material_def.is_a('IfcMaterialLayerSet'):
                            layered_material = self._extract_layered_material(material_def)
                            materials.append(layered_material)
                        
                        elif material_def.is_a('IfcMaterialConstituentSet'):
                            constituent_material = self._extract_constituent_material(material_def)
                            materials.append(constituent_material)
            
            return materials
            
        except Exception as e:
            logger.error(f"Error extracting materials for element {element.id()}: {e}")
            return []
    
    def _normalize_property_value(self, prop_name: str, prop_value: Any, 
                                 pset_name: str, is_quantity: bool) -> PropertyValue:
        """Normalize a property value to standard format."""
        
        # Handle None values
        if prop_value is None:
            return PropertyValue(
                value=None,
                property_type=PropertyType.SINGLE_VALUE,
                source_value=prop_value
            )
        
        # Handle simple types
        if isinstance(prop_value, (int, float, bool)):
            unit = self._determine_property_unit(prop_name, pset_name, is_quantity)
            converted_value = self._convert_to_standard_units(prop_value, unit)
            
            return PropertyValue(
                value=converted_value,
                property_type=PropertyType.SINGLE_VALUE,
                unit=unit,
                data_type=type(prop_value).__name__,
                source_value=prop_value
            )
        
        # Handle string values
        elif isinstance(prop_value, str):
            # Check if it's an enumerated value
            if self._is_enumerated_property(prop_name, pset_name):
                return PropertyValue(
                    value=prop_value,
                    property_type=PropertyType.ENUMERATED_VALUE,
                    data_type="string",
                    source_value=prop_value
                )
            else:
                return PropertyValue(
                    value=prop_value,
                    property_type=PropertyType.SINGLE_VALUE,
                    data_type="string",
                    source_value=prop_value
                )
        
        # Handle lists
        elif isinstance(prop_value, (list, tuple)):
            normalized_list = []
            unit = self._determine_property_unit(prop_name, pset_name, is_quantity)
            
            for item in prop_value:
                if isinstance(item, (int, float)):
                    converted_item = self._convert_to_standard_units(item, unit)
                    normalized_list.append(converted_item)
                else:
                    normalized_list.append(item)
            
            return PropertyValue(
                value=normalized_list,
                property_type=PropertyType.LIST_VALUE,
                unit=unit,
                source_value=prop_value
            )
        
        # Handle complex IFC property types (if available)
        else:
            # For complex types, convert to string representation
            return PropertyValue(
                value=str(prop_value),
                property_type=PropertyType.SINGLE_VALUE,
                data_type="complex",
                source_value=prop_value
            )
    
    def _determine_property_unit(self, prop_name: str, pset_name: str, is_quantity: bool) -> Optional[str]:
        """Determine the unit for a property based on its name and context."""
        prop_name_lower = prop_name.lower()
        
        # Quantity sets have more standardized units
        if is_quantity:
            if any(keyword in prop_name_lower for keyword in ['area', 'grossarea', 'netarea']):
                return self.units_info.area
            elif any(keyword in prop_name_lower for keyword in ['volume', 'grossvolume', 'netvolume']):
                return self.units_info.volume
            elif any(keyword in prop_name_lower for keyword in ['length', 'width', 'height', 'thickness', 'perimeter']):
                return self.units_info.length
            elif any(keyword in prop_name_lower for keyword in ['angle']):
                return self.units_info.angle
        
        # Property sets - infer from property name
        else:
            if any(keyword in prop_name_lower for keyword in ['area']):
                return self.units_info.area
            elif any(keyword in prop_name_lower for keyword in ['volume']):
                return self.units_info.volume
            elif any(keyword in prop_name_lower for keyword in ['length', 'width', 'height', 'thickness', 'depth', 'diameter']):
                return self.units_info.length
            elif any(keyword in prop_name_lower for keyword in ['temperature']):
                return "°C"
            elif any(keyword in prop_name_lower for keyword in ['pressure']):
                return "Pa"
            elif any(keyword in prop_name_lower for keyword in ['rate', 'flow']):
                return "m³/s"
        
        return None
    
    def _convert_to_standard_units(self, value: Union[int, float], unit: Optional[str]) -> Union[int, float]:
        """Convert value to standard units (meters for length, etc.)."""
        if unit is None or not isinstance(value, (int, float)):
            return value
        
        # Length unit conversion
        if unit == self.units_info.length and self.units_info.scale_to_meters != 1.0:
            return value * self.units_info.scale_to_meters
        
        # Area unit conversion
        elif unit == self.units_info.area and self.units_info.scale_to_meters != 1.0:
            return value * (self.units_info.scale_to_meters ** 2)
        
        # Volume unit conversion
        elif unit == self.units_info.volume and self.units_info.scale_to_meters != 1.0:
            return value * (self.units_info.scale_to_meters ** 3)
        
        return value
    
    def _is_enumerated_property(self, prop_name: str, pset_name: str) -> bool:
        """Check if a property is typically an enumerated value."""
        # Known enumerated properties
        enumerated_props = {
            'IsExternal', 'LoadBearing', 'Combustible', 'SurfaceSpreadOfFlame',
            'ThermalTransmittance', 'IsFireResistant', 'FireRating',
            'AcousticRating', 'Status', 'AccessibilityType'
        }
        
        if prop_name in enumerated_props:
            return True
        
        # Check for rating or type properties
        if any(keyword in prop_name.lower() for keyword in ['rating', 'type', 'classification']):
            return True
        
        return False
    
    def _extract_type_properties(self, element) -> Dict[str, PropertySet]:
        """Extract properties from the element's type definition."""
        type_properties = {}
        
        try:
            if hasattr(element, 'IsDefinedBy'):
                for rel in element.IsDefinedBy:
                    if rel.is_a('IfcRelDefinesByType'):
                        type_element = rel.RelatingType
                        if type_element:
                            # Get type property sets
                            type_psets = ifcopenshell.util.element.get_psets(type_element)
                            
                            for pset_name, pset_data in type_psets.items():
                                is_quantity_set = pset_name.startswith('Qto_')
                                
                                prop_set = PropertySet(
                                    name=pset_name,
                                    pset_type="quantity" if is_quantity_set else "property",
                                    source="type"
                                )
                                
                                for prop_name, prop_value in pset_data.items():
                                    try:
                                        normalized_prop = self._normalize_property_value(
                                            prop_name, prop_value, pset_name, is_quantity_set
                                        )
                                        prop_set.properties[prop_name] = normalized_prop
                                    except Exception as e:
                                        logger.warning(f"Error normalizing type property {prop_name}: {e}")
                                        continue
                                
                                type_properties[pset_name] = prop_set
            
            return type_properties
            
        except Exception as e:
            logger.warning(f"Error extracting type properties: {e}")
            return {}
    
    def _extract_single_material(self, material) -> MaterialInfo:
        """Extract information from a single material."""
        material_info = MaterialInfo(
            name=getattr(material, 'Name', 'Unknown Material'),
            category=getattr(material, 'Category', None)
        )
        
        # Extract material properties
        try:
            material_psets = ifcopenshell.util.element.get_psets(material)
            
            for pset_name, pset_data in material_psets.items():
                for prop_name, prop_value in pset_data.items():
                    try:
                        normalized_prop = self._normalize_property_value(
                            prop_name, prop_value, pset_name, False
                        )
                        material_info.properties[prop_name] = normalized_prop
                    except Exception as e:
                        logger.warning(f"Error normalizing material property {prop_name}: {e}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error extracting material properties: {e}")
        
        return material_info
    
    def _extract_layered_material(self, material_layer_set) -> MaterialInfo:
        """Extract information from a layered material set."""
        material_info = MaterialInfo(
            name=getattr(material_layer_set, 'LayerSetName', 'Layered Material')
        )
        
        # Extract individual layers
        if hasattr(material_layer_set, 'MaterialLayers'):
            for layer in material_layer_set.MaterialLayers:
                layer_material = self._extract_single_material(layer.Material)
                
                layer_info = MaterialLayerInfo(
                    material=layer_material,
                    thickness=getattr(layer, 'LayerThickness', None),
                    is_ventilated=getattr(layer, 'IsVentilated', None),
                    priority=getattr(layer, 'Priority', None)
                )
                
                # Convert thickness to standard units
                if layer_info.thickness:
                    layer_info.thickness = self._convert_to_standard_units(
                        layer_info.thickness, self.units_info.length
                    )
                
                material_info.layers.append(layer_info)
        
        return material_info
    
    def _extract_constituent_material(self, material_constituent_set) -> MaterialInfo:
        """Extract information from a constituent material set."""
        material_info = MaterialInfo(
            name=getattr(material_constituent_set, 'Name', 'Constituent Material')
        )
        
        # Extract individual constituents
        if hasattr(material_constituent_set, 'MaterialConstituents'):
            for constituent in material_constituent_set.MaterialConstituents:
                if constituent.Material:
                    constituent_material = self._extract_single_material(constituent.Material)
                    material_info.constituents.append(constituent_material)
        
        return material_info
    
    def _load_known_psets(self) -> Dict[str, Dict[str, str]]:
        """Load known property set definitions for better parsing."""
        # This would typically load from IFC schema definitions
        # For now, we'll define some common ones
        known_psets = {
            'Pset_WallCommon': {
                'IsExternal': 'boolean',
                'ExtendToStructure': 'boolean',
                'LoadBearing': 'boolean',
                'Compartmentation': 'boolean',
                'Reference': 'string',
                'Status': 'enum',
                'AcousticRating': 'string',
                'FireRating': 'string',
                'Combustible': 'boolean',
                'SurfaceSpreadOfFlame': 'string',
                'ThermalTransmittance': 'real'
            },
            'Pset_DoorCommon': {
                'IsExternal': 'boolean',
                'Infiltration': 'real',
                'ThermalTransmittance': 'real',
                'GlazingAreaFraction': 'real',
                'HandicapAccessible': 'boolean',
                'FireExit': 'boolean',
                'SelfClosing': 'boolean',
                'SmokeStop': 'boolean',
                'FireRating': 'string',
                'AcousticRating': 'string'
            },
            'Pset_SpaceCommon': {
                'Reference': 'string',
                'IsExternal': 'boolean',
                'GrossPlannedArea': 'real',
                'NetPlannedArea': 'real',
                'PubliclyAccessible': 'boolean',
                'HandicapAccessible': 'boolean',
                'Category': 'string'
            }
        }
        
        return known_psets
    
    def get_property_summary(self, property_sets: Dict[str, PropertySet]) -> Dict[str, Any]:
        """Generate a summary of extracted properties for an element."""
        summary = {
            'total_psets': len(property_sets),
            'property_count': 0,
            'quantity_count': 0,
            'has_materials': False,
            'key_properties': {}
        }
        
        for pset_name, prop_set in property_sets.items():
            if prop_set.pset_type == 'quantity':
                summary['quantity_count'] += len(prop_set.properties)
            else:
                summary['property_count'] += len(prop_set.properties)
            
            # Extract key properties
            for prop_name, prop_value in prop_set.properties.items():
                if self._is_key_property(prop_name, pset_name):
                    summary['key_properties'][f"{pset_name}.{prop_name}"] = prop_value.value
        
        return summary
    
    def _is_key_property(self, prop_name: str, pset_name: str) -> bool:
        """Check if a property is considered a key property for summary."""
        key_props = {
            'FireRating', 'LoadBearing', 'IsExternal', 'Reference',
            'Width', 'Height', 'Length', 'Area', 'Volume',
            'ThermalTransmittance', 'AcousticRating'
        }
        
        return prop_name in key_props
    
    def to_dict(self, property_sets: Dict[str, PropertySet], 
               materials: List[MaterialInfo]) -> Dict[str, Any]:
        """Convert properties and materials to dictionary format."""
        result = {
            'properties': {},
            'materials': []
        }
        
        # Convert property sets
        for pset_name, prop_set in property_sets.items():
            pset_dict = {}
            
            for prop_name, prop_value in prop_set.properties.items():
                prop_dict = {
                    'value': prop_value.value,
                    'type': prop_value.property_type.value
                }
                
                if prop_value.unit:
                    prop_dict['unit'] = prop_value.unit
                
                if prop_value.data_type:
                    prop_dict['data_type'] = prop_value.data_type
                
                pset_dict[prop_name] = prop_dict
            
            result['properties'][pset_name] = pset_dict
        
        # Convert materials
        for material in materials:
            material_dict = {
                'name': material.name,
                'category': material.category,
                'properties': {}
            }
            
            # Convert material properties
            for prop_name, prop_value in material.properties.items():
                material_dict['properties'][prop_name] = {
                    'value': prop_value.value,
                    'type': prop_value.property_type.value
                }
                
                if prop_value.unit:
                    material_dict['properties'][prop_name]['unit'] = prop_value.unit
            
            # Add layers if present
            if material.layers:
                material_dict['layers'] = []
                for layer in material.layers:
                    layer_dict = {
                        'material': layer.material.name,
                        'thickness': layer.thickness
                    }
                    if layer.is_ventilated is not None:
                        layer_dict['is_ventilated'] = layer.is_ventilated
                    if layer.priority is not None:
                        layer_dict['priority'] = layer.priority
                    
                    material_dict['layers'].append(layer_dict)
            
            # Add constituents if present
            if material.constituents:
                material_dict['constituents'] = [
                    constituent.name for constituent in material.constituents
                ]
            
            result['materials'].append(material_dict)
        
        return result