"""
JSON Schema Validation

This module provides JSON schema validation for IFC LLM export format.
Includes schemas for project metadata and individual elements.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

# Get schema directory
SCHEMA_DIR = Path(__file__).parent

# Load schemas
with open(SCHEMA_DIR / "project_schema.json", 'r') as f:
    PROJECT_SCHEMA = json.load(f)

with open(SCHEMA_DIR / "element_schema.json", 'r') as f:
    ELEMENT_SCHEMA = json.load(f)


def validate_project_json(project_data: Dict[str, Any]) -> bool:
    """
    Validate project JSON against schema.
    
    Args:
        project_data: Project data dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        validate(instance=project_data, schema=PROJECT_SCHEMA)
        logger.debug("Project JSON validation passed")
        return True
    except ValidationError as e:
        logger.error(f"Project JSON validation failed: {e.message}")
        raise


def validate_element_json(element_data: Dict[str, Any]) -> bool:
    """
    Validate element JSON against schema.
    
    Args:
        element_data: Element data dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        validate(instance=element_data, schema=ELEMENT_SCHEMA)
        logger.debug(f"Element JSON validation passed for {element_data.get('guid', 'unknown')}")
        return True
    except ValidationError as e:
        logger.error(f"Element JSON validation failed: {e.message}")
        raise


def validate_elements_batch(elements: List[Dict[str, Any]], 
                           stop_on_error: bool = False) -> List[Dict[str, Any]]:
    """
    Validate a batch of elements, optionally filtering out invalid ones.
    
    Args:
        elements: List of element dictionaries
        stop_on_error: If True, raise on first error; if False, filter invalid elements
        
    Returns:
        List of valid elements (may be filtered if stop_on_error=False)
        
    Raises:
        ValidationError: If stop_on_error=True and validation fails
    """
    valid_elements = []
    errors = []
    
    for i, element in enumerate(elements):
        try:
            if validate_element_json(element):
                valid_elements.append(element)
        except ValidationError as e:
            error_msg = f"Element {i} ({element.get('guid', 'unknown')}): {e.message}"
            errors.append(error_msg)
            
            if stop_on_error:
                raise ValidationError(error_msg)
            else:
                logger.warning(f"Skipping invalid element: {error_msg}")
    
    if errors and not stop_on_error:
        logger.warning(f"Filtered out {len(errors)} invalid elements out of {len(elements)}")
    
    return valid_elements


def get_schema_info() -> Dict[str, Any]:
    """Get information about available schemas."""
    return {
        "project_schema": {
            "title": PROJECT_SCHEMA.get("title"),
            "description": PROJECT_SCHEMA.get("description"),
            "version": PROJECT_SCHEMA.get("$id")
        },
        "element_schema": {
            "title": ELEMENT_SCHEMA.get("title"), 
            "description": ELEMENT_SCHEMA.get("description"),
            "version": ELEMENT_SCHEMA.get("$id")
        },
        "schema_directory": str(SCHEMA_DIR)
    }


def validate_guid_format(guid: str) -> bool:
    """
    Validate IFC GUID format.
    
    IFC GUIDs are 22-character base64-encoded strings using a custom alphabet.
    
    Args:
        guid: GUID string to validate
        
    Returns:
        True if valid IFC GUID format
    """
    if not isinstance(guid, str) or len(guid) != 22:
        return False
    
    # IFC GUID uses this specific alphabet
    ifc_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
    
    return all(c in ifc_alphabet for c in guid)


def sanitize_element_for_schema(element: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize an element dictionary to conform to schema requirements.
    
    Args:
        element: Raw element dictionary
        
    Returns:
        Sanitized element dictionary
    """
    sanitized = element.copy()
    
    # Ensure required fields exist
    required_fields = {
        'guid': 'unknown_guid',
        'ifc_class': 'IfcElement',
        'name': 'Unnamed Element',
        'spatial': {},
        'coordinates': {
            'transform_local_to_model': [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
            'transform_model_to_world': [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
            'reference_point_model_m': [0, 0, 0],
            'bbox_model_m': [0, 0, 0, 0, 0, 0]
        },
        'geometry_summary': {
            'shape': 'unknown'
        },
        'properties': {},
        'relations': {},
        'source': {}
    }
    
    for field, default_value in required_fields.items():
        if field not in sanitized:
            sanitized[field] = default_value
        elif field in ['coordinates', 'geometry_summary', 'spatial', 'properties', 'relations', 'source']:
            # Ensure nested objects have required fields
            if isinstance(default_value, dict):
                for subfield, subdefault in default_value.items():
                    if subfield not in sanitized[field]:
                        sanitized[field][subfield] = subdefault
    
    # Validate and fix GUID format
    if not validate_guid_format(sanitized['guid']):
        logger.warning(f"Invalid GUID format: {sanitized['guid']}")
        # Generate a valid GUID format (placeholder)
        sanitized['guid'] = 'invalid_guid_' + str(hash(sanitized['guid']))[:15]
    
    # Ensure IFC class starts with 'Ifc'
    if not sanitized['ifc_class'].startswith('Ifc'):
        sanitized['ifc_class'] = 'Ifc' + sanitized['ifc_class']
    
    return sanitized


__all__ = [
    'PROJECT_SCHEMA',
    'ELEMENT_SCHEMA', 
    'validate_project_json',
    'validate_element_json',
    'validate_elements_batch',
    'validate_guid_format',
    'sanitize_element_for_schema',
    'get_schema_info'
]