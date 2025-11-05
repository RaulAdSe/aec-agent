"""
TOON (Token-Oriented Object Notation) converter utilities for AEC Compliance Agent.

This module provides conversion utilities between JSON and TOON formats,
optimized for LLM token efficiency while maintaining data integrity.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from toon import encode as toon_encode, decode as toon_decode
except ImportError:
    logging.warning("python-toon library not found. TOON functionality will be limited.")
    toon_encode = None
    toon_decode = None

from pydantic import BaseModel, Field


class ToonConversionConfig(BaseModel):
    """Configuration for TOON conversion operations."""
    
    preserve_formatting: bool = Field(
        default=True,
        description="Whether to preserve human-readable formatting"
    )
    validate_conversion: bool = Field(
        default=True,
        description="Whether to validate data integrity after conversion"
    )
    token_optimization: bool = Field(
        default=True,
        description="Enable token optimization features"
    )


class ToonConverter:
    """
    Converter for JSON to TOON format and vice versa.
    
    Optimizes data representation for LLM consumption while maintaining
    full data fidelity and human readability.
    """
    
    def __init__(self, config: Optional[ToonConversionConfig] = None):
        """Initialize the TOON converter with optional configuration."""
        self.config = config or ToonConversionConfig()
        self.logger = logging.getLogger(__name__)
        
        if toon_encode is None:
            self.logger.warning("TOON library not available. Install with: pip install python-toon")
    
    def json_to_toon(self, data: Union[Dict, List, str, Path]) -> str:
        """
        Convert JSON data to TOON format.
        
        Args:
            data: JSON data as dict/list, JSON string, or file path
            
        Returns:
            TOON formatted string
            
        Raises:
            ValueError: If TOON library is not available or conversion fails
        """
        if toon_encode is None:
            raise ValueError("TOON library not available. Install with: pip install python-toon")
        
        # Handle different input types
        json_data = self._normalize_input(data)
        
        try:
            # Convert to TOON format
            toon_result = toon_encode(json_data)
            
            # Validate conversion if enabled
            if self.config.validate_conversion:
                self._validate_conversion(json_data, toon_result)
            
            self.logger.info(f"Successfully converted to TOON format")
            return toon_result
            
        except Exception as e:
            self.logger.error(f"Failed to convert to TOON: {e}")
            raise ValueError(f"TOON conversion failed: {e}")
    
    def toon_to_json(self, toon_data: Union[str, Path]) -> Dict[str, Any]:
        """
        Convert TOON format to JSON data.
        
        Args:
            toon_data: TOON formatted string or file path
            
        Returns:
            JSON data as dictionary
            
        Raises:
            ValueError: If TOON library is not available or conversion fails
        """
        if toon_decode is None:
            raise ValueError("TOON library not available. Install with: pip install python-toon")
        
        # Handle file path input
        if isinstance(toon_data, Path) or (isinstance(toon_data, str) and Path(toon_data).exists()):
            with open(toon_data, 'r', encoding='utf-8') as f:
                toon_string = f.read()
        else:
            toon_string = toon_data
        
        try:
            # Convert from TOON format
            json_result = toon_decode(toon_string)
            
            self.logger.info(f"Successfully converted from TOON format")
            return json_result
            
        except Exception as e:
            self.logger.error(f"Failed to convert from TOON: {e}")
            raise ValueError(f"TOON decoding failed: {e}")
    
    def convert_file(self, input_path: Path, output_path: Path, 
                    target_format: str = "toon") -> None:
        """
        Convert a file between JSON and TOON formats.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            target_format: Target format ("toon" or "json")
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if target_format.lower() == "toon":
            # JSON to TOON
            toon_result = self.json_to_toon(input_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(toon_result)
                
        elif target_format.lower() == "json":
            # TOON to JSON
            json_result = self.toon_to_json(input_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
                
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
        
        self.logger.info(f"Converted {input_path} to {output_path} ({target_format})")
    
    def get_token_savings(self, json_data: Union[Dict, List, str, Path]) -> Dict[str, Any]:
        """
        Calculate token savings when converting to TOON format.
        
        Args:
            json_data: JSON data to analyze
            
        Returns:
            Dictionary with token usage statistics
        """
        if toon_encode is None:
            return {"error": "TOON library not available"}
        
        json_data = self._normalize_input(json_data)
        
        # Convert to both formats
        json_string = json.dumps(json_data, separators=(',', ':'))
        toon_string = self.json_to_toon(json_data)
        
        # Calculate approximate token counts (rough estimation)
        json_tokens = len(json_string) // 4  # Rough GPT tokenization estimate
        toon_tokens = len(toon_string) // 4
        
        savings = json_tokens - toon_tokens
        savings_percent = (savings / json_tokens) * 100 if json_tokens > 0 else 0
        
        return {
            "json_size": len(json_string),
            "toon_size": len(toon_string),
            "json_tokens_est": json_tokens,
            "toon_tokens_est": toon_tokens,
            "token_savings": savings,
            "savings_percent": round(savings_percent, 2),
            "compression_ratio": round(len(toon_string) / len(json_string), 3) if len(json_string) > 0 else 0
        }
    
    def _normalize_input(self, data: Union[Dict, List, str, Path]) -> Union[Dict, List]:
        """Normalize input data to Python objects."""
        if isinstance(data, (dict, list)):
            return data
        elif isinstance(data, Path) or (isinstance(data, str) and Path(data).exists()):
            with open(data, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif isinstance(data, str):
            return json.loads(data)
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
    
    def _validate_conversion(self, original_data: Union[Dict, List], 
                           toon_result: str) -> None:
        """Validate that TOON conversion preserves data integrity."""
        try:
            decoded_data = toon_decode(toon_result)
            
            # Deep comparison of data structures
            if original_data != decoded_data:
                self.logger.warning("Data integrity check failed - structures don't match")
                # You might want to implement more sophisticated comparison here
                
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")


# Convenience functions for quick conversions
def json_to_toon(data: Union[Dict, List, str, Path]) -> str:
    """Quick conversion from JSON to TOON format."""
    converter = ToonConverter()
    return converter.json_to_toon(data)


def toon_to_json(toon_data: Union[str, Path]) -> Dict[str, Any]:
    """Quick conversion from TOON to JSON format."""
    converter = ToonConverter()
    return converter.toon_to_json(toon_data)


def analyze_token_savings(data: Union[Dict, List, str, Path]) -> Dict[str, Any]:
    """Quick analysis of potential token savings."""
    converter = ToonConverter()
    return converter.get_token_savings(data)


# Example usage and testing
if __name__ == "__main__":
    # Example building data for testing
    sample_building_data = {
        "metadata": {
            "project_name": "Sample Building",
            "total_area": 1000.0,
            "levels": 3
        },
        "rooms": [
            {"id": "R001", "name": "Office 1", "area": 25.0, "type": "office"},
            {"id": "R002", "name": "Office 2", "area": 30.0, "type": "office"},
            {"id": "R003", "name": "Conference", "area": 50.0, "type": "meeting"}
        ],
        "doors": [
            {"id": "D001", "width": 0.9, "height": 2.1, "fire_rated": True},
            {"id": "D002", "width": 1.2, "height": 2.1, "fire_rated": False}
        ]
    }
    
    # Test conversion
    converter = ToonConverter()
    
    try:
        print("Testing TOON conversion...")
        toon_result = converter.json_to_toon(sample_building_data)
        print(f"TOON Result:\n{toon_result}\n")
        
        # Test round-trip conversion
        json_result = converter.toon_to_json(toon_result)
        print(f"Round-trip successful: {sample_building_data == json_result}")
        
        # Analyze savings
        savings = converter.get_token_savings(sample_building_data)
        print(f"Token savings analysis: {savings}")
        
    except Exception as e:
        print(f"Test failed: {e}")