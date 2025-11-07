"""IFC to JSON Converter

Extracts geometrical building elements from IFC files into structured JSON.
Focuses on architectural elements: spaces, walls, doors, slabs, stairs.

Usage:
    # Save to JSON file
    python src/utils/ifc_to_json.py building.ifc -o output.json
    
    # Print JSON to terminal
    python src/utils/ifc_to_json.py building.ifc
    
    # Verbose output with logging
    python src/utils/ifc_to_json.py building.ifc -o output.json -v

Example:
    python src/utils/ifc_to_json.py data/in/DigitalHub_FM-ARC_v2.ifc -o data/out/building.json
"""

import ifcopenshell
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging


class IFCToJSONConverter:
    """Convert IFC building models to structured JSON."""

    def __init__(self, ifc_path: str):
        """Initialize converter with IFC file."""
        self.ifc_path = Path(ifc_path)
        self.model = ifcopenshell.open(str(self.ifc_path))
        self.logger = logging.getLogger(__name__)

    def extract_to_dict(self) -> Dict[str, Any]:
        """Extract all geometrical building elements to dictionary."""
        return {
            "file_info": self._get_file_info(),
            "spaces": self._extract_spaces(),
            "walls": self._extract_walls(),
            "doors": self._extract_doors(),
            "slabs": self._extract_slabs(),
            "stairs": self._extract_stairs()
        }
    
    def extract_to_json(self) -> str:
        """Extract all geometrical building elements to JSON string."""
        data = self.extract_to_dict()
        return json.dumps(data, indent=2, ensure_ascii=False)

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

    def _extract_spaces(self) -> List[Dict[str, Any]]:
        """Extract all spaces/rooms from IFC file."""
        spaces = []

        for space in self.model.by_type("IfcSpace"):
            space_data = {
                "id": space.GlobalId,
                "name": space.Name or f"Space_{space.id()}",
                "long_name": space.LongName if hasattr(space, 'LongName') else None,
                "description": space.Description,
                "object_type": space.ObjectType,
            }

            # Get space properties and quantities
            space_data.update(self._get_element_properties(space))
            space_data.update(self._get_element_quantities(space))
            space_data.update(self._get_element_geometry(space))
            spaces.append(space_data)

        return spaces

    def _extract_walls(self) -> List[Dict[str, Any]]:
      """Extract all walls from IFC file."""
      walls = []

      for wall in self.model.by_type("IfcWall"):
          wall_data = {
              "id": wall.GlobalId,
              "name": wall.Name or f"Wall_{wall.id()}",
              "description": wall.Description,
              "object_type": wall.ObjectType,
              "predefined_type": wall.PredefinedType if hasattr(wall, 'PredefinedType') else None,
          }

          # Get wall properties and quantities
          wall_data.update(self._get_element_properties(wall))
          wall_data.update(self._get_element_quantities(wall))
          wall_data.update(self._get_element_geometry(wall))

          walls.append(wall_data)

      return walls


    def _extract_doors(self) -> List[Dict[str, Any]]:
        """Extract all doors from IFC file."""
        doors = []

        for door in self.model.by_type("IfcDoor"):
            door_data = {
                "id": door.GlobalId,
                "name": door.Name or f"Door_{door.id()}",
                "description": door.Description,
                "object_type": door.ObjectType,
                "predefined_type": door.PredefinedType if hasattr(door, 'PredefinedType') else None,
            }

            # Get door properties, quantities, and geometry
            door_data.update(self._get_element_properties(door))
            door_data.update(self._get_element_quantities(door))
            door_data.update(self._get_element_geometry(door))

            doors.append(door_data)

        return doors       

    def _extract_slabs(self) -> List[Dict[str, Any]]:
      """Extract all slabs/floors from IFC file."""
      slabs = []

      for slab in self.model.by_type("IfcSlab"):
          slab_data = {
              "id": slab.GlobalId,
              "name": slab.Name or f"Slab_{slab.id()}",
              "description": slab.Description,
              "object_type": slab.ObjectType,
              "predefined_type": slab.PredefinedType if hasattr(slab, 'PredefinedType') else None,
          }

          # Get slab properties, quantities, and geometry
          slab_data.update(self._get_element_properties(slab))
          slab_data.update(self._get_element_quantities(slab))
          slab_data.update(self._get_element_geometry(slab))

          slabs.append(slab_data)

      return slabs 

    def _extract_stairs(self) -> List[Dict[str, Any]]:
      """Extract all stairs from IFC file."""
      stairs = []

      for stair in self.model.by_type("IfcStair"):
          stair_data = {
              "id": stair.GlobalId,
              "name": stair.Name or f"Stair_{stair.id()}",
              "description": stair.Description,
              "object_type": stair.ObjectType,
              "predefined_type": stair.PredefinedType if hasattr(stair, 'PredefinedType') else None,
          }

          # Get stair properties, quantities, and geometry
          stair_data.update(self._get_element_properties(stair))
          stair_data.update(self._get_element_quantities(stair))
          stair_data.update(self._get_element_geometry(stair))

          stairs.append(stair_data)

      return stairs


    def _get_element_properties(self, element) -> Dict[str, Any]:
      """Extract property sets from an IFC element."""
      properties = {}

      # Get all property sets for this element
      for rel in element.IsDefinedBy:
          if rel.is_a("IfcRelDefinesByProperties"):
              prop_set = rel.RelatingPropertyDefinition
              if prop_set.is_a("IfcPropertySet"):
                  for prop in prop_set.HasProperties:
                      if prop.is_a("IfcPropertySingleValue"):
                          properties[prop.Name] = prop.NominalValue.wrappedValue if prop.NominalValue else None

      return {"properties": properties}


    def _get_element_quantities(self, element) -> Dict[str, Any]:
      """Extract quantities from an IFC element."""
      quantities = {}

      for rel in element.IsDefinedBy:
          if rel.is_a("IfcRelDefinesByProperties"):
              quantity_set = rel.RelatingPropertyDefinition
              if quantity_set.is_a("IfcElementQuantity"):
                  for quantity in quantity_set.Quantities:
                      quantities[quantity.Name] = quantity[3]  # Value is at index 3

      return {"quantities": quantities}


    def _get_element_geometry(self, element) -> Dict[str, Any]:
        """Extract bounding box from real triangulated geometry."""
        try:
            import ifcopenshell.geom
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)

            # Compute triangulated geometry
            shape = ifcopenshell.geom.create_shape(settings, element)
            geom = shape.geometry

            verts = geom.verts
            # Group into (x,y,z) triplets efficiently
            coords = list(zip(*(iter(verts),) * 3))
            xs, ys, zs = zip(*coords)

            return {
                "geometry": {
                    "bbox": {
                        "min": [min(xs), min(ys), min(zs)],
                        "max": [max(xs), max(ys), max(zs)]
                    }
                }
            }

        except Exception:
            return {"geometry": None}
    


    def save_to_json(self, output_path: str) -> None:
        """Extract all data and save to JSON file."""
        json_string = self.extract_to_json()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_string)
        
        self.logger.info(f"Saved building data to {output_path}")


def main():
    """CLI interface for IFC to JSON conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert IFC files to structured JSON')
    parser.add_argument('ifc_file', help='Path to IFC file')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Convert IFC to JSON
    converter = IFCToJSONConverter(args.ifc_file)
    
    if args.output:
        # Save to file
        converter.save_to_json(args.output)
        print(f"✅ Converted {args.ifc_file} → {args.output}")
    else:
        # Print JSON to stdout
        json_output = converter.extract_to_json()
        print(json_output)


if __name__ == "__main__":
    main()