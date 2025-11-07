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
      """Extract geometry and spatial location from an IFC element."""
      geometry = {}

      # Get placement/location
      if hasattr(element, 'ObjectPlacement') and element.ObjectPlacement:
          placement = element.ObjectPlacement
          if placement.is_a('IfcLocalPlacement'):
              if placement.RelativePlacement:
                  if placement.RelativePlacement.is_a('IfcAxis2Placement3D'):
                      location = placement.RelativePlacement.Location
                      geometry['location'] = {
                          'x': location.Coordinates[0] if location.Coordinates else 0,
                          'y': location.Coordinates[1] if len(location.Coordinates) > 1 else 0,
                          'z': location.Coordinates[2] if len(location.Coordinates) > 2 else 0
                      }

      # Get representation/shape info
      if hasattr(element, 'Representation') and element.Representation:
          geometry['has_geometry'] = True
          geometry['representations'] = len(element.Representation.Representations)

      return {"geometry": geometry}


if __name__ == "__main__":
    # Test with Architecture IFC file
    converter = IFCToDictConverter("data/DigitalHub_FM-ARC_v2.ifc")

    # Test just file info first
    file_info = converter._get_file_info()
    print("File Info:", file_info)

    # Test spaces extraction
    spaces = converter._extract_spaces()
    print(f"Found {len(spaces)} spaces")
    if spaces:
        print("First space:", json.dumps(spaces[0], indent=2))

    # Test walls extraction
    walls = converter._extract_walls()
    print(f"Found {len(walls)} walls")
    if walls:
        print("First wall:", json.dumps(walls[0], indent=2))

    # Test doors extraction
    doors = converter._extract_doors()
    print(f"Found {len(doors)} doors")
    if doors:
        print("First door:", json.dumps(doors[0], indent=2))

    # Test slabs extraction
    slabs = converter._extract_slabs()
    print(f"Found {len(slabs)} slabs")
    if slabs:
        print("First slab:", json.dumps(slabs[0], indent=2))

    # Test stairs extraction
    stairs = converter._extract_stairs()
    print(f"Found {len(stairs)} stairs")
    if stairs:
        print("First stair:", json.dumps(stairs[0], indent=2))