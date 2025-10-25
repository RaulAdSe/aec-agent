# IFC Extraction Guide

This guide explains how to extract building elements from IFC (Industry Foundation Classes) files using the AEC Compliance Agent.

## Overview

IFC is an open international standard for Building Information Modeling (BIM) that provides a comprehensive representation of building elements, their properties, and relationships. The AEC Compliance Agent can extract building data from IFC files and convert it to our standardized JSON format for compliance analysis.

## Features

### Supported IFC Versions
- IFC2X3
- IFC4
- IFC4X3 (with appropriate IfcOpenShell version)

### Extracted Elements
- **Spaces/Rooms** (IfcSpace, IfcRoom)
- **Doors** (IfcDoor) 
- **Walls** (IfcWall)
- **Windows** (IfcWindow) - basic support
- **Building Stories** (IfcBuildingStorey)
- **Building Structure** (IfcProject, IfcBuilding)

### Extracted Properties
- **Geometric Properties**: Areas, dimensions, positions
- **Material Information**: Wall materials, fire ratings
- **Functional Properties**: Room uses, door types
- **Relationships**: Spatial containment, connections
- **Property Sets**: Standard IFC property sets (Pset_*, Qto_*)

## Quick Start

### Installation

Ensure you have ifcopenshell installed:

```bash
pip install ifcopenshell
```

### Basic Usage

#### Extract Single IFC File

```bash
# Using dedicated IFC script
python scripts/extract_ifc_files.py -f building.ifc

# Using unified extractor
python -m src.extraction.unified_extractor --file building.ifc --output building.json
```

#### Extract Multiple Files

```bash
# Process directory of IFC files
python scripts/extract_ifc_files.py -d /path/to/ifc/files/ -o /output/directory/

# Analyze file structure without extraction
python scripts/extract_ifc_files.py -a building.ifc
```

#### Python API

```python
from src.extraction.ifc_extractor import extract_from_ifc
from pathlib import Path

# Extract project data
project = extract_from_ifc(Path("building.ifc"))

# Access extracted data
print(f"Project: {project.metadata.project_name}")
print(f"Levels: {len(project.levels)}")
print(f"Rooms: {len(project.get_all_rooms())}")
print(f"Doors: {len(project.get_all_doors())}")
```

## Detailed Usage

### Command Line Options

#### IFC Extraction Script

```bash
python scripts/extract_ifc_files.py [OPTIONS]

Options:
  -f, --file PATH        Extract from single IFC file
  -d, --directory PATH   Extract from all IFC files in directory
  -a, --analyze PATH     Analyze IFC file without extraction
  -o, --output PATH      Output file/directory path
  -v, --verbose          Enable verbose output and debugging
```

#### Examples

```bash
# Extract with custom output
python scripts/extract_ifc_files.py -f building.ifc -o extracted_data.json

# Process directory with output directory
python scripts/extract_ifc_files.py -d /ifc/files/ -o /extracted/

# Analyze file structure
python scripts/extract_ifc_files.py -a building.ifc

# Verbose extraction
python scripts/extract_ifc_files.py -f building.ifc -v
```

### Unified Extractor

The unified extractor automatically detects file types and handles both IFC and DWG files:

```bash
# Automatically detects IFC vs DWG
python -m src.extraction.unified_extractor --file building.ifc
python -m src.extraction.unified_extractor --file drawing.dwg

# Analyze any CAD file type
python -m src.extraction.unified_extractor --analyze building.ifc
```

## Data Mapping

### IFC to Schema Mapping

| IFC Entity | Our Schema | Properties Extracted |
|------------|------------|---------------------|
| IfcSpace | Room | Name, area, use, occupancy load |
| IfcRoom | Room | Name, area, use, occupancy load |
| IfcDoor | Door | Dimensions, type, fire rating, emergency exit |
| IfcWall | Wall | Start/end points, thickness, height, material |
| IfcBuildingStorey | Level | Name, elevation, contained elements |
| IfcProject | ProjectMetadata | Project name, building type |

### Room Use Detection

The extractor automatically determines room uses based on:

1. **IFC Properties**: Function and use properties in property sets
2. **Name Analysis**: Keywords in space names
3. **Spatial Context**: Relationships to other spaces

**Supported Room Uses**:
- Office, Commercial, Retail
- Meeting, Assembly
- Restroom, Storage
- Corridor, Stair, Elevator
- Reception, Educational
- Healthcare, Industrial

### Door Type Detection

Door types are determined from:

1. **PredefinedType**: IFC predefined door types
2. **Name Keywords**: Emergency, fire, sliding, double
3. **Properties**: Door function properties

**Supported Door Types**:
- Single, Double, Sliding, Revolving
- Fire Door, Emergency Exit

## Property Extraction

### Standard Property Sets

The extractor looks for these standard IFC property sets:

#### Spaces/Rooms
- `Qto_SpaceBaseQuantities`: Areas, volumes, perimeters
- `Pset_SpaceCommon`: Space function, occupancy
- `Pset_SpaceFireSafetyRequirements`: Fire safety properties

#### Doors
- `Qto_DoorBaseQuantities`: Width, height, area
- `Pset_DoorCommon`: Fire rating, acoustic properties
- `Pset_DoorWindowGlazingType`: Glazing information

#### Walls
- `Qto_WallBaseQuantities`: Length, width, height, area
- `Pset_WallCommon`: Fire rating, thermal properties
- Material associations for material information

### Custom Properties

You can extend the extractor to handle custom property sets by modifying the property extraction methods in `IFCExtractor`.

## Advanced Features

### Spatial Relationships

The extractor preserves IFC spatial relationships:

- **Containment**: Which elements are contained in which spaces/stories
- **Adjacency**: Which spaces are adjacent (through shared walls/doors)
- **Connectivity**: Which doors connect which spaces

### Level Management

Building stories (IfcBuildingStorey) are extracted as levels with:

- **Elevation**: Story elevation in building coordinates
- **Element Organization**: Rooms, doors, walls organized by level
- **Hierarchical Structure**: Proper building > story > space hierarchy

### Material Information

Wall materials are extracted from:

- **Material Associations**: IfcRelAssociatesMaterial relationships
- **Material Definitions**: IfcMaterial entities with properties
- **Layered Materials**: IfcMaterialLayerSet for composite walls

## Troubleshooting

### Common Issues

#### IfcOpenShell Not Installed
```
ImportError: ifcopenshell is required for IFC extraction
```
**Solution**: Install ifcopenshell: `pip install ifcopenshell`

#### Missing Spaces
```
Warning: No spaces found, creating default room
```
**Causes**:
- IFC file exported without spaces
- Spaces not properly defined in source model
- Space boundaries missing

**Solutions**:
- Re-export IFC with space boundaries enabled
- Check source model for proper space definition
- Use IfcRoom entities if IfcSpace not available

#### Incorrect Room Areas
```
Room areas seem incorrect
```
**Causes**:
- Missing quantity sets (Qto_SpaceBaseQuantities)
- Incorrect units in IFC file
- Space geometry issues

**Solutions**:
- Verify quantity sets in IFC file
- Check IFC file units
- Validate space geometry in IFC viewer

#### Missing Door Connections
```
Doors not connecting to rooms
```
**Causes**:
- Missing spatial containment relationships
- Doors not properly associated with spaces

**Solutions**:
- Check IFC spatial structure
- Verify door-space relationships in source model

### Performance Issues

For large IFC files:

1. **Use Verbose Mode**: Track extraction progress
   ```bash
   python scripts/extract_ifc_files.py -f large_building.ifc -v
   ```

2. **Check File Size**: Very large files (>100MB) may take time
   ```bash
   python scripts/extract_ifc_files.py -a large_building.ifc
   ```

3. **Memory Usage**: Large files may require significant RAM

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from src.extraction.ifc_extractor import extract_from_ifc
project = extract_from_ifc(Path("building.ifc"))
```

## Integration with Compliance Workflow

### Next Steps After Extraction

1. **Geometry Analysis**: Use extracted data for spatial calculations
   ```python
   # Calculate egress distances
   from src.calculations.geometry import calculate_egress_distances
   distances = calculate_egress_distances(project)
   ```

2. **RAG System**: Query building regulations
   ```python
   # Load into RAG system
   from src.rag.qa_chain import query_normativa
   compliance_info = query_normativa("fire safety requirements")
   ```

3. **AI Agent**: Automated compliance checking
   ```python
   # Run AI agent analysis
   from src.agent.simple_agent import run_compliance_check
   report = run_compliance_check(project)
   ```

### JSON Output Structure

The extracted JSON follows our standard schema:

```json
{
  "metadata": {
    "project_name": "Building Name",
    "file_name": "building.ifc",
    "building_type": "commercial",
    "total_area": 1500.0,
    "number_of_levels": 3,
    "created_date": "2024-01-01T00:00:00",
    "modified_date": "2024-01-01T00:00:00"
  },
  "levels": [
    {
      "name": "Ground Floor",
      "elevation": 0.0,
      "rooms": [...],
      "doors": [...],
      "walls": [...]
    }
  ]
}
```

## Best Practices

### IFC File Preparation

1. **Export Settings**:
   - Include space boundaries
   - Export with property sets
   - Use consistent coordinate system
   - Include material information

2. **Model Quality**:
   - Properly define spaces/rooms
   - Ensure door-wall-space relationships
   - Use standard IFC entities
   - Include necessary property sets

3. **Naming Conventions**:
   - Use descriptive space names
   - Include function keywords
   - Consistent door naming
   - Clear level/story names

### Extraction Workflow

1. **Analyze First**: Always analyze files before extraction
2. **Validate Results**: Check extracted data for completeness
3. **Iterate**: Refine source model if needed
4. **Document**: Note any model-specific quirks

### Performance Optimization

1. **Filter Entities**: Focus on required entity types
2. **Batch Processing**: Process multiple files together
3. **Caching**: Cache property sets for repeated access
4. **Streaming**: For very large files, consider streaming approaches

## Comparison: IFC vs DWG Extraction

| Aspect | IFC | DWG/DXF |
|--------|-----|---------|
| **Data Richness** | Very High | Medium |
| **Semantic Information** | Built-in | Inferred |
| **Relationships** | Explicit | Calculated |
| **Properties** | Extensive | Limited |
| **Standards Compliance** | High | Variable |
| **File Size** | Larger | Smaller |
| **Processing Speed** | Slower | Faster |
| **Accuracy** | High | Depends on drawing |

### When to Use IFC

- Modern BIM workflows
- Detailed compliance analysis
- Multi-disciplinary coordination
- Rich property requirements
- Standardized data exchange

### When to Use DWG

- Legacy CAD workflows
- Simple geometric analysis
- Faster processing needs
- Limited property requirements
- Traditional architecture practice

## Resources

### Documentation
- [buildingSMART IFC](https://www.buildingsmart.org/standards/bsi-standards/industry-foundation-classes/)
- [IfcOpenShell Docs](https://docs.ifcopenshell.org/)
- [IFC Schema Browser](https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2_TC1/HTML/)

### Tools
- [IFC Viewer](https://ifcviewer.com/) - Online IFC viewer
- [FreeCAD](https://www.freecadweb.org/) - Open source BIM software
- [BlenderBIM](https://blenderbim.org/) - Open source BIM addon for Blender

### Sample Files
- [buildingSMART Sample Files](https://www.buildingsmart.org/sample-ifc-files/)
- [IFC Examples](https://github.com/buildingSMART/Sample-Test-Files)

## Contributing

To contribute IFC extraction improvements:

1. **Test Cases**: Add test IFC files and expected results
2. **Entity Support**: Extend support for more IFC entity types
3. **Property Mapping**: Improve property set extraction
4. **Performance**: Optimize for large file processing
5. **Documentation**: Improve this guide and code comments

See the main project contributing guidelines for details.