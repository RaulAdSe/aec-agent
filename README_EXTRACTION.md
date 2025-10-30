# 🏗️ CAD File Extraction System

## Overview

The AEC Compliance Agent includes a comprehensive CAD file extraction system that can read building information from DWG and DXF files. This system extracts structured data about rooms, doors, walls, fire equipment, and fire sectors for building code compliance analysis.

## 🚀 Quick Start

### 1. Test the System
```bash
# Run the test suite to verify everything works
python3 scripts/test_extraction.py
```

### 2. Analyze CAD Files
```bash
# Analyze files to understand their structure
python3 scripts/extract_real_data.py --analyze data/blueprints/cad/
```

### 3. Extract Building Data
```bash
# Extract from single file
python3 scripts/extract_real_data.py --file "data/blueprints/cad/building.dwg" --output "building.json"

# Extract from directory
python3 scripts/extract_real_data.py --input data/blueprints/ --output-dir data/extracted/
```

## 📁 File Structure

```
src/extraction/
├── dwg_extractor.py          # DWG/DXF extraction engine
├── unified_extractor.py      # Unified interface
└── json_validator.py         # Data validation

scripts/
├── extract_cad_files.py      # Original extraction script
├── extract_real_data.py      # Enhanced extraction script
├── test_extraction.py        # Test suite
└── qcad_converter.js         # QCAD conversion script

data/
├── blueprints/               # Input CAD files
│   ├── cad/                  # DWG/DXF files
└── extracted/                # Output JSON files
```

## 🎯 Supported File Types

### DWG/DXF Files (AutoCAD)
- **DWG**: Native AutoCAD format
- **DXF**: AutoCAD Drawing Exchange Format
- **Status**: ✅ Fully supported
- **Method**: Python with `ezdxf` library


## 🏗️ Extracted Building Elements

### Rooms
- **Boundaries**: Polygon coordinates
- **Areas**: Calculated in square meters
- **Use Types**: office, corridor, stairs, storage, etc.
- **Properties**: Occupancy, emergency lighting, fire detection

### Doors
- **Dimensions**: Width and height
- **Types**: single, double, sliding, emergency
- **Properties**: Fire rating, egress status, opening direction
- **Connections**: Room-to-room relationships

### Walls
- **Geometry**: Start/end points, length, thickness
- **Properties**: Fire rating (EI-30, EI-60, EI-90, EI-120)
- **Classification**: Exterior vs interior walls

### Fire Equipment
- **Types**: Extinguishers, hydrants, sprinklers, alarms
- **Positions**: X,Y coordinates
- **Coverage**: Radius and status
- **Classification**: Automatic detection from block names

### Fire Sectors
- **Boundaries**: Compartment polygons
- **Fire Resistance**: EI ratings
- **Areas**: Calculated sector areas
- **Types**: Risk categories and compartment types

## 🔧 Usage Examples

### Programmatic Usage

```python
from src.extraction.unified_extractor import UnifiedExtractor

# Create extractor
extractor = UnifiedExtractor()

# Extract from single file
project = extractor.extract_from_file(
    file_path="building.dwg",
    project_name="My Building",
    level_name="Ground Floor",
    output_path="building.json"
)

# Extract from directory
projects = extractor.extract_from_directory(
    directory_path="data/blueprints/",
    output_dir="data/extracted/"
)

# Analyze file structure
analysis = extractor.analyze_file("building.dwg")
print(f"File has {analysis['total_entities']} entities")
```

### Command Line Usage

```bash
# Show capabilities
python3 scripts/extract_real_data.py --demo

# Analyze files
python3 scripts/extract_real_data.py --analyze data/blueprints/cad/building.dwg

# Extract single file
python3 scripts/extract_real_data.py --file building.dwg --output building.json

# Extract directory
python3 scripts/extract_real_data.py --input data/blueprints/ --output-dir data/extracted/

# Test system
python3 scripts/test_extraction.py
```

## 📊 Output Format

The extraction system produces structured JSON files with the following schema:

```json
{
  "metadata": {
    "project_name": "Building Project",
    "level_name": "Planta Baja",
    "extraction_date": "2025-01-18T10:30:00",
    "source_file": "building.dwg",
    "building_use": "commercial"
  },
  "rooms": [
    {
      "id": "R001",
      "name": "Office A",
      "level": "Planta Baja",
      "boundary": [[0, 0], [5, 0], [5, 4], [0, 4]],
      "use_type": "office",
      "area": 20.0,
      "occupancy_load": 2,
      "has_emergency_lighting": true,
      "has_fire_detection": true
    }
  ],
  "doors": [
    {
      "id": "D001",
      "position": [2.5, 0],
      "width": 0.9,
      "height": 2.1,
      "door_type": "single",
      "is_egress": false,
      "fire_rating": "EI-60",
      "opening_direction": "inward"
    }
  ],
  "walls": [
    {
      "id": "W001",
      "start": [0, 0],
      "end": [5, 0],
      "length": 5.0,
      "thickness": 0.15,
      "fire_rating": "EI-90",
      "is_exterior": true
    }
  ],
  "fire_equipment": [
    {
      "id": "FE001",
      "equipment_type": "extinguisher",
      "position": [2.5, 3.8],
      "coverage_radius": 15.0,
      "floor_level": "Planta Baja",
      "status": "active"
    }
  ],
  "sectors": [
    {
      "id": "S001",
      "name": "Sector 1",
      "boundary": [[0, 0], [10, 0], [10, 8], [0, 8]],
      "area": 80.0,
      "fire_resistance": "EI-60",
      "compartment_type": "general"
    }
  ]
}
```

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Install Python dependencies
pip install ezdxf pydantic shapely

```

### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd aec-compliance-agent

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/blueprints/cad
mkdir -p data/extracted
```

## 🔍 Entity Recognition

### DWG/DXF Recognition
The system automatically recognizes building elements based on:

- **Block Names**: INSERT entities with specific naming patterns
- **Layer Names**: Entities on layers with building-related keywords
- **Entity Types**: LINE, LWPOLYLINE, CIRCLE, HATCH entities
- **Geometric Properties**: Size, position, and relationships

### Recognition Patterns
```python
# Door detection
if any(keyword in block_name for keyword in ['DOOR', 'PUERTA', 'GATE']):
    # Extract door properties

# Fire equipment detection  
if any(keyword in name for keyword in ['EXTINTOR', 'BIE', 'SPRINKLER']):
    # Classify equipment type

# Room detection
if entity.is_closed and 'ROOM' in layer_name:
    # Extract room boundary
```

## 🚨 Error Handling

### Common Issues

#### DWG Files Not Supported by ezdxf
```bash
❌ File 'building.dwg' is not a DXF file
```
**Solution**: Use QCAD to convert DWG to DXF:
```bash
qcad -exec scripts/qcad_converter.js convert building.dwg building.dxf
```


#### File Not Found
```bash
❌ File not found: building.dwg
```
**Solution**: Check file path and ensure file exists.

## 🔗 Integration

### With RAG System
```python
# Extract building data
project = extract_from_file("building.dwg")

# Use in RAG queries
from src.rag.qa_chain import create_qa_chain
qa_chain = create_qa_chain(retriever)

# Query about compliance
result = qa_chain({
    "query": f"What are the fire safety requirements for a building with {len(project.rooms)} rooms?"
})
```

### With Agent System
```python
# Load project data for agent
from src.agent.tools import load_project_data
load_project_data("data/extracted/building.json")

# Agent can now verify compliance
```

## 📈 Performance

### Optimization Tips
- Process files individually for large datasets
- Use DXF format for better compatibility
- Clear memory between large file processing
- Use file analysis before full extraction

### Benchmarks
- **Small files** (< 1MB): ~1-2 seconds
- **Medium files** (1-10MB): ~5-10 seconds  
- **Large files** (> 10MB): ~30+ seconds

## 🧪 Testing

### Run Test Suite
```bash
python3 scripts/test_extraction.py
```

### Test Results
The test suite verifies:
- ✅ Mock data extraction
- ✅ DWG extractor functionality
- ✅ Unified extractor interface
- ✅ JSON validation
- ✅ Existing mock data compatibility

## 📚 Documentation

- [CAD Extraction Guide](docs/cad_extraction_guide.md)
- [Project Structure](docs/project-structure-md.md)
- [Development Guide](docs/development_guide_md.md)
- [API Reference](docs/api_reference_md.md)

## 🎯 Next Steps

1. **Convert DWG to DXF**: Use QCAD for better compatibility
2. **Test with Real Files**: Extract from actual CAD files
4. **Integrate with RAG**: Connect to compliance analysis
5. **Deploy Agent**: Use extracted data for automated compliance

---

**Version**: 1.0  
**Last Updated**: January 2025
