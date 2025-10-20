# 🏗️ CAD File Extraction Guide

## Overview

The AEC Compliance Agent provides comprehensive tools for extracting building data from CAD files (DWG and DXF). This guide explains how to use these tools to extract structured building information for compliance analysis.

## 🎯 Supported File Types

### DWG/DXF Files (AutoCAD)
- **DWG**: Native AutoCAD format
- **DXF**: AutoCAD Drawing Exchange Format
- **Extraction Method**: Python with `ezdxf` library
- **Capabilities**: Full building element extraction


## 🛠️ Installation & Setup

### Prerequisites
```bash
# Install required Python packages
pip install ezdxf pydantic shapely

```

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd aec-compliance-agent

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/blueprints/cad
mkdir -p data/extracted
```

## 📁 File Structure

```
aec-compliance-agent/
├── src/extraction/
│   ├── dwg_extractor.py          # DWG/DXF extraction
│   ├── unified_extractor.py      # Unified interface
│   └── json_validator.py         # Data validation
├── scripts/
│   ├── extract_cad_files.py      # Original extraction script
│   ├── extract_real_data.py      # Enhanced extraction script
│   └── qcad_converter.js         # QCAD conversion script
└── data/
    ├── blueprints/               # Input CAD files
    │   ├── cad/                  # DWG/DXF files
    └── extracted/                # Output JSON files
```

## 🚀 Quick Start

### 1. Analyze CAD Files
```bash
# Analyze a single file
python scripts/extract_real_data.py --analyze data/blueprints/cad/building.dwg

# Analyze all files in directory
python scripts/extract_real_data.py --analyze data/blueprints/
```

### 2. Extract from Single File
```bash
# Extract from DWG file
python scripts/extract_real_data.py --file data/blueprints/cad/building.dwg --output building.json

```

### 3. Extract from Directory
```bash
# Extract from all CAD files in directory
python scripts/extract_real_data.py --input data/blueprints/ --output-dir data/extracted/
```

## 📊 Extracted Data Structure

### Project Schema
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
      "area": 20.0
    }
  ],
  "doors": [
    {
      "id": "D001",
      "position": [2.5, 0],
      "width": 0.9,
      "door_type": "single",
      "is_egress": false,
      "fire_rating": null
    }
  ],
  "walls": [
    {
      "id": "W001",
      "start": [0, 0],
      "end": [5, 0],
      "length": 5.0,
      "thickness": 0.15,
      "fire_rating": "EI-60",
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
      "fire_resistance": "EI-60"
    }
  ]
}
```

## 🔧 Advanced Usage

### Using the Unified Extractor Programmatically

```python
from src.extraction.unified_extractor import UnifiedExtractor

# Create extractor instance
extractor = UnifiedExtractor()

# Extract from single file
project = extractor.extract_from_file(
    file_path="data/blueprints/cad/building.dwg",
    project_name="My Building",
    level_name="Ground Floor",
    output_path="data/extracted/building.json"
)

# Extract from directory
projects = extractor.extract_from_directory(
    directory_path="data/blueprints/",
    output_dir="data/extracted/",
    level_name="Ground Floor"
)

# Analyze file without extraction
analysis = extractor.analyze_file("data/blueprints/cad/building.dwg")
print(f"File has {analysis['total_entities']} entities")
```

### Using Individual Extractors

```python
from src.extraction.dwg_extractor import DWGExtractor
# DWG/DXF extraction
dwg_extractor = DWGExtractor()
project = dwg_extractor.extract_from_file(
    file_path="building.dwg",
    project_name="My Building"
)
```

## 🎨 QCAD Conversion

### Using QCAD for DWG to DXF Conversion

```bash
# Basic conversion
qcad -exec scripts/qcad_converter.js convert input.dwg output.dxf

# Conversion with cleanup
qcad -exec scripts/qcad_converter.js convert input.dwg output.dxf --cleanup

# Analyze drawing without conversion
qcad -exec scripts/qcad_converter.js analyze input.dwg
```

### QCAD Script Features
- **Entity Analysis**: Counts and categorizes drawing elements
- **Layer Information**: Lists all layers and their contents
- **Building Element Detection**: Identifies doors, walls, rooms, fire equipment
- **Duplicate Removal**: Cleans up duplicate entities
- **Error Handling**: Robust error handling and reporting

## 🔍 Entity Recognition

### DWG/DXF Entity Types
- **INSERT**: Blocks (doors, equipment, symbols)
- **LINE**: Wall lines, door lines
- **LWPOLYLINE**: Room boundaries, wall outlines
- **CIRCLE**: Equipment symbols, column bases
- **HATCH**: Fire sectors, room fills
- **TEXT/MTEXT**: Room names, labels

### Recognition Logic
```python
# Door detection
if any(keyword in block_name for keyword in ['DOOR', 'PUERTA', 'GATE']):
    # Extract door properties
    width = get_door_width(entity)
    door_type = determine_door_type(block_name)
    is_egress = check_egress_status(entity)

# Fire equipment detection
if any(keyword in name for keyword in ['EXTINTOR', 'BIE', 'SPRINKLER']):
    equipment_type = classify_equipment(name)
    coverage_radius = get_coverage_radius(equipment_type)
```

## 🏗️ Building Element Classification

### Room Types
- `office`: Office spaces
- `corridor`: Circulation areas
- `stairs`: Stairwells
- `storage`: Storage areas
- `bathroom`: Restrooms
- `meeting_room`: Meeting spaces
- `lobby`: Reception areas
- `retail`: Retail spaces
- `general`: Unspecified spaces

### Door Types
- `single`: Single door
- `double`: Double door
- `sliding`: Sliding door
- `emergency`: Emergency exit door

### Fire Equipment Types
- `extinguisher`: Fire extinguishers
- `hydrant`: Fire hydrants (BIE)
- `sprinkler`: Sprinkler heads
- `alarm`: Fire alarms
- `emergency_light`: Emergency lighting
- `exit_sign`: Exit signs

### Fire Resistance Ratings
- `EI-30`: 30-minute fire resistance
- `EI-60`: 60-minute fire resistance
- `EI-90`: 90-minute fire resistance
- `EI-120`: 120-minute fire resistance

## 🚨 Error Handling

### Common Issues and Solutions

#### 1. File Not Found
```bash
❌ File not found: building.dwg
```
**Solution**: Check file path and ensure file exists

#### 2. Unsupported File Type
```bash
❌ Unsupported file extension: .pdf
```
**Solution**: Use supported formats (DWG, DXF)

#### 3. DXF Parsing Errors
```bash
❌ DXFStructureError: Invalid DXF file
```
**Solution**: 
- Verify DXF file integrity
- Try converting DWG to DXF with QCAD
- Check file encoding

#### 4. Memory Issues
```bash
❌ Out of memory when processing large file
```
**Solution**:
- Process files individually
- Increase system memory
- Use file chunking for very large files

## 📈 Performance Optimization

### Large File Handling
```python
# Process large files in chunks
def process_large_file(file_path, chunk_size=1000):
    extractor = DWGExtractor()
    # Custom processing logic for large files
    pass
```

### Memory Management
```python
# Clear extracted data between files
extractor = DWGExtractor()
for file_path in large_file_list:
    project = extractor.extract_from_file(file_path)
    # Process project
    del project  # Free memory
    extractor.rooms.clear()
    extractor.doors.clear()
```

## 🧪 Testing

### Test with Sample Files
```bash
# Test with existing sample files
python scripts/extract_real_data.py --file data/blueprints/cad/I01.4\ PCI\ -\ EXTINCIÓN\ AUTOMÁTICA.dwg

# Test directory processing
python scripts/extract_real_data.py --input data/blueprints/cad/ --output-dir data/extracted/
```

### Validation
```python
from src.extraction.json_validator import validate_project_json

# Validate extracted JSON
with open('extracted_data.json', 'r') as f:
    data = json.load(f)

try:
    project = validate_project_json(data)
    print("✅ JSON is valid!")
except ValidationError as e:
    print(f"❌ Validation failed: {e}")
```

## 🔗 Integration with Other Components

### With RAG System
```python
# Extract building data
project = extract_from_file("building.dwg")

# Use in RAG queries
from src.rag.qa_chain import create_qa_chain
qa_chain = create_qa_chain(retriever)

# Query about building compliance
result = qa_chain({"query": f"What are the fire safety requirements for a building with {len(project.rooms)} rooms?"})
```

### With Agent System
```python
# Load project data for agent
from src.agent.tools import load_project_data
load_project_data("data/extracted/building.json")

# Agent can now use building data for compliance verification
```

## 📚 Additional Resources

### Documentation
- [Project Structure Guide](project-structure-md.md)
- [Development Guide](development_guide_md.md)
- [API Reference](api_reference_md.md)

### Examples
- [Extraction Tutorial Notebook](../notebooks/01_data_extraction_simple.ipynb)
- [Sample Data](../data/blueprints/)

### Tools
- [QCAD Community Edition](https://qcad.org/)
- [ezdxf Documentation](https://ezdxf.readthedocs.io/)

---

**Version**: 1.0  
**Last Updated**: January 2025
