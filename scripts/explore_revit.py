#!/usr/bin/env python3
"""
Explore Revit files to understand their structure and extract data.

This script tries multiple approaches to extract data from .rvt files.
"""

import os
from pathlib import Path
import json
import subprocess
import tempfile

def try_ifcopenshell_approach(rvt_file):
    """Try to use IFC approach if the file can be exported to IFC."""
    print("\n🔄 Trying IFC approach...")
    
    # Check if we have ifcopenshell
    try:
        import ifcopenshell
        print("✅ ifcopenshell is available")
        
        # Look for IFC files in the directory
        ifc_files = list(rvt_file.parent.glob("*.ifc"))
        if ifc_files:
            print(f"Found {len(ifc_files)} IFC files:")
            for ifc_file in ifc_files:
                print(f"  - {ifc_file.name}")
                try:
                    model = ifcopenshell.open(str(ifc_file))
                    print(f"    Schema: {model.schema}")
                    
                    # Get spaces (rooms)
                    spaces = model.by_type("IfcSpace")
                    print(f"    Spaces: {len(spaces)}")
                    
                    # Get doors
                    doors = model.by_type("IfcDoor")
                    print(f"    Doors: {len(doors)}")
                    
                    # Get walls
                    walls = model.by_type("IfcWall")
                    print(f"    Walls: {len(walls)}")
                    
                    return True
                except Exception as e:
                    print(f"    Error reading {ifc_file.name}: {e}")
        else:
            print("No IFC files found. You can export from Revit: File → Export → IFC")
            
    except ImportError:
        print("ifcopenshell not installed. Install with: pip install ifcopenshell")
    
    return False

def try_revit_automation(rvt_file):
    """Try to use Revit API automation."""
    print("\n🔄 Trying Revit automation...")
    
    # Check if Revit is installed
    revit_paths = [
        "/Applications/Autodesk Revit 2024/Revit 2024.app",
        "/Applications/Autodesk Revit 2023/Revit 2023.app",
        "/Applications/Autodesk Revit 2022/Revit 2022.app"
    ]
    
    revit_path = None
    for path in revit_paths:
        if Path(path).exists():
            revit_path = path
            break
    
    if revit_path:
        print(f"✅ Found Revit at: {revit_path}")
        
        # Create a Dynamo script to extract data
        dynamo_script = create_dynamo_extraction_script(rvt_file)
        print(f"📝 Created Dynamo script: {dynamo_script}")
        print("\nTo extract data:")
        print("1. Open Revit")
        print("2. Open the RVT file")
        print("3. Open Dynamo")
        print(f"4. Run the script: {dynamo_script}")
        
        return dynamo_script
    else:
        print("❌ Revit not found in standard locations")
        return None

def create_dynamo_extraction_script(rvt_file):
    """Create a Dynamo script to extract building data."""
    script_dir = Path("scripts/dynamo")
    script_dir.mkdir(exist_ok=True)
    
    script_content = f'''
# Dynamo Script for Building Data Extraction
# File: {rvt_file.name}

import clr
clr.AddReference('RevitAPI')
clr.AddReference('RevitAPIUI')
clr.AddReference('System')

from Autodesk.Revit.DB import *
from System.Collections.Generic import List
import json

# Get current document
doc = __revit__.ActiveUIDocument.Document

# Extract Rooms
rooms_data = []
room_collector = FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_Rooms)
for room in room_collector:
    if room.Area > 0:  # Only rooms with area
        boundary_points = []
        try:
            # Get room boundary
            boundary_segments = room.GetBoundarySegments(SpatialElementBoundaryOptions())
            for boundary_loop in boundary_segments:
                for segment in boundary_loop:
                    curve = segment.GetCurve()
                    start = curve.GetEndPoint(0)
                    boundary_points.append([start.X * 304.8, start.Y * 304.8])  # Convert to mm
        except:
            pass
        
        room_data = {{
            "id": room.Id.IntegerValue,
            "name": room.get_Parameter(BuiltInParameter.ROOM_NAME).AsString() or f"Room {{room.Number}}",
            "number": room.Number,
            "area": room.Area * 0.092903,  # Convert to m²
            "level": room.Level.Name if room.Level else "Unknown",
            "boundary": boundary_points
        }}
        rooms_data.append(room_data)

# Extract Doors
doors_data = []
door_collector = FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_Doors)
for door in door_collector:
    try:
        # Get door location
        location = door.Location
        if hasattr(location, 'Point'):
            point = location.Point
            position = [point.X * 304.8, point.Y * 304.8]  # Convert to mm
        else:
            position = [0, 0]
        
        # Get width
        width_param = door.get_Parameter(BuiltInParameter.DOOR_WIDTH)
        width = width_param.AsDouble() * 304.8 if width_param else 900  # mm
        
        door_data = {{
            "id": door.Id.IntegerValue,
            "name": door.Name,
            "width": width / 1000,  # Convert to meters
            "position": position,
            "level": door.Level.Name if door.Level else "Unknown",
            "type": door.Symbol.Family.Name
        }}
        doors_data.append(door_data)
    except Exception as e:
        continue

# Extract Walls
walls_data = []
wall_collector = FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_Walls)
for wall in wall_collector:
    try:
        # Get wall location line
        location = wall.Location
        if hasattr(location, 'Curve'):
            curve = location.Curve
            start = curve.GetEndPoint(0)
            end = curve.GetEndPoint(1)
            
            wall_data = {{
                "id": wall.Id.IntegerValue,
                "start": [start.X * 304.8, start.Y * 304.8],
                "end": [end.X * 304.8, end.Y * 304.8],
                "length": curve.Length * 304.8 / 1000,  # Convert to meters
                "type": wall.WallType.Name,
                "level": wall.Level.Name if wall.Level else "Unknown"
            }}
            walls_data.append(wall_data)
    except Exception as e:
        continue

# Create output data
output_data = {{
    "project_name": doc.Title,
    "file_name": "{rvt_file.name}",
    "rooms": rooms_data,
    "doors": doors_data, 
    "walls": walls_data,
    "extraction_date": str(System.DateTime.Now)
}}

# Save to JSON file
output_path = r"{rvt_file.parent.absolute()}/{rvt_file.stem}_extracted.json"
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Data extracted to: {{output_path}}")
print(f"Rooms: {{len(rooms_data)}}")
print(f"Doors: {{len(doors_data)}}")
print(f"Walls: {{len(walls_data)}}")
'''
    
    script_path = script_dir / f"extract_{rvt_file.stem}.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def analyze_rvt_file_properties(rvt_file):
    """Analyze RVT file properties and metadata."""
    print(f"\n📊 Analyzing RVT file: {rvt_file.name}")
    
    # File size and basic info
    file_size = rvt_file.stat().st_size / (1024 * 1024)  # MB
    print(f"File size: {file_size:.1f} MB")
    
    # Try to extract basic metadata
    try:
        # Check if file has compound document structure (OLE)
        with open(rvt_file, 'rb') as f:
            header = f.read(512)
            
        # Check for Revit signature
        if b'Revit' in header or b'ADSK' in header:
            print("✅ Confirmed Revit file format")
            
            # Look for text strings that might indicate content
            with open(rvt_file, 'rb') as f:
                content = f.read(10000)  # Read first 10KB
                
            # Look for common Revit strings
            indicators = [b'Level', b'Wall', b'Door', b'Room', b'Family']
            found_indicators = []
            for indicator in indicators:
                if indicator in content:
                    found_indicators.append(indicator.decode())
            
            if found_indicators:
                print(f"Found indicators: {', '.join(found_indicators)}")
        else:
            print("⚠️  File format not recognized as standard Revit")
            
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")

def create_mock_revit_data(rvt_file):
    """Create mock data based on the Revit file name (BAUHAUS Leganés)."""
    print(f"\n📦 Creating mock data for {rvt_file.name}...")
    
    # Based on the filename "BAUHAUS LEGANÉS", create realistic commercial building data
    mock_data = {
        "project_name": "BAUHAUS Leganés",
        "file_name": rvt_file.name,
        "building_type": "commercial_retail",
        "levels": [
            {
                "name": "Planta Baja",
                "elevation": 0.0,
                "rooms": [
                    {"id": "R001", "name": "Recepción", "area": 45.5, "use": "reception"},
                    {"id": "R002", "name": "Zona Venta 1", "area": 180.0, "use": "retail"},
                    {"id": "R003", "name": "Zona Venta 2", "area": 165.0, "use": "retail"},
                    {"id": "R004", "name": "Almacén Principal", "area": 120.0, "use": "storage"},
                    {"id": "R005", "name": "Oficina Gerencia", "area": 25.0, "use": "office"},
                    {"id": "R006", "name": "Sala Reuniones", "area": 30.0, "use": "meeting"},
                    {"id": "R007", "name": "Aseos Públicos", "area": 15.0, "use": "restroom"},
                    {"id": "R008", "name": "Cuarto Técnico", "area": 12.0, "use": "technical"},
                    {"id": "R009", "name": "Pasillo Principal", "area": 85.0, "use": "corridor"},
                    {"id": "R010", "name": "Zona Carga/Descarga", "area": 40.0, "use": "loading"}
                ]
            }
        ],
        "doors": [
            {"id": "D001", "width": 1.2, "type": "entrance", "fire_rated": False},
            {"id": "D002", "width": 1.0, "type": "interior", "fire_rated": False},
            {"id": "D003", "width": 1.2, "type": "emergency_exit", "fire_rated": True},
            {"id": "D004", "width": 0.9, "type": "office", "fire_rated": False},
            {"id": "D005", "width": 2.0, "type": "loading_dock", "fire_rated": True}
        ],
        "fire_safety": {
            "extinguishers": 12,
            "hydrants": 4,
            "emergency_lights": 18,
            "exit_signs": 8,
            "fire_sectors": 3
        },
        "compliance_notes": [
            "Building designed for commercial retail use",
            "Complies with CTE DB-SI requirements",
            "Maximum evacuation distance: 35m",
            "Fire resistance: EI-60 for main structure"
        ]
    }
    
    # Save mock data
    output_file = rvt_file.parent / f"{rvt_file.stem}_mock_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mock_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created mock data: {output_file.name}")
    return output_file

def main():
    """Main function to explore Revit file."""
    print("🏗️  Revit File Explorer")
    print("=" * 60)
    
    # Find Revit file
    revit_dir = Path("data/blueprints/revit")
    if not revit_dir.exists():
        print(f"❌ Directory not found: {revit_dir}")
        return
    
    rvt_files = list(revit_dir.glob("*.rvt"))
    if not rvt_files:
        print("❌ No RVT files found")
        return
    
    print(f"Found {len(rvt_files)} Revit files:")
    for rvt_file in rvt_files:
        print(f"  - {rvt_file.name}")
    
    # Process each file
    for rvt_file in rvt_files:
        print(f"\n{'='*60}")
        print(f"Processing: {rvt_file.name}")
        print(f"{'='*60}")
        
        # Analyze file properties
        analyze_rvt_file_properties(rvt_file)
        
        # Try different extraction methods
        success = False
        
        # Method 1: IFC approach
        if try_ifcopenshell_approach(rvt_file):
            success = True
        
        # Method 2: Revit automation
        dynamo_script = try_revit_automation(rvt_file)
        if dynamo_script:
            print(f"\n💡 Manual extraction option available")
            success = True
        
        # Method 3: Create mock data for testing
        mock_file = create_mock_revit_data(rvt_file)
        print(f"\n📋 Mock data created for testing: {mock_file}")
        
        if not success:
            print("\n⚠️  Automatic extraction not possible without Revit license")
            print("Options:")
            print("1. Export to IFC from Revit: File → Export → IFC")
            print("2. Use Dynamo script (requires Revit)")
            print("3. Use mock data for development/testing")

if __name__ == "__main__":
    main()