#!/usr/bin/env python3
"""
Convert DWG files to DXF format for processing with ezdxf.

This script provides multiple methods for converting DWG to DXF:
1. Using AutoCAD automation (if installed)
2. Instructions for manual conversion
3. Mock data generation for testing
"""

import os
import sys
from pathlib import Path
import subprocess
import time
import json
from typing import List, Optional

def convert_with_autocad_script(dwg_files: List[Path], output_dir: Path):
    """
    Generate AutoCAD script for batch conversion.
    
    This creates a script file that can be run in AutoCAD.
    """
    script_path = output_dir / "convert_to_dxf.scr"
    
    with open(script_path, 'w') as f:
        for dwg_file in dwg_files:
            dxf_file = output_dir / f"{dwg_file.stem}.dxf"
            f.write(f'_OPEN "{dwg_file.absolute()}"\n')
            f.write(f'_SAVEAS DXF V 2018 "{dxf_file.absolute()}"\n')
            f.write('_CLOSE\n')
    
    print(f"✅ AutoCAD script created: {script_path}")
    print("\nTo use this script:")
    print("1. Open AutoCAD")
    print("2. Type 'SCRIPT' command")
    print(f"3. Select: {script_path}")
    print("4. Wait for all files to be converted")
    
    return script_path

def convert_with_autocad_mac(dwg_file: Path, output_dir: Path) -> Optional[Path]:
    """
    Convert DWG to DXF using AutoCAD for Mac command line.
    
    Note: This requires AutoCAD for Mac to be installed.
    """
    dxf_file = output_dir / f"{dwg_file.stem}.dxf"
    
    # AutoCAD for Mac command line approach
    applescript = f'''
    tell application "AutoCAD"
        activate
        open POSIX file "{dwg_file.absolute()}"
        delay 2
        -- Export as DXF
        do script "_.SAVEAS DXF V 2018 \\"{dxf_file.absolute()}\\" "
        delay 2
        do script "_.CLOSE"
    end tell
    '''
    
    try:
        # Run AppleScript
        process = subprocess.Popen(['osascript', '-e', applescript], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=30)
        
        if process.returncode == 0:
            print(f"✅ Converted: {dwg_file.name} → {dxf_file.name}")
            return dxf_file
        else:
            print(f"❌ Error converting {dwg_file.name}: {stderr.decode()}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout converting {dwg_file.name}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def manual_conversion_instructions():
    """Print instructions for manual conversion."""
    print("""
    ============================================================
    MANUAL DWG TO DXF CONVERSION INSTRUCTIONS
    ============================================================
    
    Since AutoCAD is being installed, here's how to convert manually:
    
    1. OPEN AutoCAD
    2. For each DWG file:
       a. File → Open → Select the DWG file
       b. File → Save As → Save as type: AutoCAD 2018 DXF
       c. Save to: data/blueprints/cad/ (same directory)
    
    Files to convert:
    - I01.4 PCI - EXTINCIÓN AUTOMÁTICA.dwg → .dxf
    - I01.6 PCI - SECTORIZACIÓN.dwg → .dxf
    
    Alternatively, use QCAD (free):
    1. Download from: https://www.qcad.org/
    2. Open DWG file
    3. File → Save As → DXF format
    
    Or use online converter:
    - https://cloudconvert.com/dwg-to-dxf
    - https://www.zamzar.com/convert/dwg-to-dxf/
    """)

def create_mock_dxf_files(output_dir: Path):
    """
    Create mock DXF-like data for testing without actual conversion.
    
    This creates simplified JSON files that simulate what we expect
    from the actual DWG files based on their names.
    """
    print("\n📦 Creating mock data for testing...")
    
    # Mock data for fire extinguishing systems
    fire_equipment_data = {
        "file": "I01.4 PCI - EXTINCIÓN AUTOMÁTICA",
        "type": "fire_extinguishing",
        "equipment": [
            {"type": "extinguisher", "position": [10, 5], "id": "EXT01"},
            {"type": "extinguisher", "position": [20, 5], "id": "EXT02"},
            {"type": "extinguisher", "position": [30, 5], "id": "EXT03"},
            {"type": "hydrant", "position": [15, 10], "id": "BIE01"},
            {"type": "hydrant", "position": [35, 10], "id": "BIE02"},
            {"type": "sprinkler", "position": [10, 15], "id": "SPR01"},
            {"type": "sprinkler", "position": [20, 15], "id": "SPR02"},
            {"type": "alarm", "position": [25, 20], "id": "ALM01"},
            {"type": "emergency_light", "position": [5, 5], "id": "EMG01"},
            {"type": "emergency_light", "position": [40, 5], "id": "EMG02"},
        ],
        "coverage_areas": [
            {"equipment_id": "EXT01", "radius": 15},
            {"equipment_id": "BIE01", "radius": 25},
            {"equipment_id": "SPR01", "radius": 3.5}
        ]
    }
    
    # Mock data for sectorization
    sectorization_data = {
        "file": "I01.6 PCI - SECTORIZACIÓN",
        "type": "sectorization", 
        "sectors": [
            {
                "id": "S01",
                "name": "Sector Comercial",
                "fire_rating": "EI-60",
                "boundary": [[0, 0], [50, 0], [50, 30], [0, 30], [0, 0]],
                "area": 1500
            },
            {
                "id": "S02", 
                "name": "Sector Almacén",
                "fire_rating": "EI-90",
                "boundary": [[50, 0], [80, 0], [80, 30], [50, 30], [50, 0]],
                "area": 900
            },
            {
                "id": "S03",
                "name": "Sector Oficinas",
                "fire_rating": "EI-60",
                "boundary": [[0, 30], [40, 30], [40, 50], [0, 50], [0, 30]],
                "area": 800
            }
        ],
        "fire_doors": [
            {"id": "PF01", "position": [50, 15], "fire_rating": "EI2-60-C5"},
            {"id": "PF02", "position": [20, 30], "fire_rating": "EI2-60-C5"}
        ]
    }
    
    # Save mock data
    mock_file1 = output_dir / "I01.4_mock_data.json"
    mock_file2 = output_dir / "I01.6_mock_data.json"
    
    with open(mock_file1, 'w', encoding='utf-8') as f:
        json.dump(fire_equipment_data, f, indent=2, ensure_ascii=False)
    
    with open(mock_file2, 'w', encoding='utf-8') as f:
        json.dump(sectorization_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created mock data: {mock_file1.name}")
    print(f"✅ Created mock data: {mock_file2.name}")
    
    return [mock_file1, mock_file2]

def main():
    """Main conversion function."""
    print("🔄 DWG to DXF Converter")
    print("=" * 60)
    
    # Set up paths
    cad_dir = Path("data/blueprints/cad")
    if not cad_dir.exists():
        print(f"❌ Directory not found: {cad_dir}")
        return
    
    # Find DWG files
    dwg_files = list(cad_dir.glob("*.dwg"))
    print(f"Found {len(dwg_files)} DWG files:")
    for f in dwg_files:
        print(f"  - {f.name}")
    
    if not dwg_files:
        print("No DWG files found!")
        return
    
    # Check for existing DXF files
    dxf_files = list(cad_dir.glob("*.dxf"))
    if dxf_files:
        print(f"\n✅ Found {len(dxf_files)} existing DXF files:")
        for f in dxf_files:
            print(f"  - {f.name}")
    
    # Options menu
    print("\n" + "=" * 60)
    print("Conversion Options:")
    print("1. Generate AutoCAD script for batch conversion")
    print("2. Try AutoCAD for Mac automation (requires AutoCAD)")
    print("3. Show manual conversion instructions")
    print("4. Create mock data for testing")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        convert_with_autocad_script(dwg_files, cad_dir)
    elif choice == "2":
        print("\n🔄 Attempting AutoCAD automation...")
        for dwg_file in dwg_files:
            convert_with_autocad_mac(dwg_file, cad_dir)
            time.sleep(2)  # Wait between files
    elif choice == "3":
        manual_conversion_instructions()
    elif choice == "4":
        create_mock_dxf_files(cad_dir)
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()