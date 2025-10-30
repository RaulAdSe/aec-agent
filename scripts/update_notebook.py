#!/usr/bin/env python3
"""
Update the notebook to use the improved BuildingDataLoader
"""

import json
from pathlib import Path

def update_notebook():
    """Update the data extraction notebook with improved code."""
    
    notebook_path = Path("notebooks/01_data_extraction_simple.ipynb")
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # New improved data loading cell
    new_data_loading_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 📂 Load Real Building Data with Enhanced Loader\n",
            "import sys\n",
            "sys.path.append('..')\n",
            "\n",
            "from scripts.load_building_data import load_vilamalla_building, quick_building_overview, analyze_door_compliance\n",
            "import matplotlib.pyplot as plt\n",
            "import pandas as pd\n",
            "\n",
            "print(\"🏗️ LOADING VILAMALLA INDUSTRIAL COMPLEX DATA\")\n",
            "print(\"=\" * 60)\n",
            "\n",
            "# Load the real building data using our enhanced loader\n",
            "try:\n",
            "    loader = load_vilamalla_building()\n",
            "    print(\"✅ Building data loaded successfully!\")\n",
            "    \n",
            "    # Show comprehensive overview\n",
            "    quick_building_overview(loader)\n",
            "    \n",
            "except FileNotFoundError as e:\n",
            "    print(f\"❌ Error: {e}\")\n",
            "    print(\"Make sure to run the IFC extraction first:\")\n",
            "    print(\"python scripts/extract_ifc_files.py -f data/blueprints/VILAMALLA_ARQ_V6_TALLER_arq_20251032.ifc\")\n",
            "    loader = None"
        ]
    }
    
    # New analysis cell
    new_analysis_cell = {
        "cell_type": "code", 
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 🔍 Detailed Building Analysis\n",
            "if loader:\n",
            "    print(\"\\n🔍 DETAILED BUILDING ANALYSIS\")\n",
            "    print(\"=\" * 50)\n",
            "    \n",
            "    # Analyze doors\n",
            "    compliance = analyze_door_compliance(loader)\n",
            "    print(f\"\\n🚪 Door Compliance Analysis:\")\n",
            "    print(f\"   Compliance rate: {compliance['compliance_rate']:.1f}%\")\n",
            "    print(f\"   Compliant doors: {compliance['compliant_count']}/{compliance['total_doors']}\")\n",
            "    \n",
            "    if compliance['non_compliant_doors']:\n",
            "        print(f\"\\n❌ Non-compliant doors:\")\n",
            "        for door in compliance['non_compliant_doors']:\n",
            "            print(f\"   {door['id']}: {door['width_mm']}mm (needs {door['deficit_mm']}mm more)\")\n",
            "    else:\n",
            "        print(\"\\n🎉 All doors meet minimum width requirements!\")\n",
            "    \n",
            "    # Analyze by level\n",
            "    print(f\"\\n📊 Level-by-Level Breakdown:\")\n",
            "    for level in loader.levels:\n",
            "        level_name = level['name']\n",
            "        rooms = level.get('rooms', [])\n",
            "        doors = level.get('doors', [])\n",
            "        walls = level.get('walls', [])\n",
            "        total_area = sum(room['area'] for room in rooms)\n",
            "        \n",
            "        print(f\"\\n   🏢 {level_name}:\")\n",
            "        print(f\"      Elevation: {level['elevation']:.2f} m\")\n",
            "        print(f\"      Elements: {len(rooms)} rooms, {len(doors)} doors, {len(walls)} walls\")\n",
            "        print(f\"      Total area: {total_area:.0f} m²\")\n",
            "        \n",
            "        if doors:\n",
            "            door_widths = [door['width_mm'] for door in doors]\n",
            "            avg_width = sum(door_widths) / len(door_widths)\n",
            "            print(f\"      Avg door width: {avg_width:.0f} mm\")\n"
        ]
    }
    
    # New visualization cell
    new_viz_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 📊 Create Comprehensive Visualizations\n",
            "if loader:\n",
            "    print(\"\\n📊 CREATING VISUALIZATIONS\")\n",
            "    print(\"=\" * 40)\n",
            "    \n",
            "    # Create statistics dashboard\n",
            "    print(\"📈 Building statistics dashboard...\")\n",
            "    loader.create_statistics_dashboard()\n",
            "    plt.show()\n",
            "    \n",
            "    # Visualize the main level (with most doors)\n",
            "    level_with_most_doors = max(loader.levels, key=lambda x: len(x.get('doors', [])))\n",
            "    level_name = level_with_most_doors['name']\n",
            "    \n",
            "    print(f\"\\n🗺️ Visualizing main level: {level_name}\")\n",
            "    print(f\"   Doors: {len(level_with_most_doors.get('doors', []))}\")\n",
            "    print(f\"   Walls: {len(level_with_most_doors.get('walls', []))}\")\n",
            "    \n",
            "    fig = loader.visualize_level(level_name, figsize=(14, 10))\n",
            "    plt.show()\n",
            "    \n",
            "    print(\"\\n✅ Visualizations complete!\")"
        ]
    }
    
    # New pandas analysis cell
    new_pandas_cell = {
        "cell_type": "code",
        "execution_count": None, 
        "metadata": {},
        "outputs": [],
        "source": [
            "# 📊 Advanced Analysis with Pandas\n",
            "if loader:\n",
            "    print(\"\\n📊 ADVANCED DATA ANALYSIS WITH PANDAS\")\n",
            "    print(\"=\" * 50)\n",
            "    \n",
            "    # Export to DataFrames\n",
            "    dataframes = loader.export_to_dataframes()\n",
            "    rooms_df = dataframes['rooms']\n",
            "    doors_df = dataframes['doors']\n",
            "    walls_df = dataframes['walls']\n",
            "    \n",
            "    print(f\"\\n📋 DataFrames created:\")\n",
            "    print(f\"   🏢 Rooms: {len(rooms_df)} entries\")\n",
            "    print(f\"   🚪 Doors: {len(doors_df)} entries\")\n",
            "    print(f\"   🧱 Walls: {len(walls_df)} entries\")\n",
            "    \n",
            "    # Room analysis\n",
            "    print(f\"\\n🏢 Room Analysis:\")\n",
            "    room_stats = rooms_df.groupby('level').agg({\n",
            "        'area_m2': ['count', 'sum', 'mean'],\n",
            "        'occupancy_load': 'sum'\n",
            "    }).round(1)\n",
            "    print(room_stats)\n",
            "    \n",
            "    # Door analysis  \n",
            "    print(f\"\\n🚪 Door Statistics:\")\n",
            "    door_stats = doors_df[['width_mm', 'height_mm']].describe().round(1)\n",
            "    print(door_stats)\n",
            "    \n",
            "    # Wall material analysis\n",
            "    print(f\"\\n🧱 Wall Materials:\")\n",
            "    wall_materials = walls_df['material'].value_counts()\n",
            "    print(wall_materials)\n",
            "    \n",
            "    # Find potential issues\n",
            "    print(f\"\\n⚠️ Potential Issues:\")\n",
            "    narrow_doors = doors_df[doors_df['width_mm'] < 800]\n",
            "    if len(narrow_doors) > 0:\n",
            "        print(f\"   - {len(narrow_doors)} doors below 800mm width\")\n",
            "    else:\n",
            "        print(f\"   - All doors meet 800mm emergency width ✅\")\n",
            "    \n",
            "    thin_walls = walls_df[walls_df['thickness_mm'] < 100]\n",
            "    if len(thin_walls) > 0:\n",
            "        print(f\"   - {len(thin_walls)} walls below 100mm thickness\")\n",
            "    else:\n",
            "        print(f\"   - All walls have adequate thickness ✅\")\n",
            "    \n",
            "    print(f\"\\n💡 Sample DataFrames:\")\n",
            "    print(f\"\\nRooms:\")\n",
            "    print(rooms_df.head(3))\n",
            "    print(f\"\\nDoors:\")\n",
            "    print(doors_df[['id', 'width_mm', 'height_mm', 'door_type']].head(3))"
        ]
    }
    
    # Find and replace the old data loading cell (index 4)
    if len(notebook['cells']) > 4:
        notebook['cells'][4] = new_data_loading_cell
        
        # Insert new analysis cells after the data loading
        notebook['cells'].insert(5, new_analysis_cell)
        notebook['cells'].insert(6, new_viz_cell) 
        notebook['cells'].insert(7, new_pandas_cell)
    
    # Update title and description
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and 'source' in cell:
            source = ''.join(cell['source'])
            if '# 🏗️ IFC Building Data Extraction' in source:
                # Update the title
                notebook['cells'][i]['source'] = [
                    "# 🏗️ Real IFC Data Extraction - Vilamalla Industrial Complex\\n",
                    "\\n",
                    "<div align=\"center\">\\n",
                    "\\n",
                    "![Building Analysis](https://img.shields.io/badge/Building-Analysis-blue?style=for-the-badge)\\n",
                    "![IFC](https://img.shields.io/badge/IFC-Extraction-green?style=for-the-badge)\\n", 
                    "![Real Data](https://img.shields.io/badge/Real-Data-orange?style=for-the-badge)\\n",
                    "\\n",
                    "</div>\\n",
                    "\\n",
                    "**Welcome to the AEC Compliance Agent Tutorial Series!** 🎯\\n",
                    "\\n",
                    "This notebook demonstrates **real IFC data extraction** using the actual Vilamalla Industrial Complex project. You'll learn how to extract, analyze, and visualize building elements from Industry Foundation Classes (IFC) files.\\n",
                    "\\n",
                    "**What you'll see:**\\n",
                    "- 🏗️ **Real building data** from a 9-level industrial complex\\n",
                    "- 🚪 **23 doors** with compliance analysis\\n", 
                    "- 🧱 **102 walls** with material analysis\\n",
                    "- 📊 **Interactive visualizations** and statistical analysis\\n",
                    "- 🔍 **Pandas integration** for advanced data analysis\\n",
                    "\\n",
                    "**Data Source:** `VILAMALLA_ARQ_V6_TALLER_arq_20251032.ifc` (6.5 MB)\\n",
                    "\\n",
                    "Let's start building the future of intelligent building analysis! 🚀\\n"
                ]
                break
    
    # Save the updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Notebook updated successfully!")
    print(f"   - Enhanced data loading with BuildingDataLoader")
    print(f"   - Added comprehensive building analysis") 
    print(f"   - Added interactive visualizations")
    print(f"   - Added pandas integration for advanced analytics")
    print(f"   - Updated to use real Vilamalla IFC data")

if __name__ == "__main__":
    update_notebook()