#!/usr/bin/env python3
"""
Simplify the notebook to focus on extraction and data showcase
"""

import json
from pathlib import Path

def simplify_notebook():
    """Create a simplified, focused notebook."""
    
    # Create a clean, simple notebook structure
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🏗️ IFC Data Extraction - What Can We Get?\n",
                    "\n",
                    "This notebook demonstrates **IFC data extraction** using a real building: the **Vilamalla Industrial Complex**.\n",
                    "\n",
                    "**Goal**: Show you what data we can extract from IFC files and how to access it.\n",
                    "\n",
                    "**Input**: `VILAMALLA_ARQ_V6_TALLER_arq_20251032.ifc` (6.5 MB)\n",
                    "**Output**: Structured building data (rooms, doors, walls, levels)\n"
                ]
            },
            {
                "cell_type": "markdown", 
                "metadata": {},
                "source": [
                    "## 🚀 Step 1: Load the Building Data\n",
                    "\n",
                    "We've already extracted the IFC data. Let's load it and see what we got:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load our building data loader\n",
                    "import sys\n",
                    "sys.path.append('..')\n",
                    "\n",
                    "from scripts.load_building_data import load_vilamalla_building\n",
                    "import json\n",
                    "\n",
                    "# Load the extracted building data\n",
                    "print(\"🏗️ Loading Vilamalla Industrial Complex...\")\n",
                    "loader = load_vilamalla_building()\n",
                    "\n",
                    "print(f\"\\n✅ Success! Here's what we extracted:\")\n",
                    "print(f\"   🏢 Project: {loader.metadata.get('project_name')}\")\n",
                    "print(f\"   📊 Levels: {len(loader.levels)}\")\n",
                    "print(f\"   🏠 Rooms: {len(loader.all_rooms)}\")\n",
                    "print(f\"   🚪 Doors: {len(loader.all_doors)}\")\n",
                    "print(f\"   🧱 Walls: {len(loader.all_walls)}\")\n",
                    "print(f\"   📐 Total area: {loader.metadata.get('total_area', 0):.0f} m²\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🏢 Step 2: Explore Building Levels\n",
                    "\n",
                    "Let's see what levels this building has:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"🏢 Building Levels:\")\n",
                    "print(\"=\" * 50)\n",
                    "\n",
                    "for i, level in enumerate(loader.levels, 1):\n",
                    "    name = level['name']\n",
                    "    elevation = level['elevation']\n",
                    "    rooms = len(level.get('rooms', []))\n",
                    "    doors = len(level.get('doors', []))\n",
                    "    walls = len(level.get('walls', []))\n",
                    "    \n",
                    "    print(f\"{i:2}. {name[:30]:30} | Elev: {elevation:6.1f}m | R:{rooms} D:{doors} W:{walls}\")\n",
                    "\n",
                    "print(f\"\\n💡 This is a {len(loader.levels)}-level industrial building!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🚪 Step 3: Look at Doors\n",
                    "\n",
                    "Doors are critical for safety. Let's see what door data we extracted:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"🚪 Door Data:\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "# Show first 5 doors as examples\n",
                    "for i, door in enumerate(loader.all_doors[:5], 1):\n",
                    "    print(f\"{i}. Door {door['id']}:\")\n",
                    "    print(f\"   Size: {door['width_mm']}mm × {door['height_mm']}mm\")\n",
                    "    print(f\"   Type: {door['door_type']}\")\n",
                    "    print(f\"   Emergency exit: {'Yes' if door['is_emergency_exit'] else 'No'}\")\n",
                    "    print(f\"   Position: ({door['position']['x']:.1f}, {door['position']['y']:.1f})\")\n",
                    "    print()\n",
                    "\n",
                    "# Quick statistics\n",
                    "widths = [door['width_mm'] for door in loader.all_doors]\n",
                    "print(f\"📊 Door Statistics:\")\n",
                    "print(f\"   Total doors: {len(loader.all_doors)}\")\n",
                    "print(f\"   Width range: {min(widths):.0f} - {max(widths):.0f} mm\")\n",
                    "print(f\"   Average width: {sum(widths)/len(widths):.0f} mm\")\n",
                    "\n",
                    "# Check compliance\n",
                    "compliant = sum(1 for w in widths if w >= 700)  # 700mm minimum\n",
                    "print(f\"   Compliant (≥700mm): {compliant}/{len(widths)} ({compliant/len(widths)*100:.1f}%)\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🧱 Step 4: Examine Walls\n",
                    "\n",
                    "Walls form the structure. Let's see what wall information we have:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"🧱 Wall Data:\")\n",
                    "print(\"=\" * 50)\n",
                    "\n",
                    "# Analyze materials\n",
                    "materials = {}\n",
                    "thicknesses = []\n",
                    "heights = []\n",
                    "\n",
                    "for wall in loader.all_walls:\n",
                    "    material = wall.get('material', 'unknown')\n",
                    "    materials[material] = materials.get(material, 0) + 1\n",
                    "    thicknesses.append(wall['thickness_mm'])\n",
                    "    heights.append(wall['height_mm'])\n",
                    "\n",
                    "print(f\"📊 Wall Statistics:\")\n",
                    "print(f\"   Total walls: {len(loader.all_walls)}\")\n",
                    "print(f\"   Materials:\")\n",
                    "for material, count in materials.items():\n",
                    "    percentage = count/len(loader.all_walls)*100\n",
                    "    print(f\"     {material}: {count} walls ({percentage:.1f}%)\")\n",
                    "\n",
                    "print(f\"\\n   Dimensions:\")\n",
                    "print(f\"     Thickness: {min(thicknesses):.0f} - {max(thicknesses):.0f} mm (avg: {sum(thicknesses)/len(thicknesses):.0f} mm)\")\n",
                    "print(f\"     Height: {min(heights):.0f} - {max(heights):.0f} mm (avg: {sum(heights)/len(heights):.0f} mm)\")\n",
                    "\n",
                    "# Show a sample wall\n",
                    "sample_wall = loader.all_walls[0]\n",
                    "print(f\"\\n💡 Sample wall data structure:\")\n",
                    "print(f\"   ID: {sample_wall['id']}\")\n",
                    "print(f\"   Start: ({sample_wall['start_point']['x']:.1f}, {sample_wall['start_point']['y']:.1f})\")\n",
                    "print(f\"   End: ({sample_wall['end_point']['x']:.1f}, {sample_wall['end_point']['y']:.1f})\")\n",
                    "print(f\"   Material: {sample_wall['material']}\")\n",
                    "print(f\"   Thickness: {sample_wall['thickness_mm']} mm\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🏠 Step 5: Check Room Information\n",
                    "\n",
                    "Rooms define the building's function. Let's see what we got:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"🏠 Room Data:\")\n",
                    "print(\"=\" * 50)\n",
                    "\n",
                    "total_area = 0\n",
                    "room_types = {}\n",
                    "\n",
                    "for room in loader.all_rooms:\n",
                    "    area = room['area']\n",
                    "    use_type = room['use']\n",
                    "    level = room['level']\n",
                    "    \n",
                    "    total_area += area\n",
                    "    room_types[use_type] = room_types.get(use_type, 0) + 1\n",
                    "    \n",
                    "    print(f\"  🏠 {room['name'][:30]:30} | {area:5.0f} m² | {use_type:12} | {level[:15]:15}\")\n",
                    "\n",
                    "print(f\"\\n📊 Room Summary:\")\n",
                    "print(f\"   Total rooms: {len(loader.all_rooms)}\")\n",
                    "print(f\"   Total area: {total_area:.0f} m²\")\n",
                    "print(f\"   Average area: {total_area/len(loader.all_rooms):.0f} m²\")\n",
                    "\n",
                    "print(f\"\\n   Room types:\")\n",
                    "for room_type, count in room_types.items():\n",
                    "    print(f\"     {room_type}: {count} rooms\")\n",
                    "\n",
                    "print(f\"\\n💡 Note: Rooms were derived from building geometry since the IFC didn't have explicit spaces defined.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 Step 6: Data in Action - Simple Analysis\n",
                    "\n",
                    "Let's show how this data can be used for building analysis:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"📊 Building Analysis Examples:\")\n",
                    "print(\"=\" * 50)\n",
                    "\n",
                    "# 1. Safety Analysis - Door Compliance\n",
                    "print(\"🚨 Safety Check - Door Widths:\")\n",
                    "emergency_min = 800  # mm\n",
                    "standard_min = 700   # mm\n",
                    "\n",
                    "emergency_compliant = sum(1 for door in loader.all_doors \n",
                    "                         if door['width_mm'] >= emergency_min)\n",
                    "standard_compliant = sum(1 for door in loader.all_doors \n",
                    "                        if door['width_mm'] >= standard_min)\n",
                    "\n",
                    "print(f\"   Emergency width (≥800mm): {emergency_compliant}/{len(loader.all_doors)} doors\")\n",
                    "print(f\"   Standard width (≥700mm): {standard_compliant}/{len(loader.all_doors)} doors\")\n",
                    "\n",
                    "# 2. Occupancy Analysis\n",
                    "print(f\"\\n👥 Occupancy Analysis:\")\n",
                    "total_occupancy = sum(room['occupancy_load'] for room in loader.all_rooms)\n",
                    "print(f\"   Total building occupancy: {total_occupancy} people\")\n",
                    "print(f\"   Area per person: {total_area/total_occupancy:.1f} m²/person\")\n",
                    "\n",
                    "# 3. Construction Analysis\n",
                    "print(f\"\\n🏗️ Construction Summary:\")\n",
                    "concrete_walls = sum(1 for wall in loader.all_walls if wall['material'] == 'concrete')\n",
                    "wall_percentage = concrete_walls / len(loader.all_walls) * 100\n",
                    "print(f\"   Concrete construction: {concrete_walls}/{len(loader.all_walls)} walls ({wall_percentage:.1f}%)\")\n",
                    "\n",
                    "avg_wall_thickness = sum(wall['thickness_mm'] for wall in loader.all_walls) / len(loader.all_walls)\n",
                    "print(f\"   Average wall thickness: {avg_wall_thickness:.0f} mm\")\n",
                    "\n",
                    "# 4. Level Distribution\n",
                    "print(f\"\\n🏢 Level Activity:\")\n",
                    "for level in loader.levels:\n",
                    "    door_count = len(level.get('doors', []))\n",
                    "    if door_count > 0:\n",
                    "        print(f\"   {level['name'][:25]:25}: {door_count} doors (active level)\")\n",
                    "    else:\n",
                    "        print(f\"   {level['name'][:25]:25}: {door_count} doors (service/structural level)\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📋 Step 7: Export for Further Analysis\n",
                    "\n",
                    "The data can be exported to different formats for use in other tools:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"📋 Data Export Options:\")\n",
                    "print(\"=\" * 40)\n",
                    "\n",
                    "# Convert to pandas DataFrames\n",
                    "try:\n",
                    "    import pandas as pd\n",
                    "    \n",
                    "    dataframes = loader.export_to_dataframes()\n",
                    "    rooms_df = dataframes['rooms']\n",
                    "    doors_df = dataframes['doors']\n",
                    "    walls_df = dataframes['walls']\n",
                    "    \n",
                    "    print(\"✅ Pandas DataFrames created:\")\n",
                    "    print(f\"   rooms_df: {len(rooms_df)} rows × {len(rooms_df.columns)} columns\")\n",
                    "    print(f\"   doors_df: {len(doors_df)} rows × {len(doors_df.columns)} columns\") \n",
                    "    print(f\"   walls_df: {len(walls_df)} rows × {len(walls_df.columns)} columns\")\n",
                    "    \n",
                    "    print(f\"\\n📊 Sample DataFrame (doors):\")\n",
                    "    print(doors_df[['id', 'width_mm', 'height_mm', 'door_type']].head(3))\n",
                    "    \n",
                    "    print(f\"\\n💾 You can save these with:\")\n",
                    "    print(f\"   rooms_df.to_csv('vilamalla_rooms.csv')\")\n",
                    "    print(f\"   doors_df.to_excel('vilamalla_doors.xlsx')\")\n",
                    "    \n",
                    "except ImportError:\n",
                    "    print(\"❌ pandas not available for DataFrame export\")\n",
                    "\n",
                    "# Raw JSON access\n",
                    "print(f\"\\n🔧 Raw data access:\")\n",
                    "print(f\"   loader.all_rooms    # List of room dictionaries\")\n",
                    "print(f\"   loader.all_doors    # List of door dictionaries\")\n",
                    "print(f\"   loader.all_walls    # List of wall dictionaries\")\n",
                    "print(f\"   loader.levels       # List of level dictionaries\")\n",
                    "print(f\"   loader.metadata     # Project metadata\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🎯 Summary: What We Can Extract from IFC Files\n",
                    "\n",
                    "### ✅ Data Successfully Extracted\n",
                    "\n",
                    "From the **6.5 MB Vilamalla IFC file**, we extracted:\n",
                    "\n",
                    "**🏢 Building Structure:**\n",
                    "- 9 levels with elevations and names\n",
                    "- Building hierarchy and organization\n",
                    "\n",
                    "**🏠 Spaces & Rooms:**\n",
                    "- 9 rooms with areas and functions\n",
                    "- Occupancy calculations\n",
                    "- Room-level relationships\n",
                    "\n",
                    "**🚪 Doors:**\n",
                    "- 23 doors with precise dimensions (width/height)\n",
                    "- Door types and classifications\n",
                    "- Exact 3D positions\n",
                    "- Emergency exit identification\n",
                    "\n",
                    "**🧱 Walls:**\n",
                    "- 102 walls with start/end coordinates\n",
                    "- Material information\n",
                    "- Thickness and height data\n",
                    "- Construction properties\n",
                    "\n",
                    "### 🚀 What You Can Do With This Data\n",
                    "\n",
                    "- **🛡️ Compliance Checking**: Verify building codes automatically\n",
                    "- **📊 Analytics**: Calculate areas, volumes, material quantities\n",
                    "- **🗺️ Visualization**: Create floor plans and 3D models\n",
                    "- **🔍 Queries**: Search for specific elements or properties\n",
                    "- **📈 Reporting**: Generate compliance and analysis reports\n",
                    "- **🤖 AI Analysis**: Feed data to AI systems for intelligent insights\n",
                    "\n",
                    "### 💡 Key Insight\n",
                    "\n",
                    "IFC files contain **rich, structured building information** that goes far beyond simple geometry. With proper extraction, you get a complete digital representation of the building suitable for automated analysis, compliance checking, and intelligent building management.\n",
                    "\n",
                    "**Next**: Learn how to use this data for calculations, compliance checking, and AI-powered building analysis! 🤖"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save the simplified notebook
    notebook_path = Path("notebooks/01_data_extraction_simple.ipynb")
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("✅ Notebook simplified successfully!")
    print("📋 New structure:")
    print("   1. Load building data")
    print("   2. Explore levels")  
    print("   3. Examine doors")
    print("   4. Analyze walls")
    print("   5. Check rooms")
    print("   6. Show analysis examples")
    print("   7. Export options")
    print("   8. Summary of what we can extract")

if __name__ == "__main__":
    simplify_notebook()