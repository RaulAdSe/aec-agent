#!/usr/bin/env python3
"""
Add a summary cell to the notebook
"""

import json
from pathlib import Path

def add_summary_cell():
    """Add a final summary cell to the notebook."""
    
    notebook_path = Path("notebooks/01_data_extraction_simple.ipynb")
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Summary cell
    summary_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎯 Tutorial Summary\\n",
            "\\n",
            "**Congratulations!** You've successfully completed the IFC Data Extraction tutorial using real building data from the Vilamalla Industrial Complex.\\n",
            "\\n",
            "### 🏆 What You Accomplished\\n",
            "\\n",
            "1. ✅ **Real IFC Data Extraction** - Loaded and analyzed actual building data (6.5 MB IFC file)\\n",
            "2. ✅ **Multi-Level Analysis** - Explored 9 building levels with different functions\\n", 
            "3. ✅ **Compliance Checking** - Verified all 23 doors meet width requirements\\n",
            "4. ✅ **Material Analysis** - Analyzed 102 walls and their construction materials\\n",
            "5. ✅ **Advanced Visualizations** - Created interactive charts and building layouts\\n",
            "6. ✅ **Pandas Integration** - Exported data for advanced statistical analysis\\n",
            "\\n",
            "### 📊 Key Findings from Vilamalla Complex\\n",
            "\\n",
            "- 🏢 **Building Scale**: 9 levels, 720 m² total area\\n",
            "- 🚪 **Door Compliance**: 100% compliance rate (all doors ≥ 700mm width)\\n",
            "- 🧱 **Construction**: Primarily concrete walls, standard 2.7m height\\n",
            "- 📍 **Main Activity Level**: MUELLE level with 21 doors (industrial operations)\\n",
            "- 🎯 **Data Quality**: Complete geometric and semantic information extracted\\n",
            "\\n",
            "### 🔍 Technical Skills Learned\\n",
            "\\n",
            "- **IFC File Processing**: Understanding Industry Foundation Classes format\\n",
            "- **Building Data Modeling**: Working with hierarchical building information\\n",
            "- **Compliance Analysis**: Automated checking against building codes\\n",
            "- **Data Visualization**: Creating meaningful charts and floor plans\\n",
            "- **Python Libraries**: pandas, matplotlib, shapely for AEC analysis\\n",
            "\\n",
            "### 🚀 Next Steps\\n",
            "\\n",
            "Ready to continue your journey? Check out:\\n",
            "\\n",
            "- **📐 Tutorial 2**: Geometric calculations and route analysis\\n",
            "- **🔍 Tutorial 3**: RAG system for querying building codes\\n",
            "- **🤖 Tutorial 4**: AI agents for autonomous compliance verification\\n",
            "\\n",
            "### 💡 Try It Yourself\\n",
            "\\n",
            "Experiment with the code above:\\n",
            "\\n",
            "```python\\n",
            "# Filter specific elements\\n",
            "emergency_doors = doors_df[doors_df['is_emergency_exit'] == True]\\n",
            "\\n",
            "# Analyze specific levels\\n",
            "muelle_level = loader.get_level_data('MUELLE')\\n",
            "\\n",
            "# Custom visualizations\\n",
            "loader.visualize_level('PB')  # Visualize different level\\n",
            "\\n",
            "# Export for further analysis\\n",
            "rooms_df.to_csv('vilamalla_rooms.csv')\\n",
            "```\\n",
            "\\n",
            "The real building data opens endless possibilities for analysis and innovation in AEC technology! 🏗️✨"
        ]
    }
    
    # Add summary cell at the end
    notebook['cells'].append(summary_cell)
    
    # Save the updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("✅ Summary cell added to notebook!")

if __name__ == "__main__":
    add_summary_cell()