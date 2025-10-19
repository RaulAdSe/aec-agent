#!/usr/bin/env python3
"""
Explore DWG/DXF files to understand their structure and content.
"""

import ezdxf
from pathlib import Path
import json
from collections import defaultdict

def explore_dwg_file(file_path):
    """Explore the contents of a DWG/DXF file."""
    print(f"\n{'='*60}")
    print(f"Exploring: {file_path.name}")
    print(f"{'='*60}")
    
    try:
        # Try to read the file
        doc = ezdxf.readfile(str(file_path))
        print(f"✅ Successfully loaded file")
        print(f"DXF version: {doc.dxfversion}")
        
    except ezdxf.DXFError as e:
        print(f"❌ Cannot read as DXF directly: {e}")
        print("\nDWG files need to be converted to DXF first.")
        print("Options:")
        print("1. Use AutoCAD to export as DXF")
        print("2. Use ODA File Converter (free)")
        print("3. Use online converters")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    # Get model space
    msp = doc.modelspace()
    
    # Count entity types
    entity_counts = defaultdict(int)
    layer_counts = defaultdict(int)
    block_names = set()
    text_samples = []
    
    print(f"\n📊 Analyzing entities...")
    
    for entity in msp:
        entity_type = entity.dxftype()
        entity_counts[entity_type] += 1
        
        # Track layers
        if hasattr(entity.dxf, 'layer'):
            layer_counts[entity.dxf.layer] += 1
        
        # Collect block names
        if entity_type == 'INSERT':
            block_names.add(entity.dxf.name)
        
        # Collect text samples
        if entity_type == 'TEXT' and len(text_samples) < 10:
            text_samples.append(entity.dxf.text)
        elif entity_type == 'MTEXT' and len(text_samples) < 10:
            text_samples.append(entity.text)
    
    # Print summary
    print(f"\n📈 Entity Types Found:")
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type:20s}: {count:5d}")
    
    print(f"\n📑 Layers Found ({len(layer_counts)}):")
    for layer, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {layer:30s}: {count:5d} entities")
    
    if block_names:
        print(f"\n🔲 Block Names Found ({len(block_names)}):")
        for block in sorted(block_names)[:20]:
            print(f"  - {block}")
    
    if text_samples:
        print(f"\n📝 Text Samples:")
        for text in text_samples:
            print(f"  - {text}")
    
    # Look for specific fire-related entities
    print(f"\n🔥 Fire Safety Related:")
    fire_keywords = ['EXTINTOR', 'BIE', 'ALARMA', 'DETECTOR', 'SPRINKLER', 
                     'EMERGENCIA', 'EVACUACION', 'SALIDA', 'SECTOR', 'PCI']
    
    for layer in layer_counts.keys():
        layer_upper = layer.upper()
        for keyword in fire_keywords:
            if keyword in layer_upper:
                print(f"  Found layer: {layer} (contains '{keyword}')")
                break
    
    for block in block_names:
        block_upper = block.upper()
        for keyword in fire_keywords:
            if keyword in block_upper:
                print(f"  Found block: {block} (contains '{keyword}')")
                break
    
    return {
        'file': file_path.name,
        'dxf_version': doc.dxfversion,
        'entity_counts': dict(entity_counts),
        'layer_count': len(layer_counts),
        'block_count': len(block_names),
        'total_entities': sum(entity_counts.values())
    }

def main():
    """Main function to explore CAD files."""
    cad_dir = Path("data/blueprints/cad")
    
    if not cad_dir.exists():
        print(f"❌ Directory not found: {cad_dir}")
        return
    
    # List available files
    dwg_files = list(cad_dir.glob("*.dwg"))
    dxf_files = list(cad_dir.glob("*.dxf"))
    
    print(f"Found {len(dwg_files)} DWG files and {len(dxf_files)} DXF files")
    
    results = []
    
    # Try to explore each file
    for file_path in dwg_files + dxf_files:
        result = explore_dwg_file(file_path)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully analyzed: {len(results)} files")
        for r in results:
            print(f"\n{r['file']}:")
            print(f"  - Total entities: {r['total_entities']}")
            print(f"  - Layers: {r['layer_count']}")
            print(f"  - Blocks: {r['block_count']}")

if __name__ == "__main__":
    main()