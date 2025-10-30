// QCAD Script for DWG to DXF conversion
// This script runs inside QCAD and provides enhanced conversion capabilities

include("scripts/Pro/File/OpenFile/OpenFile.js");
include("scripts/Pro/File/SaveAs/SaveAs.js");

function convertDwgToDxf(inputFile, outputFile, options) {
    options = options || {};
    
    print("Starting DWG to DXF conversion...");
    print("Input file: " + inputFile);
    print("Output file: " + outputFile);
    
    try {
        // Create new document
        var di = new RDocumentInterface();
        var doc = new RDocument(di);
        
        // Set up document interface
        di.setDocument(doc);
        
        // Import the DWG file
        print("Importing DWG file...");
        var importResult = di.importFile(inputFile);
        
        if (!importResult) {
            throw new Error("Failed to import DWG file: " + inputFile);
        }
        
        print("DWG file imported successfully");
        
        // Get model space for analysis
        var msp = doc.getModelSpace();
        var entities = msp.queryAllEntities();
        print("Found " + entities.length + " entities in the drawing");
        
        // Analyze entity types
        var entityTypes = {};
        for (var i = 0; i < entities.length; i++) {
            var entity = entities[i];
            var type = entity.getType();
            entityTypes[type] = (entityTypes[type] || 0) + 1;
        }
        
        print("Entity types found:");
        for (var type in entityTypes) {
            print("  " + type + ": " + entityTypes[type]);
        }
        
        // Clean up drawing if requested
        if (options.cleanup) {
            print("Cleaning up drawing...");
            cleanupDrawing(doc);
        }
        
        // Save as DXF
        print("Saving as DXF...");
        var saveResult = di.exportFile(outputFile, "DXF 2018");
        
        if (!saveResult) {
            throw new Error("Failed to save DXF file: " + outputFile);
        }
        
        print("Conversion completed successfully!");
        print("Converted: " + inputFile + " -> " + outputFile);
        
        return true;
        
    } catch (error) {
        print("Error during conversion: " + error.message);
        return false;
    }
}

function cleanupDrawing(doc) {
    // Remove duplicate entities
    var msp = doc.getModelSpace();
    var entities = msp.queryAllEntities();
    
    // Group entities by type and position to find duplicates
    var entityGroups = {};
    
    for (var i = 0; i < entities.length; i++) {
        var entity = entities[i];
        var type = entity.getType();
        var key = type + "_" + getEntityKey(entity);
        
        if (!entityGroups[key]) {
            entityGroups[key] = [];
        }
        entityGroups[key].push(entity);
    }
    
    // Remove duplicates (keep first, remove rest)
    var removedCount = 0;
    for (var key in entityGroups) {
        var group = entityGroups[key];
        if (group.length > 1) {
            for (var i = 1; i < group.length; i++) {
                msp.removeEntity(group[i]);
                removedCount++;
            }
        }
    }
    
    if (removedCount > 0) {
        print("Removed " + removedCount + " duplicate entities");
    }
}

function getEntityKey(entity) {
    // Create a key for entity comparison
    var type = entity.getType();
    
    if (type === "Line") {
        var start = entity.getStartPoint();
        var end = entity.getEndPoint();
        return start.x + "," + start.y + "," + end.x + "," + end.y;
    } else if (type === "Circle") {
        var center = entity.getCenter();
        var radius = entity.getRadius();
        return center.x + "," + center.y + "," + radius;
    } else if (type === "Arc") {
        var center = entity.getCenter();
        var radius = entity.getRadius();
        var startAngle = entity.getStartAngle();
        var endAngle = entity.getEndAngle();
        return center.x + "," + center.y + "," + radius + "," + startAngle + "," + endAngle;
    } else if (type === "Insert") {
        var pos = entity.getPosition();
        var name = entity.getName();
        return pos.x + "," + pos.y + "," + name;
    }
    
    // Fallback for other entity types
    return type + "_" + entity.getId();
}

function analyzeDrawing(inputFile) {
    // Analyze drawing without converting
    print("Analyzing drawing: " + inputFile);
    
    try {
        var di = new RDocumentInterface();
        var doc = new RDocument(di);
        di.setDocument(doc);
        
        var importResult = di.importFile(inputFile);
        if (!importResult) {
            throw new Error("Failed to import file: " + inputFile);
        }
        
        var msp = doc.getModelSpace();
        var entities = msp.queryAllEntities();
        
        print("Drawing Analysis:");
        print("  Total entities: " + entities.length);
        
        // Count by type
        var entityTypes = {};
        var layers = {};
        
        for (var i = 0; i < entities.length; i++) {
            var entity = entities[i];
            var type = entity.getType();
            var layer = entity.getLayerName();
            
            entityTypes[type] = (entityTypes[type] || 0) + 1;
            layers[layer] = (layers[layer] || 0) + 1;
        }
        
        print("  Entity types:");
        for (var type in entityTypes) {
            print("    " + type + ": " + entityTypes[type]);
        }
        
        print("  Layers:");
        for (var layer in layers) {
            print("    " + layer + ": " + layers[layer]);
        }
        
        // Check for specific building elements
        checkBuildingElements(msp);
        
        return true;
        
    } catch (error) {
        print("Error analyzing drawing: " + error.message);
        return false;
    }
}

function checkBuildingElements(msp) {
    print("  Building elements found:");
    
    // Check for doors
    var doorBlocks = msp.queryBlockReferences();
    var doorCount = 0;
    for (var i = 0; i < doorBlocks.length; i++) {
        var block = doorBlocks[i];
        var name = block.getName().toUpperCase();
        if (name.indexOf("DOOR") >= 0 || name.indexOf("PUERTA") >= 0) {
            doorCount++;
        }
    }
    print("    Doors: " + doorCount);
    
    // Check for walls
    var lines = msp.queryAllEntities().filter(function(e) { return e.getType() === "Line"; });
    var wallCount = 0;
    for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        var layer = line.getLayerName().toUpperCase();
        if (layer.indexOf("WALL") >= 0 || layer.indexOf("MURO") >= 0) {
            wallCount++;
        }
    }
    print("    Wall lines: " + wallCount);
    
    // Check for rooms
    var polylines = msp.queryAllEntities().filter(function(e) { 
        return e.getType() === "Polyline" && e.isClosed(); 
    });
    print("    Closed polylines (potential rooms): " + polylines.length);
    
    // Check for fire equipment
    var fireEquipment = 0;
    for (var i = 0; i < doorBlocks.length; i++) {
        var block = doorBlocks[i];
        var name = block.getName().toUpperCase();
        if (name.indexOf("EXTINTOR") >= 0 || name.indexOf("BIE") >= 0 || 
            name.indexOf("SPRINKLER") >= 0 || name.indexOf("ALARMA") >= 0) {
            fireEquipment++;
        }
    }
    print("    Fire equipment: " + fireEquipment);
}

// Main execution
var args = QCoreApplication.arguments();

if (args.length >= 3) {
    var command = args[1];
    var inputFile = args[2];
    
    if (command === "convert") {
        var outputFile = args[3] || inputFile.replace(/\.dwg$/i, ".dxf");
        var options = {
            cleanup: args.indexOf("--cleanup") >= 0
        };
        convertDwgToDxf(inputFile, outputFile, options);
    } else if (command === "analyze") {
        analyzeDrawing(inputFile);
    } else {
        // Legacy mode: direct conversion
        var outputFile = args[2];
        convertDwgToDxf(inputFile, outputFile);
    }
    
    // Exit QCAD
    QCoreApplication.quit();
} else {
    print("Usage:");
    print("  qcad -exec script.js convert inputFile.dwg [outputFile.dxf] [--cleanup]");
    print("  qcad -exec script.js analyze inputFile.dwg");
    print("  qcad -exec script.js inputFile.dwg outputFile.dxf  (legacy mode)");
}