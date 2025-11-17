# Building Analysis Tools

## Overview

The AEC Compliance Agent provides a comprehensive suite of building analysis tools designed for processing IFC (Industry Foundation Classes) files and performing architectural compliance analysis. These tools are deterministic, fast-executing, and provide rich contextual information for AI-powered reasoning.

## Tool Architecture

### Design Principles

1. **Deterministic Operation**: No LLM calls within tools - pure data processing
2. **Fast Execution**: Optimized for sub-second response times
3. **Rich Context**: Detailed building insights for intelligent decision-making
4. **Error Resilience**: Comprehensive validation and error handling
5. **Spatial Awareness**: Advanced geometric and spatial analysis capabilities

### Tool Categories

```
Building Analysis Tools
├── Data Loading & Validation
│   └── load_building_data
├── Element Extraction & Querying  
│   ├── get_all_elements
│   └── query_elements
├── Spatial & Geometric Analysis
│   ├── calculate_distances
│   ├── analyze_spatial_structure
│   └── get_element_properties
├── Compliance Analysis
│   ├── calculate_metrics
│   └── check_relationships
└── Compliance Search
    └── search_compliance_documents
```

## Core Building Tools

### 1. Data Loading & Validation

#### `load_building_data`
**Purpose**: Load and parse IFC building files with comprehensive validation

**Usage**:
```python
result = load_building_data(file_path="data/building.ifc")
```

**Input Parameters**:
- `file_path` (str): Path to IFC file
- `validate_structure` (bool, optional): Perform structural validation (default: True)
- `extract_geometry` (bool, optional): Extract geometric data (default: False)

**Return Structure**:
```json
{
    "status": "success|error",
    "data": {
        "file_info": {
            "file_path": "data/building.ifc",
            "file_size_mb": 12.5,
            "schema_version": "IFC4",
            "timestamp": "2024-01-01T10:00:00Z"
        },
        "building_info": {
            "total_elements": 1026,
            "element_types": {
                "IfcWall": 234,
                "IfcDoor": 45,
                "IfcWindow": 67,
                "IfcSpace": 89
            },
            "spatial_structure": {
                "buildings": 1,
                "storeys": 3,
                "spaces": 89
            }
        },
        "validation": {
            "structure_valid": true,
            "warnings": [],
            "errors": []
        }
    },
    "execution_time_ms": 156.7,
    "context_insights": [
        "3-storey residential building",
        "45 doors distributed across floors",
        "Well-structured spatial hierarchy"
    ]
}
```

**Error Handling**:
- File not found → Clear error message with path validation
- Invalid IFC format → Schema validation errors
- Corrupted data → Partial loading with warnings
- Performance monitoring → Execution time tracking

**Performance**: 
- Small files (< 1MB): 50-150ms
- Medium files (1-10MB): 150-500ms  
- Large files (10-50MB): 500-2000ms

### 2. Element Extraction & Querying

#### `get_all_elements`
**Purpose**: Extract all elements of specific types from loaded building data

**Usage**:
```python
result = get_all_elements(
    element_type="IfcDoor",
    include_properties=True,
    include_relationships=False
)
```

**Input Parameters**:
- `element_type` (str): IFC element type (e.g., "IfcDoor", "IfcWindow")
- `include_properties` (bool): Include element properties (default: True)
- `include_relationships` (bool): Include spatial relationships (default: False)
- `filter_criteria` (dict, optional): Additional filtering criteria

**Return Structure**:
```json
{
    "status": "success",
    "data": {
        "element_count": 45,
        "elements": [
            {
                "id": "door_001",
                "global_id": "3vB$lK8fX3wQYTCaFlI2$6",
                "type": "IfcDoor",
                "name": "Main Entrance Door",
                "properties": {
                    "overall_height": 2100,
                    "overall_width": 900,
                    "fire_rating": "60min",
                    "material": "Steel"
                },
                "location": {
                    "storey": "Ground Floor",
                    "space": "Lobby",
                    "coordinates": [12.5, 8.3, 0.0]
                }
            }
        ],
        "summary": {
            "by_type": {"Fire Door": 12, "Standard Door": 33},
            "by_storey": {"Ground": 18, "First": 15, "Second": 12},
            "total_area": 36.45
        }
    },
    "execution_time_ms": 87.3
}
```

#### `query_elements`
**Purpose**: Advanced element querying with complex filters and spatial queries

**Usage**:
```python
result = query_elements(
    query={
        "element_types": ["IfcDoor", "IfcWindow"],
        "properties": {
            "fire_rating": {"exists": True},
            "overall_width": {"min": 800}
        },
        "spatial": {
            "storey": "Ground Floor",
            "near_element": "emergency_exit_001",
            "distance": {"max": 10.0}
        }
    }
)
```

**Advanced Query Capabilities**:
- **Property Filters**: Range queries, existence checks, pattern matching
- **Spatial Filters**: Distance-based, containment, adjacency
- **Relationship Filters**: Connected elements, hierarchical relationships
- **Complex Logic**: AND/OR combinations, nested conditions

### 3. Spatial & Geometric Analysis

#### `calculate_distances`
**Purpose**: Calculate distances between building elements with pathfinding

**Usage**:
```python
result = calculate_distances(
    from_elements=["door_001", "door_002"],
    to_elements=["exit_001", "exit_002"],
    calculation_type="shortest_path",
    consider_obstacles=True
)
```

**Distance Calculation Types**:
- `euclidean`: Direct 3D distance
- `manhattan`: Grid-based distance
- `shortest_path`: Pathfinding through navigable space
- `accessibility_path`: ADA-compliant routing

**Return Structure**:
```json
{
    "status": "success", 
    "data": {
        "distances": [
            {
                "from_element": "door_001",
                "to_element": "exit_001", 
                "distance": 15.6,
                "path_type": "shortest_path",
                "travel_time_seconds": 18.2,
                "accessibility_compliant": true,
                "path_coordinates": [[x1,y1,z1], [x2,y2,z2], ...]
            }
        ],
        "summary": {
            "average_distance": 18.4,
            "max_distance": 28.7,
            "min_distance": 8.3,
            "accessibility_compliance": 0.89
        }
    },
    "execution_time_ms": 234.1
}
```

**Pathfinding Features**:
- **Obstacle Avoidance**: Navigate around walls and furniture
- **Multi-Level Routing**: Handle stairs and elevators
- **Accessibility Analysis**: Check wheelchair accessibility
- **Emergency Egress**: Specialized emergency exit routing

#### `analyze_spatial_structure` 
**Purpose**: Analyze building spatial hierarchy and relationships

**Usage**:
```python
result = analyze_spatial_structure(
    analysis_type="hierarchy",
    include_adjacencies=True,
    calculate_areas=True
)
```

**Analysis Types**:
- `hierarchy`: Building → Storey → Space → Element structure
- `adjacencies`: Which spaces/elements are adjacent
- `circulation`: Circulation paths and bottlenecks
- `zones`: Fire zones, accessibility zones, etc.

### 4. Property & Relationship Analysis

#### `get_element_properties`
**Purpose**: Extract detailed properties and attributes for specific elements

**Usage**:
```python
result = get_element_properties(
    element_ids=["door_001", "window_003"],
    property_categories=["dimensions", "materials", "performance"],
    include_calculated=True
)
```

**Property Categories**:
- `dimensions`: Width, height, depth, area, volume
- `materials`: Material types, thermal properties, fire ratings
- `performance`: R-values, U-values, acoustic properties
- `relationships`: Connections, containment, spatial relationships
- `metadata`: Creation date, modification history, standards compliance

#### `check_relationships`
**Purpose**: Analyze relationships between building elements

**Usage**:
```python
result = check_relationships(
    element_id="door_001",
    relationship_types=["spatial", "connectivity", "dependency"],
    max_depth=2
)
```

**Relationship Types**:
- `spatial`: Contains, adjacent_to, connects, separates
- `connectivity`: Structural connections, service connections
- `dependency`: Load bearing, support relationships
- `functional`: Access relationships, service relationships

### 5. Compliance Analysis

#### `calculate_metrics`
**Purpose**: Calculate compliance-related metrics and measurements

**Usage**:
```python
result = calculate_metrics(
    metric_types=["egress_analysis", "accessibility_compliance"],
    elements=["all_doors", "all_exits"],
    standards=["IBC_2021", "ADA_2010"]
)
```

**Metric Types**:

**Egress Analysis**:
- Travel distances to exits
- Exit capacity calculations
- Egress width requirements
- Exit sign visibility
- Emergency lighting coverage

**Accessibility Compliance**:
- Door width compliance (≥32" clear)
- Maneuvering clearances
- Accessible route identification
- Ramp slope calculations
- Elevator access analysis

**Fire Safety**:
- Fire exit distances
- Fire door ratings and locations
- Smoke compartmentalization
- Fire alarm coverage
- Sprinkler system coverage

**Return Structure**:
```json
{
    "status": "success",
    "data": {
        "egress_analysis": {
            "total_exits": 8,
            "average_travel_distance": 45.2,
            "max_travel_distance": 67.8,
            "exit_capacity": {
                "required": 340,
                "provided": 420,
                "compliance": "pass"
            },
            "violations": [
                {
                    "type": "max_travel_distance_exceeded",
                    "location": "Second Floor East Wing",
                    "actual": 67.8,
                    "required": 60.0,
                    "severity": "high"
                }
            ]
        },
        "accessibility_compliance": {
            "doors_analyzed": 45,
            "compliant_doors": 42,
            "compliance_rate": 0.933,
            "violations": [
                {
                    "door_id": "door_023",
                    "issue": "insufficient_maneuvering_clearance",
                    "required": 18,
                    "actual": 14,
                    "location": "First Floor Restroom"
                }
            ]
        }
    },
    "execution_time_ms": 445.7
}
```

## Compliance Search Tool

### `search_compliance_documents`
**Purpose**: Search building codes and compliance documents for relevant regulations

**Usage**:
```python
result = search_compliance_documents(
    query="fire door requirements commercial buildings",
    standards=["IBC_2021", "NFPA_101"],
    context="fire_safety_analysis"
)
```

**Supported Standards**:
- **International Building Code (IBC)**: 2018, 2021 editions
- **NFPA Life Safety Code**: NFPA 101
- **Americans with Disabilities Act (ADA)**: 2010 Standards
- **International Fire Code (IFC)**: 2018, 2021 editions
- **Local Codes**: City/state specific amendments

**Return Structure**:
```json
{
    "status": "success",
    "data": {
        "total_results": 12,
        "results": [
            {
                "standard": "IBC_2021",
                "section": "716.2.6.1", 
                "title": "Fire Door Requirements",
                "content": "Fire doors shall be tested in accordance with NFPA 252 and meet the fire protection rating required for the opening in which they are installed...",
                "relevance_score": 0.94,
                "related_sections": ["716.2.6.2", "716.2.6.3"],
                "keywords": ["fire doors", "commercial", "ratings"]
            }
        ],
        "summary": "Found comprehensive fire door requirements across IBC and NFPA standards covering testing, installation, and maintenance requirements.",
        "related_queries": [
            "fire door testing requirements",
            "commercial building fire ratings",
            "fire door maintenance standards"
        ]
    },
    "execution_time_ms": 234.5
}
```

## Tool Integration & Context

### Context-Aware Tool Selection

The reasoning agent uses context from previous tool executions to make intelligent tool selections:

**Example Context Flow**:
```
1. load_building_data → Building context: "3-story office building"
2. get_all_elements(IfcDoor) → Door context: "45 doors, 12 fire-rated"  
3. calculate_distances → Travel context: "Max egress distance 67.8m"
4. search_compliance_documents → Code context: "IBC requires ≤60m travel distance"
5. calculate_metrics → Violation context: "3 spaces exceed travel distance limits"
```

### Tool Chaining Patterns

#### Fire Safety Analysis Pattern
```
load_building_data
    ↓
get_all_elements(fire_doors + fire_exits)
    ↓
calculate_distances(egress_paths)
    ↓  
search_compliance_documents(fire_code_requirements)
    ↓
calculate_metrics(fire_safety_compliance)
```

#### Accessibility Analysis Pattern
```
load_building_data
    ↓
get_all_elements(doors + ramps + elevators)
    ↓
get_element_properties(accessibility_features)
    ↓
check_relationships(accessible_routes)
    ↓
calculate_metrics(ADA_compliance)
```

#### Door Inventory Pattern
```
load_building_data
    ↓
get_all_elements(doors)
    ↓
get_element_properties(door_specifications)
    ↓
analyze_spatial_structure(door_locations)
    ↓
calculate_metrics(door_summary_statistics)
```

## Performance Characteristics

### Execution Timing

| Tool | Small Building | Medium Building | Large Building |
|------|---------------|-----------------|----------------|
| load_building_data | 50-150ms | 150-500ms | 500-2000ms |
| get_all_elements | 20-80ms | 80-200ms | 200-500ms |
| query_elements | 50-120ms | 120-300ms | 300-800ms |
| calculate_distances | 100-300ms | 300-800ms | 800-2000ms |
| calculate_metrics | 200-500ms | 500-1200ms | 1200-3000ms |
| search_compliance_documents | 100-300ms | 100-300ms | 100-300ms |

### Memory Usage

- **Building Data**: 5-50MB per loaded building
- **Element Cache**: 1-10MB per element type
- **Spatial Index**: 2-20MB for pathfinding
- **Total Per Session**: 10-100MB typical

### Optimization Features

#### Caching System
```python
# Element data caching
@cache_result(ttl=3600)  # 1 hour cache
def get_all_elements(element_type):
    # Expensive IFC parsing cached

# Spatial index caching  
@cache_result(ttl=1800)  # 30 minute cache
def calculate_distances():
    # Pathfinding index cached
```

#### Parallel Processing
```python
# Parallel element processing
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(process_element, element) 
        for element in elements
    ]
    results = [f.result() for f in futures]
```

#### Incremental Loading
```python
# Load building data incrementally
loader = IncrementalIFCLoader(file_path)
loader.load_spatial_structure()  # Fast initial load
loader.load_elements_on_demand(element_types)  # Load as needed
```

## Error Handling & Resilience

### Comprehensive Error Recovery

```python
def load_building_data(file_path: str) -> Dict[str, Any]:
    """Load building data with comprehensive error handling."""
    try:
        # Validate file existence and format
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error_type": "file_not_found",
                "message": f"File not found: {file_path}",
                "suggestions": [
                    "Check file path spelling",
                    "Verify file permissions",
                    "Ensure file is in expected location"
                ]
            }
        
        # Attempt IFC parsing with fallback strategies
        try:
            data = parse_ifc_file(file_path)
        except IFCSchemaError as e:
            # Try with schema tolerance
            data = parse_ifc_file(file_path, strict=False)
            warnings.append(f"Schema warning: {e}")
        except IFCCorruptionError as e:
            # Partial loading for corrupted files
            data = parse_ifc_partial(file_path)
            warnings.append(f"Partial loading due to corruption: {e}")
        
        return {
            "status": "success",
            "data": data,
            "warnings": warnings,
            "execution_time_ms": execution_time
        }
        
    except Exception as e:
        # Comprehensive error reporting
        return {
            "status": "error", 
            "error_type": type(e).__name__,
            "message": str(e),
            "file_info": get_file_debug_info(file_path),
            "troubleshooting": get_error_troubleshooting(e)
        }
```

### Error Categories

#### File & Format Errors
- **File Not Found**: Clear path validation and suggestions
- **Invalid IFC Format**: Schema validation with helpful error messages
- **Corruption**: Partial loading with warnings about missing data
- **Permission Denied**: File access troubleshooting

#### Data Processing Errors  
- **Element Not Found**: Alternative element suggestions
- **Invalid Query**: Query syntax help and examples
- **Geometric Errors**: Fallback to simplified calculations
- **Memory Limits**: Streaming processing for large datasets

#### Performance Issues
- **Timeout Handling**: Graceful degradation for complex calculations
- **Resource Limits**: Memory usage monitoring and cleanup
- **API Limits**: Rate limiting and retry strategies

## Tool Configuration

### Building Tool Settings

```python
@dataclass
class BuildingToolConfig:
    # Performance settings
    tool_timeout: float = 30.0              # Individual tool timeout
    max_elements_per_query: int = 10000     # Query size limits
    enable_caching: bool = True             # Result caching
    cache_ttl_seconds: int = 3600           # Cache time-to-live
    
    # IFC processing settings
    ifc_parsing_timeout: float = 60.0       # IFC file parsing timeout
    ifc_strict_mode: bool = False           # Strict schema validation
    enable_geometry_extraction: bool = True # Extract geometric data
    spatial_index_precision: float = 0.1   # Spatial indexing precision
    
    # Pathfinding settings
    pathfinding_algorithm: str = "A*"       # Pathfinding algorithm
    pathfinding_timeout: float = 10.0      # Pathfinding timeout
    consider_accessibility: bool = True     # Include accessibility constraints
    emergency_egress_mode: bool = False     # Emergency egress routing
    
    # Compliance settings
    default_building_codes: List[str] = ["IBC_2021", "ADA_2010"]
    compliance_tolerance: float = 0.05      # 5% tolerance for measurements
    enable_violation_details: bool = True   # Detailed violation reporting
```

### Environment Configuration

```bash
# Tool-specific environment variables
export AEC_TOOLS_TIMEOUT="45.0"
export AEC_TOOLS_ENABLE_CACHING="true" 
export AEC_TOOLS_MAX_ELEMENTS="15000"
export AEC_TOOLS_IFC_STRICT_MODE="false"
export AEC_TOOLS_PATHFINDING_ALGORITHM="A*"
export AEC_TOOLS_DEFAULT_CODES="IBC_2021,ADA_2010,NFPA_101"
```

## Use Cases & Examples

### Complete Fire Safety Analysis

```python
# 1. Load building
building_result = load_building_data("office_building.ifc")

# 2. Get fire safety elements  
doors_result = get_all_elements("IfcDoor", include_properties=True)
exits_result = get_all_elements("IfcDoor", filter_criteria={"fire_exit": True})

# 3. Calculate egress distances
distances_result = calculate_distances(
    from_elements="all_spaces",
    to_elements="fire_exits", 
    calculation_type="emergency_egress"
)

# 4. Check compliance codes
codes_result = search_compliance_documents(
    "fire egress distance requirements commercial",
    standards=["IBC_2021", "NFPA_101"]
)

# 5. Generate compliance metrics
metrics_result = calculate_metrics(
    metric_types=["fire_safety_compliance"],
    standards=["IBC_2021"]
)
```

### Accessibility Compliance Audit

```python
# 1. Load building
building_result = load_building_data("public_library.ifc")

# 2. Get accessibility elements
doors_result = get_all_elements("IfcDoor", include_properties=True)
ramps_result = get_all_elements("IfcRamp", include_properties=True)
elevators_result = get_all_elements("IfcTransportElement")

# 3. Analyze door compliance
door_metrics = calculate_metrics(
    metric_types=["door_accessibility"],
    elements=doors_result["data"]["elements"],
    standards=["ADA_2010"]
)

# 4. Check accessible routes
routes_result = calculate_distances(
    from_elements="main_entrance",
    to_elements="all_public_spaces",
    calculation_type="accessibility_path"
)

# 5. Generate compliance report
compliance_result = calculate_metrics(
    metric_types=["full_accessibility_audit"],
    standards=["ADA_2010"]
)
```

## Integration with Reasoning Agent

### Tool Selection Intelligence

The reasoning agent uses sophisticated logic to select appropriate tools:

```python
# ToolPlanner component analyzes context and selects tools
class ToolPlanner:
    def select_tool(self, task: str, context: Dict[str, Any]) -> str:
        """Select optimal tool based on task and execution context."""
        
        # Analyze task requirements
        if "load" in task.lower() and "building" in task.lower():
            return "load_building_data"
        
        # Check if building data is already loaded
        if not context.get("building_loaded"):
            return "load_building_data"  # Always load first
        
        # Element extraction logic
        if any(element_type in task.lower() for element_type in ["door", "window", "wall"]):
            return "get_all_elements"
        
        # Spatial analysis logic
        if any(keyword in task.lower() for keyword in ["distance", "path", "route"]):
            return "calculate_distances"
        
        # Compliance analysis logic
        if any(keyword in task.lower() for keyword in ["compliance", "code", "regulation"]):
            return "search_compliance_documents"
        
        # Default to metric calculation for analysis tasks
        return "calculate_metrics"
```

### Context Accumulation

Tools provide rich context that influences subsequent tool selections:

```python
# Context builds across tool executions
execution_context = {
    "building_info": {
        "type": "office_building",
        "floors": 3,
        "total_elements": 1026
    },
    "elements_discovered": {
        "doors": 45,
        "fire_doors": 12, 
        "exits": 8
    },
    "spatial_insights": {
        "max_travel_distance": 67.8,
        "accessibility_compliance": 0.89
    },
    "compliance_requirements": {
        "max_travel_distance_ibc": 60.0,
        "min_door_width_ada": 32.0
    }
}
```

## Future Enhancements

### Planned Features

1. **Advanced Spatial Analysis**
   - 3D visibility analysis for signage placement
   - Crowd simulation for egress planning
   - Acoustic analysis for sound transmission
   - Daylight and artificial lighting analysis

2. **Machine Learning Integration**
   - Predictive compliance scoring
   - Automated violation detection
   - Design optimization suggestions
   - Pattern recognition across building types

3. **Enhanced Visualization**
   - Interactive 3D building models
   - Compliance heat maps
   - Path visualization for egress routes
   - Real-time violation highlighting

4. **Additional Standards Support**
   - International codes (Eurocode, etc.)
   - Specialized standards (healthcare, education)
   - Historical code versions
   - Local jurisdiction amendments

### Research Directions

1. **AI-Enhanced Analysis**
   - Natural language query interface
   - Automated code interpretation
   - Design recommendation engine
   - Risk assessment modeling

2. **Real-Time Integration**
   - BIM software plugins
   - Live design validation
   - Change impact analysis
   - Collaborative review workflows

## Summary

The building analysis tools provide:

✅ **Comprehensive Coverage**: Full building analysis from data loading to compliance reporting  
✅ **Deterministic Performance**: Fast, reliable execution without LLM dependencies  
✅ **Rich Context Generation**: Detailed insights that enhance AI reasoning quality  
✅ **Advanced Spatial Analysis**: Sophisticated pathfinding and geometric calculations  
✅ **Standards Integration**: Support for major building codes and regulations  
✅ **Error Resilience**: Robust error handling with graceful degradation  
✅ **Performance Optimization**: Caching, parallel processing, and incremental loading  

These tools enable the AEC Compliance Agent to perform sophisticated building analysis while maintaining the fast, deterministic execution required for effective AI-powered reasoning.