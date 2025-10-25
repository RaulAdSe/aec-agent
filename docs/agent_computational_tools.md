# Agent Computational Tools Documentation

This document describes the **practical computational tools** available to the AEC Compliance Agent for autonomous building analysis.

## Overview

The agent has access to a comprehensive suite of tools for analyzing building data extracted from IFC files. These tools are designed to be:
- **Autonomous**: Can be called without human intervention
- **Practical**: Focus on real-world compliance scenarios  
- **Reliable**: Work with actual building data
- **Compliance-Focused**: Built-in regulatory requirements

## Core Analysis Tools

### 1. Building Connectivity Analysis

**Tool**: `analyze_building_circulation()`

**Purpose**: Analyze overall building circulation and identify potential issues.

**Returns**:
- Critical circulation points
- Evacuation compliance summary
- Room adjacency mapping
- Overall circulation health assessment

**Usage Example**:
```python
result = analyze_building_circulation()
print(f"Critical rooms: {result['critical_circulation_points']['critical_room_count']}")
print(f"Compliance rate: {result['evacuation_compliance']['compliance_rate']:.1%}")
```

**Agent Applications**:
- Identify circulation bottlenecks
- Assess building connectivity risks
- Generate circulation improvement recommendations

---

### 2. Evacuation Route Analysis

**Tool**: `find_all_evacuation_routes_tool()`

**Purpose**: Find evacuation routes for all rooms in the building.

**Returns**:
- Complete evacuation analysis for all rooms
- Longest and shortest routes
- Average evacuation distances
- Compliance summary statistics

**Usage Example**:
```python
routes = find_all_evacuation_routes_tool()
longest = routes['longest_route']
print(f"Longest route: {longest['distance']:.1f}m from {longest['room_id']}")
```

**Agent Applications**:
- Validate evacuation compliance
- Identify rooms with evacuation issues
- Plan emergency response strategies

---

### 3. Room Connectivity Assessment

**Tool**: `check_room_connectivity(room_id)`

**Purpose**: Check how well-connected a specific room is to the rest of the building.

**Parameters**:
- `room_id`: The ID of the room to analyze

**Returns**:
- Direct connection count
- Reachability score
- Average distance to other rooms
- Connectivity grade (low/medium/high)

**Usage Example**:
```python
connectivity = check_room_connectivity("R001")
print(f"Connectivity grade: {connectivity['connectivity_grade']}")
print(f"Direct connections: {connectivity['direct_connections']}")
```

**Agent Applications**:
- Assess room accessibility
- Identify isolated areas
- Plan circulation improvements

---

### 4. Occupancy Load Calculations

**Tool**: `calculate_occupancy_load(room_id)`

**Purpose**: Calculate the occupancy load and egress capacity for a room.

**Parameters**:
- `room_id`: The ID of the room

**Returns**:
- Calculated occupancy based on use type
- Required egress width
- Available egress width
- Egress adequacy assessment

**Usage Example**:
```python
occupancy = calculate_occupancy_load("R001")
print(f"Occupancy: {occupancy['calculated_occupancy']} people")
print(f"Egress adequate: {occupancy['egress_adequate']}")
```

**Agent Applications**:
- Verify occupancy limits
- Check egress capacity compliance
- Calculate building capacity

---

### 5. Comprehensive Door Analysis

**Tool**: `analyze_door_compliance_comprehensive()`

**Purpose**: Comprehensive analysis of all doors for various compliance requirements.

**Returns**:
- Width compliance statistics
- Emergency exit compliance
- Accessibility compliance
- Overall compliance rate

**Usage Example**:
```python
doors = analyze_door_compliance_comprehensive()
print(f"Overall compliance: {doors['overall_compliance_rate']:.1%}")
print(f"Non-compliant doors: {len(doors['width_compliance']['non_compliant'])}")
```

**Agent Applications**:
- Audit door compliance
- Identify non-compliant doors
- Generate compliance reports

---

### 6. Egress Distance Calculations

**Tool**: `calculate_egress_distance(room_id)`

**Purpose**: Calculate evacuation distance from a room to the nearest exit.

**Parameters**:
- `room_id`: The ID of the room

**Returns**:
- Distance to nearest exit
- Evacuation path
- Accessibility assessment
- Compliance status

**Usage Example**:
```python
egress = calculate_egress_distance("R001")
print(f"Distance to exit: {egress['distance']:.1f}m")
print(f"Path: {' → '.join(egress['path'])}")
```

**Agent Applications**:
- Validate travel distances
- Check evacuation accessibility
- Identify compliance issues

---

## Advanced Analysis Capabilities

### Spatial Relationship Analysis

The tools can analyze complex spatial relationships including:

- **Room Adjacency**: Which rooms are physically connected
- **Circulation Patterns**: How people move through the building
- **Critical Paths**: Essential routes for evacuation
- **Bottleneck Identification**: Points of circulation congestion

### Compliance Validation

Built-in compliance checking against common regulations:

- **Evacuation Distances**: 25m protected, 35m unprotected routes
- **Door Widths**: 800mm minimum, 900mm for emergency exits
- **Accessibility**: Clear width and accessible route requirements
- **Occupancy Loads**: Based on room use and area

### Performance Metrics

Comprehensive building performance assessment:

- **Safety Grade**: A-D rating based on evacuation compliance
- **Circulation Grade**: A-C rating based on connectivity
- **Door Grade**: A-D rating based on door compliance
- **Overall Score**: 0-100 building performance score

## Integration with Agent System

### Tool Discovery

All tools are automatically registered with the agent system and can be discovered via:

```python
tools = get_available_tools()
for tool in tools:
    print(f"{tool['name']}: {tool['description']}")
```

### Error Handling

All tools include robust error handling and return structured error information:

```python
result = some_tool()
if 'error' in result:
    print(f"Tool failed: {result['error']}")
else:
    # Process successful result
    pass
```

### Data Requirements

Tools require building data to be loaded first:

```python
# Load project data first
load_project_data("path/to/building/data.json")

# Then use analysis tools
result = analyze_building_circulation()
```

## Usage Patterns for Autonomous Operation

### 1. Building Assessment Workflow

```python
# 1. Load and validate data
load_project_data()

# 2. Assess overall circulation
circulation = analyze_building_circulation()

# 3. Check evacuation compliance
evacuation = find_all_evacuation_routes_tool()

# 4. Analyze door compliance
doors = analyze_door_compliance_comprehensive()

# 5. Generate comprehensive report
# Combine results for final assessment
```

### 2. Room-Specific Analysis

```python
# For each room of interest
for room_id in important_rooms:
    connectivity = check_room_connectivity(room_id)
    occupancy = calculate_occupancy_load(room_id)
    egress = calculate_egress_distance(room_id)
    
    # Assess room performance
    # Generate recommendations
```

### 3. Compliance Verification

```python
# Check all compliance aspects
door_compliance = analyze_door_compliance_comprehensive()
evacuation_compliance = find_all_evacuation_routes_tool()
circulation_health = analyze_building_circulation()

# Generate compliance report
overall_compliance = calculate_overall_score(
    door_compliance, evacuation_compliance, circulation_health
)
```

## Best Practices for Agent Use

### 1. Always Validate Data First
- Check that building data is loaded
- Verify data completeness
- Handle missing or incomplete data gracefully

### 2. Use Comprehensive Analysis
- Don't rely on single metrics
- Combine multiple tool outputs
- Cross-validate results

### 3. Provide Context in Responses
- Explain what the metrics mean
- Reference regulatory requirements
- Suggest specific improvements

### 4. Handle Edge Cases
- Empty buildings or levels
- Disconnected building sections
- Missing door or room connections

## Example Agent Response Pattern

```python
def analyze_building_safety(building_id):
    \"\"\"Example of how agent might use tools comprehensively.\"\"\"
    
    # Load building data
    if not load_project_data(building_id):
        return "Unable to load building data"
    
    # Comprehensive analysis
    circulation = analyze_building_circulation()
    evacuation = find_all_evacuation_routes_tool()
    doors = analyze_door_compliance_comprehensive()
    
    # Generate structured response
    response = f\"\"\"
    Building Safety Analysis for {building_id}:
    
    🔗 Circulation: {circulation['analysis_summary']['overall_circulation_health']}
    - Critical rooms: {circulation['critical_circulation_points']['critical_room_count']}
    
    🚨 Evacuation: {evacuation['compliance_summary']['compliance_rate']:.1%} compliant
    - Average distance: {evacuation['average_distance']:.1f}m
    
    🚪 Doors: {doors['overall_compliance_rate']:.1%} compliant
    - Non-compliant: {len(doors['width_compliance']['non_compliant'])} doors
    
    Overall Status: {"GOOD" if all_metrics_good else "NEEDS ATTENTION"}
    \"\"\"
    
    return response
```

This comprehensive tool suite enables the agent to perform sophisticated building analysis autonomously, providing detailed insights and recommendations for compliance verification and building performance optimization.