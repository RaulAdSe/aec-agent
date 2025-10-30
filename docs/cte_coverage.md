# CTE Coverage vs Toolkit Capabilities

This document maps the current spatial reasoning toolkit to CTE sections, highlighting implemented checks, missing tools, and required data to reach full automation.

## Visual Coverage Overview

| CTE Section | Coverage   | Tools                         | Missing                   |
| ----------- | ---------- | ----------------------------- | ------------------------- |
| DB-SI 3     | ✅ Full     | Graph & pathfinding           | —                         |
| DB-SUA 9    | 🟨 Partial | Clearance, nearest door       | Door width (agent tool OK) |
| DB-SI 1     | 🟨 Partial | Clearance                      | Wall continuity           |
| DB-SI 6     | ❌ Limited  | Basic connectivity             | Structural info           |

## Detailed Coverage With Data Requirements

| CTE Section | Coverage | Available Tools | Missing Tools | Data Needed |
| ----------- | -------- | --------------- | ------------- | ----------- |
| DB-SI 1 (Propagación interior) | 60% | `calculate_clearance_between_elements`, `find_nearest_door` | `compute_wall_connectivity`, `check_wall_continuity` | Wall start/end, thickness, fire rating (EI), layer/material |
| DB-SI 3 (Evacuación de ocupantes) | 100% | `create_circulation_graph` (pathfinding, centrality, critical points), `find_nearest_door` | — | Exit flags on doors; optional door types and travel weights |
| DB-SI 6 (Resistencia estructural al fuego) | 40% | `calculate_angle_between_walls`, `calculate_clearance_between_elements` | `identify_structural_walls` | Load-bearing flag, wall material, thickness/height |
| DB-SUA 9 (Accesibilidad) | 80% | `find_nearest_door`, `calculate_clearance_between_elements`, graph pathfinding (filterable) | Accessible-path filters; door width in primitives (agent tool exists) | Door clear width, slope/step constraints for accessible edges |

Notes:
- Door width compliance exists in the agent layer as `check_door_width_compliance(door_id)` and can be used today.
- For accessible routing, we need edge attributes (e.g., slope ≤ 6%, no steps) to filter graph paths.

## Naming and Design Tweaks

- Rename missing continuity primitive to emphasize graph alignment:
  - `compute_wall_connectivity(walls, tolerance=0.05) -> nx.Graph`
  - Build on the same mental model as `create_circulation_graph`.
- Keep higher-level rule helpers thin and declarative:
  - `check_wall_continuity(wall_a, wall_b, connectivity_graph)` uses the graph to assert continuity within tolerance and fire rating continuity.

## Compliance Brain (Rule Layer)

Next step is a light-weight `rules/` layer that consumes geometry/topology outputs and returns structured compliance results.

- Input: geometry primitives + graphs + metadata
- Process: apply thresholds and rule logic (per CTE section)
- Output: JSON payloads suitable for reporting/agents

Example rule stubs:

```python
# DB-SI 3
CTE_SI3_MaxDistance.check(project) -> { room_id, distance_m, max_allowed_m, path, compliant }

# DB-SUA 9
CTE_SUA9_DoorWidth.check(door) -> { door_id, clear_width_mm, required_mm, compliant }

# DB-SI 1
CTE_SI1_CompartmentContinuity.check(boundary) -> { boundary_id, gaps_m, rating_ok, compliant }
```

## Current Toolkit Reference

Geometry primitives:
- `calculate_angle_between_walls`
- `find_nearest_door`
- `find_walls_within_radius`
- `calculate_wall_cardinal_direction`
- `calculate_clearance_between_elements`

Graph/topology:
- `create_circulation_graph` (NetworkX, pathfinding, centrality, articulation points)

## Roadmap (Next Sprints)

1) Implement `compute_wall_connectivity` and `check_wall_continuity` (DB-SI 1)
2) Add accessible-edge filtering to pathfinding (DB-SUA 9)
3) Integrate structural attributes and `identify_structural_walls` (DB-SI 6)
4) Add minimal `rules/` layer with JSON outputs for top 3 rules
