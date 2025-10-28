"""
Light-weight rule-layer stubs to wrap geometry/topology outputs into
structured compliance checks. Each function is tagged by CTE section.

These are intentionally minimal to serve as integration points for the
future Compliance Brain.
"""

from typing import Any, Dict, List, Optional
import networkx as nx


# ----------------------------- DB-SI 1 ---------------------------------
def compute_wall_connectivity(walls: List[Dict[str, Any]], tolerance: float = 0.05) -> nx.Graph:
    """
    Build a connectivity graph where nodes are wall segments and edges
    indicate continuous joints within a geometric tolerance.

    Returns a NetworkX graph with node attributes including wall id and
    optional fire rating if available.
    """
    G = nx.Graph()
    # Minimal stub: populate nodes; edges to be implemented
    for wall in walls:
        wall_id = wall.get('id') or f"wall_{len(G.nodes)+1}"
        G.add_node(wall_id, **wall)
    return G


def check_wall_continuity(wall_a: Dict[str, Any], wall_b: Dict[str, Any], connectivity_graph: nx.Graph,
                          require_fire_rating_match: bool = True) -> Dict[str, Any]:
    """
    Use the wall connectivity graph to assert continuity between two walls,
    optionally ensuring consistent fire rating across the joint.
    """
    a_id = wall_a.get('id')
    b_id = wall_b.get('id')
    if a_id not in connectivity_graph or b_id not in connectivity_graph:
        return {"success": False, "error": "Wall(s) not present in connectivity graph"}

    # Minimal stub result
    continuity = False
    rating_ok = True

    if require_fire_rating_match:
        a_rating = wall_a.get('fire_rating')
        b_rating = wall_b.get('fire_rating')
        if a_rating and b_rating and a_rating != b_rating:
            rating_ok = False

    return {
        "success": True,
        "continuous": continuity,
        "fire_rating_continuous": rating_ok
    }


# ----------------------------- DB-SI 3 ---------------------------------
def find_shortest_room_path(graph: nx.Graph, start_room_id: str, end_room_id: str) -> Dict[str, Any]:
    """
    Wrapper around NetworkX shortest path that returns a structured payload
    suitable for compliance checks and reporting.
    """
    start = f"room_{start_room_id}" if not start_room_id.startswith('room_') else start_room_id
    end = f"room_{end_room_id}" if not end_room_id.startswith('room_') else end_room_id

    try:
        if not nx.has_path(graph, start, end):
            return {"success": False, "error": "No path found"}
        path = nx.shortest_path(graph, start, end, weight='weight')
        length = nx.shortest_path_length(graph, start, end, weight='weight')
        return {"success": True, "path": path, "distance_m": float(length)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------- DB-SI 6 ---------------------------------
def identify_structural_walls(walls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Partition walls into structural vs non-structural if metadata is available.
    """
    structural = []
    non_structural = []
    unknown = []
    for wall in walls:
        if wall.get('load_bearing') is True:
            structural.append(wall)
        elif wall.get('load_bearing') is False:
            non_structural.append(wall)
        else:
            unknown.append(wall)
    return {
        "success": True,
        "structural": structural,
        "non_structural": non_structural,
        "unknown": unknown
    }


# ----------------------------- DB-SUA 9 -------------------------------
def filter_accessible_edges(graph: nx.Graph) -> nx.Graph:
    """
    Return a subgraph containing edges marked as accessible (e.g., slope ≤ threshold,
    no steps). Requires edge attributes to be present.
    """
    H = nx.Graph()
    H.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        if data.get('accessible', False):
            H.add_edge(u, v, **data)
    return H


