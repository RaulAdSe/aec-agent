"""
Utilities to enrich connectivity information (e.g., infer door → room links)
from plan data when extractor did not populate it.

Non-destructive: functions work on Python dict/list structures (like loader.levels)
and fill missing 'from_room' / 'to_room' on door dictionaries when possible.
"""

from typing import Dict, List, Tuple, Optional, Any


def _centroid_from_points(points: List[Dict[str, float]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    try:
        cx = sum(p.get("x", 0.0) for p in points) / len(points)
        cy = sum(p.get("y", 0.0) for p in points) / len(points)
        return cx, cy
    except Exception:
        return None


def _room_centroid(room: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    # Prefer explicit centroid
    c = room.get("centroid")
    if isinstance(c, dict) and "x" in c and "y" in c:
        return float(c["x"]), float(c["y"])

    # Try common polygon-like attributes
    for key in ("boundary", "polygon", "outline", "footprint", "points"):
        poly = room.get(key)
        if isinstance(poly, dict) and isinstance(poly.get("points"), list):
            pts = poly["points"]
        elif isinstance(poly, list):
            pts = poly
        else:
            pts = None
        if isinstance(pts, list) and pts and isinstance(pts[0], dict) and "x" in pts[0] and "y" in pts[0]:
            return _centroid_from_points(pts)

    return None


def enrich_door_room_connectivity(levels: List[Dict[str, Any]], max_link_distance_m: float = 15.0) -> Dict[str, int]:
    """
    For each level structure { rooms: [...], doors: [...] }, fill missing
    door['from_room'] / door['to_room'] by assigning the nearest two room centroids
    to each door position within a maximum distance.

    Args:
        levels: list of level dicts with 'rooms' and 'doors' arrays
        max_link_distance_m: cap distance to link a door to a room

    Returns:
        Counters: { 'updated_doors': int, 'skipped_doors': int }
    """
    updated = 0
    skipped = 0

    for level in levels or []:
        rooms = level.get("rooms", []) or []
        doors = level.get("doors", []) or []

        # Build centroid map for rooms in this level
        room_centroids: Dict[str, Tuple[float, float]] = {}
        for r in rooms:
            rid = r.get("id")
            c = _room_centroid(r)
            if rid and c is not None:
                room_centroids[rid] = (float(c[0]), float(c[1]))

        # Assign for each door with position
        for d in doors:
            pos = d.get("position") or {}
            px, py = pos.get("x"), pos.get("y")
            if px is None or py is None:
                skipped += 1
                continue

            # Already filled (skip if both are non-null)
            if d.get("from_room") is not None and d.get("to_room") is not None:
                continue

            # Rank rooms by distance
            ranked: List[Tuple[float, str]] = []
            for rid, (cx, cy) in room_centroids.items():
                dx = float(px) - cx
                dy = float(py) - cy
                dist = (dx * dx + dy * dy) ** 0.5
                ranked.append((dist, rid))
            ranked.sort(key=lambda t: t[0])

            # Choose up to two within threshold
            if not ranked:
                skipped += 1
                continue

            first = ranked[0] if ranked else None
            second = ranked[1] if len(ranked) > 1 else None

            if first and first[0] <= max_link_distance_m:
                if d.get("from_room") is None:
                    d["from_room"] = first[1]
                # if second is far, leave to_room None (exterior)
                if second and second[0] <= max_link_distance_m and second[1] != first[1]:
                    if d.get("to_room") is None:
                        d["to_room"] = second[1]
                updated += 1
            else:
                skipped += 1

    return {"updated_doors": updated, "skipped_doors": skipped}


