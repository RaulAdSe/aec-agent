import math
from src.calculations.geometry import (
    calculate_polygon_centroid,
    calculate_room_centroid,
)
from src.schemas import Point2D, Boundary, Room


def nearly_equal(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def test_calculate_polygon_centroid_square():
    # Square from (0,0) to (2,2) -> centroid at (1,1)
    points = [
        Point2D(x=0.0, y=0.0),
        Point2D(x=2.0, y=0.0),
        Point2D(x=2.0, y=2.0),
        Point2D(x=0.0, y=2.0),
    ]
    c = calculate_polygon_centroid(points)
    assert nearly_equal(c.x, 1.0)
    assert nearly_equal(c.y, 1.0)


def test_calculate_room_centroid_from_boundary():
    # Rectangle 4m x 3m -> centroid at (2,1.5)
    boundary = Boundary(points=[
        Point2D(x=0.0, y=0.0),
        Point2D(x=4.0, y=0.0),
        Point2D(x=4.0, y=3.0),
        Point2D(x=0.0, y=3.0),
    ])
    room = Room(
        id="R_TEST",
        name="Test Room",
        area=12.0,
        use="commercial",
        level="L1",
        boundary=boundary,
    )
    c = calculate_room_centroid(room)
    assert c is not None
    assert nearly_equal(c.x, 2.0)
    assert nearly_equal(c.y, 1.5)


