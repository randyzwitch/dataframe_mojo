"""Independent ISO WKB and ESRI/DBASE fixture encoders, using format bytes.

No production geometry decoder, dtype, dataframe, or spatial implementation is
imported. Shape expectations are literal coordinates, not decoded roundtrips.
"""
from std.memory import Pointer


def uint(
    mut out: List[UInt8], value: UInt64, width: Int = 4, big: Bool = False
):
    for at in range(width):
        var shift = 8 * (width - 1 - at if big else at)
        out.append(UInt8((value >> UInt64(shift)) & 255))


def number(mut out: List[UInt8], value: Float64, big: Bool = False):
    var scalar = value
    uint(out, Pointer(to=scalar).unsafe_bitcast[UInt64]()[], 8, big)


def nan_ordinate(bytes: List[UInt8], offset: Int) -> Bool:
    var bits = UInt64(0)
    for at in range(8):
        bits |= UInt64(bytes[offset + at]) << UInt64(8 * at)
    return (bits & UInt64(0x7FF0000000000000)) == UInt64(
        0x7FF0000000000000
    ) and (bits & UInt64(0x000FFFFFFFFFFFFF)) != 0


def coordinate(dim: Int, x: Float64, y: Float64) -> List[Float64]:
    var out: List[Float64] = [x, y]
    if dim == 1 or dim == 3:
        out.append(100 + x)
    if dim == 2 or dim == 3:
        out.append(-100 + y)
    return out^


def header(kind: Int, dim: Int = 0, big: Bool = False) -> List[UInt8]:
    var out: List[UInt8] = [UInt8(0 if big else 1)]
    uint(out, UInt64(kind + 1000 * dim), big=big)
    return out^


def point(
    values: List[Float64], dim: Int = 0, big: Bool = False
) -> List[UInt8]:
    var out = header(1, dim, big)
    for value in values:
        number(out, value, big)
    return out^


def line(
    values: List[List[Float64]], dim: Int = 0, big: Bool = False
) -> List[UInt8]:
    var out = header(2, dim, big)
    uint(out, UInt64(len(values)), big=big)
    for position in values:
        for value in position:
            number(out, value, big)
    return out^


def polygon(
    rings: List[List[List[Float64]]], dim: Int = 0, big: Bool = False
) -> List[UInt8]:
    var out = header(3, dim, big)
    uint(out, UInt64(len(rings)), big=big)
    for ring in rings:
        uint(out, UInt64(len(ring)), big=big)
        for position in ring:
            for value in position:
                number(out, value, big)
    return out^


def compound(
    kind: Int, children: List[List[UInt8]], dim: Int = 0, big: Bool = False
) -> List[UInt8]:
    var out = header(kind, dim, big)
    uint(out, UInt64(len(children)), big=big)
    for child in children:
        out.extend(Span(child))
    return out^


def empty(kind: Int, dim: Int = 0, big: Bool = False) -> List[UInt8]:
    var out = header(kind, dim, big)
    if kind == 1:
        for _ in range(2 + Int(dim != 0) + Int(dim == 3)):
            uint(out, UInt64(0x7FF8000000000000), 8, big)
    else:
        uint(out, 0, big=big)
    return out^


def family(kind: Int, dim: Int = 0, big: Bool = False) -> List[UInt8]:
    var a = coordinate(dim, -2, 3)
    var b = coordinate(dim, 5, -7)
    var segment = line([a.copy(), b.copy()], dim, big)
    var ring: List[List[Float64]] = [
        coordinate(dim, -2, -7),
        coordinate(dim, 5, -7),
        coordinate(dim, 5, 3),
        coordinate(dim, -2, 3),
        coordinate(dim, -2, -7),
    ]
    var area = polygon([ring^], dim, big)
    if kind == 1:
        return point(a, dim, big)
    if kind == 2:
        return segment^
    if kind == 3:
        return area^
    if kind == 4:
        return compound(4, [point(a, dim, big), point(b, dim, big)], dim, big)
    if kind == 5:
        return compound(
            5,
            [
                segment.copy(),
                line(
                    [coordinate(dim, 10, 11), coordinate(dim, -13, 17)],
                    dim,
                    big,
                ),
            ],
            dim,
            big,
        )
    if kind == 6:
        var second: List[List[Float64]] = [
            coordinate(dim, 10, 11),
            coordinate(dim, 12, 11),
            coordinate(dim, 10, 17),
            coordinate(dim, 10, 11),
        ]
        return compound(
            6, [area.copy(), polygon([second^], dim, big)], dim, big
        )
    return compound(
        7, [point(a, dim, big), segment.copy(), empty(3, dim, big)], dim, big
    )


def expected_bounds(kind: Int) -> List[Float64]:
    if kind == 1:
        return [-2, 3, -2, 3]
    if kind == 5:
        return [-13, -7, 10, 17]
    if kind == 6:
        return [-2, -7, 12, 17]
    return [-2, -7, 5, 3]


def write_bytes(path: String, bytes: List[UInt8]) raises:
    with open(path, "w") as file:
        file.write_bytes(bytes)


def shape_header(length: Int, kind: Int) -> List[UInt8]:
    var out = List[UInt8]()
    uint(out, 9994, big=True)
    for _ in range(5):
        uint(out, 0, big=True)
    uint(out, UInt64(length // 2), big=True)
    uint(out, 1000)
    uint(out, UInt64(kind))
    for _ in range(8):
        number(out, 0)
    return out^


def shape_record(base: Int, dim: Int = 0) -> List[UInt8]:
    var kind = base + (10 if dim == 1 or dim == 3 else 20 if dim == 2 else 0)
    var out = List[UInt8]()
    uint(out, UInt64(kind))
    var points = List[List[Float64]]()
    var parts = List[Int]()
    if base == 1:
        points.append(coordinate(dim, -2, 3))
    elif base == 8:
        points = [coordinate(dim, -2, 3), coordinate(dim, 5, -7)]
    elif base == 3:
        parts = [0, 2]
        points = [
            coordinate(dim, -2, 3),
            coordinate(dim, 5, -7),
            coordinate(dim, 10, 11),
            coordinate(dim, -13, 17),
        ]
    else:
        # Clockwise exterior, counterclockwise hole, then disjoint exterior.
        parts = [0, 5, 10]
        var xy: List[List[Float64]] = [
            [0, 0],
            [0, 4],
            [4, 4],
            [4, 0],
            [0, 0],
            [1, 1],
            [3, 1],
            [3, 3],
            [1, 3],
            [1, 1],
            [10, 10],
            [10, 12],
            [12, 12],
            [12, 10],
            [10, 10],
        ]
        for xy_point in xy:
            points.append(coordinate(dim, xy_point[0], xy_point[1]))
    if base != 1:
        for _ in range(4):
            number(out, 0)  # Header bbox deliberately not used as value oracle.
        if base != 8:
            uint(out, UInt64(len(parts)))
        uint(out, UInt64(len(points)))
        for start in parts:
            uint(out, UInt64(start))
    for position in points:
        number(out, position[0])
        number(out, position[1])
    if dim == 1 or dim == 3:
        if base != 1:
            number(out, 0)
            number(out, 0)
        for position in points:
            number(out, position[2])
    if dim == 2 or dim == 3:
        if base != 1:
            number(out, 0)
            number(out, 0)
        for position in points:
            number(out, position[2 if dim == 2 else 3])
    return out^


def shape_expected(base: Int, dim: Int) -> List[UInt8]:
    if base == 1:
        return family(1, dim)
    if base == 8:
        return family(4, dim)
    if base == 3:
        return family(5, dim)
    var outer: List[List[Float64]] = [
        coordinate(dim, 0, 0),
        coordinate(dim, 0, 4),
        coordinate(dim, 4, 4),
        coordinate(dim, 4, 0),
        coordinate(dim, 0, 0),
    ]
    var hole: List[List[Float64]] = [
        coordinate(dim, 1, 1),
        coordinate(dim, 3, 1),
        coordinate(dim, 3, 3),
        coordinate(dim, 1, 3),
        coordinate(dim, 1, 1),
    ]
    var island: List[List[Float64]] = [
        coordinate(dim, 10, 10),
        coordinate(dim, 10, 12),
        coordinate(dim, 12, 12),
        coordinate(dim, 12, 10),
        coordinate(dim, 10, 10),
    ]
    return compound(
        6, [polygon([outer^, hole^], dim), polygon([island^], dim)], dim
    )


def dbf_fixture() -> List[UInt8]:
    # Five DBASE III fields and three records. Deleted middle record deliberately
    # contains invalid field values; its shape and attributes must both vanish.
    var names: List[String] = ["name", "count", "score", "flag", "day"]
    var kinds: List[UInt8] = [67, 78, 70, 76, 68]
    var widths: List[Int] = [8, 5, 7, 1, 8]
    var out: List[UInt8] = [3, 126, 10, 11]
    uint(out, 3)
    uint(out, 32 + 5 * 32 + 1, 2)
    uint(out, 1 + 8 + 5 + 7 + 1 + 8, 2)
    for _ in range(20):
        out.append(0)
    for field in range(5):
        var name_bytes = List[UInt8]()
        name_bytes.extend(names[field].as_bytes())
        out.extend(Span(name_bytes))
        for _ in range(11 - len(name_bytes)):
            out.append(0)
        out.append(kinds[field])
        uint(out, 0)
        out.append(UInt8(widths[field]))
        out.append(UInt8(2 if field == 2 else 0))
        for _ in range(14):
            out.append(0)
    out.append(13)
    # Author each fixed-width row explicitly, including UTF-8 byte lengths.
    var cells: List[List[String]] = [
        ["café   ", "    7", "   1.25", "T", "20240229"],
        ["deleted ", "  bad", "bad!!!!", "X", "bad!!!!!"],
        ["        ", "*****", "       ", "?", "00000000"],
    ]
    for row in range(3):
        out.append(UInt8(42 if row == 1 else 32))
        for value in cells[row]:
            out.extend(value.as_bytes())
    out.append(26)
    return out^


def write_shape(
    stem: String,
    base: Int,
    dim: Int = 0,
    with_crs: Bool = True,
    with_index: Bool = True,
) raises:
    var body = List[UInt8]()
    var offsets = List[Int]()
    var sizes = List[Int]()
    for row in range(3):
        var record = shape_record(base, dim) if row < 2 else List[UInt8](
            [UInt8(0), UInt8(0), UInt8(0), UInt8(0)]
        )
        offsets.append(100 + len(body))
        sizes.append(len(record))
        uint(body, UInt64(row + 1), big=True)
        uint(body, UInt64(len(record) // 2), big=True)
        body.extend(record^)
    var kind = base + (10 if dim == 1 or dim == 3 else 20 if dim == 2 else 0)
    var shp = shape_header(100 + len(body), kind)
    shp.extend(body^)
    write_bytes(stem + ".shp", shp)
    var shx = shape_header(100 + 3 * 8, kind)
    for row in range(3):
        uint(shx, UInt64(offsets[row] // 2), big=True)
        uint(shx, UInt64(sizes[row] // 2), big=True)
    if with_index:
        write_bytes(stem + ".shx", shx)
    write_bytes(stem + ".dbf", dbf_fixture())
    if with_crs:
        with open(stem + ".prj", "w") as file:
            file.write('LOCAL_CS["fixture coordinates",UNIT["metre",1]]')
