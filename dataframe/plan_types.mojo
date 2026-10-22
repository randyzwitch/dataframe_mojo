"""Value types the planner owns and a backend only reads.

PlanEstimate is what a backend reports for automatic placement. WindowKey,
CategoryOwner, CategoryField and TextField are the key and text provenance
that plan nodes carry. They hold host metadata only. Backends import them
from here; nothing in this module knows a backend exists.
"""
from std.memory import ArcPointer
from .column import Column
from .dtype import DataType


@fieldwise_init
struct PlanEstimate(Copyable):
    var cpu: List[Float64]
    var accelerator: List[Float64]
    var peak_bytes: Int
    var family: String
    var placement: String


struct WindowKey(Copyable):
    var name: String
    var dtype: DataType
    var string_hash: Bool
    var values: ArcPointer[List[UInt64]]

    def __init__(out self, name: String, dtype: DataType, string_hash: Bool):
        self.name = name
        self.dtype = dtype
        self.string_hash = string_hash
        self.values = ArcPointer(List[UInt64]())


@fieldwise_init
struct CategoryOwner(Copyable):
    """Exact CPU dictionary and immutable original scan code buffers.

    Equal labels at distinct dictionary positions retain distinct identities.
    The buffers preserve offsets, chunk boundaries and their original owners.
    """

    var dtype: DataType
    var codes: List[Column[UInt32]]


struct CategoryField(Copyable):
    var owner: ArcPointer[CategoryOwner]
    var string_codes: Bool

    def __init__(
        out self, owner: ArcPointer[CategoryOwner], string_codes: Bool = False
    ):
        self.owner = owner
        self.string_codes = string_codes


struct TextField(Copyable):
    var name: String
    var dtype: DataType
    var category: Optional[CategoryField]

    def __init__(
        out self,
        name: String,
        dtype: DataType,
        category: Optional[CategoryField] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.category = category.copy()
