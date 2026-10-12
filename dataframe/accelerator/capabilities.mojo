"""Backend capabilities for bound row plans; no device imports or discovery."""
from dataframe.dtype import DataType
from dataframe.expr import (
    SUM,
    COUNT,
    MEAN,
    LEN,
    MIN,
    MAX,
    FIRST,
    LAST,
    ANY,
    ALL,
    NULL_COUNT,
    ARG_MIN,
    ARG_MAX,
)


struct RowCapabilities(Copyable):
    """Semantic capabilities, independent of storage and device availability.

    Defaults describe the existing NVIDIA row subset. Metal distinguishes
    integer-only Float64 emulation from native Float64, and uses optional
    range-guarded native integer accumulation. Float32 sum/mean still requires
    native or emulated Float64 accumulation.
    """

    var backend: String
    var float64: Bool
    var float64_storage: Bool
    var float64_emulated: Bool
    var float_division: Bool
    var float_modulo: Bool
    var float_sqrt: Bool
    var float_rounding: Bool
    var wide_integer: Bool
    var bounded_integer_sum: Bool
    var int64_arithmetic: Bool
    var extra_reductions: Bool
    var grouped: Bool
    var order: Bool
    var row_extras: Bool
    var fixed_logical: Bool
    var extrema: Bool
    var casts: Bool
    var mixed_types: Bool
    var extended_integers: Bool
    var slices: Bool
    var scalar_regions: Bool
    var temporal_fields: Bool
    var ranks: Bool

    def __init__(
        out self,
        backend: String = "accelerator",
        *,
        float64: Bool = True,
        float64_storage: Bool = False,
        float64_emulated: Bool = False,
        float_division: Bool = False,
        float_modulo: Bool = False,
        float_sqrt: Bool = False,
        float_rounding: Bool = False,
        wide_integer: Bool = True,
        bounded_integer_sum: Bool = False,
        int64_arithmetic: Bool = True,
        extended_integers: Bool = False,
        mixed_types: Bool = False,
        casts: Bool = False,
        extrema: Bool = False,
        fixed_logical: Bool = False,
        row_extras: Bool = False,
        order: Bool = False,
        grouped: Bool = False,
        extra_reductions: Bool = False,
        slices: Bool = False,
        scalar_regions: Bool = False,
        temporal_fields: Bool = False,
        ranks: Bool = False,
    ):
        self.backend = backend
        self.float64 = float64
        self.float64_storage = float64_storage
        self.float64_emulated = float64_emulated
        self.float_division = float_division
        self.float_modulo = float_modulo
        self.float_sqrt = float_sqrt
        self.float_rounding = float_rounding
        self.wide_integer = wide_integer
        self.bounded_integer_sum = bounded_integer_sum
        self.int64_arithmetic = int64_arithmetic
        self.extended_integers = extended_integers
        self.mixed_types = mixed_types
        self.casts = casts
        self.extrema = extrema
        self.fixed_logical = fixed_logical
        self.row_extras = row_extras
        self.order = order
        self.grouped = grouped
        self.extra_reductions = extra_reductions
        self.slices = slices
        self.scalar_regions = scalar_regions
        self.temporal_fields = temporal_fields
        self.ranks = ranks

    def reject(self, category: String, reason: String) raises:
        raise Error(self.backend + " unsupported [" + category + "]: " + reason)

    def require_dtype(self, dtype: DataType) raises:
        if (
            dtype == DataType.FLOAT64
            and not self.float64
            and not self.float64_storage
        ):
            self.reject("dtype", "Float64 requires CPU execution")
        if self.fixed_logical:
            if dtype.is_temporal():
                return
            if dtype.is_decimal():
                if dtype.decimal_width() <= 64:
                    return
                self.reject("dtype", "Decimal128 requires CPU execution")
        if self.extended_integers and dtype.is_integer():
            return
        if (
            dtype != DataType.FLOAT32
            and dtype != DataType.FLOAT64
            and dtype != DataType.INT32
            and dtype != DataType.INT64
            and dtype != DataType.BOOL
        ):
            self.reject("dtype", "row input dtype is not supported")

    def require_arithmetic(self, dtype: DataType) raises:
        if (
            dtype == DataType.FLOAT64
            and not self.float64
            and not self.float64_emulated
        ):
            self.reject(
                "precision", "Float64 requires CPU execution for arithmetic"
            )
        if dtype == DataType.INT64 and not self.int64_arithmetic:
            self.reject(
                "arithmetic", "checked Int64 arithmetic requires CPU execution"
            )

    def require_key(self, dtype: DataType) raises:
        if (
            dtype == DataType.FLOAT64
            and not self.float64
            and not self.float64_emulated
        ):
            self.reject(
                "precision",
                "Float64 requires CPU execution for ordering and keys",
            )

    def require_reduction(self, dtype: DataType, op: Int, rows: Int) raises:
        if (
            dtype == DataType.FLOAT64
            and not self.float64
            and not self.float64_emulated
            and (op == MIN or op == MAX or op == ARG_MIN or op == ARG_MAX)
        ):
            self.reject(
                "precision", "Float64 requires CPU execution for extrema"
            )
        if (
            op == FIRST
            or op == LAST
            or op == ANY
            or op == ALL
            or op == NULL_COUNT
            or op == ARG_MIN
            or op == ARG_MAX
        ):
            if not self.extra_reductions:
                self.reject(
                    "reduction", "additional reductions are not supported"
                )
            self.require_dtype(dtype)
            return
        if op == MIN or op == MAX:
            if not self.extrema:
                self.reject(
                    "reduction", "minimum and maximum are not supported"
                )
            self.require_dtype(dtype)
            return
        if op == COUNT or op == LEN:
            return
        if op == MEAN and dtype.is_decimal() and self.float64_emulated:
            self.reject(
                "logical", "decimal mean requires scaled Float64 conversion"
            )
        if op == MEAN and not self.float64 and not self.float64_emulated:
            self.reject("accumulator", "mean requires Float64 arithmetic")
        if (
            op == MEAN
            and (dtype == DataType.INT64 or dtype == DataType.UINT64)
            and not self.wide_integer
            and not self.bounded_integer_sum
        ):
            self.reject(
                "accumulator", "integer mean requires exact wide accumulation"
            )
        if op == SUM:
            if dtype.is_decimal() and not self.wide_integer:
                self.reject(
                    "accumulator",
                    "decimal sum requires exact Decimal128 accumulation",
                )
            if (
                dtype.is_float()
                and not self.float64
                and not self.float64_emulated
            ):
                self.reject(
                    "accumulator", "floating sum requires Float64 accumulation"
                )
            if (
                (dtype.physical() == DataType.INT64 or dtype == DataType.UINT64)
                and not self.wide_integer
                and not self.bounded_integer_sum
            ):
                self.reject(
                    "accumulator",
                    dtype.name() + " sum requires exact wide accumulation",
                )
            if (
                dtype == DataType.INT32
                and not self.wide_integer
                and rows > Int(Int32.MAX)
            ):
                self.reject(
                    "accumulator", "Int32 sum exceeds the exact Int64 row bound"
                )
