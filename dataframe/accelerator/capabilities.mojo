"""Backend capabilities for bound row plans; no device imports or discovery."""
from dataframe.dtype import DataType
from dataframe.expr import SUM, COUNT, MEAN, LEN, MIN, MAX


struct RowCapabilities(Copyable):
    """Semantic capabilities, independent of storage and device availability.

    Defaults describe the existing NVIDIA row subset. A native Metal provider
    disables Float64 and wide integer accumulation. Float32 input alone does
    not make sum/mean eligible: their shared accumulator contract is Float64.
    """

    var backend: String
    var float64: Bool
    var wide_integer: Bool
    var int64_arithmetic: Bool
    var order: Bool
    var row_extras: Bool
    var fixed_logical: Bool
    var extrema: Bool
    var casts: Bool
    var mixed_types: Bool
    var extended_integers: Bool

    def __init__(
        out self,
        backend: String = "accelerator",
        *,
        float64: Bool = True,
        wide_integer: Bool = True,
        int64_arithmetic: Bool = True,
        extended_integers: Bool = False,
        mixed_types: Bool = False,
        casts: Bool = False,
        extrema: Bool = False,
        fixed_logical: Bool = False,
        row_extras: Bool = False,
        order: Bool = False,
    ):
        self.backend = backend
        self.float64 = float64
        self.wide_integer = wide_integer
        self.int64_arithmetic = int64_arithmetic
        self.extended_integers = extended_integers
        self.mixed_types = mixed_types
        self.casts = casts
        self.extrema = extrema
        self.fixed_logical = fixed_logical
        self.row_extras = row_extras
        self.order = order

    def reject(self, category: String, reason: String) raises:
        raise Error(self.backend + " unsupported [" + category + "]: " + reason)

    def require_dtype(self, dtype: DataType) raises:
        if dtype == DataType.FLOAT64 and not self.float64:
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
        if dtype == DataType.INT64 and not self.int64_arithmetic:
            self.reject(
                "arithmetic", "checked Int64 arithmetic requires CPU execution"
            )

    def require_reduction(self, dtype: DataType, op: Int, rows: Int) raises:
        if op == MIN or op == MAX:
            if not self.extrema:
                self.reject(
                    "reduction", "minimum and maximum are not supported"
                )
            self.require_dtype(dtype)
            return
        if op == COUNT or op == LEN:
            return
        if op == MEAN and not self.float64:
            self.reject("accumulator", "mean requires Float64 arithmetic")
        if op == SUM:
            if dtype.is_decimal() and not self.wide_integer:
                self.reject(
                    "accumulator",
                    "decimal sum requires exact Decimal128 accumulation",
                )
            if dtype.is_float() and not self.float64:
                self.reject(
                    "accumulator", "floating sum requires Float64 accumulation"
                )
            if (
                dtype == DataType.INT64 or dtype == DataType.UINT64
            ) and not self.wide_integer:
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
