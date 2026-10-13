"""Backend capabilities for bound row plans; no device imports or discovery."""
from dataframe.dtype import DataType
from dataframe.expr import (
    SUM,
    COUNT,
    MEAN,
    VAR,
    SKEW,
    KURTOSIS,
    COV,
    CORR,
    STD,
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
    var duration_totals: Bool
    var decimal128: Bool
    var decimal128_reductions: Bool
    var decimal_mean: Bool
    var decimal_float_cast: Bool
    var decimal_casts: Bool
    var decimal_comparisons: Bool
    var decimal_multiplicative: Bool
    var decimal_arithmetic: Bool
    var ranks: Bool
    var paired_moments: Bool
    var moments: Bool
    var higher_moments: Bool
    var rolling_windows: Bool
    var rolling_moments: Bool
    var empty_outputs: Bool
    var empty_inputs: Bool
    var shift_fill_windows: Bool
    var cumulative_windows: Bool
    var reuse_column_slots: Bool
    var proven_integer_operations: Bool
    var max_nodes: Int
    var max_slots: Int
    var max_steps: Int
    var max_join_columns: Int

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
        duration_totals: Bool = False,
        ranks: Bool = False,
        cumulative_windows: Bool = False,
        paired_moments: Bool = False,
        moments: Bool = False,
        higher_moments: Bool = False,
        rolling_windows: Bool = False,
        rolling_moments: Bool = False,
        empty_outputs: Bool = False,
        empty_inputs: Bool = False,
        shift_fill_windows: Bool = False,
        decimal128: Bool = False,
        decimal128_reductions: Bool = False,
        decimal_mean: Bool = False,
        decimal_float_cast: Bool = False,
        decimal_casts: Bool = False,
        decimal_comparisons: Bool = False,
        decimal_arithmetic: Bool = False,
        decimal_multiplicative: Bool = False,
        max_nodes: Int = 64,
        max_slots: Int = 64,
        max_steps: Int = 64,
        max_join_columns: Int = 64,
        reuse_column_slots: Bool = False,
        proven_integer_operations: Bool = False,
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
        self.duration_totals = duration_totals
        self.ranks = ranks
        self.cumulative_windows = cumulative_windows
        self.paired_moments = paired_moments
        self.moments = moments
        self.higher_moments = higher_moments
        self.rolling_windows = rolling_windows
        self.rolling_moments = rolling_moments
        self.empty_outputs = empty_outputs
        self.empty_inputs = empty_inputs
        self.shift_fill_windows = shift_fill_windows
        self.decimal128 = decimal128
        self.decimal128_reductions = decimal128_reductions
        self.decimal_mean = decimal_mean
        self.decimal_float_cast = decimal_float_cast
        self.decimal_casts = decimal_casts
        self.decimal_comparisons = decimal_comparisons
        self.decimal_arithmetic = decimal_arithmetic
        self.decimal_multiplicative = decimal_multiplicative
        self.reuse_column_slots = reuse_column_slots
        self.proven_integer_operations = proven_integer_operations
        self.max_nodes = max_nodes
        self.max_slots = max_slots
        self.max_steps = max_steps
        self.max_join_columns = max_join_columns

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
                if dtype.decimal_width() <= 64 or self.decimal128:
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
            op == VAR
            or op == STD
            or op == COV
            or op == CORR
            or op == SKEW
            or op == KURTOSIS
        ):
            if (op == SKEW or op == KURTOSIS) and not self.higher_moments:
                self.reject(
                    "reduction", "skewness and kurtosis require backend support"
                )
            if (op == COV or op == CORR) and not self.paired_moments:
                self.reject(
                    "reduction",
                    "covariance and correlation require backend support",
                )
            if (op == VAR or op == STD) and not self.moments:
                self.reject(
                    "reduction",
                    "variance and standard deviation require backend support",
                )
            if dtype.is_decimal():
                self.reject(
                    "logical",
                    "decimal moments require an explicit Float64 cast",
                )
            if not dtype.is_numeric():
                self.reject("dtype", "moments require primitive numeric input")
            if not self.float64 and not self.float64_emulated:
                self.reject("precision", "moments require Float64 arithmetic")
            self.require_dtype(dtype)
            return
        if (
            dtype.is_decimal()
            and dtype.decimal_width() == 128
            and op != COUNT
            and op != LEN
            and op != NULL_COUNT
            and op != MIN
            and op != MAX
            and op != FIRST
            and op != LAST
            and not (self.decimal128_reductions and (op == SUM or op == MEAN))
        ):
            self.reject(
                "accumulator",
                "Decimal128 input reductions require CPU execution",
            )
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
        if (
            op == MEAN
            and dtype.is_decimal()
            and self.float64_emulated
            and not self.decimal_mean
        ):
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
            if (
                dtype.is_decimal()
                and not self.wide_integer
                and not self.decimal128
            ):
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
