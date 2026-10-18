"""Elementwise math functions, product and cum_prod against Polars 1.44.

Expected values were computed by Polars 1.44.2 (`pl.col(c).<function>()`,
`pl.arctan2`) on the same inputs. Transcendental results come from the
platform libm on both sides, so finite values agree to a few ulps (log10 and
log(2) are exact here where Polars divides natural logarithms); NaN,
infinities and signed zeros must match exactly.
"""
# contract-group: expr.arithmetic
# contract-group: reduce.global
# contract-group: reduce.grouped
# contract-group: expr.order_windows
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    arctan2,
    col,
)
from tests.support.coverage_generators import offset_layout, chunk_layout


@fieldwise_init
struct MathCase(Copyable):
    var name: String
    var source: String
    var expression: Expr
    var dtype: DataType
    var expected: List[Float64]
    var null: List[Bool]


def _inputs() raises -> DataFrame:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var valid: List[Bool] = [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    var floats: List[Float64] = [1, -1, 0, 0, nan, inf, -0.0, 8, 0.5, -27]
    var ints: List[Int64] = [1, -2, 0, 0, 5, -27, 8, 1000, 3, -1]
    var singles: List[Float32] = [
        1,
        -1,
        0,
        0,
        2,
        3,
        -8,
        8,
        0.25,
        (-inf).cast[DType.float32](),
    ]
    return DataFrame(
        [
            Series("f", Column[Float64](floats^, valid.copy())),
            Series("i", Column[Int64](ints^, valid.copy())),
            Series("f32", Column[Float32](singles^, valid^)),
        ]
    )


def _assert_case(actual: Series, expected: MathCase) raises:
    var context = expected.name + "(" + expected.source + ")"
    assert_true(actual.dtype() == expected.dtype, context + " dtype")
    var values = actual.cast(DataType.FLOAT64)
    var tolerance = 2e-6 if expected.dtype == DataType.FLOAT32 else 1e-14
    for row in range(len(expected.expected)):
        var item = values.get(row)
        assert_equal(item.is_null(), expected.null[row], context)
        if expected.null[row]:
            continue
        var got = item.float64()
        var want = expected.expected[row]
        if isnan(want):
            assert_true(isnan(got), context + " row " + String(row))
        elif want == 0 or abs(want) == Float64(1) / Float64(0):
            # Zeros keep their sign; infinities match exactly.
            assert_equal(
                bitcast[DType.uint64](got),
                bitcast[DType.uint64](want),
                context + " row " + String(row),
            )
        else:
            assert_true(
                abs(got - want) <= tolerance * abs(want),
                context
                + " row "
                + String(row)
                + ": "
                + String(got)
                + " vs "
                + String(want),
            )


def _math_cases() -> List[MathCase]:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var cases = List[MathCase]()
    cases.append(
        MathCase(
            "log10",
            "f",
            col("f").log10(),
            DataType.FLOAT64,
            [
                0.0,
                nan,
                -inf,
                0.0,
                nan,
                inf,
                -inf,
                0.9030899869919434,
                -0.30102999566398114,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log10",
            "i",
            col("i").log10(),
            DataType.FLOAT64,
            [
                0.0,
                nan,
                -inf,
                0.0,
                0.6989700043360187,
                nan,
                0.9030899869919434,
                2.9999999999999996,
                0.47712125471966244,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log10",
            "f32",
            col("f32").log10(),
            DataType.FLOAT32,
            [
                0.0,
                nan,
                -inf,
                0.0,
                0.3010299801826477,
                0.4771212637424469,
                nan,
                0.9030900001525879,
                -0.6020599603652954,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log1p",
            "f",
            col("f").log1p(),
            DataType.FLOAT64,
            [
                0.6931471805599453,
                -inf,
                0.0,
                0.0,
                nan,
                inf,
                -0.0,
                2.1972245773362196,
                0.4054651081081644,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log1p",
            "i",
            col("i").log1p(),
            DataType.FLOAT64,
            [
                0.6931471805599453,
                nan,
                0.0,
                0.0,
                1.791759469228055,
                nan,
                2.1972245773362196,
                6.90875477931522,
                1.3862943611198906,
                -inf,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log1p",
            "f32",
            col("f32").log1p(),
            DataType.FLOAT32,
            [
                0.6931471824645996,
                -inf,
                0.0,
                0.0,
                1.0986123085021973,
                1.3862943649291992,
                nan,
                2.1972246170043945,
                0.2231435477733612,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cbrt",
            "f",
            col("f").cbrt(),
            DataType.FLOAT64,
            [
                1.0,
                -1.0,
                0.0,
                0.0,
                nan,
                inf,
                -0.0,
                2.0,
                0.7937005259840998,
                -3.0,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cbrt",
            "i",
            col("i").cbrt(),
            DataType.FLOAT64,
            [
                1.0,
                -1.2599210498948732,
                0.0,
                0.0,
                1.709975946676697,
                -3.0,
                2.0,
                10.0,
                1.4422495703074083,
                -1.0,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cbrt",
            "f32",
            col("f32").cbrt(),
            DataType.FLOAT32,
            [
                1.0,
                -1.0,
                0.0,
                0.0,
                1.2599210739135742,
                1.4422495365142822,
                -2.0,
                2.0,
                0.6299605369567871,
                -inf,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "sin",
            "f",
            col("f").sin(),
            DataType.FLOAT64,
            [
                0.8414709848078965,
                -0.8414709848078965,
                0.0,
                0.0,
                nan,
                nan,
                -0.0,
                0.9893582466233818,
                0.479425538604203,
                -0.956375928404503,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "sin",
            "i",
            col("i").sin(),
            DataType.FLOAT64,
            [
                0.8414709848078965,
                -0.9092974268256817,
                0.0,
                0.0,
                -0.9589242746631385,
                -0.956375928404503,
                0.9893582466233818,
                0.8268795405320025,
                0.1411200080598672,
                -0.8414709848078965,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "sin",
            "f32",
            col("f32").sin(),
            DataType.FLOAT32,
            [
                0.8414709568023682,
                -0.8414709568023682,
                0.0,
                0.0,
                0.9092974066734314,
                0.14112000167369843,
                -0.9893582463264465,
                0.9893582463264465,
                0.24740396440029144,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cos",
            "f",
            col("f").cos(),
            DataType.FLOAT64,
            [
                0.5403023058681398,
                0.5403023058681398,
                1.0,
                0.0,
                nan,
                nan,
                1.0,
                -0.14550003380861354,
                0.8775825618903728,
                -0.2921388087338362,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cos",
            "i",
            col("i").cos(),
            DataType.FLOAT64,
            [
                0.5403023058681398,
                -0.4161468365471424,
                1.0,
                0.0,
                0.28366218546322625,
                -0.2921388087338362,
                -0.14550003380861354,
                0.5623790762907029,
                -0.9899924966004454,
                0.5403023058681398,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cos",
            "f32",
            col("f32").cos(),
            DataType.FLOAT32,
            [
                0.5403022766113281,
                0.5403022766113281,
                1.0,
                0.0,
                -0.416146844625473,
                -0.9899924993515015,
                -0.1455000340938568,
                -0.1455000340938568,
                0.9689124226570129,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "tan",
            "f",
            col("f").tan(),
            DataType.FLOAT64,
            [
                1.5574077246549023,
                -1.5574077246549023,
                0.0,
                0.0,
                nan,
                nan,
                -0.0,
                -6.799711455220379,
                0.5463024898437905,
                3.273703800428119,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "tan",
            "i",
            col("i").tan(),
            DataType.FLOAT64,
            [
                1.5574077246549023,
                2.185039863261519,
                0.0,
                0.0,
                -3.380515006246586,
                3.273703800428119,
                -6.799711455220379,
                1.4703241557027185,
                -0.1425465430742778,
                -1.5574077246549023,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "tan",
            "f32",
            col("f32").tan(),
            DataType.FLOAT32,
            [
                1.5574077367782593,
                -1.5574077367782593,
                0.0,
                0.0,
                -2.185039758682251,
                -0.14254654943943024,
                6.799711227416992,
                -6.799711227416992,
                0.25534191727638245,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cot",
            "f",
            col("f").cot(),
            DataType.FLOAT64,
            [
                0.6420926159343306,
                -0.6420926159343306,
                inf,
                0.0,
                nan,
                nan,
                -inf,
                -0.1470650639494805,
                1.830487721712452,
                0.305464410026718,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cot",
            "i",
            col("i").cot(),
            DataType.FLOAT64,
            [
                0.6420926159343306,
                0.45765755436028577,
                inf,
                0.0,
                -0.2958129155327455,
                0.305464410026718,
                -0.1470650639494805,
                0.6801221323348698,
                -7.015252551434534,
                -0.6420926159343306,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cot",
            "f32",
            col("f32").cot(),
            DataType.FLOAT32,
            [
                0.6420925855636597,
                -0.6420925855636597,
                inf,
                0.0,
                -0.4576575756072998,
                -7.015252113342285,
                0.14706507325172424,
                -0.14706507325172424,
                3.9163174629211426,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arcsin",
            "f",
            col("f").arcsin(),
            DataType.FLOAT64,
            [
                1.5707963267948966,
                -1.5707963267948966,
                0.0,
                0.0,
                nan,
                nan,
                -0.0,
                nan,
                0.5235987755982989,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arcsin",
            "i",
            col("i").arcsin(),
            DataType.FLOAT64,
            [
                1.5707963267948966,
                nan,
                0.0,
                0.0,
                nan,
                nan,
                nan,
                nan,
                nan,
                -1.5707963267948966,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arcsin",
            "f32",
            col("f32").arcsin(),
            DataType.FLOAT32,
            [
                1.5707963705062866,
                -1.5707963705062866,
                0.0,
                0.0,
                nan,
                nan,
                nan,
                nan,
                0.252680242061615,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arccos",
            "f",
            col("f").arccos(),
            DataType.FLOAT64,
            [
                0.0,
                3.141592653589793,
                1.5707963267948966,
                0.0,
                nan,
                nan,
                1.5707963267948966,
                nan,
                1.0471975511965979,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arccos",
            "i",
            col("i").arccos(),
            DataType.FLOAT64,
            [
                0.0,
                nan,
                1.5707963267948966,
                0.0,
                nan,
                nan,
                nan,
                nan,
                nan,
                3.141592653589793,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arccos",
            "f32",
            col("f32").arccos(),
            DataType.FLOAT32,
            [
                0.0,
                3.1415927410125732,
                1.5707963705062866,
                0.0,
                nan,
                nan,
                nan,
                nan,
                1.3181160688400269,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arctan",
            "f",
            col("f").arctan(),
            DataType.FLOAT64,
            [
                0.7853981633974483,
                -0.7853981633974483,
                0.0,
                0.0,
                nan,
                1.5707963267948966,
                -0.0,
                1.446441332248135,
                0.4636476090008061,
                -1.5337762109209665,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arctan",
            "i",
            col("i").arctan(),
            DataType.FLOAT64,
            [
                0.7853981633974483,
                -1.1071487177940904,
                0.0,
                0.0,
                1.373400766945016,
                -1.5337762109209665,
                1.446441332248135,
                1.5697963271282298,
                1.2490457723982544,
                -0.7853981633974483,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arctan",
            "f32",
            col("f32").arctan(),
            DataType.FLOAT32,
            [
                0.7853981852531433,
                -0.7853981852531433,
                0.0,
                0.0,
                1.1071487665176392,
                1.249045729637146,
                -1.4464412927627563,
                1.4464412927627563,
                0.244978666305542,
                -1.5707963705062866,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "sinh",
            "f",
            col("f").sinh(),
            DataType.FLOAT64,
            [
                1.1752011936438014,
                -1.1752011936438014,
                0.0,
                0.0,
                nan,
                inf,
                -0.0,
                1490.4788257895502,
                0.5210953054937474,
                -266024120300.89932,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "sinh",
            "i",
            col("i").sinh(),
            DataType.FLOAT64,
            [
                1.1752011936438014,
                -3.626860407847019,
                0.0,
                0.0,
                74.20321057778875,
                -266024120300.89932,
                1490.4788257895502,
                inf,
                10.017874927409903,
                -1.1752011936438014,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "sinh",
            "f32",
            col("f32").sinh(),
            DataType.FLOAT32,
            [
                1.175201177597046,
                -1.175201177597046,
                0.0,
                0.0,
                3.6268603801727295,
                10.017874717712402,
                -1490.4788818359375,
                1490.4788818359375,
                0.25261232256889343,
                -inf,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cosh",
            "f",
            col("f").cosh(),
            DataType.FLOAT64,
            [
                1.5430806348152437,
                1.5430806348152437,
                1.0,
                0.0,
                nan,
                inf,
                1.0,
                1490.479161252178,
                1.1276259652063807,
                266024120300.89932,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cosh",
            "i",
            col("i").cosh(),
            DataType.FLOAT64,
            [
                1.5430806348152437,
                3.7621956910836314,
                1.0,
                0.0,
                74.20994852478785,
                266024120300.89932,
                1490.479161252178,
                inf,
                10.067661995777765,
                1.5430806348152437,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "cosh",
            "f32",
            col("f32").cosh(),
            DataType.FLOAT32,
            [
                1.5430805683135986,
                1.5430805683135986,
                1.0,
                0.0,
                3.7621958255767822,
                10.067662239074707,
                1490.4791259765625,
                1490.4791259765625,
                1.0314130783081055,
                inf,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "tanh",
            "f",
            col("f").tanh(),
            DataType.FLOAT64,
            [
                0.7615941559557649,
                -0.7615941559557649,
                0.0,
                0.0,
                nan,
                1.0,
                -0.0,
                0.9999997749296758,
                0.46211715726000974,
                -1.0,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "tanh",
            "i",
            col("i").tanh(),
            DataType.FLOAT64,
            [
                0.7615941559557649,
                -0.9640275800758169,
                0.0,
                0.0,
                0.9999092042625951,
                -1.0,
                0.9999997749296758,
                1.0,
                0.9950547536867305,
                -0.7615941559557649,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "tanh",
            "f32",
            col("f32").tanh(),
            DataType.FLOAT32,
            [
                0.7615941762924194,
                -0.7615941762924194,
                0.0,
                0.0,
                0.9640275835990906,
                0.9950547814369202,
                -0.9999997615814209,
                0.9999997615814209,
                0.24491865932941437,
                -1.0,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arcsinh",
            "f",
            col("f").arcsinh(),
            DataType.FLOAT64,
            [
                0.881373587019543,
                -0.881373587019543,
                0.0,
                0.0,
                nan,
                inf,
                -0.0,
                2.7764722807237177,
                0.48121182505960347,
                -3.989326805819546,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arcsinh",
            "i",
            col("i").arcsinh(),
            DataType.FLOAT64,
            [
                0.881373587019543,
                -1.4436354751788103,
                0.0,
                0.0,
                2.3124383412727525,
                -3.989326805819546,
                2.7764722807237177,
                7.600902709541988,
                1.8184464592320668,
                -0.881373587019543,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arcsinh",
            "f32",
            col("f32").arcsinh(),
            DataType.FLOAT32,
            [
                0.8813735842704773,
                -0.8813735842704773,
                0.0,
                0.0,
                1.4436354637145996,
                1.8184465169906616,
                -2.776472330093384,
                2.776472330093384,
                0.24746647477149963,
                -inf,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arccosh",
            "f",
            col("f").arccosh(),
            DataType.FLOAT64,
            [0.0, nan, nan, 0.0, nan, inf, nan, 2.7686593833135738, nan, nan],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arccosh",
            "i",
            col("i").arccosh(),
            DataType.FLOAT64,
            [
                0.0,
                nan,
                nan,
                0.0,
                2.2924316695611777,
                nan,
                2.7686593833135738,
                7.600902209541989,
                1.762747174039086,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arccosh",
            "f32",
            col("f32").arccosh(),
            DataType.FLOAT32,
            [
                0.0,
                nan,
                nan,
                0.0,
                1.316957950592041,
                1.7627471685409546,
                nan,
                2.7686593532562256,
                nan,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arctanh",
            "f",
            col("f").arctanh(),
            DataType.FLOAT64,
            [inf, -inf, 0.0, 0.0, nan, nan, -0.0, nan, 0.5493061443340548, nan],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arctanh",
            "i",
            col("i").arctanh(),
            DataType.FLOAT64,
            [inf, nan, 0.0, 0.0, nan, nan, nan, nan, nan, -inf],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arctanh",
            "f32",
            col("f32").arctanh(),
            DataType.FLOAT32,
            [inf, -inf, 0.0, 0.0, nan, nan, nan, nan, 0.2554128170013428, nan],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "degrees",
            "f",
            col("f").degrees(),
            DataType.FLOAT64,
            [
                57.29577951308232,
                -57.29577951308232,
                0.0,
                0.0,
                nan,
                inf,
                -0.0,
                458.3662361046586,
                28.64788975654116,
                -1546.9860468532227,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "degrees",
            "i",
            col("i").degrees(),
            DataType.FLOAT64,
            [
                57.29577951308232,
                -114.59155902616465,
                0.0,
                0.0,
                286.4788975654116,
                -1546.9860468532227,
                458.3662361046586,
                57295.77951308232,
                171.88733853924697,
                -57.29577951308232,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "degrees",
            "f32",
            col("f32").degrees(),
            DataType.FLOAT32,
            [
                57.295780181884766,
                -57.295780181884766,
                0.0,
                0.0,
                114.59156036376953,
                171.88734436035156,
                -458.3662414550781,
                458.3662414550781,
                14.323945045471191,
                -inf,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "radians",
            "f",
            col("f").radians(),
            DataType.FLOAT64,
            [
                0.017453292519943295,
                -0.017453292519943295,
                0.0,
                0.0,
                nan,
                inf,
                -0.0,
                0.13962634015954636,
                0.008726646259971648,
                -0.47123889803846897,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "radians",
            "i",
            col("i").radians(),
            DataType.FLOAT64,
            [
                0.017453292519943295,
                -0.03490658503988659,
                0.0,
                0.0,
                0.08726646259971647,
                -0.47123889803846897,
                0.13962634015954636,
                17.453292519943297,
                0.05235987755982989,
                -0.017453292519943295,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "radians",
            "f32",
            col("f32").radians(),
            DataType.FLOAT32,
            [
                0.01745329238474369,
                -0.01745329238474369,
                0.0,
                0.0,
                0.03490658476948738,
                0.05235987901687622,
                -0.13962633907794952,
                0.13962633907794952,
                0.004363323096185923,
                -inf,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "sign",
            "f",
            col("f").sign(),
            DataType.FLOAT64,
            [1.0, -1.0, 0.0, 0.0, nan, 1.0, -0.0, 1.0, 1.0, -1.0],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "sign",
            "i",
            col("i").sign(),
            DataType.INT64,
            [1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "sign",
            "f32",
            col("f32").sign(),
            DataType.FLOAT32,
            [1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "truncate",
            "f",
            col("f").truncate(),
            DataType.FLOAT64,
            [1.0, -1.0, 0.0, 0.0, nan, inf, -0.0, 8.0, 0.0, -27.0],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "truncate",
            "i",
            col("i").truncate(),
            DataType.INT64,
            [1.0, -2.0, 0.0, 0.0, 5.0, -27.0, 8.0, 1000.0, 3.0, -1.0],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "truncate",
            "f32",
            col("f32").truncate(),
            DataType.FLOAT32,
            [1.0, -1.0, 0.0, 0.0, 2.0, 3.0, -8.0, 8.0, 0.0, -inf],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log2.0",
            "f",
            col("f").log(2.0),
            DataType.FLOAT64,
            [0.0, nan, -inf, 0.0, nan, inf, -inf, 3.0, -1.0, nan],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log2.0",
            "i",
            col("i").log(2.0),
            DataType.FLOAT64,
            [
                0.0,
                nan,
                -inf,
                0.0,
                2.321928094887362,
                nan,
                3.0,
                9.965784284662087,
                1.5849625007211563,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log2.0",
            "f32",
            col("f32").log(2.0),
            DataType.FLOAT32,
            [0.0, nan, -inf, 0.0, 1.0, 1.5849624872207642, nan, 3.0, -2.0, nan],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log0.5",
            "f",
            col("f").log(0.5),
            DataType.FLOAT64,
            [-0.0, nan, inf, 0.0, nan, -inf, inf, -3.0, 1.0, nan],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log0.5",
            "i",
            col("i").log(0.5),
            DataType.FLOAT64,
            [
                -0.0,
                nan,
                inf,
                0.0,
                -2.321928094887362,
                nan,
                -3.0,
                -9.965784284662087,
                -1.5849625007211563,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log0.5",
            "f32",
            col("f32").log(0.5),
            DataType.FLOAT32,
            [
                -0.0,
                nan,
                inf,
                0.0,
                -1.0,
                -1.5849624872207642,
                nan,
                -3.0,
                2.0,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log2.718281828459045",
            "f",
            col("f").log(2.718281828459045),
            DataType.FLOAT64,
            [
                0.0,
                nan,
                -inf,
                0.0,
                nan,
                inf,
                -inf,
                2.0794415416798357,
                -0.6931471805599453,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log2.718281828459045",
            "i",
            col("i").log(2.718281828459045),
            DataType.FLOAT64,
            [
                0.0,
                nan,
                -inf,
                0.0,
                1.6094379124341003,
                nan,
                2.0794415416798357,
                6.907755278982137,
                1.0986122886681098,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "log2.718281828459045",
            "f32",
            col("f32").log(2.718281828459045),
            DataType.FLOAT32,
            [
                0.0,
                nan,
                -inf,
                0.0,
                0.6931472420692444,
                1.0986124277114868,
                nan,
                2.079441785812378,
                -1.3862944841384888,
                nan,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arctan2_f_i",
            "f",
            arctan2(col("f"), col("i")),
            DataType.FLOAT64,
            [
                0.7853981633974483,
                -2.677945044588987,
                0.0,
                0.0,
                nan,
                1.5707963267948966,
                -0.0,
                0.007999829339886634,
                0.16514867741462683,
                -1.6078164426688266,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arctan2_i_f",
            "i",
            arctan2(col("i"), col("f")),
            DataType.FLOAT64,
            [
                0.7853981633974483,
                -2.0344439357957027,
                0.0,
                0.0,
                nan,
                -0.0,
                1.5707963267948966,
                1.56279649745501,
                1.4056476493802699,
                -3.104572537715863,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arctan2_f32_f",
            "f32",
            arctan2(col("f32"), col("f")),
            DataType.FLOAT32,
            [
                0.7853981852531433,
                -2.356194496154785,
                0.0,
                0.0,
                nan,
                0.0,
                -1.5707963705062866,
                0.7853981852531433,
                0.46364760398864746,
                -1.5707963705062866,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    cases.append(
        MathCase(
            "arctan2_i_i",
            "i",
            arctan2(col("i"), col("i")),
            DataType.FLOAT64,
            [
                0.7853981633974483,
                -2.356194490192345,
                0.0,
                0.0,
                0.7853981633974483,
                -2.356194490192345,
                0.7853981633974483,
                0.7853981633974483,
                0.7853981633974483,
                -2.356194490192345,
            ],
            [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
    )
    return cases^


def test_math_functions_match_polars_eager_and_lazy_on_every_layout() raises:
    var original = _inputs()
    var cases = _math_cases()
    var expressions = List[Expr]()
    for k in range(len(cases)):
        expressions.append(cases[k].expression.alias("c" + String(k)))
    for layout in range(3):
        var frame = (
            original.copy() if layout
            == 0 else offset_layout(original) if layout
            == 1 else chunk_layout(original)
        )
        for lazy in [False, True]:
            var actual = (
                frame.lazy()
                .select_exprs(expressions)
                .collect() if lazy else frame.select_exprs(expressions)
            )
            for k in range(len(cases)):
                _assert_case(actual.column("c" + String(k)), cases[k])


def test_math_functions_keep_types_reject_non_numeric_and_handle_decimals() raises:
    var frame = DataFrame(
        [
            Series("u", Column[UInt8]([0, 7, 200])),
            Series("i8", Column[Int8]([-5, 0, 3])),
            Series("s", Column[String](["a", "b", "c"])),
            Series("b", Column[Bool]([True, False, True])),
            Series(
                "d",
                Column[Int128]([-275, 0, 310], [True, True, True]),
            )
            .with_dtype(DataType.decimal(38, 2))
            .cast(DataType.decimal(10, 2, 64)),
        ]
    )
    var signs = frame.select_exprs(
        [
            col("u").sign().alias("u"),
            col("i8").sign().alias("i8"),
            col("i8").truncate().alias("t"),
            col("d").sign().alias("d"),
            col("d").truncate().alias("dt"),
            col("d").log10().alias("dl"),
            col("u").sin().alias("us"),
        ]
    )
    assert_true(signs.column("u").dtype() == DataType.UINT8)
    assert_true(signs.column("i8").dtype() == DataType.INT8)
    assert_true(signs.column("t").dtype() == DataType.INT8)
    assert_true(signs.column("d").dtype() == DataType.decimal(10, 2, 64))
    assert_true(signs.column("dt").dtype() == DataType.decimal(10, 2, 64))
    assert_true(signs.column("dl").dtype() == DataType.FLOAT64)
    assert_true(signs.column("us").dtype() == DataType.FLOAT64)
    var expected_u: List[Int] = [0, 1, 1]
    var expected_i8: List[Int] = [-1, 0, 1]
    var expected_t: List[Int] = [-5, 0, 3]
    # Polars: sign of decimal(10, 2) is -1.00, 0.00, 1.00; truncate keeps
    # the scale: -2.00, 0.00, 3.00.
    var expected_d: List[Float64] = [-1, 0, 1]
    var expected_dt: List[Float64] = [-2, 0, 3]
    for row in range(3):
        assert_equal(
            Int(signs.item(row, "u").numeric[DType.uint8]()), expected_u[row]
        )
        assert_equal(
            Int(signs.item(row, "i8").numeric[DType.int8]()), expected_i8[row]
        )
        assert_equal(
            Int(signs.item(row, "t").numeric[DType.int8]()), expected_t[row]
        )
        assert_equal(
            signs.column("d").cast(DataType.FLOAT64).get(row).float64(),
            expected_d[row],
        )
        assert_equal(
            signs.column("dt").cast(DataType.FLOAT64).get(row).float64(),
            expected_dt[row],
        )
    assert_true(isnan(signs.item(0, "dl").float64()))
    assert_equal(signs.item(1, "dl").float64(), -Float64(1) / Float64(0))
    for bad in [
        col("s").sin(),
        col("s").log10(),
        col("b").sign(),
        col("b").cos(),
        arctan2(col("s"), col("u")),
    ]:
        with assert_raises():
            _ = frame.select(bad)


def test_arctan2_casts_x_to_y_float_type_and_broadcasts_scalars() raises:
    var frame = DataFrame(
        [
            Series("y", Column[Float32]([1, -1])),
            Series("x", Column[Int64]([1, -1])),
        ]
    )
    var result = frame.select_exprs(
        [
            arctan2(col("y"), col("x")).alias("a"),
            arctan2(col("x"), 2.0).alias("b"),
            arctan2(col("x"), col("x").max()).alias("c"),
        ]
    )
    assert_true(result.column("a").dtype() == DataType.FLOAT32)
    assert_true(result.column("b").dtype() == DataType.FLOAT64)
    assert_true(
        abs(Float64(result.item(1, "a").numeric[DType.float32]()) + 2.3561945)
        < 1e-6
    )
    assert_true(abs(result.item(0, "b").float64() - 0.4636476090008061) < 1e-15)
    assert_true(abs(result.item(1, "c").float64() + 0.7853981633974483) < 1e-15)


def test_product_and_cum_prod_follow_polars_types_and_null_policy() raises:
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([1, 1, 2, 2, 3, 3])),
            Series(
                "i",
                Column[Int64](
                    [2, 3, 0, 4, 0, 0], [True, True, False, True, False, False]
                ),
            ),
            Series(
                "f",
                Column[Float64](
                    [1.5, 0, 2, Float64(0) / Float64(0), 0, 0],
                    [True, False, True, True, False, False],
                ),
            ),
            Series("u", Column[UInt64]([2, 3, 1, 4, 1, 1])),
            Series("i32", Column[Int32]([2, 3, 1, 4, 1, 1])),
            Series("f32", Column[Float32]([2, 3, 1, 4, 1, 1])),
            Series("b", Column[Bool]([True, True, True, False, True, True])),
        ]
    )
    for lazy in [False, True]:
        var reductions: List[Expr] = [
            col("i").product().alias("i"),
            col("u").product().alias("u"),
            col("i32").product().alias("i32"),
            col("f32").product().alias("f32"),
            col("b").product().alias("b"),
        ]
        var whole = (
            frame.lazy()
            .select_exprs(reductions)
            .collect() if lazy else frame.select_exprs(reductions)
        )
        assert_true(whole.column("i").dtype() == DataType.INT64)
        assert_true(whole.column("u").dtype() == DataType.UINT64)
        assert_true(whole.column("i32").dtype() == DataType.INT64)
        assert_true(whole.column("f32").dtype() == DataType.FLOAT32)
        assert_true(whole.column("b").dtype() == DataType.INT64)
        assert_equal(whole.item(0, "i").int64(), 24)
        assert_equal(whole.item(0, "u").numeric[DType.uint64](), UInt64(24))
        assert_equal(whole.item(0, "i32").int64(), 24)
        assert_equal(whole.item(0, "f32").numeric[DType.float32](), 24)
        assert_equal(whole.item(0, "b").int64(), 0)
        var grouped_exprs: List[Expr] = [
            col("i").product().alias("p"),
            col("f").product().alias("fp"),
        ]
        var grouped = (
            frame.lazy()
            .group_by("g", maintain_order=True)
            .agg(grouped_exprs)
            .collect() if lazy else frame.group_by("g", maintain_order=True)
            .agg(grouped_exprs)
        )
        # Polars: an all-null group's product is 1.
        var products: List[Int64] = [6, 4, 1]
        for row in range(3):
            assert_equal(grouped.item(row, "p").int64(), products[row])
        assert_equal(grouped.item(0, "fp").float64(), 1.5)
        assert_true(isnan(grouped.item(1, "fp").float64()))
        assert_equal(grouped.item(2, "fp").float64(), 1.0)
        var windows: List[Expr] = [
            col("i").product().over("g").alias("p"),
            col("i").cum_prod().over("g").alias("cp"),
            col("i").cum_prod(reverse=True).alias("rev"),
            col("u").cum_prod().alias("u"),
            col("b").cum_prod().alias("b"),
        ]
        var spread = (
            frame.lazy()
            .with_columns(windows)
            .collect() if lazy else frame.with_columns(windows)
        )
        var over: List[Int64] = [6, 6, 4, 4, 1, 1]
        var running: List[Int64] = [2, 6, -1, 4, -1, -1]
        var reverse: List[Int64] = [24, 12, -1, 4, -1, -1]
        var unsigned: List[UInt64] = [2, 6, 6, 24, 24, 24]
        var flags: List[Int64] = [1, 1, 1, 0, 0, 0]
        assert_true(spread.column("u").dtype() == DataType.UINT64)
        assert_true(spread.column("b").dtype() == DataType.INT64)
        for row in range(6):
            assert_equal(spread.item(row, "p").int64(), over[row])
            assert_equal(spread.item(row, "cp").is_null(), running[row] < 0)
            if running[row] >= 0:
                assert_equal(spread.item(row, "cp").int64(), running[row])
            assert_equal(spread.item(row, "rev").is_null(), reverse[row] < 0)
            if reverse[row] >= 0:
                assert_equal(spread.item(row, "rev").int64(), reverse[row])
            assert_equal(
                spread.item(row, "u").numeric[DType.uint64](), unsigned[row]
            )
            assert_equal(spread.item(row, "b").int64(), flags[row])
    var empty = frame.clear().select(col("i").product())
    assert_equal(empty.item(0, "i").int64(), 1)


def test_integer_products_raise_on_overflow_and_decimals_give_float64() raises:
    # Polars wraps integer products silently; here they are checked, as
    # cum_sum is.
    var big = DataFrame(
        [
            Series("k", Column[Int64]([1, 1])),
            Series("i", Column[Int64]([1 << 40, 1 << 40])),
            Series("u", Column[UInt64]([UInt64(1) << 62, 2])),
            Series("w", Column[UInt64]([UInt64(1) << 63, 2])),
        ]
    )
    for expression in [col("i").product(), col("i").cum_prod()]:
        with assert_raises(contains="Int64 product overflow"):
            _ = big.select(expression)
    with assert_raises(contains="Int64 product overflow"):
        _ = big.group_by("k").agg([col("i").product()])
    with assert_raises(contains="Int64 product overflow"):
        _ = big.select(col("i").cum_prod().over("k"))
    assert_equal(
        big.select(col("u").product()).item(0, "u").numeric[DType.uint64](),
        UInt64(1) << 63,
    )
    with assert_raises(contains="UInt64 product overflow"):
        _ = big.select(col("w").product())
    var decimals = DataFrame(
        [
            Series("d", Column[Int128]([150, -200, 0], [True, True, False]))
            .with_dtype(DataType.decimal(38, 2))
            .cast(DataType.decimal(10, 2, 64))
        ]
    )
    # DuckDB's product of a decimal is DOUBLE.
    var product = decimals.select(col("d").product())
    assert_true(product.column("d").dtype() == DataType.FLOAT64)
    assert_equal(product.item(0, "d").float64(), -3.0)
    var running = decimals.select(col("d").cum_prod())
    assert_true(running.column("d").dtype() == DataType.FLOAT64)
    assert_equal(running.item(1, "d").float64(), -3.0)
    assert_true(running.item(2, "d").is_null())


def test_products_merge_across_workers_and_routes() raises:
    from tests.support.group_strategy_reference import ScopedEnvironment

    var n = 40_000
    var groups = List[Int64](capacity=n)
    var values = List[Int64](capacity=n)
    var negatives = List[Int](length=7, fill=0)
    for i in range(n):
        groups.append(Int64(i % 7))
        values.append(Int64(-1) if i % 3 == 0 else Int64(1))
        negatives[i % 7] += Int(i % 3 == 0)
    var frame = DataFrame(
        [
            Series("g", Column[Int64](groups^)),
            Series("x", Column[Int64](values^)),
        ]
    )
    for threads in ["1", "4"]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", threads)
        for lazy in [False, True]:
            var grouped = (
                frame.lazy()
                .group_by("g", maintain_order=True)
                .agg([col("x").product()])
                .collect() if lazy else frame.group_by("g", maintain_order=True)
                .agg([col("x").product()])
            )
            for row in range(7):
                assert_equal(
                    grouped.item(row, "x").int64(),
                    Int64(-1) if negatives[row] % 2 == 1 else Int64(1),
                )
        _ = workers^
