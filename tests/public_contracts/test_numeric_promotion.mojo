"""Mixed numeric operands promote before the operation: integers of different
widths or signedness, integers with floats, and decimals with integers or
floats. Result types follow the matrix below, checked against duckdb 2.0.0
(`typeof(1::L op 1::R)`), eager and lazy."""
# contract-group: expr.arithmetic
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    coalesce,
    col,
    lit,
    null,
    when,
)


# "left op right result" cells. "err" marks pairs that need 128-bit integers
# (UInt64 beside a signed type) and decimal `//` and `%`, which raise.
comptime _ARITHMETIC = (
    "i8 + i16 i16;i8 + i32 i32;i8 + i64 i64;i8 + u8 i64;i8 + u16 i64;i8 + u32 i64;i8 + u64 err;i8 + f32 f32;i8 + f64 f64;i8 + d4.1 d5.1;i8 + d10.2 d11.2;i8 + d38.2 d38.2;"
    "i16 + i8 i16;i16 + i32 i32;i16 + i64 i64;i16 + u8 i16;i16 + u16 i64;i16 + u32 i64;i16 + u64 err;i16 + f32 f32;i16 + f64 f64;i16 + d4.1 d7.1;i16 + d10.2 d11.2;i16 + d38.2 d38.2;"
    "i32 + i8 i32;i32 + i16 i32;i32 + i64 i64;i32 + u8 i32;i32 + u16 i32;i32 + u32 i64;i32 + u64 err;i32 + f32 f32;i32 + f64 f64;i32 + d4.1 d12.1;i32 + d10.2 d13.2;i32 + d38.2 d38.2;"
    "i64 + i8 i64;i64 + i16 i64;i64 + i32 i64;i64 + u8 i64;i64 + u16 i64;i64 + u32 i64;i64 + u64 err;i64 + f32 f32;i64 + f64 f64;i64 + d4.1 d21.1;i64 + d10.2 d22.2;i64 + d38.2 d38.2;"
    "u8 + i8 i64;u8 + i16 i16;u8 + i32 i32;u8 + i64 i64;u8 + u16 u16;u8 + u32 u32;u8 + u64 u64;u8 + f32 f32;u8 + f64 f64;u8 + d4.1 d5.1;u8 + d10.2 d11.2;u8 + d38.2 d38.2;"
    "u16 + i8 i64;u16 + i16 i64;u16 + i32 i32;u16 + i64 i64;u16 + u8 u16;u16 + u32 u32;u16 + u64 u64;u16 + f32 f32;u16 + f64 f64;u16 + d4.1 d7.1;u16 + d10.2 d11.2;u16 + d38.2 d38.2;"
    "u32 + i8 i64;u32 + i16 i64;u32 + i32 i64;u32 + i64 i64;u32 + u8 u32;u32 + u16 u32;u32 + u64 u64;u32 + f32 f32;u32 + f64 f64;u32 + d4.1 d12.1;u32 + d10.2 d13.2;u32 + d38.2 d38.2;"
    "u64 + i8 err;u64 + i16 err;u64 + i32 err;u64 + i64 err;u64 + u8 u64;u64 + u16 u64;u64 + u32 u64;u64 + f32 f32;u64 + f64 f64;u64 + d4.1 d22.1;u64 + d10.2 d23.2;u64 + d38.2 d38.2;"
    "f32 + i8 f32;f32 + i16 f32;f32 + i32 f32;f32 + i64 f32;f32 + u8 f32;f32 + u16 f32;f32 + u32 f32;f32 + u64 f32;f32 + f64 f64;f32 + d4.1 f32;f32 + d10.2 f32;f32 + d38.2 f32;"
    "f64 + i8 f64;f64 + i16 f64;f64 + i32 f64;f64 + i64 f64;f64 + u8 f64;f64 + u16 f64;f64 + u32 f64;f64 + u64 f64;f64 + f32 f64;f64 + d4.1 f64;f64 + d10.2 f64;f64 + d38.2 f64;"
    "d4.1 + i8 d5.1;d4.1 + i16 d7.1;d4.1 + i32 d12.1;d4.1 + i64 d21.1;d4.1 + u8 d5.1;d4.1 + u16 d7.1;d4.1 + u32 d12.1;d4.1 + u64 d22.1;d4.1 + f32 f32;d4.1 + f64 f64;d4.1 + d10.2 d11.2;d4.1 + d38.2 d38.2;"
    "d10.2 + i8 d11.2;d10.2 + i16 d11.2;d10.2 + i32 d13.2;d10.2 + i64 d22.2;d10.2 + u8 d11.2;d10.2 + u16 d11.2;d10.2 + u32 d13.2;d10.2 + u64 d23.2;d10.2 + f32 f32;d10.2 + f64 f64;d10.2 + d4.1 d11.2;d10.2 + d38.2 d38.2;"
    "d38.2 + i8 d38.2;d38.2 + i16 d38.2;d38.2 + i32 d38.2;d38.2 + i64 d38.2;d38.2 + u8 d38.2;d38.2 + u16 d38.2;d38.2 + u32 d38.2;d38.2 + u64 d38.2;d38.2 + f32 f32;d38.2 + f64 f64;d38.2 + d4.1 d38.2;d38.2 + d10.2 d38.2;"
    "i8 - i16 i16;i8 - i32 i32;i8 - i64 i64;i8 - u8 i64;i8 - u16 i64;i8 - u32 i64;i8 - u64 err;i8 - f32 f32;i8 - f64 f64;i8 - d4.1 d5.1;i8 - d10.2 d11.2;i8 - d38.2 d38.2;"
    "i16 - i8 i16;i16 - i32 i32;i16 - i64 i64;i16 - u8 i16;i16 - u16 i64;i16 - u32 i64;i16 - u64 err;i16 - f32 f32;i16 - f64 f64;i16 - d4.1 d7.1;i16 - d10.2 d11.2;i16 - d38.2 d38.2;"
    "i32 - i8 i32;i32 - i16 i32;i32 - i64 i64;i32 - u8 i32;i32 - u16 i32;i32 - u32 i64;i32 - u64 err;i32 - f32 f32;i32 - f64 f64;i32 - d4.1 d12.1;i32 - d10.2 d13.2;i32 - d38.2 d38.2;"
    "i64 - i8 i64;i64 - i16 i64;i64 - i32 i64;i64 - u8 i64;i64 - u16 i64;i64 - u32 i64;i64 - u64 err;i64 - f32 f32;i64 - f64 f64;i64 - d4.1 d21.1;i64 - d10.2 d22.2;i64 - d38.2 d38.2;"
    "u8 - i8 i64;u8 - i16 i16;u8 - i32 i32;u8 - i64 i64;u8 - u16 u16;u8 - u32 u32;u8 - u64 u64;u8 - f32 f32;u8 - f64 f64;u8 - d4.1 d5.1;u8 - d10.2 d11.2;u8 - d38.2 d38.2;"
    "u16 - i8 i64;u16 - i16 i64;u16 - i32 i32;u16 - i64 i64;u16 - u8 u16;u16 - u32 u32;u16 - u64 u64;u16 - f32 f32;u16 - f64 f64;u16 - d4.1 d7.1;u16 - d10.2 d11.2;u16 - d38.2 d38.2;"
    "u32 - i8 i64;u32 - i16 i64;u32 - i32 i64;u32 - i64 i64;u32 - u8 u32;u32 - u16 u32;u32 - u64 u64;u32 - f32 f32;u32 - f64 f64;u32 - d4.1 d12.1;u32 - d10.2 d13.2;u32 - d38.2 d38.2;"
    "u64 - i8 err;u64 - i16 err;u64 - i32 err;u64 - i64 err;u64 - u8 u64;u64 - u16 u64;u64 - u32 u64;u64 - f32 f32;u64 - f64 f64;u64 - d4.1 d22.1;u64 - d10.2 d23.2;u64 - d38.2 d38.2;"
    "f32 - i8 f32;f32 - i16 f32;f32 - i32 f32;f32 - i64 f32;f32 - u8 f32;f32 - u16 f32;f32 - u32 f32;f32 - u64 f32;f32 - f64 f64;f32 - d4.1 f32;f32 - d10.2 f32;f32 - d38.2 f32;"
    "f64 - i8 f64;f64 - i16 f64;f64 - i32 f64;f64 - i64 f64;f64 - u8 f64;f64 - u16 f64;f64 - u32 f64;f64 - u64 f64;f64 - f32 f64;f64 - d4.1 f64;f64 - d10.2 f64;f64 - d38.2 f64;"
    "d4.1 - i8 d5.1;d4.1 - i16 d7.1;d4.1 - i32 d12.1;d4.1 - i64 d21.1;d4.1 - u8 d5.1;d4.1 - u16 d7.1;d4.1 - u32 d12.1;d4.1 - u64 d22.1;d4.1 - f32 f32;d4.1 - f64 f64;d4.1 - d10.2 d11.2;d4.1 - d38.2 d38.2;"
    "d10.2 - i8 d11.2;d10.2 - i16 d11.2;d10.2 - i32 d13.2;d10.2 - i64 d22.2;d10.2 - u8 d11.2;d10.2 - u16 d11.2;d10.2 - u32 d13.2;d10.2 - u64 d23.2;d10.2 - f32 f32;d10.2 - f64 f64;d10.2 - d4.1 d11.2;d10.2 - d38.2 d38.2;"
    "d38.2 - i8 d38.2;d38.2 - i16 d38.2;d38.2 - i32 d38.2;d38.2 - i64 d38.2;d38.2 - u8 d38.2;d38.2 - u16 d38.2;d38.2 - u32 d38.2;d38.2 - u64 d38.2;d38.2 - f32 f32;d38.2 - f64 f64;d38.2 - d4.1 d38.2;d38.2 - d10.2 d38.2;"
    "i8 * i16 i16;i8 * i32 i32;i8 * i64 i64;i8 * u8 i64;i8 * u16 i64;i8 * u32 i64;i8 * u64 err;i8 * f32 f32;i8 * f64 f64;i8 * d4.1 d7.1;i8 * d10.2 d13.2;i8 * d38.2 d38.2;"
    "i16 * i8 i16;i16 * i32 i32;i16 * i64 i64;i16 * u8 i16;i16 * u16 i64;i16 * u32 i64;i16 * u64 err;i16 * f32 f32;i16 * f64 f64;i16 * d4.1 d9.1;i16 * d10.2 d15.2;i16 * d38.2 d38.2;"
    "i32 * i8 i32;i32 * i16 i32;i32 * i64 i64;i32 * u8 i32;i32 * u16 i32;i32 * u32 i64;i32 * u64 err;i32 * f32 f32;i32 * f64 f64;i32 * d4.1 d14.1;i32 * d10.2 d18.2;i32 * d38.2 d38.2;"
    "i64 * i8 i64;i64 * i16 i64;i64 * i32 i64;i64 * u8 i64;i64 * u16 i64;i64 * u32 i64;i64 * u64 err;i64 * f32 f32;i64 * f64 f64;i64 * d4.1 d23.1;i64 * d10.2 d29.2;i64 * d38.2 d38.2;"
    "u8 * i8 i64;u8 * i16 i16;u8 * i32 i32;u8 * i64 i64;u8 * u16 u16;u8 * u32 u32;u8 * u64 u64;u8 * f32 f32;u8 * f64 f64;u8 * d4.1 d7.1;u8 * d10.2 d13.2;u8 * d38.2 d38.2;"
    "u16 * i8 i64;u16 * i16 i64;u16 * i32 i32;u16 * i64 i64;u16 * u8 u16;u16 * u32 u32;u16 * u64 u64;u16 * f32 f32;u16 * f64 f64;u16 * d4.1 d9.1;u16 * d10.2 d15.2;u16 * d38.2 d38.2;"
    "u32 * i8 i64;u32 * i16 i64;u32 * i32 i64;u32 * i64 i64;u32 * u8 u32;u32 * u16 u32;u32 * u64 u64;u32 * f32 f32;u32 * f64 f64;u32 * d4.1 d14.1;u32 * d10.2 d18.2;u32 * d38.2 d38.2;"
    "u64 * i8 err;u64 * i16 err;u64 * i32 err;u64 * i64 err;u64 * u8 u64;u64 * u16 u64;u64 * u32 u64;u64 * f32 f32;u64 * f64 f64;u64 * d4.1 d24.1;u64 * d10.2 d30.2;u64 * d38.2 d38.2;"
    "f32 * i8 f32;f32 * i16 f32;f32 * i32 f32;f32 * i64 f32;f32 * u8 f32;f32 * u16 f32;f32 * u32 f32;f32 * u64 f32;f32 * f64 f64;f32 * d4.1 f32;f32 * d10.2 f32;f32 * d38.2 f32;"
    "f64 * i8 f64;f64 * i16 f64;f64 * i32 f64;f64 * i64 f64;f64 * u8 f64;f64 * u16 f64;f64 * u32 f64;f64 * u64 f64;f64 * f32 f64;f64 * d4.1 f64;f64 * d10.2 f64;f64 * d38.2 f64;"
    "d4.1 * i8 d7.1;d4.1 * i16 d9.1;d4.1 * i32 d14.1;d4.1 * i64 d23.1;d4.1 * u8 d7.1;d4.1 * u16 d9.1;d4.1 * u32 d14.1;d4.1 * u64 d24.1;d4.1 * f32 f32;d4.1 * f64 f64;d4.1 * d10.2 d14.3;d4.1 * d38.2 d38.3;"
    "d10.2 * i8 d13.2;d10.2 * i16 d15.2;d10.2 * i32 d18.2;d10.2 * i64 d29.2;d10.2 * u8 d13.2;d10.2 * u16 d15.2;d10.2 * u32 d18.2;d10.2 * u64 d30.2;d10.2 * f32 f32;d10.2 * f64 f64;d10.2 * d4.1 d14.3;d10.2 * d38.2 d38.4;"
    "d38.2 * i8 d38.2;d38.2 * i16 d38.2;d38.2 * i32 d38.2;d38.2 * i64 d38.2;d38.2 * u8 d38.2;d38.2 * u16 d38.2;d38.2 * u32 d38.2;d38.2 * u64 d38.2;d38.2 * f32 f32;d38.2 * f64 f64;d38.2 * d4.1 d38.3;d38.2 * d10.2 d38.4;"
    "i8 / i16 f64;i8 / i32 f64;i8 / i64 f64;i8 / u8 f64;i8 / u16 f64;i8 / u32 f64;i8 / u64 f64;i8 / f32 f32;i8 / f64 f64;i8 / d4.1 f64;i8 / d10.2 f64;i8 / d38.2 f64;"
    "i16 / i8 f64;i16 / i32 f64;i16 / i64 f64;i16 / u8 f64;i16 / u16 f64;i16 / u32 f64;i16 / u64 f64;i16 / f32 f32;i16 / f64 f64;i16 / d4.1 f64;i16 / d10.2 f64;i16 / d38.2 f64;"
    "i32 / i8 f64;i32 / i16 f64;i32 / i64 f64;i32 / u8 f64;i32 / u16 f64;i32 / u32 f64;i32 / u64 f64;i32 / f32 f32;i32 / f64 f64;i32 / d4.1 f64;i32 / d10.2 f64;i32 / d38.2 f64;"
    "i64 / i8 f64;i64 / i16 f64;i64 / i32 f64;i64 / u8 f64;i64 / u16 f64;i64 / u32 f64;i64 / u64 f64;i64 / f32 f32;i64 / f64 f64;i64 / d4.1 f64;i64 / d10.2 f64;i64 / d38.2 f64;"
    "u8 / i8 f64;u8 / i16 f64;u8 / i32 f64;u8 / i64 f64;u8 / u16 f64;u8 / u32 f64;u8 / u64 f64;u8 / f32 f32;u8 / f64 f64;u8 / d4.1 f64;u8 / d10.2 f64;u8 / d38.2 f64;"
    "u16 / i8 f64;u16 / i16 f64;u16 / i32 f64;u16 / i64 f64;u16 / u8 f64;u16 / u32 f64;u16 / u64 f64;u16 / f32 f32;u16 / f64 f64;u16 / d4.1 f64;u16 / d10.2 f64;u16 / d38.2 f64;"
    "u32 / i8 f64;u32 / i16 f64;u32 / i32 f64;u32 / i64 f64;u32 / u8 f64;u32 / u16 f64;u32 / u64 f64;u32 / f32 f32;u32 / f64 f64;u32 / d4.1 f64;u32 / d10.2 f64;u32 / d38.2 f64;"
    "u64 / i8 f64;u64 / i16 f64;u64 / i32 f64;u64 / i64 f64;u64 / u8 f64;u64 / u16 f64;u64 / u32 f64;u64 / f32 f32;u64 / f64 f64;u64 / d4.1 f64;u64 / d10.2 f64;u64 / d38.2 f64;"
    "f32 / i8 f32;f32 / i16 f32;f32 / i32 f32;f32 / i64 f32;f32 / u8 f32;f32 / u16 f32;f32 / u32 f32;f32 / u64 f32;f32 / f64 f64;f32 / d4.1 f32;f32 / d10.2 f32;f32 / d38.2 f32;"
    "f64 / i8 f64;f64 / i16 f64;f64 / i32 f64;f64 / i64 f64;f64 / u8 f64;f64 / u16 f64;f64 / u32 f64;f64 / u64 f64;f64 / f32 f64;f64 / d4.1 f64;f64 / d10.2 f64;f64 / d38.2 f64;"
    "d4.1 / i8 f64;d4.1 / i16 f64;d4.1 / i32 f64;d4.1 / i64 f64;d4.1 / u8 f64;d4.1 / u16 f64;d4.1 / u32 f64;d4.1 / u64 f64;d4.1 / f32 f32;d4.1 / f64 f64;d4.1 / d10.2 f64;d4.1 / d38.2 f64;"
    "d10.2 / i8 f64;d10.2 / i16 f64;d10.2 / i32 f64;d10.2 / i64 f64;d10.2 / u8 f64;d10.2 / u16 f64;d10.2 / u32 f64;d10.2 / u64 f64;d10.2 / f32 f32;d10.2 / f64 f64;d10.2 / d4.1 f64;d10.2 / d38.2 f64;"
    "d38.2 / i8 f64;d38.2 / i16 f64;d38.2 / i32 f64;d38.2 / i64 f64;d38.2 / u8 f64;d38.2 / u16 f64;d38.2 / u32 f64;d38.2 / u64 f64;d38.2 / f32 f32;d38.2 / f64 f64;d38.2 / d4.1 f64;d38.2 / d10.2 f64;"
    "i8 // i16 i16;i8 // i32 i32;i8 // i64 i64;i8 // u8 i64;i8 // u16 i64;i8 // u32 i64;i8 // u64 err;i8 // f32 f32;i8 // f64 f64;i8 // d4.1 err;i8 // d10.2 err;i8 // d38.2 err;"
    "i16 // i8 i16;i16 // i32 i32;i16 // i64 i64;i16 // u8 i16;i16 // u16 i64;i16 // u32 i64;i16 // u64 err;i16 // f32 f32;i16 // f64 f64;i16 // d4.1 err;i16 // d10.2 err;i16 // d38.2 err;"
    "i32 // i8 i32;i32 // i16 i32;i32 // i64 i64;i32 // u8 i32;i32 // u16 i32;i32 // u32 i64;i32 // u64 err;i32 // f32 f32;i32 // f64 f64;i32 // d4.1 err;i32 // d10.2 err;i32 // d38.2 err;"
    "i64 // i8 i64;i64 // i16 i64;i64 // i32 i64;i64 // u8 i64;i64 // u16 i64;i64 // u32 i64;i64 // u64 err;i64 // f32 f32;i64 // f64 f64;i64 // d4.1 err;i64 // d10.2 err;i64 // d38.2 err;"
    "u8 // i8 i64;u8 // i16 i16;u8 // i32 i32;u8 // i64 i64;u8 // u16 u16;u8 // u32 u32;u8 // u64 u64;u8 // f32 f32;u8 // f64 f64;u8 // d4.1 err;u8 // d10.2 err;u8 // d38.2 err;"
    "u16 // i8 i64;u16 // i16 i64;u16 // i32 i32;u16 // i64 i64;u16 // u8 u16;u16 // u32 u32;u16 // u64 u64;u16 // f32 f32;u16 // f64 f64;u16 // d4.1 err;u16 // d10.2 err;u16 // d38.2 err;"
    "u32 // i8 i64;u32 // i16 i64;u32 // i32 i64;u32 // i64 i64;u32 // u8 u32;u32 // u16 u32;u32 // u64 u64;u32 // f32 f32;u32 // f64 f64;u32 // d4.1 err;u32 // d10.2 err;u32 // d38.2 err;"
    "u64 // i8 err;u64 // i16 err;u64 // i32 err;u64 // i64 err;u64 // u8 u64;u64 // u16 u64;u64 // u32 u64;u64 // f32 f32;u64 // f64 f64;u64 // d4.1 err;u64 // d10.2 err;u64 // d38.2 err;"
    "f32 // i8 f32;f32 // i16 f32;f32 // i32 f32;f32 // i64 f32;f32 // u8 f32;f32 // u16 f32;f32 // u32 f32;f32 // u64 f32;f32 // f64 f64;f32 // d4.1 f32;f32 // d10.2 f32;f32 // d38.2 f32;"
    "f64 // i8 f64;f64 // i16 f64;f64 // i32 f64;f64 // i64 f64;f64 // u8 f64;f64 // u16 f64;f64 // u32 f64;f64 // u64 f64;f64 // f32 f64;f64 // d4.1 f64;f64 // d10.2 f64;f64 // d38.2 f64;"
    "d4.1 // i8 err;d4.1 // i16 err;d4.1 // i32 err;d4.1 // i64 err;d4.1 // u8 err;d4.1 // u16 err;d4.1 // u32 err;d4.1 // u64 err;d4.1 // f32 f32;d4.1 // f64 f64;d4.1 // d10.2 err;d4.1 // d38.2 err;"
    "d10.2 // i8 err;d10.2 // i16 err;d10.2 // i32 err;d10.2 // i64 err;d10.2 // u8 err;d10.2 // u16 err;d10.2 // u32 err;d10.2 // u64 err;d10.2 // f32 f32;d10.2 // f64 f64;d10.2 // d4.1 err;d10.2 // d38.2 err;"
    "d38.2 // i8 err;d38.2 // i16 err;d38.2 // i32 err;d38.2 // i64 err;d38.2 // u8 err;d38.2 // u16 err;d38.2 // u32 err;d38.2 // u64 err;d38.2 // f32 f32;d38.2 // f64 f64;d38.2 // d4.1 err;d38.2 // d10.2 err;"
    "i8 % i16 i16;i8 % i32 i32;i8 % i64 i64;i8 % u8 i64;i8 % u16 i64;i8 % u32 i64;i8 % u64 err;i8 % f32 f32;i8 % f64 f64;i8 % d4.1 err;i8 % d10.2 err;i8 % d38.2 err;"
    "i16 % i8 i16;i16 % i32 i32;i16 % i64 i64;i16 % u8 i16;i16 % u16 i64;i16 % u32 i64;i16 % u64 err;i16 % f32 f32;i16 % f64 f64;i16 % d4.1 err;i16 % d10.2 err;i16 % d38.2 err;"
    "i32 % i8 i32;i32 % i16 i32;i32 % i64 i64;i32 % u8 i32;i32 % u16 i32;i32 % u32 i64;i32 % u64 err;i32 % f32 f32;i32 % f64 f64;i32 % d4.1 err;i32 % d10.2 err;i32 % d38.2 err;"
    "i64 % i8 i64;i64 % i16 i64;i64 % i32 i64;i64 % u8 i64;i64 % u16 i64;i64 % u32 i64;i64 % u64 err;i64 % f32 f32;i64 % f64 f64;i64 % d4.1 err;i64 % d10.2 err;i64 % d38.2 err;"
    "u8 % i8 i64;u8 % i16 i16;u8 % i32 i32;u8 % i64 i64;u8 % u16 u16;u8 % u32 u32;u8 % u64 u64;u8 % f32 f32;u8 % f64 f64;u8 % d4.1 err;u8 % d10.2 err;u8 % d38.2 err;"
    "u16 % i8 i64;u16 % i16 i64;u16 % i32 i32;u16 % i64 i64;u16 % u8 u16;u16 % u32 u32;u16 % u64 u64;u16 % f32 f32;u16 % f64 f64;u16 % d4.1 err;u16 % d10.2 err;u16 % d38.2 err;"
    "u32 % i8 i64;u32 % i16 i64;u32 % i32 i64;u32 % i64 i64;u32 % u8 u32;u32 % u16 u32;u32 % u64 u64;u32 % f32 f32;u32 % f64 f64;u32 % d4.1 err;u32 % d10.2 err;u32 % d38.2 err;"
    "u64 % i8 err;u64 % i16 err;u64 % i32 err;u64 % i64 err;u64 % u8 u64;u64 % u16 u64;u64 % u32 u64;u64 % f32 f32;u64 % f64 f64;u64 % d4.1 err;u64 % d10.2 err;u64 % d38.2 err;"
    "f32 % i8 f32;f32 % i16 f32;f32 % i32 f32;f32 % i64 f32;f32 % u8 f32;f32 % u16 f32;f32 % u32 f32;f32 % u64 f32;f32 % f64 f64;f32 % d4.1 f32;f32 % d10.2 f32;f32 % d38.2 f32;"
    "f64 % i8 f64;f64 % i16 f64;f64 % i32 f64;f64 % i64 f64;f64 % u8 f64;f64 % u16 f64;f64 % u32 f64;f64 % u64 f64;f64 % f32 f64;f64 % d4.1 f64;f64 % d10.2 f64;f64 % d38.2 f64;"
    "d4.1 % i8 err;d4.1 % i16 err;d4.1 % i32 err;d4.1 % i64 err;d4.1 % u8 err;d4.1 % u16 err;d4.1 % u32 err;d4.1 % u64 err;d4.1 % f32 f32;d4.1 % f64 f64;d4.1 % d10.2 err;d4.1 % d38.2 err;"
    "d10.2 % i8 err;d10.2 % i16 err;d10.2 % i32 err;d10.2 % i64 err;d10.2 % u8 err;d10.2 % u16 err;d10.2 % u32 err;d10.2 % u64 err;d10.2 % f32 f32;d10.2 % f64 f64;d10.2 % d4.1 err;d10.2 % d38.2 err;"
    "d38.2 % i8 err;d38.2 % i16 err;d38.2 % i32 err;d38.2 % i64 err;d38.2 % u8 err;d38.2 % u16 err;d38.2 % u32 err;d38.2 % u64 err;d38.2 % f32 f32;d38.2 % f64 f64;d38.2 % d4.1 err;d38.2 % d10.2 err;"
)


# The same for conditional branches, fill values and coalesce
# (`typeof(CASE WHEN ... THEN 1::L ELSE 1::R END)`): the narrowest type that
# holds both inputs.
comptime _BRANCHES = (
    "i8 case i16 i16;i8 case i32 i32;i8 case i64 i64;i8 case u8 i16;i8 case u16 i32;i8 case u32 i64;i8 case u64 err;i8 case f32 f32;i8 case f64 f64;i8 case d4.1 d4.1;i8 case d10.2 d10.2;i8 case d38.2 d38.2;"
    "i16 case i8 i16;i16 case i32 i32;i16 case i64 i64;i16 case u8 i16;i16 case u16 i32;i16 case u32 i64;i16 case u64 err;i16 case f32 f32;i16 case f64 f64;i16 case d4.1 d6.1;i16 case d10.2 d10.2;i16 case d38.2 d38.2;"
    "i32 case i8 i32;i32 case i16 i32;i32 case i64 i64;i32 case u8 i32;i32 case u16 i32;i32 case u32 i64;i32 case u64 err;i32 case f32 f32;i32 case f64 f64;i32 case d4.1 d11.1;i32 case d10.2 d12.2;i32 case d38.2 d38.2;"
    "i64 case i8 i64;i64 case i16 i64;i64 case i32 i64;i64 case u8 i64;i64 case u16 i64;i64 case u32 i64;i64 case u64 err;i64 case f32 f32;i64 case f64 f64;i64 case d4.1 d20.1;i64 case d10.2 d21.2;i64 case d38.2 d38.2;"
    "u8 case i8 i16;u8 case i16 i16;u8 case i32 i32;u8 case i64 i64;u8 case u16 u16;u8 case u32 u32;u8 case u64 u64;u8 case f32 f32;u8 case f64 f64;u8 case d4.1 d4.1;u8 case d10.2 d10.2;u8 case d38.2 d38.2;"
    "u16 case i8 i32;u16 case i16 i32;u16 case i32 i32;u16 case i64 i64;u16 case u8 u16;u16 case u32 u32;u16 case u64 u64;u16 case f32 f32;u16 case f64 f64;u16 case d4.1 d6.1;u16 case d10.2 d10.2;u16 case d38.2 d38.2;"
    "u32 case i8 i64;u32 case i16 i64;u32 case i32 i64;u32 case i64 i64;u32 case u8 u32;u32 case u16 u32;u32 case u64 u64;u32 case f32 f32;u32 case f64 f64;u32 case d4.1 d11.1;u32 case d10.2 d12.2;u32 case d38.2 d38.2;"
    "u64 case i8 err;u64 case i16 err;u64 case i32 err;u64 case i64 err;u64 case u8 u64;u64 case u16 u64;u64 case u32 u64;u64 case f32 f32;u64 case f64 f64;u64 case d4.1 d21.1;u64 case d10.2 d22.2;u64 case d38.2 d38.2;"
    "f32 case i8 f32;f32 case i16 f32;f32 case i32 f32;f32 case i64 f32;f32 case u8 f32;f32 case u16 f32;f32 case u32 f32;f32 case u64 f32;f32 case f64 f64;f32 case d4.1 f32;f32 case d10.2 f32;f32 case d38.2 f32;"
    "f64 case i8 f64;f64 case i16 f64;f64 case i32 f64;f64 case i64 f64;f64 case u8 f64;f64 case u16 f64;f64 case u32 f64;f64 case u64 f64;f64 case f32 f64;f64 case d4.1 f64;f64 case d10.2 f64;f64 case d38.2 f64;"
    "d4.1 case i8 d4.1;d4.1 case i16 d6.1;d4.1 case i32 d11.1;d4.1 case i64 d20.1;d4.1 case u8 d4.1;d4.1 case u16 d6.1;d4.1 case u32 d11.1;d4.1 case u64 d21.1;d4.1 case f32 f32;d4.1 case f64 f64;d4.1 case d10.2 d10.2;d4.1 case d38.2 d38.2;"
    "d10.2 case i8 d10.2;d10.2 case i16 d10.2;d10.2 case i32 d12.2;d10.2 case i64 d21.2;d10.2 case u8 d10.2;d10.2 case u16 d10.2;d10.2 case u32 d12.2;d10.2 case u64 d22.2;d10.2 case f32 f32;d10.2 case f64 f64;d10.2 case d4.1 d10.2;d10.2 case d38.2 d38.2;"
    "d38.2 case i8 d38.2;d38.2 case i16 d38.2;d38.2 case i32 d38.2;d38.2 case i64 d38.2;d38.2 case u8 d38.2;d38.2 case u16 d38.2;d38.2 case u32 d38.2;d38.2 case u64 d38.2;d38.2 case f32 f32;d38.2 case f64 f64;d38.2 case d4.1 d38.2;d38.2 case d10.2 d38.2;"
)


def _types() -> List[String]:
    return [
        "i8",
        "i16",
        "i32",
        "i64",
        "u8",
        "u16",
        "u32",
        "u64",
        "f32",
        "f64",
        "d4.1",
        "d10.2",
        "d38.2",
    ]


def _short(dtype: DataType) raises -> String:
    """A dtype as the matrix spells it: i64, u8, f32, d10.2, bool."""
    if dtype.is_decimal():
        return "d" + String(dtype.precision()) + "." + String(dtype.scale())
    if dtype == DataType.BOOL:
        return "bool"
    var name = dtype.name()
    if name.startswith("uint"):
        return "u" + String(name[byte=4:])
    if name.startswith("int"):
        return "i" + String(name[byte=3:])
    return "f" + String(name[byte=5:])


def _column(name: String, short: String, value: Int) raises -> Series:
    """A one-row column of the type `short` holding `value`."""
    return _column(name, short, [value], [True])


def _column(
    name: String, short: String, values: List[Int], valid: List[Bool]
) raises -> Series:
    """A column of the type `short` holding `values`, null where not
    `valid`."""
    var rows = List[Int128]()
    for value in values:
        rows.append(Int128(value))
    var scale = 0
    var precision = 38
    if short.startswith("d"):
        var parts = String(short[byte=1:]).split(".")
        precision = Int(parts[0])
        scale = Int(parts[1])
        for k in range(len(rows)):
            for _ in range(scale):
                rows[k] *= 10
    var raw = Series(name, Column[Int128](rows^, valid.copy())).with_dtype(
        DataType.decimal(38, scale)
    )
    if short.startswith("d"):
        return raw.cast(DataType.decimal(precision, scale))
    return raw.cast(_dtype(short))


def _dtype(short: String) raises -> DataType:
    var names: List[String] = [
        "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64"
    ]
    var types: List[DataType] = [
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.INT64,
        DataType.UINT8,
        DataType.UINT16,
        DataType.UINT32,
        DataType.UINT64,
        DataType.FLOAT32,
        DataType.FLOAT64,
    ]
    for k in range(len(names)):
        if names[k] == short:
            return types[k]
    raise Error("unknown type " + short)


def _apply(op: String, left: Expr, right: Expr) raises -> Expr:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right
    if op == "//":
        return left // right
    if op == "%":
        return left % right
    if op == "==":
        return left == right
    if op == "<":
        return left < right
    raise Error("unknown operator " + op)


def _result(frame: DataFrame, expr: Expr, lazy: Bool) raises -> DataFrame:
    if lazy:
        return frame.lazy().select(expr).collect()
    return frame.select(expr)


def test_arithmetic_result_types_match_the_matrix_eager_and_lazy() raises:
    var cells = String(_ARITHMETIC).split(";")
    var checked = 0
    for cell in cells:
        if cell.byte_length() == 0:
            continue
        var parts = String(cell).split(" ")
        var left = String(parts[0])
        var op = String(parts[1])
        var right = String(parts[2])
        var expected = String(parts[3])
        var frame = DataFrame([_column("a", left, 7), _column("b", right, 2)])
        var expr = _apply(op, col("a"), col("b")).alias("x")
        for lazy in [False, True]:
            if expected == "err":
                with assert_raises():
                    _ = _result(frame, expr, lazy)
                continue
            var actual = _result(frame, expr, lazy)
            var found = _short(actual.column("x").dtype())
            if found != expected:
                raise Error(
                    left
                    + " "
                    + op
                    + " "
                    + right
                    + ": expected "
                    + expected
                    + ", found "
                    + found
                    + (" (lazy)" if lazy else " (eager)")
                )
        checked += 1
    assert_equal(checked, 13 * 12 * 6)


def test_comparisons_of_mixed_types_compare_values_exactly() raises:
    # Every pair compares by value, whatever the widths: no wraparound of
    # negative values against unsigned ones, and decimals at their scale.
    for left in _types():
        for right in _types():
            if left == right:
                continue
            var signed_left = left.startswith("i") or left.startswith("d")
            var signed_right = right.startswith("i") or right.startswith("d")
            if (
                left == "u64" and signed_right and not right.startswith("d")
            ) or (right == "u64" and signed_left and not left.startswith("d")):
                continue
            var a = 3 if left.startswith("u") else -3
            var frame = DataFrame(
                [_column("a", left, a), _column("b", right, 2)]
            )
            for lazy in [False, True]:
                var actual = _result(
                    frame,
                    _apply("<", col("a"), col("b")).alias("lt"),
                    lazy,
                )
                assert_true(actual.column("lt").dtype() == DataType.BOOL)
                assert_equal(actual.item(0, "lt").bool(), a < 2)
                var equal = _result(
                    frame,
                    _apply("==", col("a"), col("b")).alias("eq"),
                    lazy,
                )
                assert_true(not equal.item(0, "eq").bool())


def test_promotion_computes_values_at_the_promoted_width() raises:
    var frame = DataFrame(
        [
            Series("i8", Column[Int8]([127, -128])),
            Series("u8", Column[UInt8]([255, 1])),
            Series("i64", Column[Int64]([9223372036854775807, -5])),
            Series("u32", Column[UInt32]([0, 4000000000])),
            Series("f32", Column[Float32]([0.5, 1.5])),
            Series("d", Column[Int128]([125, -250])).with_dtype(
                DataType.decimal(10, 2)
            ),
        ]
    )
    var exprs: List[Expr] = [
        (col("i8") + col("u8")).alias("small"),
        (col("i64") + col("u32")).alias("wide"),
        (col("i64") * col("f32")).alias("float"),
        (col("d") + col("i64").filter(col("i64") < 0).first()).alias("dec"),
        (col("d") * col("u8")).alias("dec_mul"),
        (col("u32") > col("i8")).alias("cmp"),
    ]
    for lazy in [False, True]:
        var out = (
            frame.lazy()
            .select_exprs(exprs)
            .collect() if lazy else (frame.select_exprs(exprs))
        )
        # Int8 + UInt8 computes in Int64, so 127 + 255 does not overflow.
        assert_equal(_short(out.column("small").dtype()), "i64")
        assert_equal(out.item(0, "small").numeric[DType.int64](), 382)
        assert_equal(out.item(1, "small").numeric[DType.int64](), -127)
        assert_equal(_short(out.column("wide").dtype()), "i64")
        assert_equal(out.item(1, "wide").numeric[DType.int64](), 3999999995)
        assert_equal(_short(out.column("float").dtype()), "f32")
        assert_equal(_short(out.column("dec").dtype()), "d22.2")
        assert_equal(out.column("dec").rechunk()._decimal_raw(0), Int128(-375))
        assert_equal(_short(out.column("dec_mul").dtype()), "d13.2")
        assert_equal(
            out.column("dec_mul").rechunk()._decimal_raw(0), Int128(31875)
        )
        assert_true(out.item(0, "cmp").bool() == False)
        assert_true(out.item(1, "cmp").bool())


def test_promotion_keeps_the_overflow_error_at_the_promoted_width() raises:
    var frame = DataFrame(
        [
            Series("i64", Column[Int64]([9223372036854775807])),
            Series("u32", Column[UInt32]([1])),
            Series("i16", Column[Int16]([32767])),
            Series("u8", Column[UInt8]([1])),
        ]
    )
    for lazy in [False, True]:
        with assert_raises(contains="overflow"):
            _ = _result(frame, (col("i64") + col("u32")).alias("x"), lazy)
        # Int16 holds UInt8, so the sum stays Int16 and overflows there.
        with assert_raises(contains="int16 addition overflow"):
            _ = _result(frame, (col("i16") + col("u8")).alias("x"), lazy)


def test_bare_literals_take_the_column_type_when_they_fit() raises:
    var frame = DataFrame(
        [
            Series("u8", Column[UInt8]([200])),
            Series("i64", Column[Int64]([3])),
            Series("d", Column[Int128]([125])).with_dtype(
                DataType.decimal(10, 2)
            ),
        ]
    )
    for lazy in [False, True]:
        var fits = _result(frame, (col("u8") + 1).alias("x"), lazy)
        assert_equal(_short(fits.column("x").dtype()), "u8")
        # 300 does not fit UInt8: it is an Int32 literal, and the sum
        # computes in Int32.
        var wide = _result(frame, (col("u8") + 300).alias("x"), lazy)
        assert_equal(_short(wide.column("x").dtype()), "i32")
        assert_equal(wide.item(0, "x").numeric[DType.int32](), 500)
        var big = _result(frame, (col("u8") + 5000000000).alias("x"), lazy)
        assert_equal(_short(big.column("x").dtype()), "i64")
        var compared = _result(frame, (col("u8") > 300).alias("x"), lazy)
        assert_true(not compared.item(0, "x").bool())
        var above = _result(frame, (col("d") > 1).alias("x"), lazy)
        assert_true(above.item(0, "x").bool())
        # A float literal beside an integer computes in Float64.
        var half = _result(frame, (col("i64") + 0.5).alias("x"), lazy)
        assert_equal(_short(half.column("x").dtype()), "f64")
        assert_equal(half.item(0, "x").numeric[DType.float64](), 3.5)
        # An integer literal beside a decimal is the number, not the
        # decimal's stored integer: an Int32 that promotes to decimal[10,0].
        var dec = _result(frame, (col("d") + 2).alias("x"), lazy)
        assert_equal(_short(dec.column("x").dtype()), "d13.2")
        assert_equal(dec.column("x").rechunk()._decimal_raw(0), Int128(325))


def test_non_numeric_types_keep_strict_matching() raises:
    var frame = DataFrame(
        [
            Series("i64", Column[Int64]([1])),
            Series("u64", Column[UInt64]([1])),
            Series("s", Column[String](["a"])),
            Series("b", Column[Bool]([True])),
        ]
    )
    for lazy in [False, True]:
        with assert_raises(contains="requires matching dtypes"):
            _ = _result(frame, (col("i64") == col("s")).alias("x"), lazy)
        with assert_raises(contains="requires matching dtypes"):
            _ = _result(frame, (col("i64") + col("b")).alias("x"), lazy)
        with assert_raises(contains="no common type holds both"):
            _ = _result(frame, (col("i64") + col("u64")).alias("x"), lazy)


def test_series_operators_promote_like_expressions() raises:
    var total = Series("x", Column[Int64]([1, 2])) + Series(
        "y", Column[Float32]([0.5, 0.25])
    )
    assert_true(total.dtype() == DataType.FLOAT32)
    var flags = Series("x", Column[Int8]([-1, 2])) < Series(
        "y", Column[UInt16]([1, 1])
    )
    assert_true(flags.dtype() == DataType.BOOL)


def test_branch_fill_and_coalesce_types_match_the_matrix_eager_and_lazy() raises:
    var checked = 0
    for cell in String(_BRANCHES).split(";"):
        if cell.byte_length() == 0:
            continue
        var parts = String(cell).split(" ")
        var left = String(parts[0])
        var right = String(parts[2])
        var expected = String(parts[3])
        var frame = DataFrame(
            [
                _column("a", left, [7, 7], [False, True]),
                _column("b", right, [2, 2], [True, True]),
                Series("c", Column[Bool]([False, True])),
            ]
        )
        var exprs: List[Expr] = [
            when(col("c")).then(col("a")).otherwise(col("b")).alias("when"),
            col("a").fill_null(col("b")).alias("fill"),
            coalesce([col("a"), col("b")]).alias("coalesce"),
        ]
        for lazy in [False, True]:
            for e in exprs:
                if expected == "err":
                    with assert_raises():
                        _ = _result(frame, e, lazy)
                    continue
                var out = _result(frame, e, lazy)
                var column = out.column(out.columns()[0])
                var found = _short(column.dtype())
                if found != expected:
                    raise Error(
                        left
                        + " / "
                        + right
                        + " in "
                        + out.columns()[0]
                        + ": expected "
                        + expected
                        + ", found "
                        + found
                        + (" (lazy)" if lazy else " (eager)")
                    )
                var values = column.cast(DataType.FLOAT64)
                assert_equal(values.item(0).float64(), 2.0)
                assert_equal(values.item(1).float64(), 7.0)
        checked += 1
    assert_equal(checked, 13 * 12)


def test_typed_null_branches_take_the_other_branch_type() raises:
    # A typed null of a narrower or other-signed type no longer forces an
    # error (#1480, #1498): the branches take the type that holds both.
    var frame = DataFrame(
        [
            Series("n", Column[Int64]([5, 6])),
            Series("u", Column[UInt16]([1, 0])),
        ]
    )
    var exprs: List[Expr] = [
        when(col("u") > 0)
        .then(col("n"))
        .otherwise(null(DataType.UINT32))
        .alias("unsigned"),
        when(col("u") > 0)
        .then(col("n"))
        .otherwise(null(DataType.INT8))
        .alias("narrow"),
        when(col("u") > 0)
        .then(null(DataType.INT8))
        .otherwise(col("u"))
        .alias("first_null"),
    ]
    for lazy in [False, True]:
        var out = (
            frame.lazy().select_exprs(exprs).collect() if lazy else (
                frame.select_exprs(exprs)
            )
        )
        assert_equal(_short(out.column("unsigned").dtype()), "i64")
        assert_equal(_short(out.column("narrow").dtype()), "i64")
        assert_equal(out.item(0, "unsigned").int64(), 5)
        assert_true(out.item(1, "narrow").is_null())
        # Int8 and UInt16 meet in Int32.
        assert_equal(_short(out.column("first_null").dtype()), "i32")
        assert_true(out.item(0, "first_null").is_null())
        assert_equal(out.item(1, "first_null").int32(), 0)


def test_literals_in_branches_and_fills_promote_when_they_do_not_fit() raises:
    var frame = DataFrame(
        [
            Series("u", Column[UInt8]([200], [False])),
            Series("d", Column[Int128]([125], [False])).with_dtype(
                DataType.decimal(10, 2)
            ),
            Series("c", Column[Bool]([False])),
        ]
    )
    var exprs: List[Expr] = [
        col("u").fill_null(1).alias("fits"),
        col("u").fill_null(300).alias("wide"),
        when(col("c")).then(col("u")).otherwise(0.5).alias("half"),
        col("d").fill_null(2).alias("dec"),
        when(col("c")).then(col("d")).otherwise(3).alias("dec_branch"),
    ]
    for lazy in [False, True]:
        var out = (
            frame.lazy().select_exprs(exprs).collect() if lazy else (
                frame.select_exprs(exprs)
            )
        )
        assert_equal(_short(out.column("fits").dtype()), "u8")
        assert_equal(_short(out.column("wide").dtype()), "i32")
        assert_equal(out.item(0, "wide").int32(), 300)
        assert_equal(_short(out.column("half").dtype()), "f64")
        assert_equal(out.item(0, "half").float64(), 0.5)
        # A number beside a decimal is the number, not a scaled integer.
        assert_equal(_short(out.column("dec").dtype()), "d12.2")
        assert_equal(out.column("dec").rechunk()._decimal_raw(0), Int128(200))
        assert_equal(_short(out.column("dec_branch").dtype()), "d12.2")
        assert_equal(
            out.column("dec_branch").rechunk()._decimal_raw(0), Int128(300)
        )


def test_is_in_compares_integers_of_other_widths_by_value() raises:
    var frame = DataFrame(
        [Series("x", Column[UInt8]([1, 200, 7], [True, True, False]))]
    )
    for lazy in [False, True]:
        var listed: List[Expr] = [lit(Int64(200)), lit(Int64(-1))]
        var out = _result(frame, col("x").is_in(listed).alias("x"), lazy)
        assert_true(not out.item(0, "x").bool())
        assert_true(out.item(1, "x").bool())
        assert_true(out.item(2, "x").is_null())
        var mixed: List[Expr] = [lit(Int16(1)), lit(Int64(200))]
        var both = _result(frame, col("x").is_in(mixed).alias("x"), lazy)
        assert_true(both.item(0, "x").bool())
        assert_true(both.item(1, "x").bool())
        with assert_raises(contains="needs values of its type"):
            _ = _result(
                DataFrame([Series("f", Column[Float64]([1.0]))]),
                col("f").is_in([lit(Int64(1))]),
                lazy,
            )
