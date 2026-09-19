# Coming from pandas or Polars

The API follows Polars' expression style (via Narwhals) closely. The main
differences: expressions are built with `col`/`lit` and there is no implicit
type promotion: a bare number adopts the other operand's dtype (`col("a") > 1`)
and errors if it cannot (`col("i") * 2.5` on an integer column); `lit(Int64(1))` fixes a type explicitly;
equality is `.eq()`; there is no index; and results are eager dataframes.

| Task | pandas | Polars | dataframe_mojo |
|---|---|---|---|
| Read CSV | `pd.read_csv(p)` | `pl.read_csv(p)` | `read_csv(p)` or `read_csv(p, CsvSchema([...]))` |
| Write CSV | `df.to_csv(p, index=False)` | `df.write_csv(p)` | `write_csv(df, p)` |
| First rows | `df.head(5)` | `df.head(5)` | `df.head(5)` |
| Show | `print(df)` | `print(df)` | `print(df)`, `df.glimpse()` |
| Select columns | `df[["a", "b"]]` | `df.select("a", "b")` | `df.select(["a", "b"])` |
| One column | `df["a"]` | `df["a"]` | `df["a"]` (a `Series`) |
| Filter | `df[df.a > 1]` | `df.filter(pl.col("a") > 1)` | `df.filter(col("a") > 1)` |
| Equality | `df.a == "x"` | `pl.col("a") == "x"` | `col("a") == "x"` |
| New column | `df.assign(b=df.a * 2)` | `df.with_columns((pl.col("a") * 2).alias("b"))` | `df.with_columns((col("a") * 2).alias("b"))` |
| Conditional | `np.where(c, x, y)` | `pl.when(c).then(x).otherwise(y)` | `when(c).then(x).otherwise(y)` |
| Group and aggregate | `df.groupby("k").agg(s=("v", "sum"))` | `df.group_by("k").agg(pl.col("v").sum())` | `df.group_by("k").agg(col("v").sum())` |
| Window | `df.groupby("k").v.cumsum()` | `pl.col("v").cum_sum().over("k")` | `col("v").cum_sum().over("k")` |
| Sort | `df.sort_values(["a", "b"])` | `df.sort(["a", "b"])` | `df.sort(["a", "b"])` |
| Join | `df.merge(o, on="k", how="left")` | `df.join(o, on="k", how="left")` | `df.join(o, "k", "left")` |
| Concatenate | `pd.concat([a, b])` | `pl.concat([a, b])` | `concat([a, b])` |
| Deduplicate | `df.drop_duplicates()` | `df.unique()` | `df.unique()` |
| Drop nulls | `df.dropna()` | `df.drop_nulls()` | `df.drop_nulls()` |
| Fill nulls | `df.fillna(0)` | `df.fill_null(0)` | `df.fill_null(0)` |
| Cast | `df.a.astype(float)` | `pl.col("a").cast(pl.Float64)` | `col("a").cast("float64")` |
| Strings | `df.s.str.upper()` | `pl.col("s").str.to_uppercase()` | `col("s").str().to_uppercase()` |
| Pivot | `df.pivot_table(...)` | `df.pivot("on", index="i", values="v")` | `df.pivot("on", index=["i"], values="v")` |
| Unpivot | `df.melt(id_vars=["i"])` | `df.unpivot(index="i")` | `df.unpivot(index=["i"])` |
| Value counts | `df.a.value_counts()` | `df["a"].value_counts()` | `df["a"].value_counts()` |

Behavioral differences worth knowing:

- Sums of empty or all-null input are 0; pass `min_count=1` for null.
- Int64 arithmetic raises on overflow instead of wrapping.
- NaN is a value, distinct from null; `fill_null` does not touch NaN (use
  `fill_nan`).
- Group output order is unspecified unless `maintain_order=True`.
- Numeric dtypes are Int8–Int64, UInt8–UInt64, Float32, and Float64, with
  no implicit promotion: mixed widths need an explicit `cast`. Sums of 8- and
  16-bit integers produce Int64, as in Polars.
