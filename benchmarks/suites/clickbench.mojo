"""ClickBench's 43 queries over the `hits` table, written with this
library's lazy API so projection pushdown reads only the columns each query
uses, as Polars' lazy versions do. engines.py holds ClickBench's DuckDB SQL
and the Polars versions; output columns follow the SQL. Usage: see
suite_common.mojo."""
from std.collections import Dict

from dataframe import DataFrame, Expr, LazyFrame, col, lit, when
from suite_common import date_lit, run, unsupported


def top(
    frame: LazyFrame,
    by: String,
    k: Int = 10,
    offset: Int = 0,
    descending: Bool = True,
) raises -> DataFrame:
    return frame.sort(by, descending=descending).slice(offset, k).collect()


def count() -> Expr:
    """COUNT(*): the length of any column, here the always-present WatchID."""
    return col("WatchID").len()


def query(q: String, t: Dict[String, DataFrame]) raises -> DataFrame:
    var hits = t["hits"].lazy()
    var n = Int(String(q[byte=1:]))
    var phrase = col("SearchPhrase").ne("")
    var july = (
        (col("CounterID") == 62)
        & (col("EventDate") >= date_lit("2013-07-01"))
        & (col("EventDate") <= date_lit("2013-07-31"))
    )
    if n == 0:
        return hits.select(count().alias("count")).collect()
    if n == 1:
        return (
            hits.filter(col("AdvEngineID") != 0)
            .select(count().alias("count"))
            .collect()
        )
    if n == 2:
        return hits.select_exprs(
            [
                col("AdvEngineID").sum().alias("s"),
                count().alias("c"),
                col("ResolutionWidth").mean().alias("a"),
            ]
        ).collect()
    if n == 3:
        return hits.select(col("UserID").mean()).collect()
    if n == 4:
        return hits.select(col("UserID").n_unique()).collect()
    if n == 5:
        return hits.select(col("SearchPhrase").n_unique()).collect()
    if n == 6:
        return hits.select_exprs(
            [
                col("EventDate").min().alias("min"),
                col("EventDate").max().alias("max"),
            ]
        ).collect()
    if n == 7:
        return (
            hits.filter(col("AdvEngineID") != 0)
            .group_by("AdvEngineID")
            .agg([count().alias("c")])
            .sort("c", descending=True)
            .collect()
        )
    if n == 8:
        return top(
            hits.group_by("RegionID").agg(
                [col("UserID").n_unique().alias("u")]
            ),
            "u",
        )
    if n == 9:
        return top(
            hits.group_by("RegionID").agg(
                [
                    col("AdvEngineID").sum().alias("s"),
                    count().alias("c"),
                    col("ResolutionWidth").mean().alias("a"),
                    col("UserID").n_unique().alias("u"),
                ]
            ),
            "c",
        )
    if n == 10 or n == 11:
        var keys: List[String] = ["MobilePhoneModel"]
        if n == 11:
            keys = ["MobilePhone", "MobilePhoneModel"]
        return top(
            hits.filter(col("MobilePhoneModel").ne(""))
            .group_by(keys)
            .agg([col("UserID").n_unique().alias("u")]),
            "u",
        )
    if n == 12:
        return top(
            hits.filter(phrase)
            .group_by("SearchPhrase")
            .agg([count().alias("c")]),
            "c",
        )
    if n == 13:
        return top(
            hits.filter(phrase)
            .group_by("SearchPhrase")
            .agg([col("UserID").n_unique().alias("u")]),
            "u",
        )
    if n == 14:
        return top(
            hits.filter(phrase)
            .group_by(["SearchEngineID", "SearchPhrase"])
            .agg([count().alias("c")]),
            "c",
        )
    if n == 15:
        return top(hits.group_by("UserID").agg([count().alias("c")]), "c")
    if n == 16:
        return top(
            hits.group_by(["UserID", "SearchPhrase"]).agg([count().alias("c")]),
            "c",
        )
    if n == 17:
        return (
            hits.group_by(["UserID", "SearchPhrase"])
            .agg([count().alias("c")])
            .head(10)
            .collect()
        )
    if n == 18:
        return top(
            hits.with_columns([col("EventTime").dt().minute().alias("m")])
            .group_by(["UserID", "m", "SearchPhrase"])
            .agg([count().alias("c")]),
            "c",
        )
    if n == 19:
        return (
            hits.filter(col("UserID") == lit(Int64(435090932899640449)))
            .select(["UserID"])
            .collect()
        )
    if n == 20:
        return (
            hits.filter(col("URL").str().contains("google"))
            .select(count().alias("count"))
            .collect()
        )
    if n == 21:
        return top(
            hits.filter(col("URL").str().contains("google") & phrase)
            .group_by("SearchPhrase")
            .agg([col("URL").min().alias("url"), count().alias("c")]),
            "c",
        )
    if n == 22:
        return top(
            hits.filter(
                col("Title").str().contains("Google")
                & ~col("URL").str().contains(".google.")
                & phrase
            )
            .group_by("SearchPhrase")
            .agg(
                [
                    col("URL").min().alias("url"),
                    col("Title").min().alias("title"),
                    count().alias("c"),
                    col("UserID").n_unique().alias("u"),
                ]
            ),
            "c",
        )
    if n == 23:
        return top(
            hits.filter(col("URL").str().contains("google")),
            "EventTime",
            descending=False,
        )
    if n == 24:
        return (
            hits.filter(phrase)
            .select(["EventTime", "SearchPhrase"])
            .sort("EventTime")
            .head(10)
            .select(["SearchPhrase"])
            .collect()
        )
    if n == 25:
        return top(
            hits.filter(phrase).select(["SearchPhrase"]),
            "SearchPhrase",
            descending=False,
        )
    if n == 26:
        return (
            hits.filter(phrase)
            .select(["EventTime", "SearchPhrase"])
            .sort(["EventTime", "SearchPhrase"])
            .head(10)
            .select(["SearchPhrase"])
            .collect()
        )
    if n == 27:
        return top(
            hits.filter(col("URL").ne(""))
            .group_by("CounterID")
            .agg(
                [
                    col("URL").str().len_chars().mean().alias("l"),
                    count().alias("c"),
                ]
            )
            .filter(col("c") > 100000),
            "l",
            k=25,
        )
    if n == 28:
        raise unsupported("regular expressions are not implemented (#219)")
    if n == 29:
        var sums = List[Expr]()
        sums.append(col("ResolutionWidth").sum().alias("s0"))
        for i in range(1, 90):
            sums.append(
                (col("ResolutionWidth") + i).sum().alias("s" + String(i))
            )
        return hits.select_exprs(sums).collect()
    if n == 30 or n == 31 or n == 32:
        var source = hits.filter(phrase) if n < 32 else hits.copy()
        var keys: List[String] = ["WatchID", "ClientIP"]
        if n == 30:
            keys = ["SearchEngineID", "ClientIP"]
        return top(
            source.group_by(keys).agg(
                [
                    count().alias("c"),
                    col("IsRefresh").sum().alias("r"),
                    col("ResolutionWidth").mean().alias("w"),
                ]
            ),
            "c",
        )
    if n == 33:
        return top(hits.group_by("URL").agg([count().alias("c")]), "c")
    if n == 34:
        return top(
            hits.group_by("URL")
            .agg([count().alias("c")])
            .select_exprs([lit(Int64(1)).alias("one"), col("URL"), col("c")]),
            "c",
        )
    if n == 35:
        return top(
            hits.group_by("ClientIP")
            .agg([count().alias("c")])
            .select_exprs(
                [
                    col("ClientIP"),
                    (col("ClientIP") - 1).alias("m1"),
                    (col("ClientIP") - 2).alias("m2"),
                    (col("ClientIP") - 3).alias("m3"),
                    col("c"),
                ]
            ),
            "c",
        )
    if n == 36 or n == 37:
        var target = "URL" if n == 36 else "Title"
        return top(
            hits.filter(
                july
                & (col("DontCountHits") == 0)
                & (col("IsRefresh") == 0)
                & col(target).ne("")
            )
            .group_by(target)
            .agg([count().alias("PageViews")]),
            "PageViews",
        )
    if n == 38:
        return top(
            hits.filter(
                july
                & (col("IsRefresh") == 0)
                & (col("IsLink") != 0)
                & (col("IsDownload") == 0)
            )
            .group_by("URL")
            .agg([count().alias("PageViews")]),
            "PageViews",
            offset=1000,
        )
    if n == 39:
        return top(
            hits.filter(july & (col("IsRefresh") == 0))
            .with_columns(
                [
                    when(
                        (col("SearchEngineID") == 0) & (col("AdvEngineID") == 0)
                    )
                    .then(col("Referer"))
                    .otherwise(lit(""))
                    .alias("Src"),
                    col("URL").alias("Dst"),
                ]
            )
            .group_by(
                [
                    "TraficSourceID",
                    "SearchEngineID",
                    "AdvEngineID",
                    "Src",
                    "Dst",
                ]
            )
            .agg([count().alias("PageViews")]),
            "PageViews",
            offset=1000,
        )
    if n == 40:
        var sources: List[Expr] = [Expr(-1), Expr(6)]
        return top(
            hits.filter(
                july
                & (col("IsRefresh") == 0)
                & col("TraficSourceID").is_in(sources)
                & (col("RefererHash") == lit(Int64(3594120000172545465)))
            )
            .group_by(["URLHash", "EventDate"])
            .agg([count().alias("PageViews")]),
            "PageViews",
            offset=100,
        )
    if n == 41:
        return top(
            hits.filter(
                july
                & (col("IsRefresh") == 0)
                & (col("DontCountHits") == 0)
                & (col("URLHash") == lit(Int64(2868770270353813622)))
            )
            .group_by(["WindowClientWidth", "WindowClientHeight"])
            .agg([count().alias("PageViews")]),
            "PageViews",
            offset=10000,
        )
    if n == 42:
        return top(
            hits.filter(
                (col("CounterID") == 62)
                & (col("EventDate") >= date_lit("2013-07-14"))
                & (col("EventDate") <= date_lit("2013-07-15"))
                & (col("IsRefresh") == 0)
                & (col("DontCountHits") == 0)
            )
            .with_columns([col("EventTime").dt().truncate("1m").alias("M")])
            .group_by("M")
            .agg([count().alias("PageViews")]),
            "M",
            offset=1000,
            descending=False,
        )
    raise Error("unknown query " + q)


def main() raises:
    run[query]()
