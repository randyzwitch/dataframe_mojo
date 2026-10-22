"""Reusable grouped typed owner checks; no GPU context initialization."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_false, assert_raises
from dataframe import Column, DataFrame, Series, col
from dataframe_metal.accelerator.grouped_key_queries import prepare_keys
from dataframe_metal.accelerator.grouped_lists import prepare
from dataframe_metal.accelerator.gpu.residency import Scope, Table
from dataframe_metal.accelerator.gpu.device_lists import HostReservation
from dataframe_metal.accelerator.gpu.grouped_key_owner import (
    GroupedKeyOwner,
    GroupedKeyField,
)


def scoped_grouped_typed_owner_contract() raises:
    var input = DataFrame(
        [
            Series("k", Column[String](List[String]())),
            Series("v", Column[Int64](List[Int64]())),
        ]
    )
    var query = (
        input.lazy().group_by("k", maintain_order=True).agg(col("v").implode())
    )
    var staged = ArcPointer(prepare_keys(query, memory_limit_bytes=0))
    var plan = ArcPointer(prepare(staged[].query))
    var scope = Scope(0, 0)
    var charge = ArcPointer(HostReservation(scope, staged[].staging_bytes))
    var source = Table.empty(plan[].source.source.clear())
    var owner = ArcPointer(GroupedKeyOwner(staged, plan, source, scope, charge))
    owner[].validate(plan[], source, scope)
    assert_equal(scope.info()[0], 0)
    assert_false(scope.initialized())
    var other = Scope(0, 0)
    with assert_raises(contains="another query"):
        owner[].validate(plan[], source, other)
    var duplicate = ArcPointer(prepare(staged[].query))
    with assert_raises(contains="prepared plan differs"):
        owner[].validate(duplicate[], source, scope)
    var key = GroupedKeyField(owner, 0)
    var values = Table.empty(
        DataFrame([staged[].domains[0].encoded.slice(0, 0)])
    )
    key.validate(values, scope)
    with assert_raises(contains="another query"):
        key.validate(values, other)
    var foreign = Table.rows_only(other, 0)
    with assert_raises(contains="another query"):
        key.validate(foreign, scope)
    var invalid = GroupedKeyField(owner, 1)
    with assert_raises(contains="outside its owner"):
        _ = invalid.logical_dtype()
    assert_false(scope.initialized())
    assert_false(other.initialized())
    print("scoped grouped typed owner: PASS")
