"""Observed-pair dictionary construction preserves text and pair semantics."""
# contract-group: public.apple_parity
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import DataType
from dataframe.dtype import CategoricalDictionary
from dataframe_metal.accelerator.gpu.residency import Scope
from dataframe_metal.accelerator.gpu.observed_concat_labels import build


def _dtype(values: List[String]) -> DataType:
    var dictionary = CategoricalDictionary()
    for value in values:
        dictionary.append(value)
    return DataType.categorical(dictionary^)


def _assert_text_pair_results(scope: Scope, source: DataType) raises:
    # Different source pairs ("ab","c") and ("a","bc") both yield abc.
    # Empty source text remains significant even with an empty separator.
    var result = build(
        scope,
        [0, 2, 4, 6, 5, 6],
        [1, 3, 5, 5, 6, 6],
        source,
        "",
        0,
    )
    var dictionary = result[0][].dtype.dictionary()
    assert_equal(len(dictionary[]), 4)
    assert_equal(dictionary[].get(0), "")
    assert_equal(dictionary[].get(1), "abc")
    assert_equal(dictionary[].get(2), "é")
    assert_equal(dictionary[].get(3), "🙂é")
    var expected_codes: List[UInt32] = [
        UInt32(1),
        UInt32(1),
        UInt32(3),
        UInt32(2),
        UInt32(2),
        UInt32(0),
    ]
    assert_equal(result[1], expected_codes)
    assert_true(result[0][].dtype.is_categorical())
    assert_false(scope.initialized())
    _ = result^

    var unicode_separator = build(scope, [2, 6], [3, 5], source, "🙂|", 0)
    var unicode_dictionary = unicode_separator[0][].dtype.dictionary()
    assert_equal(len(unicode_dictionary[]), 2)
    assert_equal(unicode_dictionary[].get(0), "a🙂|bc")
    assert_equal(unicode_dictionary[].get(1), "🙂|é")
    var expected_unicode_codes: List[UInt32] = [UInt32(0), UInt32(1)]
    assert_equal(unicode_separator[1], expected_unicode_codes)
    assert_false(scope.initialized())
    _ = unicode_separator^


def _assert_budget_recovery_result(scope: Scope, source: DataType) raises:
    var result = build(scope, [0, 1], [1, 2], source, ":", 0)
    assert_true(scope.info()[0] > 0)
    assert_false(scope.initialized())
    _ = result^


def _assert_empty_pair_result(scope: Scope, source: DataType) raises:
    var result = build(scope, List[UInt32](), List[UInt32](), source, "|", 0)
    assert_equal(len(result[0][].dtype.dictionary()[]), 0)
    assert_equal(len(result[1]), 0)
    assert_false(scope.initialized())
    _ = result^


# requires: mojo_gpu
def test_observed_concat_labels_sort_deduplicate_unicode_and_empty_text() raises:
    var scope = Scope(99999)
    var source = _dtype(["ab", "c", "a", "bc", "🙂", "é", ""])
    _assert_text_pair_results(scope, source)
    assert_equal(scope.info()[0], 0)


# requires: mojo_gpu
def test_observed_concat_labels_reject_bad_codes_and_pair_shapes() raises:
    var scope = Scope(99999)
    var source = _dtype(["a", "b"])
    with assert_raises(contains="compact pair shape"):
        _ = build(scope, [0, 1], [1], source, ":", 0)
    with assert_raises(contains="outside its source dictionary"):
        _ = build(scope, [0], [2], source, ":", 0)
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 0)
    assert_false(scope.initialized())


# requires: mojo_gpu
def test_observed_concat_label_budget_rejection_and_recovery() raises:
    var scope = Scope(99999, 1024)
    var source = _dtype(["a", "b", "c"])
    with assert_raises(contains="budget"):
        _ = build(scope, [0, 1], [1, 2], source, ":", 1024)
    # The failed preflight allocates no retained owner, and the same scope can
    # accept a later operation once the caller's reservation is valid.
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 0)
    assert_false(scope.initialized())
    _assert_budget_recovery_result(scope, source)
    assert_equal(scope.info()[0], 0)
    assert_false(scope.initialized())


# requires: mojo_gpu
def test_observed_concat_labels_accept_empty_pair_domain() raises:
    var scope = Scope(99999)
    var source = _dtype(["x"])
    _assert_empty_pair_result(scope, source)
    assert_equal(scope.info()[0], 0)
