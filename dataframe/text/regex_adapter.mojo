"""Regular-expression string kernels (#219): `str.contains`,
`count_matches`, `extract`, `extract_all`, `replace`, `replace_all` and
`split`.

The pattern is compiled once into a `Regex` whose programs every worker
shares; each worker takes a range of rows and searches them with its own
`Matcher` (its DFA caches and Pike VM). Row bytes are read in
place (`_Bytes`). Predicates write whole bitmap bytes; string and list
results are built per worker and joined once.

A pattern that is one literal string (`"foo"`, `"a\\.b"`) is searched as a
substring by the engine itself, and `contains` with such a pattern uses the
literal kernel of `string_predicates`.
"""
from std.memory import Pointer

from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column, _copy_validity
from dataframe.storage.nested import ListColumn, StructColumn
from dataframe.parallel import Job, run_jobs, worker_count
from dataframe.regex import Matcher, Regex, Template, escape
from dataframe.text.regex_cache import cached_pattern
from dataframe.series import Series
from dataframe.text.bytes import _Bytes
from dataframe.storage.string import StringColumn, StringBuilder
from dataframe.text.predicates import string_contains

comptime _CONTAINS = 0
comptime _COUNT = 1
comptime _EXTRACT = 2
comptime _REPLACE = 3
comptime _EXTRACT_ALL = 4
comptime _SPLIT = 5
comptime _SPLIT_INCLUSIVE = 6
comptime _FIND = 7


def compile_pattern(pattern: String, literal: Bool) raises -> Regex:
    """The regex for a pattern argument; with `literal` the pattern's text
    is matched as is."""
    try:
        return cached_pattern(escape(pattern) if literal else pattern)
    except e:
        raise Error("Invalid regular expression: " + String(e))


def is_literal_pattern(pattern: String) -> Bool:
    """A pattern with no ASCII punctuation is replaced as a literal (and
    its replacement is not expanded)."""
    for b in pattern.as_bytes():
        if (
            (b >= 33 and b <= 47)
            or (b >= 58 and b <= 64)
            or (b >= 91 and b <= 96)
            or (b >= 123 and b <= 126)
        ):
            return False
    return True


struct _Output(Movable):
    """One worker's string results: the bytes, each row's end offset, and
    per row (list kernels) the number of parts."""

    var bytes: List[UInt8]
    var ends: List[Int64]
    var valid: List[Bool]
    var counts: List[Int]

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.ends = List[Int64]()
        self.valid = List[Bool]()
        self.counts = List[Int]()

    def push(mut self, part: Span[UInt8, ImmutAnyOrigin], start: Int, end: Int):
        var p = part.unsafe_ptr()
        for i in range(start, end):
            self.bytes.append(p[unsafe_offset=i])
        self.ends.append(Int64(len(self.bytes)))
        self.valid.append(True)

    def push_null(mut self):
        self.ends.append(Int64(len(self.bytes)))
        self.valid.append(False)


struct _RegexJob[mode: Int](Job):
    """Rows [first, last) of one regex kernel. Predicate modes write their
    output through `output` (a bitmap for contains, whose `first` is a
    multiple of 8; UInt32 counts for count); string modes fill `out`."""

    var rows: _Bytes
    var nulls: Bool
    var regex: Regex
    var template: Template
    var group: Int
    var limit: Int
    var output: Int
    var first: Int
    var last: Int
    var out: _Output

    def __init__(
        out self,
        column: StringColumn,
        nulls: Bool,
        regex: Regex,
        template: Template,
        group: Int,
        limit: Int,
        output: Int,
        first: Int,
        last: Int,
    ):
        self.rows = _Bytes(column)
        self.nulls = nulls
        self.regex = regex.copy()
        self.template = template.copy()
        self.group = group
        self.limit = limit
        self.output = output
        self.first = first
        self.last = last
        self.out = _Output()

    def run(mut self) raises:
        var matcher = Matcher(self.regex)
        comptime if Self.mode == _CONTAINS:
            var bits = Pointer[List[UInt8], MutAnyOrigin](
                unsafe_from_address=self.output
            )[].unsafe_ptr()
            var row = self.first
            while row < self.last:
                var end = min(row + 8, self.last)
                var byte = UInt8(0)
                for i in range(row, end):
                    if self.nulls and not self.rows.column._valid(i):
                        continue
                    if matcher.is_match(self.rows.get(i)):
                        byte |= UInt8(1) << UInt8(i - row)
                bits.unsafe_offset(row // 8)[] = byte
                row = end
        elif Self.mode == _COUNT:
            var counts = Pointer[List[UInt32], MutAnyOrigin](
                unsafe_from_address=self.output
            )[].unsafe_ptr()
            for i in range(self.first, self.last):
                var total = 0
                if not self.nulls or self.rows.column._valid(i):
                    total = matcher.count(self.rows.get(i))
                counts.unsafe_offset(i)[] = UInt32(total)
        elif Self.mode == _FIND:
            var found = Pointer[List[Int64], MutAnyOrigin](
                unsafe_from_address=self.output
            )[].unsafe_ptr()
            for i in range(self.first, self.last):
                var at = -1
                if not self.nulls or self.rows.column._valid(i):
                    at = matcher.find(self.rows.get(i))[0]
                found.unsafe_offset(i)[] = Int64(at)
        else:
            for i in range(self.first, self.last):
                if self.nulls and not self.rows.column._valid(i):
                    self.out.push_null()
                    self.out.counts.append(0)
                    continue
                var text = self.rows.get(i)
                comptime if Self.mode == _EXTRACT:
                    if (
                        matcher.captures(text)
                        and 2 * self.group + 1 < len(matcher.slots)
                        and matcher.slots[2 * self.group] >= 0
                    ):
                        self.out.push(
                            text,
                            matcher.slots[2 * self.group],
                            matcher.slots[2 * self.group + 1],
                        )
                    else:
                        self.out.push_null()
                elif Self.mode == _REPLACE:
                    _ = matcher.replace(
                        text, self.template, self.limit, self.out.bytes
                    )
                    self.out.ends.append(Int64(len(self.out.bytes)))
                    self.out.valid.append(True)
                else:
                    self._parts(matcher, text)

    def _parts(
        mut self, mut matcher: Matcher, text: Span[UInt8, ImmutAnyOrigin]
    ):
        """One row's list: every match (extract_all), or the text between
        matches (split), or each piece through its match (inclusive split)."""
        var parts = 0
        var last = 0
        matcher.reset()
        while True:
            var m = matcher.next(text)
            if m[0] < 0:
                break
            comptime if Self.mode == _EXTRACT_ALL:
                self.out.push(text, m[0], m[1])
            elif Self.mode == _SPLIT:
                self.out.push(text, last, m[0])
            else:
                self.out.push(text, last, m[1])
            last = m[1]
            parts += 1
        comptime if Self.mode == _SPLIT:
            self.out.push(text, last, len(text))
            parts += 1
        elif Self.mode == _SPLIT_INCLUSIVE:
            # An empty text has one empty part, except when splitting into
            # code points (group -1), which gives none.
            if last < len(text) or (parts == 0 and self.group >= 0):
                self.out.push(text, last, len(text))
                parts += 1
        self.out.counts.append(parts)


def _jobs[
    mode: Int
](
    column: StringColumn,
    regex: Regex,
    template: Template,
    group: Int,
    limit: Int,
    output: Int,
) raises -> List[_RegexJob[mode]]:
    var n = len(column)
    var nulls = column.null_count() > 0
    var workers = worker_count(n)
    var jobs = List[_RegexJob[mode]](capacity=workers)
    for w in range(workers):
        var first = (n * w // workers) // 8 * 8
        var last = n if w == workers - 1 else (n * (w + 1) // workers) // 8 * 8
        jobs.append(
            _RegexJob[mode](
                column,
                nulls,
                regex,
                template,
                group,
                limit,
                output,
                first,
                last,
            )
        )
    if workers == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    return jobs^


def regex_contains(column: StringColumn, regex: Regex) raises -> Series:
    """Whether each row matches anywhere; null rows stay null."""
    var literal = regex.literal()
    if literal:
        return string_contains(column, literal.value())
    var n = len(column)
    # Every byte is written by exactly one job.
    var values = List[UInt8](unsafe_uninit_length=(n + 7) // 8)
    _ = _jobs[_CONTAINS](
        column,
        regex,
        regex.template(""),
        0,
        0,
        Int(Pointer(to=values)),
    )
    var valid = _copy_validity(column._bits[], column._offset, n)
    return Series("", BoolColumn(values=values^, bits=valid^, length=n))


def regex_count(column: StringColumn, regex: Regex) raises -> Series:
    """Non-overlapping matches per row, as UInt32."""
    var n = len(column)
    var counts = List[UInt32](unsafe_uninit_length=n)
    _ = _jobs[_COUNT](
        column, regex, regex.template(""), 0, 0, Int(Pointer(to=counts))
    )
    var valid = List[Bool](capacity=n)
    for i in range(n):
        valid.append(column._valid(i))
    return Series("", Column[UInt32](counts^, valid))


def regex_find(column: StringColumn, regex: Regex) raises -> Series:
    """The byte offset (UInt32) of each row's first match; null when there
    is none."""
    var n = len(column)
    var found = List[Int64](unsafe_uninit_length=n)
    _ = _jobs[_FIND](
        column, regex, regex.template(""), 0, 0, Int(Pointer(to=found))
    )
    var values = List[UInt32](length=n, fill=0)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        valid.append(found[i] >= 0)
        if found[i] >= 0:
            values[i] = UInt32(found[i])
    return Series("", Column[UInt32](values^, valid))


def _strings[mode: Int](var jobs: List[_RegexJob[mode]]) -> StringColumn:
    var total = 0
    var rows = 0
    for j in range(len(jobs)):
        ref job = jobs[j]
        total += len(job.out.bytes)
        rows += len(job.out.ends)
    var bytes = List[UInt8](capacity=total)
    var offsets = List[Int64](capacity=rows + 1)
    offsets.append(0)
    var valid = List[Bool](capacity=rows)
    var any_null = False
    for j in range(len(jobs)):
        ref job = jobs[j]
        var base = Int64(len(bytes))
        bytes.extend(Span(job.out.bytes))
        for end in job.out.ends:
            offsets.append(base + end)
        for flag in job.out.valid:
            valid.append(flag)
            any_null = any_null or not flag
    var bits = List[UInt8]()
    if any_null:
        bits = List[UInt8](length=(rows + 7) // 8, fill=0)
        for i in range(rows):
            if valid[i]:
                bits[i // 8] |= UInt8(1) << UInt8(i % 8)
    return StringColumn(bytes=bytes^, offsets=offsets^, bits=bits^, length=rows)


def regex_extract(
    column: StringColumn, regex: Regex, group: Int
) raises -> Series:
    """Capture group `group` of each row's first match; null when there is
    no match or the group took no part."""
    var jobs = _jobs[_EXTRACT](column, regex, regex.template(""), group, 0, 0)
    return Series("", _strings(jobs^))


def regex_replace(
    column: StringColumn, regex: Regex, template: Template, limit: Int
) raises -> Series:
    """Each row with its first `limit` matches (all when negative)
    replaced; null rows stay null."""
    var jobs = _jobs[_REPLACE](column, regex, template, 0, limit, 0)
    return Series("", _strings(jobs^))


def _lists[
    mode: Int
](column: StringColumn, regex: Regex, group: Int = 0) raises -> Series:
    var n = len(column)
    var jobs = _jobs[mode](column, regex, regex.template(""), group, 0, 0)
    var offsets = List[Int64](capacity=n + 1)
    offsets.append(0)
    var valid = List[Bool](capacity=n)
    var any_null = False
    var total = 0
    for j in range(len(jobs)):
        ref job = jobs[j]
        for k in range(len(job.out.counts)):
            total += job.out.counts[k]
            offsets.append(Int64(total))
    for i in range(n):
        var ok = column._valid(i)
        valid.append(ok)
        any_null = any_null or not ok
    # Null rows pushed one null part placeholder; drop those parts.
    var parts = _strings_without_nulls(jobs^)
    var bits = List[UInt8]()
    if any_null:
        bits = List[UInt8](length=(n + 7) // 8, fill=0)
        for i in range(n):
            if valid[i]:
                bits[i // 8] |= UInt8(1) << UInt8(i % 8)
    var child = Series("item", parts^)
    return Series("", ListColumn(offsets^, child^, bits^))


def _strings_without_nulls[
    mode: Int
](var jobs: List[_RegexJob[mode]]) -> StringColumn:
    """The parts of list kernels: every pushed value except the null
    placeholders of null rows."""
    var bytes = List[UInt8]()
    var offsets = List[Int64]()
    offsets.append(0)
    for j in range(len(jobs)):
        ref job = jobs[j]
        var base = Int64(len(bytes))
        bytes.extend(Span(job.out.bytes))
        for k in range(len(job.out.ends)):
            if job.out.valid[k]:
                offsets.append(base + job.out.ends[k])
    var rows = len(offsets) - 1
    return StringColumn(
        bytes=bytes^, offsets=offsets^, bits=List[UInt8](), length=rows
    )


def split_codepoints(column: StringColumn) raises -> Series:
    """Each row as a list of its code points (an empty row gives an empty
    list) when splitting on an empty pattern."""
    return _lists[_SPLIT_INCLUSIVE](column, Regex("(?s)."), -1)


def regex_extract_all(column: StringColumn, regex: Regex) raises -> Series:
    """Every non-overlapping match of each row, as a list of strings."""
    return _lists[_EXTRACT_ALL](column, regex)


def regex_split(
    column: StringColumn, regex: Regex, inclusive: Bool
) raises -> Series:
    """Each row split on the pattern's matches."""
    if inclusive:
        return _lists[_SPLIT_INCLUSIVE](column, regex)
    return _lists[_SPLIT](column, regex)


struct _GroupsJob(Job):
    var rows: _Bytes
    var regex: Regex
    var first: Int
    var last: Int
    var fields: List[Series]

    def __init__(
        out self, column: StringColumn, regex: Regex, first: Int, last: Int
    ):
        self.rows = _Bytes(column)
        self.regex = regex.copy()
        self.first = first
        self.last = last
        self.fields = List[Series]()

    def run(mut self) raises:
        var names = self.regex.group_names()
        var builders = List[StringBuilder]()
        for _ in names:
            builders.append(StringBuilder(self.last - self.first))
        var matcher = Matcher(self.regex)
        for row in range(self.first, self.last):
            if not self.rows.column._valid(row):
                for k in range(len(names)):
                    builders[k].append_null()
                continue
            var text = self.rows.get(row)
            var matched = matcher.captures(text)
            for k in range(len(names)):
                var start = matcher.slots[2 * (k + 1)] if matched else -1
                if start < 0:
                    builders[k].append_null()
                else:
                    var end = matcher.slots[2 * (k + 1) + 1]
                    builders[k].append(
                        StringSlice(unsafe_from_utf8=text[start:end])
                    )
        var reversed = List[Series]()
        for k in range(len(names) - 1, -1, -1):
            var builder = builders.pop()
            reversed.append(Series(names[k], builder^.finish()))
        for k in range(len(names)):
            self.fields.append(reversed[len(names) - 1 - k].copy())


def regex_extract_groups(column: StringColumn, regex: Regex) raises -> Series:
    """One capture search per valid row; each worker builds all fields."""
    var names = regex.group_names()
    var n = len(column)
    if not len(names):
        # Polars returns a valid empty struct even for null input strings.
        return Series("", StructColumn(List[Series](), length=n))
    var workers = worker_count(n)
    var jobs = List[_GroupsJob]()
    for w in range(workers):
        jobs.append(
            _GroupsJob(column, regex, n * w // workers, n * (w + 1) // workers)
        )
    if workers == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    var fields = List[Series]()
    for k in range(len(names)):
        var pieces = List[Series]()
        for j in range(len(jobs)):
            pieces.append(jobs[j].fields[k].copy())
        fields.append(Series._from_chunks(pieces).rechunk())
    var bits = _copy_validity(column._bits[], column._offset, n)
    return Series("", StructColumn(fields^, bits^))
