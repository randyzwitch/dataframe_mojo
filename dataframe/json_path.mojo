"""Compiled JSONPath selectors and predicates over the shared JSON tape."""
from .json_reader import JsonReader, json_bytes, _utf8
from .json_number import is_json_number, json_number_text
from .parse import parse_float64
from .json_path_number import parse_jsonpath_float

# Selector kinds: field,index,wildcard,index union,name union,slice,filter.
@fieldwise_init
struct _Step(Copyable, Movable):
    var kind: Int
    var text: String
    var indices: List[Int]
    var names: List[String]
    var start: Int
    var end: Int
    var stride: Int
    var has_start: Bool
    var has_end: Bool
    var predicate: Int
    var recursive: Bool


def _step(kind: Int) -> _Step:
    return _Step(
        kind, "", List[Int](), List[String](), 0, 0, 1, False, False, -1, False
    )


# Predicate kinds: numeric,string,bool,path,comparison,and,or.
@fieldwise_init
struct _Expr(Copyable, Movable):
    var kind: Int
    var op: Int
    var left: Int
    var right: Int
    var number: Float64
    var text: String
    var path: Int
    var root: Bool
    var boolean: Bool


def _expr(
    kind: Int,
    op: Int = 0,
    left: Int = -1,
    right: Int = -1,
    number: Float64 = 0,
    text: String = "",
    path: Int = -1,
    root: Bool = False,
    boolean: Bool = False,
) -> _Expr:
    return _Expr(kind, op, left, right, number, text, path, root, boolean)


struct _Parser(Movable):
    var text: String
    var at: Int
    var depth: Int
    var steps: List[_Step]
    var paths: List[List[Int]]
    var expressions: List[_Expr]

    def __init__(out self, text: String):
        self.text = text
        self.at = 0
        self.depth = 0
        self.steps = List[_Step]()
        self.paths = List[List[Int]]()
        self.expressions = List[_Expr]()

    def byte(self) -> Int:
        return (
            Int(self.text.as_bytes()[self.at]) if self.at
            < self.text.byte_length() else -1
        )

    def peek(self, offset: Int) -> Int:
        return (
            Int(self.text.as_bytes()[self.at + offset]) if self.at + offset
            < self.text.byte_length() else -1
        )

    def space(mut self):
        while self.byte() in [32, 9, 10, 13]:
            self.at += 1

    def expect(mut self, byte: Int) raises:
        if self.byte() != byte:
            raise Error("Invalid JSONPath at byte " + String(self.at))
        self.at += 1

    def quoted(mut self) raises -> String:
        var quote = self.byte()
        if quote != 34 and quote != 39:
            raise Error("Expected quoted JSONPath field")
        self.at += 1
        var out = List[UInt8]()
        while self.at < self.text.byte_length():
            var byte = self.byte()
            self.at += 1
            if byte == quote:
                return String(unsafe_from_utf8=out)
            if byte < 32:
                raise Error("Control character in JSONPath string")
            if byte != 92:
                out.append(UInt8(byte))
                continue
            byte = self.byte()
            self.at += 1
            if byte == 117:
                var reader = JsonReader(json_bytes(self.text), self.at)
                _utf8(out, reader.escaped_codepoint())
                self.at = reader.pos
            elif byte == 34 or byte == 39 or byte == 92 or byte == 47:
                out.append(UInt8(byte))
            elif (
                byte == 98
                or byte == 102
                or byte == 110
                or byte == 114
                or byte == 116
            ):
                out.append(
                    UInt8(
                        8 if byte
                        == 98 else (
                            12 if byte
                            == 102 else (
                                10 if byte
                                == 110 else (13 if byte == 114 else 9)
                            )
                        )
                    )
                )
            else:
                raise Error("Invalid JSONPath escape")
        raise Error("Unterminated JSONPath string")

    def integer(mut self) raises -> Int:
        var negative = self.byte() == 45
        if negative:
            self.at += 1
        var first = self.at
        var value = 0
        while self.byte() >= 48 and self.byte() <= 57:
            var digit = self.byte() - 48
            if value > (9223372036854775807 - digit) // 10:
                raise Error("JSONPath index overflow")
            value = value * 10 + digit
            self.at += 1
        if self.at == first:
            raise Error("Expected JSONPath index")
        return -value if negative else value

    def field_name(mut self) raises -> String:
        var start = self.at
        while self.at < self.text.byte_length():
            var b = self.byte()
            if b <= 32 or b in [46, 91, 93, 40, 41, 61, 33, 60, 62, 38, 124]:
                break
            self.at += 1
        if start == self.at:
            raise Error("Expected JSONPath field")
        return String(self.text[byte = start : self.at])

    def bracket(mut self) raises -> _Step:
        self.expect(91)
        self.space()
        if self.byte() == 42:
            self.at += 1
            self.space()
            self.expect(93)
            return _step(2)
        if self.byte() == 63:
            self.at += 1
            self.space()
            self.expect(40)
            self.depth += 1
            if self.depth > 64:
                raise Error("JSONPath predicate nesting exceeds 64")
            var predicate = self.expression(0)
            self.depth -= 1
            if self.expressions[predicate].kind <= 2:
                raise Error("JSONPath filter requires a path or comparison")
            self.space()
            self.expect(41)
            self.space()
            self.expect(93)
            var step = _step(6)
            step.predicate = predicate
            return step^
        if self.byte() == 34 or self.byte() == 39:
            var step = _step(4)
            step.names.append(self.quoted())
            self.space()
            while self.byte() == 44:
                self.at += 1
                self.space()
                step.names.append(self.quoted())
                self.space()
            self.expect(93)
            if len(step.names) == 1:
                step.kind = 0
                step.text = step.names[0]
            return step^
        var step = _step(1)
        if self.byte() != 58:
            step.start = self.integer()
            step.has_start = True
        self.space()
        if self.byte() == 58:
            step.kind = 5
            self.at += 1
            self.space()
            if self.byte() != 58 and self.byte() != 93:
                step.end = self.integer()
                step.has_end = True
            self.space()
            if self.byte() == 58:
                self.at += 1
                self.space()
                step.stride = self.integer()
                if step.stride <= 0:
                    raise Error("JSONPath slice step must be positive")
            self.space()
        elif self.byte() == 44:
            step.kind = 3
            step.indices.append(step.start)
            while self.byte() == 44:
                self.at += 1
                self.space()
                step.indices.append(self.integer())
                self.space()
        elif not step.has_start:
            raise Error("Empty JSONPath subscript")
        self.expect(93)
        return step^

    def path(mut self, predicate: Bool = False) raises -> Tuple[Int, Bool]:
        var root = self.byte() == 36
        if not root and not (predicate and self.byte() == 64):
            raise Error("JSONPath must start with '$'")
        self.at += 1
        var sequence = List[Int]()
        while self.byte() == 46 or self.byte() == 91:
            var recursive = False
            var step = _step(0)
            if self.byte() == 46:
                self.at += 1
                if self.byte() == 46:
                    recursive = True
                    self.at += 1
                if self.byte() == 91:
                    step = self.bracket()
                elif self.byte() == 42:
                    self.at += 1
                    step.kind = 2
                else:
                    step.text = self.field_name()
            else:
                step = self.bracket()
            step.recursive = recursive
            sequence.append(len(self.steps))
            self.steps.append(step^)
        var result = len(self.paths)
        self.paths.append(sequence^)
        return (result, root)

    def add(mut self, var expr: _Expr) -> Int:
        var index = len(self.expressions)
        self.expressions.append(expr^)
        return index

    def primary(mut self) raises -> Int:
        self.space()
        if self.byte() == 40:
            self.at += 1
            self.depth += 1
            if self.depth > 64:
                raise Error("JSONPath predicate nesting exceeds 64")
            var value = self.expression(0)
            self.space()
            self.expect(41)
            self.depth -= 1
            return value
        if self.byte() == 34 or self.byte() == 39:
            return self.add(_expr(1, text=self.quoted()))
        if self.byte() == 64 or self.byte() == 36:
            var path = self.path(True)
            return self.add(_expr(3, path=path[0], root=path[1]))
        if self.byte() == 45 or (self.byte() >= 48 and self.byte() <= 57):
            var start = self.at
            var reader = JsonReader(json_bytes(self.text), start)
            reader.scan_number()
            self.at = reader.pos
            var number = parse_float64(reader.slice(start, self.at))
            if number != number or abs(number) == Float64("inf"):
                raise Error("Invalid JSONPath number")
            return self.add(_expr(0, number=number))
        if (
            StringSlice(
                self.text[
                    byte = self.at : self.at
                    + min(4, self.text.byte_length() - self.at)
                ]
            )
            == "true"
        ):
            self.at += 4
            return self.add(_expr(2, boolean=True))
        if (
            StringSlice(
                self.text[
                    byte = self.at : self.at
                    + min(5, self.text.byte_length() - self.at)
                ]
            )
            == "false"
        ):
            self.at += 5
            return self.add(_expr(2))
        raise Error(
            "Expected JSONPath predicate operand at byte " + String(self.at)
        )

    def expression(mut self, minimum: Int) raises -> Int:
        var left = self.primary()
        while True:
            self.space()
            var a = self.byte()
            var b = self.peek(1)
            var op = 0
            var precedence = 3
            var width = 2
            if a == 61 and b == 61:
                op = 1
            elif a == 33 and b == 61:
                op = 2
            elif a == 60:
                op = 4 if b == 61 else 3
                width = 2 if b == 61 else 1
            elif a == 62:
                op = 6 if b == 61 else 5
                width = 2 if b == 61 else 1
            elif a == 38 and b == 38:
                precedence = 2
            elif a == 124 and b == 124:
                precedence = 1
            else:
                break
            if precedence < minimum:
                break
            self.at += width
            var right = self.expression(precedence + 1)
            if precedence < 3 and (
                self.expressions[left].kind <= 2
                or self.expressions[right].kind <= 2
            ):
                raise Error(
                    "JSONPath logical operands require paths or comparisons"
                )
            left = self.add(
                _expr(
                    4 if precedence == 3 else (5 if precedence == 2 else 6),
                    op=op,
                    left=left,
                    right=right,
                )
            )
        return left


struct JsonPath(Copyable, Movable):
    var steps: List[_Step]
    var paths: List[List[Int]]
    var expressions: List[_Expr]
    var simple: List[Bool]
    var references: List[Bool]
    var root: Int

    def __init__(out self, text: String) raises:
        var parser = _Parser(text)
        var path = parser.path()
        if parser.at != text.byte_length():
            raise Error(
                "Unexpected JSONPath content at byte " + String(parser.at)
            )
        self.steps = List[_Step]()
        self.paths = List[List[Int]]()
        self.expressions = List[_Expr]()
        swap(self.steps, parser.steps)
        swap(self.paths, parser.paths)
        swap(self.expressions, parser.expressions)
        self.root = path[0]
        self.simple = List[Bool]()
        self.references = List[Bool]()
        for expression in self.expressions:
            var reference = expression.kind == 3
            if expression.kind >= 4:
                reference = (
                    self.references[expression.left]
                    or self.references[expression.right]
                )
            self.references.append(reference)
        for i in range(len(self.paths)):
            var simple = True
            for index in self.paths[i]:
                if self.steps[index].recursive or (
                    self.steps[index].kind != 0 and self.steps[index].kind != 1
                ):
                    simple = False
            self.simple.append(simple)

    def first(self, reader: JsonReader, mut scratch: List[Int]) raises -> Int:
        if self.simple[self.root]:
            return _resolve(self, reader, self.root, 0)
        scratch.clear()
        _ = _walk(self, reader, self.root, 0, 0, scratch, 1)
        return scratch[0] if len(scratch) else -1


def _key(reader: JsonReader, node: Int) raises -> String:
    ref field = reader.nodes[node]
    if field.key_escaped:
        return reader.string_at(field.key_start)
    return String(reader.slice(field.key_start + 1, field.key_end - 1))


def _field(reader: JsonReader, node: Int, name: String) raises -> Int:
    if node < 0 or reader.nodes[node].kind != 123:
        return -1
    var found = -1
    var child = node + 1
    while child < reader.nodes[node].after:
        ref token = reader.nodes[child]
        if token.key_escaped:
            if reader.string_at(token.key_start) == name:
                found = child
        else:
            var bytes = reader.slice(
                token.key_start + 1, token.key_end - 1
            ).as_bytes()
            var key = name.as_bytes()
            if len(bytes) == len(key):
                var equal = True
                for i in range(len(bytes)):
                    if (
                        bytes.unsafe_ptr()[unsafe_offset=i]
                        != key.unsafe_ptr()[unsafe_offset=i]
                    ):
                        equal = False
                        break
                if equal:
                    found = child
        child = reader.nodes[child].after
    return found


def _children(reader: JsonReader, node: Int) raises -> List[Int]:
    var out = List[Int]()
    var kind = reader.nodes[node].kind
    if kind != 123 and kind != 91:
        return out^
    var positions = Dict[String, Int]()
    var child = node + 1
    while child < reader.nodes[node].after:
        if kind == 123:
            var name = _key(reader, child)
            if name in positions:
                out[positions[name]] = child
            else:
                positions[name] = len(out)
                out.append(child)
        else:
            out.append(child)
        child = reader.nodes[child].after
    return out^


def _array(reader: JsonReader, node: Int, index: Int) -> Int:
    if node < 0 or reader.nodes[node].kind != 91:
        return -1
    var target = index
    if target < 0:
        var count = 0
        var child = node + 1
        while child < reader.nodes[node].after:
            count += 1
            child = reader.nodes[child].after
        target += count
    if target < 0:
        return -1
    var child = node + 1
    var row = 0
    while child < reader.nodes[node].after:
        if row == target:
            return child
        row += 1
        child = reader.nodes[child].after
    return -1


def _resolve(
    program: JsonPath, reader: JsonReader, path: Int, current: Int
) raises -> Int:
    var node = current
    for index in program.paths[path]:
        ref step = program.steps[index]
        node = _field(reader, node, step.text) if step.kind == 0 else _array(
            reader, node, step.start
        )
        if node < 0:
            break
    return node


def _send(
    program: JsonPath,
    reader: JsonReader,
    path: Int,
    next: Int,
    node: Int,
    mut out: List[Int],
    limit: Int,
) raises -> Bool:
    return node >= 0 and _walk(program, reader, path, next, node, out, limit)


def _select(
    program: JsonPath,
    reader: JsonReader,
    path: Int,
    position: Int,
    node: Int,
    mut out: List[Int],
    limit: Int,
) raises -> Bool:
    ref step = program.steps[program.paths[path][position]]
    var next = position + 1
    if step.kind == 0:
        return _send(
            program,
            reader,
            path,
            next,
            _field(reader, node, step.text),
            out,
            limit,
        )
    if step.kind == 1:
        return _send(
            program,
            reader,
            path,
            next,
            _array(
                reader,
                node,
                max(0, step.start) if step.recursive else step.start,
            ),
            out,
            limit,
        )
    if step.kind == 3:
        for index in step.indices:
            if _send(
                program,
                reader,
                path,
                next,
                _array(reader, node, index),
                out,
                limit,
            ):
                return True
        return False
    if step.kind == 4:
        for name in step.names:
            if _send(
                program,
                reader,
                path,
                next,
                _field(reader, node, name),
                out,
                limit,
            ):
                return True
        return False
    if step.kind == 6:
        ref predicate = program.expressions[step.predicate]
        if not program.references[step.predicate]:
            if _predicate(program, reader, step.predicate, node):
                return _send(program, reader, path, next, node, out, limit)
            return False
        # Recursive predicate paths select the matching descendants in the
        # Polars evaluator; ordinary field predicates retain their owners.
        if _recursive_operand(program, step.predicate):
            for selected in _values(program, reader, step.predicate, node):
                if _send(program, reader, path, next, selected, out, limit):
                    return True
            return False
        if predicate.kind == 4:
            var left_recursive = _recursive_operand(program, predicate.left)
            var right_recursive = _recursive_operand(program, predicate.right)
            var other = predicate.right if left_recursive else predicate.left
            if (left_recursive or right_recursive) and program.expressions[
                other
            ].kind <= 2:
                var operand = (
                    predicate.left if left_recursive else predicate.right
                )
                var literal = _value(program, reader, other, node)
                for selected in _values(program, reader, operand, node):
                    var value = _Value(1, selected)
                    if _compare_values(
                        program,
                        reader,
                        value.copy() if left_recursive else literal.copy(),
                        literal.copy() if left_recursive else value.copy(),
                        predicate.op,
                    ):
                        if _send(
                            program, reader, path, next, selected, out, limit
                        ):
                            return True
                return False
        if predicate.kind == 5 or predicate.kind == 6:
            var term = _filter_term(program, reader, step.predicate, node)
            if term.scalar:
                if term.boolean:
                    return _send(program, reader, path, next, node, out, limit)
                return False
            for selected in term.nodes:
                if _send(program, reader, path, next, selected, out, limit):
                    return True
            return False
        var matched = False
        if reader.nodes[node].kind == 91 and not _bare_current(
            program, step.predicate
        ):
            var child = node + 1
            while child < reader.nodes[node].after:
                if _predicate(program, reader, step.predicate, child):
                    matched = True
                    if _send(program, reader, path, next, child, out, limit):
                        return True
                child = reader.nodes[child].after
        elif _predicate(program, reader, step.predicate, node):
            matched = True
            if _send(program, reader, path, next, node, out, limit):
                return True
        # Polars' JSONPath inequality retains its current input selection
        # when no elements satisfy that direct inequality predicate.
        if not matched and predicate.kind == 4 and predicate.op == 2:
            return _send(program, reader, path, next, node, out, limit)
        return False
    if step.kind == 5:
        if reader.nodes[node].kind != 91:
            return False
        var count = 0
        var child = node + 1
        while child < reader.nodes[node].after:
            count += 1
            child = reader.nodes[child].after
        var start = step.start if step.has_start else 0
        var end = step.end if step.has_end else count
        if start < 0:
            start += count
        if end < 0:
            end += count
        start = min(max(start, 0), count)
        end = min(max(end, 0), count)
        child = node + 1
        var at = 0
        while child < reader.nodes[node].after and at < end:
            if (
                at >= start
                and (at - start) % step.stride == 0
                and _send(program, reader, path, next, child, out, limit)
            ):
                return True
            child = reader.nodes[child].after
            at += 1
        return False
    for child in _children(reader, node):
        if _send(program, reader, path, next, child, out, limit):
            return True
    return False


def _walk(
    program: JsonPath,
    reader: JsonReader,
    path: Int,
    position: Int,
    node: Int,
    mut out: List[Int],
    limit: Int,
) raises -> Bool:
    if position == len(program.paths[path]):
        out.append(node)
        return limit > 0 and len(out) >= limit
    ref step = program.steps[program.paths[path][position]]
    if not step.recursive or step.kind == 4:
        return _select(program, reader, path, position, node, out, limit)
    return _descend(program, reader, path, position, node, out, limit)


def _descend(
    program: JsonPath,
    reader: JsonReader,
    path: Int,
    position: Int,
    node: Int,
    mut out: List[Int],
    limit: Int,
) raises -> Bool:
    var kind = reader.nodes[node].kind
    if kind != 123 and kind != 91:
        return False
    if _select(program, reader, path, position, node, out, limit):
        return True
    # Only visit visible members. Scanning the raw tape would also visit
    # descendants of duplicate object keys superseded by a later value.
    if kind == 91:
        var child = node + 1
        while child < reader.nodes[node].after:
            if _descend(program, reader, path, position, child, out, limit):
                return True
            child = reader.nodes[child].after
    else:
        for child in _children(reader, node):
            if _descend(program, reader, path, position, child, out, limit):
                return True
    return False


# A predicate value is a tape node, a compiled literal, or an evaluated bool.
@fieldwise_init
struct _Value(Copyable, Movable):
    var kind: Int
    var index: Int


def _value(
    program: JsonPath, reader: JsonReader, index: Int, current: Int
) raises -> _Value:
    ref expression = program.expressions[index]
    if expression.kind <= 2:
        return _Value(2, index)
    if expression.kind == 3:
        var node = _resolve(
            program, reader, expression.path, 0 if expression.root else current
        )
        return _Value(1, node) if node >= 0 else _Value(0, 0)
    return _Value(3, Int(_predicate(program, reader, index, current)))


def _type(program: JsonPath, reader: JsonReader, value: _Value) -> Int:
    if value.kind == 0:
        return -1
    if value.kind == 3:
        return 1
    if value.kind == 2:
        var kind = program.expressions[value.index].kind
        return 2 if kind == 0 else (3 if kind == 1 else 1)
    var kind = reader.nodes[value.index].kind
    return 0 if kind == 110 else (
        1 if kind == 116
        or kind
        == 102 else (
            2 if is_json_number(kind) else (
                3 if kind == 34 else (4 if kind == 91 else 5)
            )
        )
    )


def _float(
    program: JsonPath, reader: JsonReader, value: _Value
) raises -> Float64:
    return program.expressions[
        value.index
    ].number if value.kind == 2 else parse_jsonpath_float(
        reader.raw(value.index)
    )


def _string(
    program: JsonPath, reader: JsonReader, value: _Value
) raises -> String:
    return program.expressions[
        value.index
    ].text.copy() if value.kind == 2 else reader.string_at(
        reader.nodes[value.index].start
    )


def _bool(program: JsonPath, reader: JsonReader, value: _Value) -> Bool:
    if value.kind == 3:
        return value.index != 0
    return (
        program.expressions[value.index].boolean if value.kind
        == 2 else reader.nodes[value.index].kind
        == 116
    )


def _equal(reader: JsonReader, a: Int, b: Int) raises -> Bool:
    if a == b:
        return True
    var x = reader.nodes[a].kind
    var y = reader.nodes[b].kind
    if is_json_number(x) and is_json_number(y):
        return parse_jsonpath_float(reader.raw(a)) == parse_jsonpath_float(
            reader.raw(b)
        )
    if x != y:
        return False
    if x == 34:
        return reader.string_at(reader.nodes[a].start) == reader.string_at(
            reader.nodes[b].start
        )
    if x == 123:
        var left = _children(reader, a)
        var right = _children(reader, b)
        if len(left) != len(right):
            return False
        for child in left:
            var other = _field(reader, b, _key(reader, child))
            if other < 0 or not _equal(reader, child, other):
                return False
        return True
    if x == 91:
        var i = a + 1
        var j = b + 1
        while i < reader.nodes[a].after and j < reader.nodes[b].after:
            if not _equal(reader, i, j):
                return False
            i = reader.nodes[i].after
            j = reader.nodes[j].after
        return i == reader.nodes[a].after and j == reader.nodes[b].after
    return True


def _compare_values(
    program: JsonPath, reader: JsonReader, a: _Value, b: _Value, op: Int
) raises -> Bool:
    var x = _type(program, reader, a)
    var y = _type(program, reader, b)
    if x < 0 or y < 0:
        return False
    if x != y:
        return op == 2
    if x == 0:
        return op == 1 or op == 4 or op == 6
    if x >= 4:
        var equal = _equal(reader, a.index, b.index)
        return equal if op == 1 else (not equal if op == 2 else False)
    var equal: Bool
    var less: Bool
    if x == 1:
        var left = _bool(program, reader, a)
        var right = _bool(program, reader, b)
        equal = left == right
        less = not left and right
    elif x == 2:
        var left = _float(program, reader, a)
        var right = _float(program, reader, b)
        equal = left == right
        less = left < right
    else:
        var left = _string(program, reader, a)
        var right = _string(program, reader, b)
        equal = left == right
        less = left < right
    return equal if op == 1 else (
        not equal if op
        == 2 else (
            less if op
            == 3 else (
                less
                or equal if op
                == 4 else (not less and not equal if op == 5 else not less)
            )
        )
    )


def _complex(program: JsonPath, index: Int) -> Bool:
    ref expression = program.expressions[index]
    return expression.kind == 3 and not program.simple[expression.path]


def _recursive_operand(program: JsonPath, index: Int) -> Bool:
    ref expression = program.expressions[index]
    if expression.kind != 3:
        return False
    for step in program.paths[expression.path]:
        if program.steps[step].recursive:
            return True
    return False


def _values(
    program: JsonPath, reader: JsonReader, index: Int, current: Int
) raises -> List[Int]:
    ref expression = program.expressions[index]
    var out = List[Int]()
    _ = _walk(
        program,
        reader,
        expression.path,
        0,
        0 if expression.root else current,
        out,
        0,
    )
    return out^


def _comparison(
    program: JsonPath, reader: JsonReader, index: Int, current: Int
) raises -> Bool:
    ref expression = program.expressions[index]
    ref a = program.expressions[expression.left]
    ref b = program.expressions[expression.right]
    if (
        a.kind == 3
        and b.kind == 3
        and not a.root
        and not b.root
        and program.simple[a.path]
        and program.simple[b.path]
    ):
        # Relative path terms in Polars compare their owning selections,
        # whereas a path versus a scalar literal compares the field value.
        var left = _resolve(program, reader, a.path, current) >= 0
        var right = _resolve(program, reader, b.path, current) >= 0
        return (
            left
            and (right if expression.op == 1 else not right) if expression.op
            <= 2 else False
        )
    if _complex(program, expression.left):
        var values = _values(program, reader, expression.left, current)
        if _complex(program, expression.right):
            var other = _values(program, reader, expression.right, current)
            for a in values:
                for b in other:
                    if _compare_values(
                        program,
                        reader,
                        _Value(1, a),
                        _Value(1, b),
                        expression.op,
                    ):
                        return True
        else:
            var right = _value(program, reader, expression.right, current)
            for a in values:
                if _compare_values(
                    program, reader, _Value(1, a), right, expression.op
                ):
                    return True
        return False
    if _complex(program, expression.right):
        var values = _values(program, reader, expression.right, current)
        var left = _value(program, reader, expression.left, current)
        for b in values:
            if _compare_values(
                program, reader, left, _Value(1, b), expression.op
            ):
                return True
        return False
    return _compare_values(
        program,
        reader,
        _value(program, reader, expression.left, current),
        _value(program, reader, expression.right, current),
        expression.op,
    )


def _predicate(
    program: JsonPath, reader: JsonReader, index: Int, current: Int
) raises -> Bool:
    ref expression = program.expressions[index]
    if expression.kind == 5:
        return _predicate(
            program, reader, expression.left, current
        ) and _predicate(program, reader, expression.right, current)
    if expression.kind == 6:
        return _predicate(
            program, reader, expression.left, current
        ) or _predicate(program, reader, expression.right, current)
    if expression.kind == 4:
        return _comparison(program, reader, index, current)
    if expression.kind == 3:
        if program.simple[expression.path]:
            return (
                _resolve(
                    program,
                    reader,
                    expression.path,
                    0 if expression.root else current,
                )
                >= 0
            )
        var values = _values(program, reader, index, current)
        return len(values) != 0
    if expression.kind == 2:
        return expression.boolean
    if expression.kind == 0:
        return expression.number != 0
    return expression.text.byte_length() != 0


def _bare_current(program: JsonPath, index: Int) -> Bool:
    ref expression = program.expressions[index]
    if expression.kind == 3:
        return not expression.root and len(program.paths[expression.path]) == 0
    if expression.kind >= 4:
        return _bare_current(program, expression.left) or _bare_current(
            program, expression.right
        )
    return False


@fieldwise_init
struct _FilterTerm(Movable):
    var scalar: Bool
    var boolean: Bool
    var nodes: List[Int]
    var key: Int


def _filter_bool(value: Bool) -> _FilterTerm:
    return _FilterTerm(True, value, List[Int](), -1)


def _filter_term(
    program: JsonPath, reader: JsonReader, index: Int, node: Int
) raises -> _FilterTerm:
    # The upstream evaluator distinguishes Boolean scalars from selections.
    # Combining two selections uses identity, in left-query order; combining
    # one with a scalar compares only its Boolean values, not its truthiness.
    ref expression = program.expressions[index]
    if not program.references[index]:
        return _filter_bool(_predicate(program, reader, index, node))
    var out = List[Int]()
    if expression.kind == 5 or expression.kind == 6:
        var left = _filter_term(program, reader, expression.left, node)
        var right = _filter_term(program, reader, expression.right, node)
        if left.scalar and right.scalar:
            return _filter_bool(
                left.boolean
                and right.boolean if expression.kind
                == 5 else (left.boolean or right.boolean)
            )
        if left.scalar or right.scalar:
            var boolean = left.boolean if left.scalar else right.boolean
            var key = right.key if left.scalar else left.key
            var nodes = List[Int]()
            if left.scalar:
                swap(nodes, right.nodes)
            else:
                swap(nodes, left.nodes)
            for selected in nodes:
                var value = selected
                if reader.nodes[value].kind == 123 and key >= 0:
                    value = _field(reader, value, program.steps[key].text)
                if value < 0:
                    continue
                var kind = reader.nodes[value].kind
                if kind == 116 or kind == 102:
                    var keep = (
                        kind == 116
                        and boolean if expression.kind
                        == 5 else (kind == 116 or boolean)
                    )
                    if keep:
                        out.append(selected)
        else:
            var seen = Dict[Int, Bool]()
            if expression.kind == 5:
                for selected in right.nodes:
                    seen[selected] = True
                for selected in left.nodes:
                    if selected in seen:
                        out.append(selected)
            else:
                for selected in left.nodes:
                    seen[selected] = True
                    out.append(selected)
                for selected in right.nodes:
                    if selected not in seen:
                        seen[selected] = True
                        out.append(selected)
        if len(out) == 0:
            return _filter_bool(False)
        return _FilterTerm(False, False, out^, -1)
    if reader.nodes[node].kind == 91 and not _bare_current(program, index):
        var child = node + 1
        while child < reader.nodes[node].after:
            if _predicate(program, reader, index, child):
                out.append(child)
            child = reader.nodes[child].after
    elif _predicate(program, reader, index, node):
        out.append(node)
    if expression.kind == 4 and len(out) == 0:
        return _filter_bool(expression.op == 2)
    var key = -1
    if expression.kind == 3 and not expression.root:
        ref path = program.paths[expression.path]
        if len(path) == 1 and program.steps[path[0]].kind == 0:
            key = path[0]
    return _FilterTerm(False, False, out^, key)


def _quoted(reader: JsonReader, start: Int, mut out: List[UInt8]) raises:
    out.append(34)
    var cursor = JsonReader(reader.span(), start + 1)
    while cursor.pos < cursor.size:
        var b = cursor.byte(cursor.pos)
        cursor.pos += 1
        if b == 34:
            out.append(34)
            return
        if b != 92:
            out.append(UInt8(b))
            continue
        b = cursor.byte(cursor.pos)
        cursor.pos += 1
        if b == 117:
            var cp = cursor.escaped_codepoint()
            if cp < 32:
                var digits = StringSlice("0123456789abcdef")
                # Serde prefers the short spelling for these five controls.
                var short = 98 if cp == 8 else (
                    102 if cp
                    == 12 else (
                        110 if cp
                        == 10 else (
                            114 if cp == 13 else (116 if cp == 9 else 0)
                        )
                    )
                )
                if short != 0:
                    out.append(92)
                    out.append(UInt8(short))
                else:
                    out.extend(StringSlice("\\u00").as_bytes())
                    out.append(digits.as_bytes()[cp // 16])
                    out.append(digits.as_bytes()[cp % 16])
            elif cp == 34 or cp == 92:
                out.append(92)
                out.append(UInt8(cp))
            else:
                _utf8(out, cp)
        elif b == 47:
            out.append(47)
        else:
            out.append(92)
            out.append(UInt8(b))
    raise Error("Unterminated JSON string")


def append_json(reader: JsonReader, node: Int, mut out: List[UInt8]) raises:
    var kind = reader.nodes[node].kind
    if kind == 34:
        _quoted(reader, reader.nodes[node].start, out)
    elif is_json_number(kind):
        var text = json_number_text(reader, node, False)
        out.extend(text.as_bytes())
    elif kind == 123:
        out.append(123)
        var fields = _children(reader, node)
        for i in range(len(fields)):
            if i:
                out.append(44)
            _quoted(reader, reader.nodes[fields[i]].key_start, out)
            out.append(58)
            append_json(reader, fields[i], out)
        out.append(125)
    elif kind == 91:
        out.append(91)
        var child = node + 1
        var first = True
        while child < reader.nodes[node].after:
            if not first:
                out.append(44)
            first = False
            append_json(reader, child, out)
            child = reader.nodes[child].after
        out.append(93)
    else:
        out.extend(reader.raw(node).as_bytes())
