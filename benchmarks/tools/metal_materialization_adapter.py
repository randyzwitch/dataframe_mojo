"""Reviewed API type flow for scalar eager SELECT after a forced collection.

This is a small token adapter for the benchmark modules, not a Mojo compiler.
Only a lexically proved collected DataFrame and a single scalar aggregate Expr
can use this bridge. Unknown syntax remains unchanged and is reported. It never
rewrites expressions, CPU workers, canonical query files, or table inputs.
"""
import hashlib
import io
import tokenize

LAZY_METHODS = {
    "select",
    "select_exprs",
    "with_columns",
    "filter",
    "join",
    "join_asof",
    "sort",
    "head",
    "slice",
    "drop",
    "unique",
    "concat",
    "copy",
    "explode",
}
REDUCTIONS = {
    "sum",
    "mean",
    "min",
    "max",
    "first",
    "last",
    "count",
    "len",
    "n_unique",
    "null_count",
    "any",
    "all",
    "var",
    "std",
    "median",
    "quantile",
}
UNKNOWN = ("unknown", False)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def strings(tokens):
    return [t.string for t in tokens]


def closing(tokens, start):
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for index in range(start, len(tokens)):
        value = tokens[index].string
        if value in pairs:
            stack.append(pairs[value])
        elif value in pairs.values():
            if not stack or stack.pop() != value:
                raise ValueError("Unbalanced materialization token boundary")
            if not stack:
                return index
    raise ValueError("Unclosed materialization token boundary")


def infer(tokens, lookup, functions):
    """Conservative result type/collection provenance for a method chain."""
    if not tokens:
        return UNKNOWN
    index = 1
    first = tokens[0].string
    if first == "(":
        end = closing(tokens, 0)
        kind, collected = infer(tokens[1:end], lookup, functions)
        index = end + 1
    elif tokens[0].type == tokenize.NAME:
        known = lookup(first)
        if index < len(tokens) and tokens[index].string == "(":
            end = closing(tokens, index)
            if known is not None:
                return UNKNOWN  # A local shadow is not a known API function.
            kind, collected = functions.get(first, UNKNOWN)
            index = end + 1
        else:
            kind, collected = known if known is not None else UNKNOWN
            if index < len(tokens) and tokens[index].string == "[":
                index = closing(tokens, index) + 1
                kind, collected = (
                    "frame",
                    False,
                ) if kind == "tables" else UNKNOWN
    else:
        return UNKNOWN
    while index < len(tokens):
        if (
            index + 2 >= len(tokens)
            or tokens[index].string != "."
            or tokens[index + 2].string != "("
        ):
            return UNKNOWN
        method = tokens[index + 1].string
        end = closing(tokens, index + 2)
        args = tokens[index + 3 : end]
        if kind == "frame" and method == "lazy" and not args:
            kind = "lazy"
        elif kind == "frame" and method == "copy" and not args:
            pass
        elif kind == "lazy" and method in LAZY_METHODS:
            pass
        elif kind == "lazy" and method == "group_by":
            kind = "lazy_group"
        elif kind == "lazy_group" and method == "agg":
            kind = "lazy"
        elif kind == "lazy" and method == "collect" and not args:
            kind, collected = "frame", True
        elif kind in {"expr_rows", "expr_scalar"}:
            if method in REDUCTIONS:
                kind = "expr_scalar"
            elif method not in {"cast", "alias"}:
                return UNKNOWN
        else:
            return UNKNOWN
        index = end + 1
    return kind, collected


def signature(tokens):
    """Read explicit benchmark helper parameter/return type annotations."""
    values = strings(tokens)
    if not values or values[0] != "def" or "(" not in values:
        return {}, None
    start = values.index("(")
    end = closing(tokens, start)
    parameters = {}
    index = start + 1
    while index < end:
        if index + 1 < end and values[index + 1] == ":":
            name, type_start = values[index], index + 2
            if values[type_start : type_start + 6] == [
                "Dict",
                "[",
                "String",
                ",",
                "DataFrame",
                "]",
            ]:
                parameters[name] = ("tables", False)
            elif values[type_start] in {"DataFrame", "LazyFrame"}:
                parameters[name] = (
                    "frame" if values[type_start] == "DataFrame" else "lazy",
                    False,
                )
        index += 1
    result = None
    if "->" in values:
        annotation = values[values.index("->") + 1]
        if annotation in {"DataFrame", "LazyFrame", "Expr"}:
            result = (
                "frame" if annotation
                == "DataFrame" else "lazy" if annotation
                == "LazyFrame" else "expr_rows",
                False,
            )
    return parameters, (values[1], result) if result else None


def lift_scalar_selects(source):
    """Insert only lazy()/collect() around proved scalar DataFrame selections."""
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    ignored = {tokenize.NL, tokenize.COMMENT, tokenize.ENCODING}
    clean = [t for t in tokens if t.type not in ignored]
    functions = {"col": ("expr_rows", False)}
    statements, current = [], []
    for token in clean:
        if token.type == tokenize.NEWLINE:
            if current:
                statements.append(current)
            current = []
        elif token.type not in {
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
        }:
            current.append(token)
    for statement in statements:
        _, result = signature(statement)
        if result:
            functions[result[0]] = result[1]
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def offset(position):
        return offsets[position[0] - 1] + position[1]

    scopes = [{}]
    barriers = [False]
    pending = {}
    pending_barrier = False
    edits, proofs, unadapted = [], [], []

    def lookup(name):
        for scope, barrier in zip(reversed(scopes), reversed(barriers)):
            if name in scope:
                return scope[name]
            if barrier:
                break
        return None

    def process(statement):
        nonlocal pending, pending_barrier
        if not statement:
            return
        values = strings(statement)
        if values[0] == "def":
            pending, _ = signature(statement)
            pending_barrier = True
            return
        if values[0] == "for" and "in" in values:
            pending = {
                name: UNKNOWN
                for name in values[1 : values.index("in")]
                if name.isidentifier()
            }
        elif values[0] == "except" and len(values) > 2:
            pending = {values[1]: UNKNOWN}
        elif values[0] == "with" and "as" in values:
            pending = {values[values.index("as") + 1]: UNKNOWN}
        # Only identifier.select(single scalar Expr).item() is reviewed. Token
        # bounds exclude comments/strings and retain the exact argument bytes.
        for index in range(len(statement) - 3):
            if statement[index].type != tokenize.NAME or values[
                index + 1 : index + 4
            ] != [".", "select", "("]:
                continue
            if index and values[index - 1] == ".":
                continue
            end = closing(statement, index + 3)
            if values[end + 1 : end + 5] != [".", "item", "(", ")"]:
                continue
            receiver = values[index]
            args = statement[index + 4 : end]
            reason = None
            if lookup(receiver) != ("frame", True):
                reason = (
                    "Receiver is not a lexically proved collected DataFrame"
                )
            elif infer(args, lookup, functions)[0] != "expr_scalar":
                reason = (
                    "SELECT is not one reviewed scalar aggregate expression"
                )
            if reason:
                unadapted.append(
                    {
                        "source_line": statement[index].start[0],
                        "receiver": receiver,
                        "reason": reason,
                    }
                )
                continue
            begin, finish = offset(statement[index].start), offset(
                statement[end].end
            )
            expression = source[
                offset(statement[index + 3].end) : offset(statement[end].start)
            ]
            edits.extend(
                [
                    (offset(statement[index].end), ".lazy()"),
                    (finish, ".collect()"),
                ]
            )
            proofs.append(
                {
                    "source_line": statement[index].start[0],
                    "receiver": receiver,
                    "receiver_type": "DataFrame returned by terminal LazyFrame.collect",
                    "selection_shape": "scalar aggregate Expr",
                    "selection_expression_sha256": digest(expression),
                    "original_selection_sha256": digest(source[begin:finish]),
                    "insertions_only": [
                        ".lazy() after receiver",
                        ".collect() after unchanged select call",
                    ],
                    "scope_policy": "lexical declarations and conservative assignment invalidation",
                }
            )
        depth, assignment = 0, None
        for index, value in enumerate(values):
            if value in {"(", "[", "{"}:
                depth += 1
            elif value in {")", "]", "}"}:
                depth -= 1
            elif depth == 0 and value in {
                "=",
                "+=",
                "-=",
                "*=",
                "/=",
                "//=",
                "%=",
                "|=",
                "&=",
                "^=",
            }:
                assignment = index
                break
        declaration = values[0] in {"var", "ref"}
        if declaration and len(values) > 2 and values[1].isidentifier():
            name = values[1]
            if "=" in values:
                # The first assignment token in a declaration precedes its RHS.
                equal = values.index("=")
                scopes[-1][name] = UNKNOWN if values[0] == "ref" else infer(
                    statement[equal + 1 :], lookup, functions
                )
            else:
                scopes[-1][name] = UNKNOWN
        elif len(values) > 1 and values[0].isidentifier() and values[1] == "=":
            name = values[0]
            inferred = infer(statement[2:], lookup, functions)
            if name in scopes[-1]:
                scopes[-1][name] = inferred
            else:
                # A conditional mutation cannot preserve an ancestor proof
                # after this block; retain only this block's known new value.
                for scope in scopes:
                    if name in scope:
                        scope[name] = UNKNOWN
                scopes[-1][name] = inferred
        elif assignment is not None:
            # Unsupported destructuring/augmented assignment cannot leave a
            # stale collected-frame proof, including a branch mutation.
            for name in values[:assignment]:
                if name.isidentifier() and name not in {"var", "ref"}:
                    for scope in scopes:
                        if name in scope:
                            scope[name] = UNKNOWN
                    scopes[-1][name] = UNKNOWN

    current = []
    for token in clean:
        if token.type == tokenize.INDENT:
            scopes.append(dict(pending))
            barriers.append(pending_barrier)
            pending = {}
            pending_barrier = False
        elif token.type == tokenize.DEDENT:
            if len(scopes) > 1:
                scopes.pop()
                barriers.pop()
        elif token.type == tokenize.NEWLINE:
            process(current)
            current = []
        elif token.type != tokenize.ENDMARKER:
            current.append(token)
    for at, text in sorted(edits, reverse=True):
        source = source[:at] + text + source[at:]
    return source, {
        "format": "dataframe-scalar-materialization-proof-v1",
        "adapted": proofs,
        "unadapted": unadapted,
        "whole_query_gpu_certificate": False,
    }
