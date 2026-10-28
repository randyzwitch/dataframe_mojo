"""Derive full relations and ordering evidence from independent reference SQL.

This is a clause scanner, not a query translator. Only the outer SELECT is
extended; nested queries, quoted text and comments are retained verbatim.
Unhandled outer set operations fail closed.
"""
import re


def tokens(sql):
    """Yield (word, start, end) for unquoted depth-zero SQL tokens."""
    depth, i = 0, 0
    while i < len(sql):
        c = sql[i]
        if sql.startswith("--", i):
            end = sql.find("\n", i + 2)
            i = len(sql) if end < 0 else end + 1
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated SQL comment")
            i = end + 2
        elif c in "'\"":
            quote = c
            i += 1
            while i < len(sql):
                if sql[i] == quote:
                    i += 1
                    if i < len(sql) and sql[i] == quote:
                        i += 1
                        continue
                    break
                i += 1
            else:
                raise ValueError("unterminated SQL quote")
        elif c == "(":
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced SQL parentheses")
            i += 1
        elif depth == 0 and (c.isalpha() or c == "_" or c.isdigit()):
            match = re.match(r"[\w]+", sql[i:])
            end = i + len(match[0])
            yield sql[i:end].upper(), i, end
            i = end
        elif depth == 0 and c == ",":
            yield ",", i, i + 1
            i += 1
        else:
            i += 1
    if depth:
        raise ValueError("unbalanced SQL parentheses")


def split_expressions(text):
    cuts = [start for word, start, _ in tokens(text) if word == ","]
    positions = [-1] + cuts + [len(text)]
    return [
        text[left + 1 : right].strip()
        for left, right in zip(positions, positions[1:])
    ]


def full_relation(sql, output_names=None):
    """Return SQL with hidden sort fields, plus LIMIT/OFFSET/order metadata."""
    sql = sql.strip().rstrip(";")
    words = list(tokens(sql))
    if any(
        word in {"UNION", "EXCEPT", "INTERSECT", "FETCH"}
        for word, _, _ in words
    ):
        raise ValueError("unhandled outer SQL set operation or FETCH")
    selects = [t for t in words if t[0] == "SELECT"]
    if len(selects) != 1:
        raise ValueError("expected one outer SELECT")
    select = selects[0]
    source = next(
        (t for t in words if t[0] == "FROM" and t[1] > select[1]), None
    )
    if source is None:
        raise ValueError("outer SELECT has no FROM")
    order = next(
        (
            words[i]
            for i in range(len(words) - 1)
            if words[i][0] == "ORDER" and words[i + 1][0] == "BY"
        ),
        None,
    )
    limit_token = next((t for t in words if t[0] == "LIMIT"), None)
    offset_token = next((t for t in words if t[0] == "OFFSET"), None)
    limit, offset = None, 0
    if limit_token:
        tail = sql[limit_token[2] :]
        match = re.fullmatch(r"\s*(\d+)\s*(?:OFFSET\s+(\d+)\s*)?", tail, re.I)
        if not match:
            raise ValueError("unhandled SQL LIMIT clause")
        limit, offset = int(match[1]), int(match[2] or 0)
    elif offset_token:
        raise ValueError("OFFSET without LIMIT is unhandled")
    end = min([t[1] for t in (order, limit_token) if t] or [len(sql)])
    relation = sql[:end].rstrip()
    order_specs, expressions = [], []
    if order:
        by = next(t for t in words if t[1] > order[1] and t[0] == "BY")
        text = sql[by[2] : limit_token[1] if limit_token else len(sql)]
        projection_text = sql[select[2] : source[1]]
        distinct = bool(re.match(r"\s*DISTINCT\b", projection_text, re.I))
        if re.match(r"\s*DISTINCT\s+ON\b", projection_text, re.I):
            raise ValueError("DISTINCT ON requires a dedicated selection proof")
        if distinct:
            projection_text = re.sub(
                r"^\s*DISTINCT\s+", "", projection_text, flags=re.I
            )
        projections = split_expressions(projection_text)
        aliases = {}
        projection_values = []
        for index, projection in enumerate(projections):
            alias = re.fullmatch(
                r'(.+?)\s+(?:AS\s+)?("(?:[^"]|"")+"|[A-Za-z_]\w*)',
                projection,
                re.I | re.S,
            )
            if alias and (
                re.search(r"\sAS\s", projection, re.I)
                or (
                    output_names is not None
                    and len(output_names) == len(projections)
                    and alias[2].strip('"').lower()
                    == output_names[index].lower()
                )
            ):
                name = alias[2].strip('"').lower()
                aliases[name] = alias[1].rstrip()
                projection_values.append(alias[1].rstrip())
            else:
                projection_values.append(projection)
        for term in split_expressions(text):
            # SQL/DuckDB defaults: ASC and NULLS LAST, including DESC.
            match = re.fullmatch(
                r"(.+?)(?:\s+(ASC|DESC))?(?:\s+NULLS\s+(FIRST|LAST))?",
                term,
                re.I | re.S,
            )
            expression = match[1].strip()
            if expression.isdigit():
                position = int(expression) - 1
                if "*" in projections:
                    if (
                        output_names is None
                        or position < 0
                        or position >= len(output_names)
                    ):
                        raise ValueError(
                            "ORDER BY star ordinal requires original output schema"
                        )
                    expression = (
                        '"' + output_names[position].replace('"', '""') + '"'
                    )
                else:
                    if position < 0 or position >= len(projections):
                        raise ValueError("unhandled ORDER BY ordinal")
                    expression = projection_values[position]
            # ORDER BY resolves output aliases before input names. In SELECT,
            # the same identifier can bind an unaggregated input instead.
            # Expand the alias expression to preserve the original meaning.
            expression = aliases.get(expression.strip('"').lower(), expression)
            # Adding a hidden source field to DISTINCT can split one original
            # row into multiple rows. Only duplicate projected expressions
            # are safe without a separate functional-dependency proof.
            if distinct and expression.strip().lower() not in {
                p.strip().lower() for p in projection_values
            }:
                raise ValueError(
                    "hidden DISTINCT ordering requires a dedicated proof"
                )
            expressions.append(expression)
            order_specs.append(
                {
                    "descending": (match[2] or "ASC").upper() == "DESC",
                    "nulls_first": (match[3] or "LAST").upper() == "FIRST",
                }
            )
    if expressions:
        extra = ", " + ", ".join(
            f'{expr} AS "__suite_rank_{i}"'
            for i, expr in enumerate(expressions)
        )
        relation = (
            relation[: source[1]].rstrip() + extra + " " + relation[source[1] :]
        )
    return relation, {"limit": limit, "offset": offset, "order": order_specs}
