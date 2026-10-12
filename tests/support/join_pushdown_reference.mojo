"""Independent scalar key membership and nested-loop join rows."""


@fieldwise_init
struct PushdownRow(Copyable, Movable):
    var values: List[Int64]
    var valid: List[Bool]


def row(var values: List[Int64], payload_valid: Bool = True) -> PushdownRow:
    var valid = List[Bool](length=len(values), fill=True)
    if len(values):
        valid[len(values) - 1] = payload_valid
    return PushdownRow(values^, valid^)


def fact_item(index: Int) -> Int64:
    return Int64(index % 1024)


def fact_day(index: Int) -> Int64:
    return Int64(((index // 1024) * 17 + index * 13) % 128)


def fact_item_valid(index: Int) -> Bool:
    return fact_item(index) != 31 and index % 61 != 7


def fact_day_valid(index: Int) -> Bool:
    return index % 37 != 5


def quantity(index: Int) -> Int64:
    return Int64((index * 11) % 41 - 20)


def payload(index: Int) -> Int64:
    return Int64(index % 211 - 100)


def payload_valid(index: Int) -> Bool:
    return index % 11 != 3


def calendar_day(index: Int) -> Int64:
    return Int64(index) if index < 128 else Int64(6)


def bucket(index: Int) -> Int64:
    return Int64((index * 3) % 7)


def selected_keys(
    count: Int, missing: Bool = False
) -> Tuple[List[Int64], List[Bool]]:
    var values = List[Int64]()
    var valid = List[Bool]()
    for index in range(count):
        values.append(Int64(7 + index * 2))
        valid.append(True)
    if missing:
        values.append(5000)
        valid.append(True)
    # Duplicate the first value and poison a null key's hidden payload.
    values.append(values[0])
    valid.append(True)
    values.append(31)
    valid.append(False)
    return (values^, valid^)


def contains(values: List[Int64], valid: List[Bool], value: Int64) -> Bool:
    for index in range(len(values)):
        if valid[index] and values[index] == value:
            return True
    return False


def key_filter(
    values: List[Int64], valid: List[Bool], value: Int64, use_range: Bool
) -> Bool:
    if not use_range:
        return contains(values, valid, value)
    var low = Int64.MAX
    var high = Int64.MIN
    for index in range(len(values)):
        if valid[index]:
            low = min(low, values[index])
            high = max(high, values[index])
    return low <= value and value <= high


def stock_rows(
    n: Int,
    item_keys: List[Int64],
    item_valid: List[Bool],
    use_range: Bool,
    days: List[Int64] = List[Int64](),
    quantity_filter: Bool = True,
) -> List[PushdownRow]:
    var result = List[PushdownRow]()
    var day_valid = List[Bool](length=len(days), fill=True)
    for index in range(n):
        if not fact_item_valid(index) or not fact_day_valid(index):
            continue
        if quantity_filter and quantity(index) <= -11:
            continue
        if not key_filter(item_keys, item_valid, fact_item(index), use_range):
            continue
        for dimension in range(129):
            if fact_day(index) != calendar_day(dimension):
                continue
            if days and not contains(days, day_valid, calendar_day(dimension)):
                continue
            result.append(
                row(
                    [
                        Int64(index),
                        fact_item(index),
                        fact_day(index),
                        bucket(dimension),
                        payload(index),
                    ],
                    payload_valid(index),
                )
            )
    return result^


def single_join_rows(
    n: Int, count: Int, missing: Bool, semi: Bool
) -> List[PushdownRow]:
    var keys = selected_keys(count, missing)
    # Exact scalar join semantics, independent of range approximation.
    var stock = stock_rows(n, keys[0], keys[1], False)
    var result = List[PushdownRow]()
    for left in range(len(keys[0])):
        if not keys[1][left]:
            continue
        var found = False
        for right in stock:
            if keys[0][left] != right.values[1]:
                continue
            found = True
            if not semi:
                result.append(
                    row(
                        [
                            Int64(left),
                            keys[0][left],
                            right.values[0],
                            right.values[2],
                            right.values[3],
                            right.values[4],
                        ],
                        right.valid[4],
                    )
                )
        if semi and found:
            result.append(row([Int64(left), keys[0][left]]))
    return result^


def order_item(index: Int) -> Int64:
    return Int64(7 + (index % 17) * 2)


def order_day(index: Int) -> Int64:
    return Int64(2 + (index * 3) % 7)


def order_kind(index: Int) -> Int64:
    return Int64(index % 5)


def order_valid(index: Int) -> Bool:
    return index % 23 != 4


def sold_rows() -> List[PushdownRow]:
    var result = List[PushdownRow]()
    for index in range(257):
        if not order_valid(index):
            continue
        for kind in range(5):
            if order_kind(index) != Int64(kind) or kind % 2 != 1:
                continue
            for day in range(128):
                if order_day(index) != Int64(day) or day >= 9:
                    continue
                result.append(
                    row([Int64(index), order_item(index), Int64(day)])
                )
    return result^


def compound_keys() -> Tuple[List[Int64], List[Int64]]:
    var items = List[Int64]()
    var days = List[Int64]()
    for sold in sold_rows():
        items.append(sold.values[1])
        days.append(sold.values[2])
    return (items^, days^)


def compound_join_rows(n: Int, semi: Bool) -> List[PushdownRow]:
    var keys = compound_keys()
    var valid = List[Bool](length=len(keys[0]), fill=True)
    var stock = stock_rows(n, keys[0], valid, False, keys[1], False)
    var result = List[PushdownRow]()
    for sold in sold_rows():
        var found = False
        for right in stock:
            if (
                sold.values[1] != right.values[1]
                or sold.values[2] != right.values[2]
            ):
                continue
            found = True
            if not semi:
                result.append(
                    row(
                        [
                            sold.values[0],
                            sold.values[1],
                            right.values[0],
                            right.values[2],
                            right.values[3],
                            right.values[4],
                        ],
                        right.valid[4],
                    )
                )
        if semi and found:
            result.append(row([sold.values[0], sold.values[1]]))
    return result^
