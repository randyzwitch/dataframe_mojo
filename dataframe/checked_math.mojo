"""Overflow-checked Int arithmetic for size and byte-count estimates."""


def checked_add(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or a > Int.MAX - b:
        raise Error("Accelerator memory estimate overflows Int")
    return a + b


def checked_mul(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or (b != 0 and a > Int.MAX // b):
        raise Error("Accelerator memory estimate overflows Int")
    return a * b
