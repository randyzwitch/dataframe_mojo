"""Value faults raised by accelerator kernels, apart from device failures.

A data fault is a failure caused by input values: overflow, a failed strict
cast, an out-of-range temporal value. Device, budget and planning failures
are not. A provider computes every checked value but does not reproduce the
CPU executor's evaluation order, so it reports a data fault with this marker
and the CPU decides which error the query raises, if any.
"""

comptime DATA_FAULT = "Mojo GPU data fault: "


def data_fault(message: String) -> Error:
    return Error(DATA_FAULT + message)


def is_data_fault(error: Error) -> Bool:
    return String(error).startswith(DATA_FAULT)
