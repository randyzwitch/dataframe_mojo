"""Provider-neutral requested execution settings attached to dispatch copies.

This additive envelope leaves existing provider trait and module signatures
unchanged. Requested stream batches are not global expression batches.
"""


@fieldwise_init
struct ExecutionSettings(Copyable):
    var optimize: Bool
    var streaming: Bool
    var batch_size: Int
