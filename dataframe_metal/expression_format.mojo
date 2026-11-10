"""Bounded postfix value-program format shared by host and GPU.

The low 32 bits of arithmetic word2 stay zero and opaque to the device
compiler, preserving Float32 rounding between nodes. Typed programs carry
their per-node dtype above those bits; legacy programs keep the whole word zero.
"""

comptime TREE_OP = 0xFFFF_FFFF
comptime MAX_EXPRESSION_NODES = 15
comptime MAX_EXPRESSION_STACK = 8
comptime EXPRESSION_NODE_WORDS = 4


comptime CASE_HEADER = 0xFFFF_FFFE
comptime CASE_TEST = 0xFFFF_FFFD
comptime CASE_JUMP = 0xFFFF_FFFC
comptime CASE_NULL = 0xFFFF_FFFB
comptime MAX_CASE_INSTRUCTIONS = 64
comptime MAX_CASE_CONDITIONS = 8
comptime MAX_CASE_STACK = 8

# A large membership predicate may have many source nodes but one GPU leaf.
comptime MAX_CASE_SOURCE_NODES = 262144

# A separate program header preserves the legacy homogeneous evaluators.
comptime TYPED_HEADER = 0xFFFF_FFFA
comptime VALUE_TYPE_SHIFT = 40

# Typed-header flags: bit zero requires Float32 payloads even for COUNT.
# Bit one proves a scalar accumulator schedule (no nested right subtree).
# Bits 8..15 contain its operation count; bits 16..30 mark unary steps.
comptime TYPED_CHAIN = 2

# Typed-header word1 records three-bit result-register slots per instruction.
# The literal word uses bits8..11 for the register count when this flag is set.
comptime TYPED_REGISTERS = 4

# Composed-scan header word14: four scalar-field bits before moment bits24..27.
comptime SCALAR_FIELD_SHIFT = 20

# Unary predicate leaves reserve zero for null tests. COL itself is zero, so
# native Bool values need a distinct ABI tag rather than their Expr opcode.
comptime BOOL_COLUMN_COMPARISON = 0x7FFF
