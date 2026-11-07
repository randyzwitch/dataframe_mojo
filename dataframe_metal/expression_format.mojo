"""Bounded postfix value-program format shared by host and GPU.

Operation word2 stays zero and is opaque to the device compiler. Scalar
Float32 fast paths use it to preserve rounding between arithmetic nodes.
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
