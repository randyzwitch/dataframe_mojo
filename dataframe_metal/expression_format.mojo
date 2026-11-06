"""Bounded postfix value-program format shared by host and GPU.

Operation word2 stays zero and is opaque to the device compiler. Scalar
Float32 fast paths use it to preserve rounding between arithmetic nodes.
"""

comptime TREE_OP = 0xFFFF_FFFF
comptime MAX_EXPRESSION_NODES = 15
comptime MAX_EXPRESSION_STACK = 8
comptime EXPRESSION_NODE_WORDS = 4
