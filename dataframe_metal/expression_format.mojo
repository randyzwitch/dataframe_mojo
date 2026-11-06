"""Bounded postfix value-program format shared by host and GPU."""

comptime TREE_OP = 0xFFFF_FFFF
comptime MAX_EXPRESSION_NODES = 15
comptime MAX_EXPRESSION_STACK = 8
comptime EXPRESSION_NODE_WORDS = 4
