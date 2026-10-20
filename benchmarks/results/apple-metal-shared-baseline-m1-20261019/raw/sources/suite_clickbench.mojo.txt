from metal_suite_common import run
from cpu_clickbench import query as cpu_query
from metal_clickbench import query as metal_query
def main() raises:
    run[cpu_query, metal_query]()
