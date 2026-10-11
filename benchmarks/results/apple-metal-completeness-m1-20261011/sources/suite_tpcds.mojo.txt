from metal_suite_common import run
from cpu_tpcds import query as cpu_query
from metal_tpcds import query as metal_query
def main() raises:
    run[cpu_query, metal_query]()
