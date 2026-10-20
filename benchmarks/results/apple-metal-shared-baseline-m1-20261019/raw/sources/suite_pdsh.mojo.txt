from metal_suite_common import run
from cpu_pdsh import query as cpu_query
from metal_pdsh import query as metal_query
def main() raises:
    run[cpu_query, metal_query]()
