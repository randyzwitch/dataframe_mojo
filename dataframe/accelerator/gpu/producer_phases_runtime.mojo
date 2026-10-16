"""One scoped reservation for immutable original producer proof owners."""
from std.memory import Pointer
from dataframe.accelerator.joins import JoinDag
from .residency import Scope, checked_add, checked_mul


struct MetadataReservation(Movable):
    var scope: Scope
    var retained: Int

    def __init__(out self, scope: Scope, dag: JoinDag) raises:
        var seen = List[Int]()
        var retained = 0
        for task in dag.tasks:
            if task.sum_receipt:
                var receipt = task.sum_receipt.value()
                var identity = Int(Pointer(to=receipt[]))
                if identity not in seen:
                    seen.append(identity)
                    retained = checked_add(retained, receipt[].retained_bytes)
                var origin = receipt[].origin
                identity = Int(Pointer(to=origin[]))
                if identity not in seen:
                    seen.append(identity)
                    retained = checked_add(retained, origin[].retained_bytes)
            for node in task.query._nodes:
                if node.producer_origin:
                    var owner = node.producer_origin.value()
                    var identity = Int(Pointer(to=owner[]))
                    if identity not in seen:
                        seen.append(identity)
                        retained = checked_add(retained, owner[].retained_bytes)
                if node.producer_region:
                    var owner = node.producer_region.value()
                    var identity = Int(Pointer(to=owner[]))
                    if identity not in seen:
                        seen.append(identity)
                        retained = checked_add(retained, owner[].retained_bytes)
        _ = scope.check(checked_add(retained, checked_mul(seen.capacity(), 8)))
        scope._retain(retained)
        self.scope = scope.copy()
        self.retained = retained

    def __deinit__(deinit self):
        self.scope._release(self.retained)
