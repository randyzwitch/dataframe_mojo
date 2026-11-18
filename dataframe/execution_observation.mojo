"""Provider-neutral evidence retained by one observed execution region."""
from std.atomic import Atomic


struct DeviceObservation(Movable):
    var provider: String
    var result_device: String
    var status: String
    var reason: String
    var launches: Atomic[Int64]
    var completed_launches: Atomic[Int64]
    var upload_bytes: Atomic[Int64]
    var download_bytes: Atomic[Int64]
    var waits: Atomic[Int64]
    var wait_ns: Atomic[Int64]
    var context_ns: Int
    var input_ns: Int
    var device_work_ns: Int
    var finish_ns: Int
    var phase_samples: Int

    def __init__(out self):
        self.provider = ""
        self.result_device = "cpu"
        self.status = "cpu"
        self.reason = ""
        self.launches = Atomic[Int64](0)
        self.completed_launches = Atomic[Int64](0)
        self.upload_bytes = Atomic[Int64](0)
        self.download_bytes = Atomic[Int64](0)
        self.waits = Atomic[Int64](0)
        self.wait_ns = Atomic[Int64](0)
        self.context_ns = 0
        self.input_ns = 0
        self.device_work_ns = 0
        self.finish_ns = 0
        self.phase_samples = 0

    def begin(mut self):
        self.result_device = "unverified"
        self.status = "running"
        self.reason = ""

    def complete(mut self):
        if self.status == "cpu_fallback":
            return
        if not self.provider:
            self.status = "unverified"
        elif self.launches.load() == 0:
            self.result_device = "host"
            self.status = "no_device_work"
        elif self.completed_launches.load() == self.launches.load():
            self.result_device = self.provider
            self.status = "completed"
        else:
            self.status = "unverified"

    def cpu(mut self, reason: String):
        self.result_device = "cpu"
        self.status = "cpu_fallback"
        self.reason = reason

    def failed(mut self, reason: String):
        self.result_device = "none"
        self.status = "failed"
        self.reason = reason

    def phases(
        mut self, context: Int, input: Int, device_work: Int, finish: Int
    ):
        self.context_ns += context
        self.input_ns += input
        self.device_work_ns += device_work
        self.finish_ns += finish
        self.phase_samples += 1
