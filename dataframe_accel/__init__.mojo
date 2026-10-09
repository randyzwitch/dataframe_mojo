"""Optional accelerator providers; the CPU package does not import them.

In the gpu environment, pass NvidiaRuntime explicitly to a supported query:
query.collect(engine="accel", accelerator=runtime).
"""
