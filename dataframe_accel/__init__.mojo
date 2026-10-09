"""Optional accelerator runtime; the CPU dataframe package does not import it.

Use the gpu environment and import a backend explicitly. Runtime storage
does not yet enable LazyFrame.collect(engine="accel").
"""
