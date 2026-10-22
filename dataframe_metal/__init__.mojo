"""The Metal backend for dataframe, built only when a Metal provider is chosen.

Core code never imports this package. A Metal build puts `providers/metal`
ahead of the repository root on the import path; its `dataframe_provider`
re-exports `dataframe_metal.provider`.
"""
