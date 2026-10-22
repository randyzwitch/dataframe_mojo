"""The Metal backend for dataframe, built only when a Metal provider is chosen.

Core code never imports this package. The root `dataframe_provider` package
re-exports `dataframe_metal.provider`, which makes Metal the default provider;
`-I providers/cpu -I .` selects the CPU-only provider instead.
"""
