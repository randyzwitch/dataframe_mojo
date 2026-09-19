# Versioning and API stability

The project uses semantic versioning. Before 1.0, a minor release (0.x.0) may
contain breaking changes and a patch release (0.x.y) may not; from 1.0 on,
breaking changes require a major release. Every release lists its changes in
[CHANGELOG.md](../CHANGELOG.md), with breaking changes under **Breaking** and
migration notes for each.

## What is public

The public API is exactly the set of names exported by
`dataframe/__init__.mojo`, their public (non-underscore) methods and fields,
and the behavior documented in [semantics](semantics.md),
[expressions](expressions.md), and [csv](csv.md). Specifically:

- Documented results, null/NaN rules, error conditions, and output ordering
  guarantees are public. Error message wording is not, except where a guide
  quotes it.
- Anything starting with an underscore (`_columns`, `_data`, `_values`, ...)
  and every module not re-exported by `__init__` (`binding`, `execution`,
  `expr_kernels`, `hashing`, ...) is internal. Tests may use internals; user
  code should not.
- Unspecified output order (for example `group_by` without
  `maintain_order=True`) may change in any release.
- Performance characteristics are not part of the contract, though benchmark
  regressions are treated as bugs.

## Deprecation

Before 1.0, APIs may be removed without a deprecation period when the
replacement is documented in the changelog, as with the legacy column kernels
in the first changelog entry. From 1.0 on, a public name is first marked
deprecated in its docstring and the changelog for at least one minor release,
then removed in the next major release.

## Releases

Pushing a `vX.Y.Z` tag runs `.github/workflows/release.yml`: it runs the tests,
checks public docstrings, precompiles `dist/dataframe.mojoc`, runs an example
from outside the source tree against only that package, regenerates
[api.md](api.md), and publishes a GitHub release with the package, the API
reference, and the matching changelog section as notes. Consumers put the
package directory on the import path: `mojo run -I path/to/dist app.mojo`.
