# Compiled test-suite packaging

Source files follow the coverage plan. Executable packaging controls how often
shared dataframe code is compiled; merging files within an existing area would
remove no compiler invocation.

The default `combined` mode builds one ordinary executable. `--packaging areas`
retains separate area executables for diagnosis and process isolation. Both modes
build fatal cases once into a shared dispatcher, then run each selected canonical
`module::function` ID in a fresh process with its own fixture directory and
one-worker baseline. Exact platform exit status and diagnostic checks remain
mandatory for every fatal case; a shared build does not establish a test result.

## Evidence

Linux experiments used unchanged production code and test functions from commit
`664a7c55f0f35e48e53972a5a89b487728f390ea`, Mojo `1.2.0.dev2026100905`, a Threadripper
3970X and approximately 128 GB RAM. Timed compiler work was serialized across
experiments. Operating-system caches were warm, compiler caches were not cleared,
and unrelated host load was uncontrolled.

| Selected workload | Separate builds | Shared build | Observed reduction |
| --- | ---: | ---: | ---: |
| Eight ordinary cases across three areas, initial builds | 374.73 s | 217.26 s | 42.0% |
| Same ordinary cases, unchanged-driver rebuilds | 50.00 s | 30.02 s | 40.0% |
| All ten fatal cases | 1,363.91 s | 142.56 s | 89.55% |

Every selected case retained its canonical identity and result. The fatal
comparison checked all ten existing diagnostics and Linux status 132 in both
arrangements. An ordinary assertion probe retained its original file/line and
function name, returned failure and allowed later cases and areas to run.

These are bounded Linux observations with one paired measurement per condition,
not full-suite or macOS speed predictions. The ordinary sample is unbalanced
(6/1/1 cases) and excludes kernels, execution, interchange, Metal and strict
ordinary failures. Fresh driver paths and fixture literals restored combined
compilation to 225.17 s; unchanged-driver timings must not be averaged with fresh
builds. The precise compiler reuse mechanism was not isolated.

Peak individual compiler RSS changed from 5.43 to 5.50 GiB for the initial ordinary
sample and from 3.85 to 3.92 GiB for fatal cases. Full combined compilation may
require more memory; subset measurements do not establish hosted-runner capacity.

## Correctness requirements

Full Linux and macOS runs must retain selected IDs, capability omissions, strict
XFAIL/XPASS behavior and exact fatal diagnostics/status, followed by independent
consumers of that run's actual interchange outputs. Actual Metal execution
requires an available device and explicit capability selection. Its existing
workflow selects one area; that hardware check does not establish Metal behavior
inside a full six-area combined executable.

`--packaging areas` remains available when a compiler failure, unexpected abort
or shared process state requires narrower isolation. No test expectations or
production algorithms need to change to consolidate compilation.
