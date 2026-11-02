# Numerical-flux regression tests

The `golden-numflux` recipe compares solver fields against CPU reference data for
the same numerical flux. It belongs to the Cartesian, regression, integrated and
full suites and uses the prepared Cartesian build supplied to the runner.

| Case | Configuration | Intended coverage |
| --- | --- | --- |
| `cpg-hll-cavity` | CPG, HLL, 2D viscous cavity, p=2 | Interior and no-slip boundary assembly with HLL |
| `cpg-roe-cavity` | CPG, Roe, 2D viscous cavity, p=2 | Interior and no-slip boundary assembly with Roe |
| `lte-hll-tgv` | LTE, HLL, 3D compressible periodic TGV, p=2 | LTE state/property evaluation and 3D assembly with HLL |

Each case runs 20 fixed-timestep steps and compares the initial, step-10 and
step-20 fields. Cavity DT is 1e-4 and TGV DT is 1e-5. The required build has PLATO
enabled, Cartesian geometry, subcell blending enabled and SUTHERLAND disabled.
The preparation stage rejects incompatible transport/geometry/blending options.

The comparisons require density, pressure and velocity; LTE additionally requires
temperature. They check finite fields, positive thermodynamic values, actual
velocity evolution, output times, the requested execution device, cell types and
cell-local geometry. Field tolerances are rtol=1e-10 and atol=1e-12. A trace is
identified by both its cell center and node coordinates, preserving discontinuous
face values while allowing rank and output ordering to differ.

Read the executable recipe with:

```bash
python3 scripts/run_tests.py show golden-numflux
```

Run against an existing build from the repository root:

```bash
python3 scripts/run_tests.py run-test golden-numflux \
  --build-standard /path/to/build --prefix /path/to/tpl/install \
  --results /path/to/new/results --device cpu --ranks 2
```

Use `--device cuda` and a separate results directory for a CUDA-enabled build.
The same CPU reference fields apply to both execution devices. The recipe retains
all configurations, launch logs and visualization outputs under its results
folder. The central runner saves the comparison output in its normal per-stage
logs.

## Reference data

`TestCases/GoldenData/NumericalFluxes/manifest.json` records the CPU source revision,
executable and CMake-cache hashes, build options, portable runtime configurations
and comparison tolerances. Compressed NPZ files contain cell-local coordinates,
cell types and fields. Normal test execution only reads these files.

Reference generation is an explicit development operation, using a separately
built reference implementation. After running the recipe's prepare/run stages
with that executable, record to a new directory:

```bash
python3 tests/numflux_regression_test.py record --source "$PWD" \
  --results /path/to/reference/results/cases \
  --build /path/to/reference/build --reference-revision FULL_COMMIT_ID \
  --golden /path/to/new/reference-directory
```

Compare a candidate against those references before proposing their inclusion.
Reference fields from different numerical fluxes are separate regression contracts;
agreement between HLL and Roe solutions is not required.

## Unit and checker tests

- `HLLFace_SupersonicBranchesMatchTensorReference`: verify both supersonic branches
  for CPG and LTE in dimensions 1–3 against full-tensor physical-flux projection.
- `RoeFace_NormalAccumulationPreservesPassiveScalarFlux`: compare all equations
  with the tensor-based Roe reference for unequal scalar concentrations; verify
  inputs are preserved.
- `NumericalFluxRegressionCheckerTests`: verify cell-order independence with
  discontinuous traces, matching data, finite corruption detection, nonfinite
  rejection, wrong-device rejection and missing-step rejection.

The flux unit cases run in `RiemannTestingSuite`. The Python checker tests have
one CTest registration. The solver recipe has one central-runner registration.
