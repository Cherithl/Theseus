# HLL/Roe field references

These references use CPU main at commit
`1c6dd52e04b4822aa5c2e01929207b48cd38a8db`, an optimized Release build,
constant transport, subcell blending enabled, and two MPI ranks.
`manifest.json` records the executable/CMake-cache hashes and case configurations.

- `cpg-hll-cavity`: 2D viscous cavity, HLL, p=2, DT=1e-4.
- `cpg-roe-cavity`: 2D viscous cavity, Roe, p=2, DT=1e-4.
- `lte-hll-tgv`: 3D compressible LTE TGV, HLL, p=2, DT=1e-5.

Each case contains cycles 0, 10 and 20. NPZ files store cell-local geometry, cell
types, density, pressure and velocity; the LTE case also stores temperature.
Different fluxes have independent reference fields. The checker preserves both
DG traces at shared coordinates and aligns traces independently of MPI ordering.

Run `golden-numflux` through the central test runner. Generation and validation
are described in `docs/numerical-flux-regression.md`.
