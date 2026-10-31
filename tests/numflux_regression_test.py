#!/usr/bin/env python3
"""Prepare numerical-flux recipes and compare cell-local fields with CPU references."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv


CASES = ("cpg-hll-cavity", "cpg-roe-cavity", "lte-hll-tgv")
CYCLES = (0, 10, 20)
RTOL = 1e-10
ATOL = 1e-12


def require_reference_build(build):
    cache = (build / "CMakeCache.txt").read_text()
    for expected in ("SUTHERLAND:BOOL=OFF", "AXISYMMETRIC:BOOL=OFF", "SUBCELL_FV_BLENDING:BOOL=ON"):
        if expected not in cache:
            raise AssertionError(f"Numerical-flux golden cases require {expected}")
    return cache


def prepare(source, results, database):
    results.mkdir(parents=True, exist_ok=True)
    for name in CASES:
        directory = results / name
        directory.mkdir()
        is_lte = name == "lte-hll-tgv"
        if is_lte:
            case = source / "TestCases/NavierStokes/3D/TaylorGreenVortex"
            spec = importlib.util.spec_from_file_location("tgv_mesh", case / "make_periodic_box.py")
            generator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(generator)
            mesh = directory / "periodic.msh"
            generator.write_mesh(mesh, 2 * math.pi, 2 * math.pi, 2 * math.pi, 4, 4, 4)
        else:
            case = source / "TestCases/NavierStokes/2D/LidDrivenCavity"
            mesh = case / "LidDrivenCavity.msh"
        config = json.loads((case / "config.json").read_text())
        runtime = config["runTime"]
        runtime.update(
            order=2, ser_ref_levels=0, mesh_file=str(mesh.resolve()),
            numerical_flux="Roe" if "roe" in name else "HLL",
            variable_dt=False, dt=1e-5 if is_lte else 1e-4,
            nsteps_max=20, final_time=1.0, print_interval=1,
            visualize=True, paraview=True, visit=False, vis_steps=10,
            nancheck=True, clock_simulation=False, checkpoint_save=False,
            output_file_path=str((directory / "solution").resolve()),
        )
        fields = ["density", "velocity", "pressure"]
        if is_lte:
            runtime.update(
                dim=3, num_equations=5, gas_model="lte", gas_mixture="air11",
                plato_solver="LTE_table_rhoT_(air11)", database_path=str(database.resolve()),
                N_rho=31, N_T=41, rho_min=0.5, rho_max=2.0,
                T_min=200.0, T_max=2000.0, rho_dist="log", T_dist="log",
            )
            runtime["conditions"] = {"initial_conditions": {
                "signature": 1, "function": "CompressibleTaylorGreenVortexIC",
                "params": {"x1": 0.2},
            }}
            fields.append("temperature")
        runtime["visualization"] = {"fields": fields, "mesh_mode": "gll_subcells"}
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def cell_local_fields(path, fields):
    """Identify DG traces by cell center AND node position, independent of rank ordering."""
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine(merge_points=False)
    missing = set(fields) - set(mesh.point_data)
    if missing:
        raise AssertionError(f"Missing fields {missing}: {path}")
    connectivity = np.asarray(mesh.cell_connectivity)
    if hasattr(mesh, "cell_offsets"):
        offsets = np.asarray(mesh.cell_offsets)
    else:
        offsets = np.asarray(mesh.offset)
    counts = np.diff(offsets)
    centers = np.asarray(mesh.cell_centers().points)
    keys = np.column_stack((np.repeat(centers, counts, axis=0), mesh.points[connectivity]))
    if not np.all(np.isfinite(keys)):
        raise AssertionError(f"Nonfinite cell-local geometry: {path}")
    # Round only the sort keys, not the coordinates being compared. This avoids
    # summation-order differences in cell centers changing the trace ordering.
    rounded = np.round(keys, decimals=12)
    order = np.lexsort(rounded.T[::-1])
    if np.any(np.all(np.diff(rounded[order], axis=0) == 0.0, axis=1)):
        raise AssertionError(f"Ambiguous cell-local trace identifiers: {path}")
    data = {"coordinates": keys[order],
            "cell_types": np.repeat(np.asarray(mesh.celltypes), counts)[order]}
    for field in fields:
        values = np.asarray(mesh.point_data[field])[connectivity][order]
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"Nonfinite {field}: {path}")
        if field != "Velocity" and np.min(values) <= 0:
            raise AssertionError(f"Nonpositive {field}: {path}")
        data[field] = values
    return data


def read_frames(results, name, device):
    directory = results / name
    log = (directory / "run.log").read_text()
    configurations = re.findall(r"Device configuration:\s*([^\n]+)", log)
    if not configurations or device not in configurations[-1].strip().split(","):
        raise AssertionError(f"{name}: requested {device}, got {configurations}")
    steps = re.findall(r"time step:\s*(\d+),", log)
    if not steps or int(steps[-1]) != 20:
        raise AssertionError(f"{name}: expected 20 steps, got {steps}")
    config = json.loads((directory / "config.json").read_text())
    dt = config["runTime"]["dt"]
    collection = directory / "solution/ParaView/ParaView.pvd"
    entries = ET.parse(collection).findall(".//DataSet")
    if len(entries) != len(CYCLES):
        raise AssertionError(f"Expected initial, step 10, step 20 outputs: {collection}")
    fields = ["Density", "Pressure", "Velocity"]
    if name.startswith("lte"):
        fields.append("Temperature")
    frames = []
    for cycle, entry in zip(CYCLES, entries):
        time = float(entry.attrib["timestep"])
        if not math.isclose(time, cycle * dt, rel_tol=1e-12, abs_tol=1e-15):
            raise AssertionError(f"{name}: cycle {cycle} has incorrect time {time}")
        if Path(entry.attrib["file"]).parts[0] != f"Cycle{cycle:06d}":
            raise AssertionError(f"{name}: unexpected cycle file {entry.attrib['file']}")
        frames.append(cell_local_fields(collection.parent / entry.attrib["file"], fields))
    change = np.max(np.abs(frames[-1]["Velocity"] - frames[0]["Velocity"]))
    if change <= 1e-8:
        raise AssertionError(f"{name}: velocity did not evolve")
    return frames


def portable_config(directory):
    config = json.loads((directory / "config.json").read_text())
    runtime = config["runTime"]
    runtime.pop("output_file_path")
    runtime.pop("database_path", None)
    runtime["mesh_file"] = Path(runtime["mesh_file"]).name
    return config


def record(source, results, golden, build, revision, ranks=2):
    """Explicit development operation: normal recipes never replace references."""
    cache = require_reference_build(build)
    golden.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": 1, "reference_revision": revision,
        "executable_sha256": hashlib.sha256((build / "theseus").read_bytes()).hexdigest(),
        "cmake_cache_sha256": hashlib.sha256(cache.encode()).hexdigest(),
        "build_options": {
            key: re.search(rf"^{key}:[^=]+=(.*)$", cache, re.MULTILINE).group(1)
            for key in ("CMAKE_BUILD_TYPE", "SUTHERLAND", "SUBCELL_FV_BLENDING", "AXISYMMETRIC")
            if re.search(rf"^{key}:[^=]+=(.*)$", cache, re.MULTILINE)
        },
        "device": "cpu", "reference_mpi_ranks": ranks,
        "cycles": list(CYCLES), "rtol": RTOL, "atol": ATOL,
        "cases": {},
    }
    for name in CASES:
        frames = read_frames(results, name, "cpu")
        metadata["cases"][name] = portable_config(results / name)
        for cycle, frame in zip(CYCLES, frames):
            np.savez_compressed(golden / f"{name}-{cycle:06d}.npz", **frame)
    (golden / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Recorded CPU references from {revision}: {golden}")


def check(results, golden, device):
    metadata = json.loads((golden / "manifest.json").read_text())
    if metadata["schema"] != 1 or metadata["cycles"] != list(CYCLES):
        raise AssertionError("Unsupported reference manifest")
    failures = []
    for name in CASES:
        if portable_config(results / name) != metadata["cases"][name]:
            raise AssertionError(f"{name}: configuration differs from the golden case")
        frames = read_frames(results, name, device)
        for cycle, actual in zip(CYCLES, frames):
            with np.load(golden / f"{name}-{cycle:06d}.npz", allow_pickle=False) as reference:
                if set(actual) != set(reference.files):
                    raise AssertionError(f"{name}: reference field inventory mismatch")
                for field, values in actual.items():
                    expected = reference[field]
                    if values.shape != expected.shape:
                        raise AssertionError(f"{name}/{cycle}/{field}: shape mismatch")
                    if field == "cell_types":
                        passed = np.array_equal(values, expected)
                    elif field == "coordinates":
                        passed = np.allclose(values, expected, rtol=0.0, atol=1e-12)
                    else:
                        passed = np.allclose(values, expected, rtol=RTOL, atol=ATOL)
                    error = float(np.max(np.abs(values.astype(float) - expected)))
                    print(f"{name} cycle={cycle} {field}: {'PASS' if passed else 'FAIL'} Linf={error:.12e}", flush=True)
                    if not passed:
                        failures.append(f"{name}/{cycle}/{field}")
    if failures:
        raise AssertionError(f"Golden comparison failed: {', '.join(failures)}")
    print(f"Numerical-flux CPU-main/{device} golden comparisons PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "record"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--build", type=Path)
    parser.add_argument("--reference-revision")
    parser.add_argument("--ranks", type=int, default=2, help="MPI ranks used to generate references")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.database is None:
            parser.error("prepare requires --database")
        if args.build is not None:
            require_reference_build(args.build)
        prepare(args.source.resolve(), args.results.resolve(), args.database)
    else:
        golden = args.golden or args.source / "TestCases/GoldenData/NumericalFluxes"
        if args.action == "record":
            if args.build is None or not args.reference_revision:
                parser.error("record requires --build and --reference-revision")
            if args.ranks < 1:
                parser.error("--ranks must be positive")
            record(args.source, args.results, golden, args.build, args.reference_revision, args.ranks)
        else:
            check(args.results, golden, args.device)


if __name__ == "__main__":
    main()
