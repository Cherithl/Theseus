#!/usr/bin/env python3
"""Check numerical-flux reference comparison ordering and failure detection."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyvista as pv

import numflux_regression_test as regression


class NumericalFluxCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def mesh(self, cycle=0, lte=False):
        # Separate DG nodes at the common x=1 face carry distinct trace values.
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], dtype=float)
        cells = np.array([4, 0, 1, 2, 3, 4, 4, 5, 6, 7])
        mesh = pv.UnstructuredGrid(cells, np.array([9, 9], dtype=np.uint8), points)
        mesh.point_data["Density"] = 1.0 + np.arange(8) * 0.01
        mesh.point_data["Pressure"] = 10000.0 + np.arange(8)
        mesh.point_data["Velocity"] = np.column_stack((np.arange(8) * 0.1 + cycle * 0.01,
                                                     np.zeros(8)))
        if lte:
            mesh.point_data["Temperature"] = np.full(8, 300.0)
        return mesh

    def dataset(self):
        results = self.root / "results"
        for name in regression.CASES:
            case = results / name
            viz = case / "solution/ParaView"
            viz.mkdir(parents=True)
            (case / "config.json").write_text(json.dumps({"runTime": {
                "dt": 1e-4, "mesh_file": "fixture.msh", "output_file_path": str(case),
            }}))
            (case / "run.log").write_text("Device configuration: cpu\ntime step: 20, time: 0.002\n")
            entries = []
            for cycle in regression.CYCLES:
                output = viz / f"Cycle{cycle:06d}"
                output.mkdir()
                self.mesh(cycle, name.startswith("lte")).save(output / "data.vtu")
                entries.append(f'<DataSet timestep="{cycle * 1e-4}" file="{output.name}/data.vtu"/>')
            (viz / "ParaView.pvd").write_text("<VTKFile><Collection>" + "".join(entries) +
                                            "</Collection></VTKFile>")
        build = self.root / "build"
        build.mkdir()
        (build / "theseus").write_bytes(b"fixture executable")
        (build / "CMakeCache.txt").write_text(
            "SUTHERLAND:BOOL=OFF\nAXISYMMETRIC:BOOL=OFF\nSUBCELL_FV_BLENDING:BOOL=ON\n")
        golden = self.root / "golden"
        with redirect_stdout(io.StringIO()):
            regression.record(self.root, results, golden, build, "fixture-reference")
        return results, golden

    def test_cell_local_order_preserves_discontinuous_traces(self):
        original = self.root / "original.vtu"
        reordered = self.root / "reordered.vtu"
        self.mesh().save(original)
        mesh = self.mesh()
        reversed_cells = mesh.cells.reshape(-1, 5)[::-1].ravel()
        shuffled = pv.UnstructuredGrid(reversed_cells, mesh.celltypes, mesh.points)
        for field in mesh.point_data:
            shuffled.point_data[field] = mesh.point_data[field]
        self.assertFalse(np.array_equal(mesh.cells, shuffled.cells))
        shuffled.save(reordered)
        fields = ["Density", "Pressure", "Velocity"]
        a = regression.cell_local_fields(original, fields)
        b = regression.cell_local_fields(reordered, fields)
        for field in a:
            np.testing.assert_array_equal(a[field], b[field])
        shared = a["coordinates"][:, 3] == 1.0
        self.assertGreater(np.ptp(a["Density"][shared]), 0.0)

    def test_complete_matching_reference_passes(self):
        results, golden = self.dataset()
        with redirect_stdout(io.StringIO()):
            regression.check(results, golden, "cpu")

    def test_finite_field_corruption_fails(self):
        results, golden = self.dataset()
        path = results / regression.CASES[0] / "solution/ParaView/Cycle000020/data.vtu"
        mesh = pv.read(path)
        mesh.point_data["Velocity"][0, 0] += 0.01
        mesh.save(path)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(AssertionError, "Golden comparison failed"):
            regression.check(results, golden, "cpu")

    def test_nonfinite_field_fails(self):
        path = self.root / "nonfinite.vtu"
        mesh = self.mesh()
        mesh.point_data["Pressure"][0] = np.nan
        mesh.save(path)
        with self.assertRaisesRegex(AssertionError, "Nonfinite Pressure"):
            regression.cell_local_fields(path, ["Pressure"])

    def test_wrong_device_fails(self):
        results, golden = self.dataset()
        with self.assertRaisesRegex(AssertionError, "requested cuda"):
            regression.check(results, golden, "cuda")

    def test_missing_final_step_fails(self):
        results, golden = self.dataset()
        (results / regression.CASES[0] / "run.log").write_text(
            "Device configuration: cpu\ntime step: 10, time: 0.001\n")
        with self.assertRaisesRegex(AssertionError, "expected 20 steps"):
            regression.check(results, golden, "cpu")


if __name__ == "__main__":
    unittest.main()
