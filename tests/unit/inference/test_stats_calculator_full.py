
import pytest
import numpy as np
from dpeva.inference.stats import StatsCalculator

class TestStatsCalculator:
    
    @pytest.fixture
    def basic_data(self):
        e_pred = np.array([-10.0, -20.0])
        f_pred = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        e_true = np.array([-10.1, -20.2])
        f_true = np.array([0.15, 0.15, 0.15, 0.25, 0.25, 0.25])
        atom_counts = [{"H": 1}, {"H": 2}]
        atom_nums = [1, 2]
        
        return {
            "energy_per_atom": e_pred,
            "force_flat": f_pred,
            "energy_true": e_true,
            "force_true": f_true,
            "atom_counts_list": atom_counts,
            "atom_num_list": atom_nums
        }

    def test_metrics_computation(self, basic_data):
        """Test RMSE and MAE calculation."""
        calc = StatsCalculator(**basic_data)
        metrics = calc.compute_metrics()
        
        assert "e_mae" in metrics
        assert "e_rmse" in metrics
        assert "f_mae" in metrics
        assert "f_rmse" in metrics
        
        # Verify correctness
        # E: abs(0.1), abs(0.2) -> mean 0.15
        assert np.isclose(metrics["e_mae"], 0.15)
        
    def test_relative_energy_least_squares(self, basic_data):
        """Test relative energy using Least Squares fitting."""
        # e_pred = -10 (1 atom), -20 (2 atoms) -> E_total = -10, -40
        # If E0(H) = -10, then E_coh = 0, -10 (per atom)
        
        # Let's verify what StatsCalculator does.
        # It solves Ax = b.
        # Frame 1: 1*H = -10.0 * 1
        # Frame 2: 2*H = -20.0 * 2 = -40.0
        # Solution: H = -10? No.
        # -10 = 1*H
        # -40 = 2*H -> H = -20
        # LS solution will be between -10 and -20.
        
        calc = StatsCalculator(**basic_data, allow_ref_energy_lstsq_completion=True)
        coh_e = calc.compute_relative_energy(calc.e_pred)
        
        assert coh_e is not None
        assert len(coh_e) == 2

    def test_relative_energy_ref_energies(self, basic_data):
        """Test relative energy using provided reference energies."""
        ref = {"H": -10.0}
        basic_data["ref_energies"] = ref
        
        calc = StatsCalculator(**basic_data)
        coh_e = calc.compute_relative_energy(calc.e_pred)
        
        # Frame 1: E_pred = -10. E_ref = 1*-10 = -10. E_coh = -10 - (-10/1) = 0.
        # Frame 2: E_pred = -20. E_ref = 2*-10 = -20. E_coh = -20 - (-20/2) = -10.
        
        assert np.isclose(coh_e[0], 0.0)
        assert np.isclose(coh_e[1], -10.0)

    def test_relative_energy_partial_ref_with_lstsq_completion(self):
        e_pred = np.array([-5.0, -6.0, -7.0])
        atom_counts = [{"H": 1, "O": 1}, {"H": 1, "O": 2}, {"H": 1, "O": 3}]
        atom_nums = [2, 3, 4]
        calc = StatsCalculator(
            energy_per_atom=e_pred,
            force_flat=None,
            atom_counts_list=atom_counts,
            atom_num_list=atom_nums,
            ref_energies={"H": -1.0},
            allow_ref_energy_lstsq_completion=True
        )
        coh_pred = calc.compute_relative_energy(e_pred)
        coh_true = calc.compute_relative_energy(e_pred - 0.1)
        assert coh_pred is not None
        assert coh_true is not None
        assert len(coh_pred) == 3
        assert np.all(np.isfinite(coh_pred))
        assert np.all(np.isfinite(coh_true))

    def test_relative_energy_partial_ref_without_completion(self):
        calc = StatsCalculator(
            energy_per_atom=np.array([-5.0, -6.0]),
            force_flat=None,
            atom_counts_list=[{"H": 1, "O": 1}, {"H": 2, "O": 1}],
            atom_num_list=[2, 3],
            ref_energies={"H": -1.0},
            allow_ref_energy_lstsq_completion=False
        )
        coh_e = calc.compute_relative_energy(calc.e_pred)
        assert coh_e is None

    def test_force_magnitude(self, basic_data):
        """Test force magnitude calculation."""
        calc = StatsCalculator(**basic_data)
        mag = calc.compute_force_magnitude(calc.f_pred)
        
        # [0.1, 0.1, 0.1] -> sqrt(0.03) ~ 0.173
        assert len(mag) == 2
        assert np.isclose(mag[0], np.sqrt(0.03))

    def test_missing_truth(self, basic_data):
        """Test behavior when ground truth is missing."""
        basic_data["energy_true"] = None
        basic_data["force_true"] = None
        
        calc = StatsCalculator(**basic_data)
        metrics = calc.compute_metrics()
        
        assert metrics == {}
        assert calc.has_truth is False

    def test_missing_composition(self, basic_data):
        """Test relative energy failure when composition is missing."""
        basic_data["atom_counts_list"] = None
        
        calc = StatsCalculator(**basic_data)
        coh_e = calc.compute_relative_energy(calc.e_pred)
        
        assert coh_e is None
