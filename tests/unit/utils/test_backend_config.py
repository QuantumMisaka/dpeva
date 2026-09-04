import pytest

from dpeva.utils.command import DPCommandBuilder


class TestDPCommandBuilderBackend:
    """The deprecated facade must remain explicit and stateless."""

    def test_explicit_backend_is_used_per_call(self):
        assert DPCommandBuilder._get_base_cmd("tf") == "dp --tf"
        assert DPCommandBuilder._get_base_cmd("jax") == "dp --jax"
        assert DPCommandBuilder._get_base_cmd("pt") == "dp --pt"

    def test_pt_expt_command(self):
        command = DPCommandBuilder.eval_desc(
            "pt-expt", model="model.pt", system="data", output="desc"
        )
        assert "dp --pt-expt eval-desc" in command

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            DPCommandBuilder._get_base_cmd("invalid")

    def test_all_commands_accept_backend(self):
        assert "dp --pt train" in DPCommandBuilder.train("pt", "input.json")
        assert "dp --pt freeze" in DPCommandBuilder.freeze("pt")
        assert "dp --pt test" in DPCommandBuilder.test("pt", "model.pt", "data", "results")

    def test_no_global_backend_state(self):
        assert not hasattr(DPCommandBuilder, "_backend")
        assert not hasattr(DPCommandBuilder, "set_backend")

    def test_facade_emits_migration_warning(self):
        with pytest.warns(DeprecationWarning, match="inject DeepMDAdapter"):
            DPCommandBuilder.freeze("pt")
