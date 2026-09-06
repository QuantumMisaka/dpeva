import pytest
import warnings

from dpeva.utils.command import DPCommandBuilder


class TestDPCommandBuilderBackend:
    """The deprecated facade preserves the historical stateful API."""

    def setup_method(self):
        DPCommandBuilder.set_backend("pt")

    def test_set_backend_is_used_by_legacy_calls(self):
        DPCommandBuilder.set_backend("tf")
        assert DPCommandBuilder._get_base_cmd() == "dp --tf"
        assert DPCommandBuilder.freeze() == "dp --tf freeze"

    def test_pt_expt_command(self):
        command = DPCommandBuilder.eval_desc(
            model="model.pt", system="data", output="desc", backend="pt-expt"
        )
        assert "dp --pt-expt eval-desc" in command

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            DPCommandBuilder.set_backend("invalid")

    def test_all_commands_keep_legacy_signatures(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert DPCommandBuilder.train("input.json") == "dp --pt train input.json"
            assert DPCommandBuilder.freeze() == "dp --pt freeze"
            assert DPCommandBuilder.freeze("frozen.pt") == "dp --pt freeze -o frozen.pt"
            assert DPCommandBuilder.eval_desc("model.pt", "data", "desc", "head", "desc.log") == (
                "dp --pt eval-desc -s data -m model.pt -o desc --head head > desc.log 2>&1"
            )
            assert DPCommandBuilder.embed("model.pt", "data", "embed", "head", "fp64", "embed.log") == (
                "dp --pt embed -s data -m model.pt -o embed --dtype fp64 --head head > embed.log 2>&1"
            )
            assert DPCommandBuilder.test("model.pt", "data", "results") == (
                "dp --pt test -s data -m model.pt -d results"
            )

    def test_optional_keyword_backend_override_is_stateless(self):
        DPCommandBuilder.set_backend("tf")
        assert DPCommandBuilder.freeze(backend="pt") == "dp --pt freeze"
        assert DPCommandBuilder.freeze() == "dp --tf freeze"

    def test_facade_emits_migration_warning(self):
        with pytest.warns(DeprecationWarning, match="inject DeepMDAdapter") as records:
            DPCommandBuilder.freeze("pt")
        assert records[0].filename != DPCommandBuilder.__module__
        assert not records[0].filename.endswith("/utils/command.py")
