import pytest
from dpeva.utils.command import DPCommandBuilder

class TestDPCommandBuilderBackend:
    
    def setup_method(self):
        # Reset to default before each test
        DPCommandBuilder.set_backend("--pt")
        
    def test_default_backend(self):
        assert DPCommandBuilder._backend == "--pt"
        assert "dp --pt train" in DPCommandBuilder.train("input.json")
        
    def test_set_backend_tf(self):
        DPCommandBuilder.set_backend("--tf")
        assert DPCommandBuilder._backend == "--tf"
        cmd = DPCommandBuilder.train("input.json")
        assert "dp --tf train" in cmd
        
    def test_set_backend_jax(self):
        DPCommandBuilder.set_backend("--jax")
        cmd = DPCommandBuilder.freeze("out.pb")
        assert "dp --jax freeze" in cmd
        
    def test_invalid_backend(self):
        with pytest.raises(ValueError) as excinfo:
            DPCommandBuilder.set_backend("--invalid")
        assert "Invalid backend" in str(excinfo.value)
        
    def test_all_commands_reflect_backend(self):
        DPCommandBuilder.set_backend("--paddle")
        
        assert "dp --paddle train" in DPCommandBuilder.train("in.json")
        assert "dp --paddle freeze" in DPCommandBuilder.freeze()
        assert "dp --paddle eval-desc" in DPCommandBuilder.eval_desc("m", "s", "o")
        assert "dp --paddle test" in DPCommandBuilder.test("m", "s", "p")

if __name__ == "__main__":
    pytest.main([__file__])
