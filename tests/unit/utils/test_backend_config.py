import pytest
from dpeva.utils.command import DPCommandBuilder

class TestDPCommandBuilderBackend:
    """Test DPCommandBuilder backend configuration."""

    def test_default_backend(self):
        """Verify default backend."""
        assert DPCommandBuilder._backend == "pt"

    def test_set_backend_tf(self):
        """Verify setting backend to TensorFlow."""
        DPCommandBuilder.set_backend("tf")
        assert DPCommandBuilder._backend == "tf"
        assert DPCommandBuilder._get_base_cmd() == "dp --tf"

    def test_set_backend_jax(self):
        """Verify setting backend to JAX."""
        DPCommandBuilder.set_backend("jax")
        assert DPCommandBuilder._backend == "jax"
        assert DPCommandBuilder._get_base_cmd() == "dp --jax"

    def test_invalid_backend(self):
        """Verify setting invalid backend raises ValueError."""
        with pytest.raises(ValueError):
            DPCommandBuilder.set_backend("invalid")

    def test_all_commands_reflect_backend(self):
        """Verify generated commands use the set backend."""
        DPCommandBuilder.set_backend("pt")
        
        train_cmd = DPCommandBuilder.train("input.json")
        assert "dp --pt train" in train_cmd
        
        freeze_cmd = DPCommandBuilder.freeze()
        assert "dp --pt freeze" in freeze_cmd
        
        # Reset to default
        DPCommandBuilder.set_backend("pt")

if __name__ == "__main__":
    pytest.main([__file__])
