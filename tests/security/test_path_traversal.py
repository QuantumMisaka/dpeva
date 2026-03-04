import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from dpeva.io.collection import CollectionIOManager
from dpeva.utils.exceptions import WorkflowError
import pandas as pd

class TestPathTraversal(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.project_dir = os.path.join(self.tmp_dir, "project")
        os.makedirs(self.project_dir)
        
        # Create a "sensitive" file outside the project structure
        self.sensitive_file = os.path.join(self.tmp_dir, "sensitive.txt")
        with open(self.sensitive_file, "w") as f:
            f.write("secret")
            
        self.io_manager = CollectionIOManager(self.project_dir, "dpeva_results")
        self.io_manager.ensure_dirs() # Create dirs
        
        # Mock dpdata system
        self.mock_sys = MagicMock()
        # Assume dpdata System has target_name
        self.mock_sys.target_name = "../../sensitive.txt"
        self.mock_sys.short_name = "../../sensitive.txt"
        self.mock_sys.__len__.return_value = 10
        
        # Mock sub_system behavior
        sub_sys = MagicMock()
        self.mock_sys.sub_system.return_value = sub_sys
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    @patch("dpeva.io.collection.load_systems")
    def test_export_path_traversal(self, mock_load_systems):
        """
        Test if sys_name with '../' allows writing outside the intended directory.
        """
        # Mock load_systems to return our malicious system
        mock_load_systems.return_value = [self.mock_sys]
        
        # Prepare df_final to trigger the loop
        # dataname format: "sys_name-index"
        df_final = pd.DataFrame({
            "dataname": ["../../sensitive.txt-0", "../../sensitive.txt-1"]
        })
        
        # Mock sub_system().to_deepmd_npy to capture the path
        def side_effect(path):
            # Check if path is outside expected directory
            expected_base = os.path.abspath(os.path.join(self.project_dir, "dpeva_results", "sampled_dpdata"))
            abs_path = os.path.abspath(path)
            print(f"Writing to: {abs_path}")
            if not abs_path.startswith(expected_base):
                # This confirms traversal is possible if code allows it
                return "TRAVERSAL_DETECTED"
            return "SAFE"

        self.mock_sys.sub_system.return_value.to_deepmd_npy.side_effect = side_effect
        
        # Run export
        try:
            self.io_manager.export_dpdata(
                testdata_dir="dummy",
                df_final=df_final,
                unique_system_names=["../../sensitive.txt"]
            )
        except Exception as e:
            print(f"Caught exception: {e}")
            pass

        # Verify calls
        call_args = self.mock_sys.sub_system.return_value.to_deepmd_npy.call_args
        if call_args:
            path_arg = call_args[0][0]
            print(f"Called with path: {path_arg}")
            
            # Check if traversal occurred
            if "../../" in path_arg or ".." in path_arg:
                 print("Traversal payload passed to function.")
            else:
                 print("Path seems sanitized (unexpected for reproduction).")
        else:
            print("to_deepmd_npy was NOT called (Sanitization/Skipping worked).")

if __name__ == '__main__':
    unittest.main()
