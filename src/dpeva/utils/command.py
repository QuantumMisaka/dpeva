import shlex
from typing import Optional, ClassVar, List
from dpeva.constants import DEFAULT_DP_BACKEND, VALID_DP_BACKENDS

class DPCommandBuilder:
    """
    Builder for DeepMD-kit CLI commands (dp [backend]).
    Centralizes command construction to avoid hardcoded strings.
    Uses shlex.quote to prevent command injection.
    """
    
    DEFAULT_BACKEND: ClassVar[str] = DEFAULT_DP_BACKEND
    VALID_BACKENDS: ClassVar[List[str]] = VALID_DP_BACKENDS
    
    _backend: ClassVar[str] = DEFAULT_BACKEND
    
    @classmethod
    def set_backend(cls, backend: str):
        """
        Set the backend flag for DeepMD-kit commands.
        
        Args:
            backend: Backend name (e.g., 'pt', 'tf').
        
        Raises:
            ValueError: If backend is not valid.
        """
        if backend not in cls.VALID_BACKENDS:
            raise ValueError(f"Invalid backend '{backend}'. Valid options: {cls.VALID_BACKENDS}")
        cls._backend = backend

    @classmethod
    def _get_base_cmd(cls) -> str:
        return f"dp --{cls._backend}"

    @classmethod
    def train(cls, input_file: str, 
              finetune_path: Optional[str] = None, 
              init_model_path: Optional[str] = None,
              skip_neighbor_stat: bool = False,
              log_file: Optional[str] = None) -> str:
        """
        Builds 'dp [backend] train' command.
        
        Args:
            input_file: Path to input.json.
            finetune_path: Path to pretrained model for finetuning (--finetune).
            init_model_path: Path to pretrained model for initialization (--init-model).
            skip_neighbor_stat: Whether to add --skip-neighbor-stat.
            log_file: If provided, appends '> log_file 2>&1'.
        """
        cmd = f"{cls._get_base_cmd()} train {shlex.quote(input_file)}"
        
        if skip_neighbor_stat:
            cmd += " --skip-neighbor-stat"
            
        if finetune_path:
            cmd += f" --finetune {shlex.quote(finetune_path)}"
        elif init_model_path:
            cmd += f" --init-model {shlex.quote(init_model_path)}"
            
        if log_file:
            cmd += f" > {shlex.quote(log_file)} 2>&1"
            
        return cmd

    @classmethod
    def freeze(cls, output: Optional[str] = None) -> str:
        """
        Builds 'dp [backend] freeze' command.
        
        Args:
            output: Optional output filename (default uses dp default 'frozen_model.pb').
        """
        cmd = f"{cls._get_base_cmd()} freeze"
        if output:
            cmd += f" -o {shlex.quote(output)}"
        return cmd

    @classmethod
    def eval_desc(cls, model: str, system: str, output: str, head: Optional[str] = None, log_file: Optional[str] = None) -> str:
        """
        Builds 'dp [backend] eval-desc' command.
        
        Args:
            model: Path to model file.
            system: Path to system directory.
            output: Output directory/file.
            head: Model head (optional).
            log_file: If provided, appends '> log_file 2>&1'.
        """
        cmd = f"{cls._get_base_cmd()} eval-desc -s {shlex.quote(system)} -m {shlex.quote(model)} -o {shlex.quote(output)}"
        if head:
            cmd += f" --head {shlex.quote(head)}"
            
        if log_file:
            cmd += f" > {shlex.quote(log_file)} 2>&1"
            
        return cmd

    @classmethod
    def test(cls, model: str, system: str, prefix: str, head: Optional[str] = None, log_file: Optional[str] = None) -> str:
        """
        Builds 'dp [backend] test' command.
        
        Args:
            model: Path to model file.
            system: Path to system directory.
            prefix: Output prefix (-d).
            head: Model head (optional).
            log_file: If provided, appends '> log_file 2>&1'.
        """
        cmd = f"{cls._get_base_cmd()} test -s {shlex.quote(system)} -m {shlex.quote(model)} -d {shlex.quote(prefix)}"
        if head:
            cmd += f" --head {shlex.quote(head)}"
            
        if log_file:
            cmd += f" > {shlex.quote(log_file)} 2>&1"
            
        return cmd
