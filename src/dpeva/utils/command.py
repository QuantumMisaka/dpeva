"""Deprecated stateless DeepMD command facade."""

from __future__ import annotations

import shlex
import warnings
from typing import Optional

from dpeva.compatibility.adapter import DeepMDAdapter
from dpeva.constants import VALID_DP_BACKENDS


class DPCommandBuilder:
    """Deprecated stateless facade around :class:`DeepMDAdapter`.

    Every method takes an explicit backend.  New code should inject an
    adapter instead of using this one-release compatibility facade.
    """

    VALID_BACKENDS = tuple(VALID_DP_BACKENDS)

    @staticmethod
    def _adapter(backend: str) -> DeepMDAdapter:
        return DeepMDAdapter.for_legacy_unchecked(backend)

    @staticmethod
    def _warn_deprecated() -> None:
        warnings.warn(
            "DPCommandBuilder is deprecated; inject DeepMDAdapter instead",
            DeprecationWarning,
            stacklevel=3,
        )

    @staticmethod
    def _get_base_cmd(backend: str) -> str:
        return shlex.join(DPCommandBuilder._adapter(backend).base_command)

    @staticmethod
    def train(
        backend: str,
        input_file: str,
        finetune_path: Optional[str] = None,
        init_model_path: Optional[str] = None,
        skip_neighbor_stat: bool = False,
        log_file: Optional[str] = None,
    ) -> str:
        DPCommandBuilder._warn_deprecated()
        return DPCommandBuilder._adapter(backend).train(
            input_file, finetune_path, init_model_path, skip_neighbor_stat, log_file
        )

    @staticmethod
    def freeze(backend: str, output: Optional[str] = None) -> str:
        DPCommandBuilder._warn_deprecated()
        return DPCommandBuilder._adapter(backend).freeze(output)

    @staticmethod
    def eval_desc(
        backend: str,
        model: str,
        system: str,
        output: str,
        head: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> str:
        DPCommandBuilder._warn_deprecated()
        return DPCommandBuilder._adapter(backend).eval_desc(
            model, system, output, head, log_file
        )

    @staticmethod
    def embed(
        backend: str,
        model: str,
        system: str,
        output: str,
        head: Optional[str] = None,
        dtype: str = "fp32",
        log_file: Optional[str] = None,
    ) -> str:
        DPCommandBuilder._warn_deprecated()
        return DPCommandBuilder._adapter(backend).embed(
            model, system, output, head, dtype, log_file
        )

    @staticmethod
    def test(
        backend: str,
        model: str,
        system: str,
        prefix: str,
        head: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> str:
        DPCommandBuilder._warn_deprecated()
        return DPCommandBuilder._adapter(backend).test(
            model, system, prefix, head, log_file
        )
