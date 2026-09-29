"""H2LM research package; tokenizer tools do not import PyTorch."""
from typing import TYPE_CHECKING, Any

from .config import H2LMConfig, load_config

if TYPE_CHECKING:
    from .modeling.h2lm import H2LM

__all__ = ["H2LM", "H2LMConfig", "load_config"]


def __getattr__(name: str) -> Any:
    if name == "H2LM":
        from .modeling.h2lm import H2LM
        return H2LM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
