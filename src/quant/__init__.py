"""Weight-only quantization — the third lossy block.

Same shape as ``src/mla`` and ``src/kernels``: a module that degrades the
model cheaply, plus a swap that installs it without touching
``src/engine/model.py``. The TVD corrector in ``src/mla/heal.py`` is meant to
repair this the same way it repairs MLA and a small draft.
"""

from .int8 import QuantLinear, dequantize, quantize_weight
from .swap import apply_weight_quant, quant_memory_report

__all__ = [
    "QuantLinear",
    "apply_weight_quant",
    "dequantize",
    "quant_memory_report",
    "quantize_weight",
]
