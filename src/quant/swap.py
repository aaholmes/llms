"""In-place swap of the engine's ``nn.Linear`` layers → :class:`QuantLinear`.

Mirror of ``src/kernels/swap.py`` and ``src/mla/swap.py``: the engine stays
agnostic and the quant package owns module replacement. Because the walk is
generic over ``nn.Linear`` rather than hardcoded to ``q/k/v/o/gate/up/down``,
it also quantizes an MLA-swapped model (``dkv``/``uk_nope``/``uv``/``kr``)
with no extra code — which is exactly the composition Stage C wants.
"""

from __future__ import annotations

from torch import nn

from .int8 import QuantLinear

# The embedding matrix and its tied output head are excluded by default. The
# head sits directly on the logits, so its quantization error lands on the
# output distribution undiluted rather than being attenuated by downstream
# layers — a separate experiment, not a free ride on this one.
DEFAULT_EXCLUDE = ("lm_head", "embed")


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    """True if any pattern appears as a dot-delimited component of ``name``."""
    parts = set(name.split("."))
    return any(p in parts for p in patterns)


def apply_weight_quant(
    model: nn.Module,
    *,
    n_bits: int = 8,
    group_size: int | None = None,
    include: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
) -> nn.Module:
    """Replace every eligible ``nn.Linear`` in ``model`` with a ``QuantLinear``.

    Args:
        model: a ``Qwen3Model``, before or after an MLA swap.
        n_bits: 8 or 4.
        group_size: per-group scales along the input dim; None = per-output-channel.
        include: if given, only layers whose qualified name contains one of
            these components are quantized (e.g. ``("gate", "up", "down")`` to
            hit the FFN only). None means all not excluded.
        exclude: layer-name components never quantized.

    Returns:
        The same model, mutated in place — returned for chaining, matching
        ``apply_triton_kernels`` / ``apply_mla``. Idempotent: an already-
        quantized layer is left alone rather than double-quantized.
    """
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            qualified = f"{name}.{child_name}" if name else child_name
            if _matches(qualified, exclude):
                continue
            if include is not None and not _matches(qualified, include):
                continue
            targets.append((module, child_name, child))

    # Collected first, mutated second: replacing during named_modules() would
    # invalidate the traversal.
    for parent, child_name, linear in targets:
        setattr(parent, child_name, QuantLinear.from_linear(linear, n_bits=n_bits, group_size=group_size))
    return model


def quant_memory_report(model: nn.Module) -> dict:
    """Weight footprint before/after quantization, in bytes.

    Reports the **packed-equivalent** size for the quantized layers — 4-bit
    codes are stored one-per-int8 here (see ``int8.py``), so the on-disk
    tensors are larger than ``quantized_bytes`` until bit-packing lands. The
    compression ratio is therefore what the scheme is worth, not what this
    implementation currently allocates; both are returned so the distinction
    stays visible.
    """
    dense_bytes = 0
    quantized_bytes = 0
    allocated_bytes = 0
    n_quant_layers = 0
    for module in model.modules():
        if isinstance(module, QuantLinear):
            n_quant_layers += 1
            n_elem = module.qweight.numel()
            # What a bf16 dense weight would have cost.
            dense_bytes += n_elem * 2
            quantized_bytes += n_elem * module.n_bits // 8 + module.scale.numel() * 4
            allocated_bytes += module.qweight.numel() + module.scale.numel() * 4
        elif isinstance(module, nn.Linear):
            n_elem = module.weight.numel()
            dense_bytes += n_elem * 2
            quantized_bytes += n_elem * 2
            allocated_bytes += n_elem * module.weight.element_size()
    return {
        "n_quant_layers": n_quant_layers,
        "dense_bytes": dense_bytes,
        "quantized_bytes": quantized_bytes,
        "allocated_bytes": allocated_bytes,
        "compression": dense_bytes / quantized_bytes if quantized_bytes else 1.0,
    }
