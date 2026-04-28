import pytest
import torch

TARGET_MODEL_ID = "Qwen/Qwen3-4B"
DRAFT_MODEL_ID = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def target_model_id() -> str:
    return TARGET_MODEL_ID


@pytest.fixture(scope="session")
def draft_model_id() -> str:
    return DRAFT_MODEL_ID


@pytest.fixture(scope="session")
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dtype() -> torch.dtype:
    return torch.bfloat16


def _config_available(model_id: str) -> bool:
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(model_id)
        return True
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_target: needs Qwen3.5-4B available locally")
    config.addinivalue_line("markers", "requires_draft: needs Qwen3.5-0.8B available locally")
    config.addinivalue_line("markers", "requires_cuda: needs an NVIDIA GPU with CUDA")
    config.addinivalue_line("markers", "requires_triton: needs Triton (Linux/CUDA)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_target = pytest.mark.skip(reason=f"{TARGET_MODEL_ID} config not available")
    skip_draft = pytest.mark.skip(reason=f"{DRAFT_MODEL_ID} config not available")
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_triton = pytest.mark.skip(reason="Triton not available on this platform")

    target_ok = _config_available(TARGET_MODEL_ID)
    draft_ok = _config_available(DRAFT_MODEL_ID)
    cuda_ok = torch.cuda.is_available()
    try:
        import triton  # noqa: F401

        triton_ok = cuda_ok
    except ImportError:
        triton_ok = False

    for item in items:
        if "requires_target" in item.keywords and not target_ok:
            item.add_marker(skip_target)
        if "requires_draft" in item.keywords and not draft_ok:
            item.add_marker(skip_draft)
        if "requires_cuda" in item.keywords and not cuda_ok:
            item.add_marker(skip_cuda)
        if "requires_triton" in item.keywords and not triton_ok:
            item.add_marker(skip_triton)
