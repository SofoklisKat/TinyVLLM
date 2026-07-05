"""Class-name captions for image–text (CLIP) training."""

from tinyvllm.data.factory import DatasetName

# torchvision Fashion-MNIST class indices
FASHION_MNIST_LABELS = [
    "T-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]

# CIFAR-10
CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Subset of CIFAR-100 fine labels (index → name); full list loaded at runtime when needed.
_CIFAR100_FINE_LABELS: list[str] | None = None


def _load_cifar100_labels() -> list[str]:
    global _CIFAR100_FINE_LABELS
    if _CIFAR100_FINE_LABELS is None:
        from torchvision.datasets import CIFAR100

        _CIFAR100_FINE_LABELS = CIFAR100.classes  # type: ignore[attr-defined]
    return _CIFAR100_FINE_LABELS


def get_label_names(dataset: DatasetName) -> list[str]:
    """Return human-readable class names for a dataset."""
    if dataset == "fashion_mnist":
        return FASHION_MNIST_LABELS
    if dataset == "cifar10":
        return CIFAR10_LABELS
    if dataset == "cifar100":
        return _load_cifar100_labels()
    if dataset == "mnist":
        return [str(i) for i in range(10)]
    raise ValueError(
        f"No built-in captions for {dataset!r}. "
        "Use fashion_mnist, cifar10, or cifar100 for Phase 2 CLIP."
    )


def label_to_caption(dataset: DatasetName, label: int) -> str:
    names = get_label_names(dataset)
    return names[int(label)]
