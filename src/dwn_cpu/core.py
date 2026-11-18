import torch

from . import _efd_fused_general_cpu as _C
from . import _efd_fused_multilayer_cpu as _M


_SPECIALIZED_N = {2, 3, 4, 5, 6, 8}


def _random_mapping(input_size, num_luts, n):
    total = num_luts * n
    num_complete, offset = divmod(total, input_size)
    parts = [torch.randperm(input_size) for _ in range(num_complete)]
    if offset:
        parts.append(torch.randperm(input_size)[:offset])
    return torch.cat(parts).reshape(num_luts, n).to(torch.int32).contiguous()


def _layer_sizes(num_luts):
    if isinstance(num_luts, int):
        sizes = [int(num_luts)]
    elif isinstance(num_luts, (list, tuple)):
        sizes = [int(v) for v in num_luts]
        if not sizes:
            raise ValueError("num_luts list cannot be empty")
    else:
        raise TypeError("num_luts must be an int or a list/tuple of ints")
    if any(v <= 0 for v in sizes):
        raise ValueError("all layer LUT counts must be > 0")
    return sizes


def _train_function(n, mapping):
    if mapping == "learnable":
        if n in _SPECIALIZED_N:
            return getattr(_C, f"train_lut{n}")
        return _C.train_general
    if mapping in {"random", "fixed"}:
        if n in _SPECIALIZED_N:
            return getattr(_C, f"train_fixed_lut{n}")
        return _C.train_fixed_general
    raise ValueError("mapping must be 'learnable', 'random', or 'fixed'")


def _inference_function(n):
    if n in _SPECIALIZED_N:
        return getattr(_C, f"predict_lut{n}")
    return _C.predict_general


def _multi_train_function(n):
    if n in _SPECIALIZED_N:
        return getattr(_M, f"train_multi_lut{n}")
    return _M.train_multi_general


def _multi_inference_function(n):
    if n in _SPECIALIZED_N:
        return getattr(_M, f"predict_multi_lut{n}")
    return _M.predict_multi_general


def train_dwn(
    x,
    y,
    n,
    tau,
    learning_rate,
    epochs,
    num_luts,
    num_classes,
    batch_size,
    lr_decay_steps,
    gamma,
    mapping="learnable",
    initial_luts=None,
    initial_mapping=None,
    verbose=True,
):
    n = int(n)
    num_classes = int(num_classes)
    mapping = mapping.lower()
    sizes = _layer_sizes(num_luts)

    if sizes[-1] % num_classes != 0:
        raise ValueError("last layer num_luts must be divisible by num_classes")
    if mapping not in {"learnable", "random", "fixed"}:
        raise ValueError("mapping must be 'learnable', 'random', or 'fixed'")

    x = x.to(device="cpu", dtype=torch.uint8).contiguous()
    y = y.to(device="cpu", dtype=torch.long).contiguous()

    if len(sizes) == 1:
        num_luts_single = sizes[0]
        if mapping == "learnable":
            if initial_mapping is None:
                initial_mapping = torch.rand(
                    x.size(1), num_luts_single * n, dtype=torch.float32
                )
            initial_mapping = initial_mapping.to(
                device="cpu", dtype=torch.float32
            ).contiguous()
        else:
            if initial_mapping is None:
                initial_mapping = _random_mapping(x.size(1), num_luts_single, n)
            initial_mapping = initial_mapping.to(
                device="cpu", dtype=torch.int32
            ).contiguous()

        if initial_luts is None:
            initial_luts = (
                torch.rand(num_luts_single, 1 << n, dtype=torch.float32) * 2.0 - 1.0
            )
        initial_luts = initial_luts.to(
            device="cpu", dtype=torch.float32
        ).contiguous()

        fn = _train_function(n, mapping)
        kwargs = dict(
            batch_size=int(batch_size),
            lr_decay_steps=int(lr_decay_steps),
            gamma=float(gamma),
            num_classes=num_classes,
            num_luts=num_luts_single,
            verbose=bool(verbose),
        )
        if n in _SPECIALIZED_N:
            return fn(
                x, y, initial_luts, initial_mapping,
                float(tau), float(learning_rate), int(epochs), **kwargs
            )
        return fn(
            x, y, initial_luts, initial_mapping, n,
            float(tau), float(learning_rate), int(epochs), **kwargs
        )

    if initial_luts is None:
        initial_luts = [
            torch.rand(v, 1 << n, dtype=torch.float32) * 2.0 - 1.0
            for v in sizes
        ]
    if not isinstance(initial_luts, (list, tuple)) or len(initial_luts) != len(sizes):
        raise ValueError("for multiple layers, initial_luts must be a list with one tensor per layer")
    initial_luts = [
        t.to(device="cpu", dtype=torch.float32).contiguous()
        for t in initial_luts
    ]

    first_learnable = mapping == "learnable"
    if first_learnable:
        if initial_mapping is None:
            first_mapping = torch.rand(
                x.size(1), sizes[0] * n, dtype=torch.float32
            )
        else:
            first_mapping = initial_mapping
        first_mapping = first_mapping.to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        fixed_mappings = []
        prev_size = sizes[0]
        for layer_size in sizes[1:]:
            fixed_mappings.append(_random_mapping(prev_size, layer_size, n))
            prev_size = layer_size
    else:
        first_mapping = torch.empty(0, dtype=torch.float32)
        fixed_mappings = []
        prev_size = x.size(1)
        for layer_idx, layer_size in enumerate(sizes):
            if layer_idx == 0 and mapping == "fixed" and initial_mapping is not None:
                m = initial_mapping.to(device="cpu", dtype=torch.int32).contiguous()
            else:
                m = _random_mapping(prev_size, layer_size, n)
            fixed_mappings.append(m)
            prev_size = layer_size

    fn = _multi_train_function(n)
    kwargs = dict(
        batch_size=int(batch_size),
        lr_decay_steps=int(lr_decay_steps),
        gamma=float(gamma),
        num_classes=num_classes,
        first_learnable=first_learnable,
        verbose=bool(verbose),
    )
    if n in _SPECIALIZED_N:
        out = fn(
            x, y, initial_luts, first_mapping, fixed_mappings,
            float(tau), float(learning_rate), int(epochs), **kwargs
        )
    else:
        out = fn(
            x, y, initial_luts, first_mapping, fixed_mappings, n,
            float(tau), float(learning_rate), int(epochs), **kwargs
        )

    binary_luts = list(out[0::2])
    mappings = list(out[1::2])
    return binary_luts, mappings


def inference_dwn(
    x,
    binary_luts,
    mapping,
    n,
    num_classes,
    num_luts,
):
    n = int(n)
    num_classes = int(num_classes)
    x = x.to(device="cpu", dtype=torch.uint8).contiguous()

    if isinstance(binary_luts, torch.Tensor):
        if num_luts is None:
            num_luts_single = int(binary_luts.size(0))
        else:
            sizes = _layer_sizes(num_luts)
            if len(sizes) != 1:
                raise ValueError("single-layer binary_luts requires one num_luts value")
            num_luts_single = sizes[0]

        binary_luts = binary_luts.to(
            device="cpu", dtype=torch.uint8
        ).contiguous()
        mapping = mapping.to(device="cpu", dtype=torch.int32).contiguous()
        fn = _inference_function(n)
        if n in _SPECIALIZED_N:
            return fn(
                x, binary_luts, mapping,
                num_classes=num_classes, num_luts=num_luts_single
            )
        return fn(
            x, binary_luts, mapping, n,
            num_classes=num_classes, num_luts=num_luts_single
        )

    binary_luts = list(binary_luts)
    mappings = list(mapping)
    if len(binary_luts) != len(mappings) or len(binary_luts) < 2:
        raise ValueError("multilayer model requires matching LUT and mapping lists")
    if num_luts is not None:
        sizes = _layer_sizes(num_luts)
        if sizes != [int(t.size(0)) for t in binary_luts]:
            raise ValueError("num_luts does not match trained model")

    binary_luts = [
        t.to(device="cpu", dtype=torch.uint8).contiguous()
        for t in binary_luts
    ]
    mappings = [
        t.to(device="cpu", dtype=torch.int32).contiguous()
        for t in mappings
    ]

    fn = _multi_inference_function(n)
    if n in _SPECIALIZED_N:
        return fn(x, binary_luts, mappings, num_classes=num_classes)
    return fn(x, binary_luts, mappings, n, num_classes=num_classes)
