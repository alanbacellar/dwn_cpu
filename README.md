# dwn_cpu: Lightweight CPU Implementation of Differentiable Weightless Neural Networks

`dwn_cpu` is a lightweight C++/PyTorch implementation of Differentiable Weightless Neural Networks (DWN) for small CPU models.

It is intended primarily for the Weightless Neural Networks course at UFRJ and for experiments where the model is small enough that a fused CPU implementation can avoid much of the Python and PyTorch autograd overhead.

Training and inference are implemented in optimized C++ extensions and exposed through two main functions:

- `train_dwn`
- `inference_dwn`

## Installation

Requirements: PyTorch and a C++17-compatible compiler.

```bash
pip install dwn_cpu
```

## Quick Docs

### `train_dwn`

```python
train_dwn(
    x,                       # Binary input tensor [num_samples, input_size].
    y,                       # Integer class labels.
    n,                       # Number of inputs per LUT. LUT size is 2^n.
    tau,                     # GroupSum temperature.
    learning_rate,           # SGD learning rate.
    epochs,                  # Number of training epochs.
    num_luts,                # LUTs per layer. Int for 1 layer, list for multiple layers.
    num_classes,             # Number of output classes.
    batch_size,              # Training batch size.
    lr_decay_steps,          # Number of epochs between learning-rate decays.
    gamma,                   # Learning-rate decay factor.
    mapping="learnable",     # First-layer mapping: "learnable", "random", or "fixed".
    initial_luts=None,       # Optional initial LUT values.
    initial_mapping=None,    # Optional initial learnable weights or fixed mapping.
    verbose=True,            # Print training statistics once per epoch.
)
```

For a single-layer model:

```python
num_luts = 40
```

For a multilayer model:

```python
num_luts = [128, 64, 40]
```

Only the first layer can use learnable mapping in this implementation. Later layers always use fixed random mappings.

The final layer LUT count must be divisible by `num_classes`.

**Returns:**

```python
binary_luts, mapping
```

- `binary_luts`: Trained LUT values binarized to `0` or `1`. For a single-layer model this is a tensor; for a multilayer model it is a list of tensors, one per layer.
- `mapping`: Final discrete LUT input mappings. For a single-layer model this is a tensor; for a multilayer model it is a list of tensors, one per layer.

These outputs can be passed directly to `inference_dwn`.

### `inference_dwn`

```python
inference_dwn(
    x,                       # Binary input tensor [num_samples, input_size].
    binary_luts,             # Binarized LUTs returned by train_dwn.
    mapping,                 # Final discrete mappings returned by train_dwn.
    n,                       # Number of inputs per LUT.
    num_classes,             # Number of output classes.
    num_luts,                # LUT count, or list of LUT counts for multiple layers.
)
```

**Returns:**

```python
predictions
```

- `predictions`: Integer tensor with shape `[num_samples]` containing the predicted class index for each sample.

### `DistributiveThermometer`

```python
DistributiveThermometer(
    num_bits,                # Number of thermometer bits per input feature.
    feature_wise=True,       # Learn thresholds independently for each feature.
)
```

Fit the encoder on the training data:

```python
thermometer = DistributiveThermometer(8).fit(x_train)
```

Binarize data:

```python
x_train = thermometer.binarize(x_train)
x_test = thermometer.binarize(x_test)
```

For use with `dwn_cpu`, flatten the thermometer dimension and convert to `uint8`:

```python
x_train = (
    thermometer.binarize(x_train)
    .flatten(start_dim=1)
    .to(torch.uint8)
    .contiguous()
)

x_test = (
    thermometer.binarize(x_test)
    .flatten(start_dim=1)
    .to(torch.uint8)
    .contiguous()
)
```

**Methods:**

- `.fit(x)`: Fits the thermometer thresholds from the training data and returns the fitted encoder.
- `.binarize(x)`: Converts the input features into their binary thermometer representation.

**Returns from `.binarize(x)`:**

A binary PyTorch tensor containing `num_bits` thermometer bits for each original input feature.
