import time
import numpy as np
import openml
from sklearn.model_selection import train_test_split
import torch
from dwn_cpu import DistributiveThermometer, inference_dwn, train_dwn

torch.set_num_threads(8)

N = 40
N_INPUTS_PER_LUT = 6
MAPPING = 'learnable'
BATCH_SIZE = 512
LEARNING_RATE = 100.0
EPOCHS = 1
LR_DECAY_STEPS = 10
GAMMA = 0.1
VERBOSE = True

dataset_name = 'higgs'
dataset_id = 4532

print(f"Dataset: {dataset_name}")

dataset = openml.datasets.get_dataset(dataset_id)
df_features, df_labels, _, _ = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute,
)

features = df_features.values.astype(np.float32)
label_names = list(df_labels.unique())
labels = np.array(df_labels.map(lambda v: label_names.index(v)).values)
num_output = labels.max() + 1

x_train, x_test, y_train, y_test = train_test_split(
    features,
    labels,
    train_size=0.8,
    random_state=0,
)

thermometer = DistributiveThermometer(8).fit(x_train)
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

y_train = torch.tensor(y_train, dtype=torch.long).contiguous()
y_test = torch.tensor(y_test, dtype=torch.long).contiguous()

final_num_luts = N if isinstance(N, int) else N[-1]
tau = (1.0 * final_num_luts / num_output) ** 0.5 / 0.5
print('Tau:', tau)

accuracies = []
train_times = []
test_times = []

for seed in range(10):
    torch.manual_seed(seed)

    train_start = time.time()
    binary_luts, mapping = train_dwn(
        x_train,
        y_train,
        n=N_INPUTS_PER_LUT,
        tau=float(tau),
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        num_luts=N,
        num_classes=num_output,
        mapping=MAPPING,
        batch_size=BATCH_SIZE,
        lr_decay_steps=LR_DECAY_STEPS,
        gamma=GAMMA,
        verbose=VERBOSE,
    )
    train_time = time.time() - train_start

    test_start = time.time()
    pred = inference_dwn(
        x_test,
        binary_luts,
        mapping,
        n=N_INPUTS_PER_LUT,
        num_classes=num_output,
        num_luts=N,
    )
    test_time = time.time() - test_start

    accuracy = (pred == y_test).float().mean().item()

    accuracies.append(accuracy)
    train_times.append(train_time)
    test_times.append(test_time)

    print(f"Seed {seed} - Test Acc: {accuracy}")
    print(f"Seed {seed} - Train Time: {train_time}s")
    print(f"Seed {seed} - Test Time: {test_time}s")

print(f"{dataset_name} - Avg Test Acc: {np.mean(accuracies)} ± {np.std(accuracies)}")
print(f"{dataset_name} - Avg Train Time: {np.mean(train_times)} ± {np.std(train_times)}s")
print(f"{dataset_name} - Avg Test Time: {np.mean(test_times)} ± {np.std(test_times)}s")
