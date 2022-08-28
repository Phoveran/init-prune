import numpy as np
from tensorboard.backend.event_processing import event_accumulator

path = f"/mnt/home/chenaoch/workspace/results/model_prune/resnet18/dense/seed777/tensorboard"
ea=event_accumulator.EventAccumulator(path)
ea.Reload()
test_accs = np.array([i.value for i in ea.scalars.Items("test/acc")])
print(test_accs.max())

for ratio in [0.5, 0.75, 0.9, 0.95, 0.99, 0.995]:
    path = f"/mnt/home/chenaoch/workspace/results/model_prune/resnet18/synflow_iterative/ratio{str(ratio).replace('.', 'p')}/seed777/tensorboard"
    ea=event_accumulator.EventAccumulator(path)
    ea.Reload()
    test_accs = np.array([i.value for i in ea.scalars.Items("test/acc")])
    print(test_accs.max())