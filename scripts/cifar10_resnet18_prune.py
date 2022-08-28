import os

for score_type in ["synflow_iterative"]:
    for prune_ratio in [0.5, 0.75, 0.9, 0.95, 0.99, 0.995]:
        for seed in range(780, 781):
            args = (
                f"cifar10_prune.py"
                f" --seed {seed}"
                f" --score-type {score_type}"
                f" --prune-ratio {prune_ratio}"
            )
            cmd = f"sbatch --export=args='{args}' --job-name cifar10-resnet18-{score_type} --time=01:00:00 scripts/exp.sb"
            # print(cmd)
            os.system(cmd)
