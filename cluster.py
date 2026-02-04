from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger
import torch
import numpy as np
import random
import sys

# 聚类评估指标
all_metrics = {
    "nmi": [],
    "ari": [],
}

def cli_main(seed: int):
    base_args = sys.argv[1:]
    seed_arg = ["--seed_everything", str(seed)]
    args = base_args + seed_arg

    cli = LightningCLI(
        model_class=None,
        datamodule_class=None,
        parser_kwargs={"parser_mode": "omegaconf"},
        subclass_mode_model=True,
        args=args,
        run=False
    )

    cli.trainer.test(cli.model, cli.datamodule)

    test_results = cli.trainer.callback_metrics
    for k in all_metrics:
        v = test_results.get(k)
        if v is not None:
            all_metrics[k].append(v.item())
        else:
            print(f"Warning: metric '{k}' not found for seed {seed}")

if __name__ == "__main__":
    seeds = list(range(10))
    for seed in seeds:
        cli_main(seed)

    print("\n====== 聚类结果汇总（均值 ± 标准差） ======")
    for metric, values in all_metrics.items():
        if len(values) == 0:
            print(f"{metric}: No data.")
            continue
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric}: {mean*100:.2f} ± {std*100:.2f}")
