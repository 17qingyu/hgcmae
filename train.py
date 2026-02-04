from lightning.pytorch.cli import LightningCLI, ArgsType
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import numpy as np
import random
import sys

all_metrics = {
    "f1_micro": [],
    "f1_macro": [],
    "auc": [],
}

class TrainLightningCLI(LightningCLI):
    
    def add_arguments_to_parser(self, parser):
        # 固定优化器为 Adam，用户仅传入参数
        parser.add_optimizer_args(torch.optim.AdamW)
        # 固定 lr_scheduler 为 StepLR，用户仅传入参数
        # parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

    def after_instantiate_classes(self):
        # 获取 CSVLogger
        logger = self.trainer.logger
        if isinstance(logger, CSVLogger):
            checkpoint_dir = logger.log_dir

            # 创建 ModelCheckpoint
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_top_k=1,
                save_last=False,
                monitor="val_loss",
                mode="min",
                filename="best_model",
                auto_insert_metric_name=False
            )

            # 添加回调
            self.trainer.callbacks.append(checkpoint_callback)
            print(f"Checkpoint will be saved in: {checkpoint_dir}")


def cli_main(seed: int):
    base_args = sys.argv[1:]

    # 添加新的参数
    seed_arg = ["--seed_everything", f"{seed}"]

    args = base_args + seed_arg
    cli = TrainLightningCLI(
        model_class=None,
        datamodule_class=None,
        parser_kwargs={"parser_mode": "omegaconf"},
        subclass_mode_model=True,
        args=args,
        run=False,
    )
    cli.trainer.fit(cli.model, cli.datamodule)
    print(cli.trainer.checkpoint_callback.best_model_path)
    test_results = cli.trainer.test(cli.model, cli.datamodule, ckpt_path="best")

    if len(test_results) > 0:
        metrics = test_results[0]  # 只取第一个 batch 的 metrics
        for k in all_metrics.keys():
            if k in metrics:
                all_metrics[k].append(metrics[k])
            else:
                print(f"Warning: metric '{k}' not found in test results for seed {seed}")


if __name__ == "__main__":
    seeds = list(range(10))
    for seed in seeds:
        cli_main(seed)

    print("\n====== 结果汇总（均值 ± 标准差） ======")
    for metric, values in all_metrics.items():
        if len(values) == 0:
            print(f"{metric}: No data.")
            continue
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric}: {mean*100:.2f}±{std*100:.2f}")