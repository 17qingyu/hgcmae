from lightning.pytorch.cli import LightningCLI, ArgsType
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch


class PretrainLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # 固定优化器为 Adam，用户仅传入参数
        parser.add_optimizer_args(torch.optim.AdamW)
        # parser.add_argument("--num_repeats", type=int, default=10)
        # 固定 lr_scheduler 为 StepLR，用户仅传入参数
        # parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

    def after_instantiate_classes(self):
        # 获取 CSVLogger
        logger = self.trainer.logger
        if isinstance(logger, CSVLogger):
            checkpoint_dir = logger.log_dir

            # 创建 ModelCheckpoint
            # checkpoint_callback = ModelCheckpoint(
            #     dirpath=checkpoint_dir,
            #     save_top_k=1,
            #     save_last=False,
            #     monitor="train_loss",
            #     mode="min",
            #     filename="best_model",
            #     auto_insert_metric_name=False
            # )
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_top_k=0,  # 设置为0表示不保存最佳模型
                save_last=True,  # 启用保存最后一次模型
                monitor=None,  # 不需要监控指标
                auto_insert_metric_name=False
            )

            # 添加回调
            self.trainer.callbacks.append(checkpoint_callback)
            print(f"Checkpoint will be saved in: {checkpoint_dir}")


def cli_main(args: ArgsType = None):
    cli = PretrainLightningCLI(
        model_class=None,
        datamodule_class=None,
        parser_kwargs={"parser_mode": "omegaconf"},
        subclass_mode_model=True,
        args=args,
        run=False
    )
    cli.trainer.fit(cli.model, cli.datamodule)
    # print(cli.trainer.checkpoint_callback.last_model_path)
    # cli.trainer.test(cli.model, cli.datamodule, ckpt_path="last")


if __name__ == "__main__":
    cli_main()
