import os
from datetime import datetime
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.callback import Callback
import torch


class MomentumUpdateCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.momentum_update()


class SaveOnlineEncoderCheckpointCallback(Callback):
    # def on_fit_end(self, trainer, pl_module):
    #     best_ckpt_path = None
    #     for callback in trainer.callbacks:
    #         if hasattr(callback, "best_model_path") and callback.best_model_path:
    #             best_ckpt_path = callback.best_model_path
    #             break

    #     if best_ckpt_path is None or not os.path.exists(best_ckpt_path):
    #         print("No checkpoint found to move.")
    #         return

    #     checkpoint = torch.load(best_ckpt_path, map_location='cpu')

    #     encoder_state_dict = {k[len("backbone."):]: v
    #                           for k, v in checkpoint['state_dict'].items()
    #                           if k.startswith("backbone.")}
    #     new_checkpoint = {'state_dict': encoder_state_dict}

    #     filename = "best_encoder.ckpt"
    #     target_dir = os.path.join(os.path.dirname(best_ckpt_path), "..")
    #     target_path = os.path.join(target_dir, filename)

    #     torch.save(new_checkpoint, target_path)
    #     print(f"Online encoder checkpoint saved to: {target_path}")

    def on_fit_end(self, trainer, pl_module):
        # 改为获取last模型路径
        last_ckpt_path = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                # 获取最后一次保存的模型路径
                last_ckpt_path = os.path.join(callback.dirpath, "last.ckpt")
                break

        if last_ckpt_path is None or not os.path.exists(last_ckpt_path):
            print("No checkpoint found to move.")
            return

        checkpoint = torch.load(last_ckpt_path, map_location='cpu')

        encoder_state_dict = {k[len("backbone."):]: v
                              for k, v in checkpoint['state_dict'].items()
                              if k.startswith("backbone.")}
        new_checkpoint = {'state_dict': encoder_state_dict}

        filename = "last_encoder.ckpt"  # 修改文件名
        target_dir = os.path.dirname(last_ckpt_path)
        target_path = os.path.join(target_dir, filename)

        torch.save(new_checkpoint, target_path)
        print(f"Online encoder checkpoint saved to: {target_path}")
