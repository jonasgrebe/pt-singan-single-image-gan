from typing import List, Dict

import torch
import numpy as np
import os
import cv2

class Logger:

    def __init__(self, run_name: str) -> None:
        self.log_dir = 'logs'
        self.run_name = run_name
        self.mode = 'training'

        self.scale = None
        self.step = 0


    def set_mode(self, mode: str) -> None:
        self.mode = mode


    def set_scale(self, scale: int) -> None:
        self.scale = scale


    def new_scale(self) -> None:
        self.scale -= 1
        self.step = 0


    def new_step(self) -> None:
        self.step += 1


    def log_losses(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError


    def log_images(self, img_batch: torch.Tensor, name: str, dataformats: str = 'NCHW') -> None:
        raise NotImplementedError


class TensorboardLogger(Logger):

    def __init__(self, run_name) -> None:
        super(TensorboardLogger, self).__init__(run_name)

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.run_name))


    def log_losses(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        for loss_name in loss_dict:
            if torch.is_tensor(loss_dict[loss_name]):
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().numpy().item()

        self.writer.add_scalars(f'{self.mode}/{self.scale}', loss_dict, self.step)


    def log_images(self, img_batch: torch.Tensor, name: str, dataformats: str = 'NCHW') -> None:

        if torch.is_tensor(img_batch):
            img_batch = img_batch.detach().cpu().numpy()

        self.writer.add_images(f'{self.mode}/{self.scale}/{name}', img_batch / 255, self.step, dataformats=dataformats)
