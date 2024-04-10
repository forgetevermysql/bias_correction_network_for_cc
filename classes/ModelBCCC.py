import os
from typing import Union, Tuple

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms import transforms

from auxiliary.utils import correct, rescale, scale
from classes.core.Model import Model
from classes.bias_corr_cc.BCCCModel import dobuleBias




class ModelBCCC(Model):

    def __init__(self):
        super().__init__()
        self._network = dobuleBias().to(self._device)

        self.check_device(self._network)
        self.check_module_weights_and_biases(self._network)
        self.alpha1 = 0.33
        self.alpha2 = 0.33

    def predict(self, img: Tensor, return_steps: bool = False) -> Union[Tensor, Tuple]:

        pred1, b1, b2, rgb, confidece = self._network(img)

        return pred1, b1, b2, rgb, confidece


    def optimize(self, img: Tensor, label: Tensor) -> float:


        self._optimizer.zero_grad()
        pred, b1, b2, rgb, confidece = self.predict(img)
        loss1 = self.get_loss(pred, label)
        loss2 = self.get_loss(torch.mul(pred, b1), label)
        loss3 = self.get_loss(torch.mul(torch.mul(pred, b1), b2), label)
        loss = 0.33 * loss1 + 0.33 * loss2 + 0.34 * loss3


        loss.backward()
        self._optimizer.step()

        return loss.item()

    def save_vis(self, model_output: dict, path_to_plot: str):

        model_output = {k: v.clone().detach().to(self._device) for k, v in model_output.items()}

        img, label, pred = model_output["img"], model_output["label"], model_output["pred"]
        rgb, c = model_output["rgb"], model_output["c"]

        original = transforms.ToPILImage()(img.squeeze(0)).convert("RGB")
        est_corrected = correct(original, pred)

        size = original.size[::-1]

        weighted_est = rescale(scale(rgb * c), size).squeeze().permute(1, 2, 0)
        rgb = rescale(rgb, size).squeeze(0).permute(1, 2, 0)
        c = rescale(c, size).squeeze(0).permute(1, 2, 0)

        # 查看原始图像的置信度分布
        masked_original = scale(F.to_tensor(original).to(self._device).permute(1, 2, 0) * c)

        plots = [(original, "original"), (masked_original, "masked_original"), (est_corrected, "correction"),
                 (rgb, "per_patch_estimate"), (c, "confidence"), (weighted_est, "weighted_estimate")]

        stages, axs = plt.subplots(2, 3, figsize=(15, 10))
        for i in range(2):
            for j in range(3):
                plot, text = plots[i * 3 + j]
                if isinstance(plot, Tensor):
                    plot = plot.cpu()
                axs[i, j].imshow(plot, cmap="gray" if "confidence" in text else None)
                axs[i, j].set_title(text)
                axs[i, j].axis("off")

                # 创建单独保存每个子图的路径
                plot_path = f"{path_to_plot}_{text}.png"
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                # 保存单个子图
                fig = plt.figure()
                plt.imshow(plot, cmap="gray" if "confidence" in text else None)
                plt.axis("off")
                plt.title(text)
                fig.savefig(plot_path, bbox_inches='tight', dpi=200)
                plt.close(fig)

        os.makedirs(os.sep.join(path_to_plot.split(os.sep)[:-1]), exist_ok=True)
        epoch, loss = path_to_plot.split(os.sep)[-1].split("_")[-1].split(".")[0], self.get_loss(pred, label)
        stages.suptitle("EPOCH {} - ERROR: {:.4f}".format(epoch, loss))
        stages.savefig(os.path.join(path_to_plot), bbox_inches='tight', dpi=200)
        plt.clf()
        plt.close('all')

    def check_device(self, model, expected_device='cuda'):
        for name, param in model.named_parameters():
            if param.device.type != expected_device:
                print(f"Module {name} is on {param.device}, not on {expected_device}.")
                return False
        print(f"All parameters are on {expected_device}.")
        return True

    def check_module_weights_and_biases(self, module):
        for sub_module in module.children():
            if len(list(sub_module.children())) == 0:  # 如果是叶子模块
                if hasattr(sub_module, 'weight') and sub_module.weight is not None:
                    if sub_module.weight.device.type != 'cuda':
                        print(f"Module {sub_module} weight is not on cuda!")
                if hasattr(sub_module, 'bias') and sub_module.bias is not None:
                    if sub_module.bias.device.type != 'cuda':
                        print(f"Module {sub_module} bias is not on cuda!")
