import os
from time import time

import numpy as np
import torch.utils.data

from auxiliary.settings import DEVICE
from classes.core.Evaluator import Evaluator
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.bias_corr_cc.ModelBCCC import ModelBCCC

MODEL_TYPE = "baseline/bccc"
SAVE_PRED = True
SAVE_CONF = True
USE_TRAINING_SET = False


def main():
    evaluator = Evaluator()
    model = ModelBCCC()
    path_to_pred, path_to_pred_fold = None, None
    path_to_conf, path_to_conf_fold = None, None

    if SAVE_PRED:
        path_to_pred = os.path.join("test", "pred", "{}_{}".format("train" if USE_TRAINING_SET else "test", time()))

    if SAVE_CONF:
        path_to_conf = os.path.join("test", "conf", "{}_{}".format("train" if USE_TRAINING_SET else "test", time()))

    for i in range(1):
        # 模型 fold_0
        num_fold = 1
        fold_evaluator = Evaluator()
        # 数据集 fold_0
        test_set = ColorCheckerDataset(train=USE_TRAINING_SET, folds_num=1)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

        path_to_pretrained = os.path.join("../trained_models", MODEL_TYPE, "fold_{}".format(num_fold))


        model.test_load(path_to_pretrained)
        model.evaluation_mode()

        if SAVE_PRED:
            path_to_pred_fold = os.path.join(path_to_pred, "fold_{}".format(num_fold))
            os.makedirs(path_to_pred_fold)

        if SAVE_CONF:
            path_to_conf_fold = os.path.join(path_to_conf, "fold_{}".format(num_fold))
            os.makedirs(path_to_conf_fold)

        print("\n *** FOLD {} *** \n".format(num_fold))
        print(" * Test set size: {}".format(len(test_set)))
        print(" * Using trained model stored at: {} \n".format(path_to_pretrained))

        with torch.no_grad():
            for i, (img, label, file_name) in enumerate(dataloader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                pred1, pred2, pred3, rgb, conf = model.predict(img, return_steps=True)
                loss = model.get_loss(torch.mul(torch.mul(pred1, pred2), pred3), label)
                fold_evaluator.add_error(loss.item())
                evaluator.add_error(loss.item())
                print('\t - Input: {} - Batch: {} | Loss: {:f}'.format(file_name[0], i, loss.item()))
                if SAVE_PRED:
                    np.save(os.path.join(path_to_pred_fold, file_name[0]), torch.mul(torch.mul(pred1, pred2), pred3).cpu())
                if SAVE_CONF:
                    np.save(os.path.join(path_to_conf_fold, file_name[0]), conf.cpu())

        metrics = fold_evaluator.compute_metrics()
        print("\n Mean ............ : {:.4f}".format(metrics["mean"]))
        print(" Median .......... : {:.4f}".format(metrics["median"]))
        print(" Trimean ......... : {:.4f}".format(metrics["trimean"]))
        print(" Best 25% ........ : {:.4f}".format(metrics["bst25"]))
        print(" Worst 25% ....... : {:.4f}".format(metrics["wst25"]))
        print(" Percentile 95 ... : {:.4f} \n".format(metrics["wst5"]))

    print("\n *** AVERAGE ACROSS FOLDS *** \n")
    metrics = evaluator.compute_metrics()
    print("\n Mean ............ : {:.4f}".format(metrics["mean"]))
    print(" Median .......... : {:.4f}".format(metrics["median"]))
    print(" Trimean ......... : {:.4f}".format(metrics["trimean"]))
    print(" Best 25% ........ : {:.4f}".format(metrics["bst25"]))
    print(" Worst 25% ....... : {:.4f}".format(metrics["wst25"]))
    print(" Percentile 95 ... : {:.4f} \n".format(metrics["wst5"]))


if __name__ == '__main__':
    main()
