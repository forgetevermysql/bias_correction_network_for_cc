import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from auxiliary.settings import DEVICE, make_deterministic
from auxiliary.utils import print_metrics, log_metrics
from classes.core.Evaluator import Evaluator
from classes.core.LossTracker import LossTracker
from classes.data.ColorCheckerDataset import ColorCheckerDataset
from classes.data.NUS8Dataset import NUS8Dataset
from classes.bias_corr_cc.ModelBCCC import ModelBCCC


# --------------------------------------------------------------------------------------------------------------------

RANDOM_SEED = 3
EPOCHS = 16000

BATCH_SIZE = 16
LEARNING_RATE = 0.0003
FOLD_NUM = 0
# nus的八个相机，数字几表示几
DATASET_NUM = 1
# false 则为 nus8
ColorChecker = True

# fold_0
TEST_VIS_IMG = ["IMG_0753", "IMG_0438", "8D5U5533"]



# Reload checkpoint
RELOAD_CHECKPOINT = False


PATH_TO_PTH_CHECKPOINT = os.path.join("../trained_models", "fold_{}".format(FOLD_NUM))

# Load pretrained model
RELOAD_PRETRAINED = False


# --------------------------------------------------------------------------------------------------------------------

def main(opt):
    fold_num, epochs, batch_size, lr, dataset_num = opt.fold_num, opt.epochs, opt.batch_size, opt.lr, opt.dataset_num

    # Save the address of the interrupt model
    path_to_checkpoint = os.path.join("train", "logs", "fold_{}_checkpoint".format(str(fold_num)))
    path_to_pretrained = os.path.join("train", "logs", "fold_{}_pretrained".format(str(fold_num)))

    os.makedirs(path_to_checkpoint, exist_ok=True)

    path_to_log = os.path.join("train", "logs", "fold_{}_best".format(str(fold_num)))

    os.makedirs(path_to_log, exist_ok=True)

    path_to_metrics_log = os.path.join("./train/logs/metrics", "metrics.csv")
    os.makedirs("./train/logs/metrics", exist_ok=True)

    model = ModelBCCC()

    model.set_optimizer(lr)

    start_epoch = 0
    best_val_loss = 100.0


    evaluator = Evaluator()
    best_val_loss, best_metrics = 100.0, evaluator.get_best_metrics()
    train_loss, val_loss = LossTracker(), LossTracker()

    if RELOAD_CHECKPOINT:
        print('\n Reloading checkpoint - pretrained model stored at: {} \n'.format(path_to_checkpoint))
        start_epoch, best_metrics, best_val_loss = model.load(path_to_checkpoint, fold_num)
        start_epoch = start_epoch + 1

    if RELOAD_PRETRAINED:
        print('\n Reloading pretrained - pretrained model stored at : {}\n'.format(path_to_pretrained))
        start_epoch, _, _ = model.load(path_to_pretrained, fold_num)
        start_epoch = 0



    os.makedirs("./train/logs/network", exist_ok=True)
    model.log_network("./train/logs/network")

    if ColorChecker:
        training_set = ColorCheckerDataset(train=True, folds_num=fold_num)
        training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True,
                                 pin_memory=True)
        print("\n Training set size ... : {}".format(len(training_set)))


        test_set = ColorCheckerDataset(train=False, folds_num=fold_num)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
        print(" Test set size ....... : {}\n".format(len(test_set)))
    else:
        training_set = NUS8Dataset(train=True, dataset_num=dataset_num, folds_num=fold_num)
        training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True,
                                     pin_memory=True)
        print("\n Training set size ... : {}".format(len(training_set)))

        test_set = NUS8Dataset(train=False, dataset_num=dataset_num, folds_num=fold_num)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
        print(" Test set size ....... : {}\n".format(len(test_set)))


    path_to_vis = os.path.join("./train/logs", "test_vis")
    if TEST_VIS_IMG:
        print("Test vis for monitored image {} will be saved at {}\n".format(TEST_VIS_IMG, path_to_vis))
        os.makedirs(path_to_vis, exist_ok=True)

    print("\n**************************************************************")
    print("\t\t\t Training - Fold {}".format(fold_num))
    print("**************************************************************\n")

    writer = SummaryWriter("train/tensorBoard_logs")
    torch.backends.cudnn.enabled = True

    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train_mode()
        train_loss.reset()
        start = time.time()

        for i, (img, label, _) in enumerate(training_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            loss = model.optimize(img, label)

            batch_size = img.size(0)
            train_loss.update(loss, batch_size)

            if i % 5 == 0:
                print("[ Epoch: {}/{} - Batch: {} ] | [ Train loss: {:.4f} ]".format(epoch, epochs, i, loss))

        train_time = time.time() - start
        writer.add_scalar("train_loss", train_loss.avg, epoch)
        val_loss.reset()
        start = time.time()

        if epoch == start_epoch or epoch % 5 == 0:
            evaluator.reset_errors()
            model.evaluation_mode()

            print("\n--------------------------------------------------------------")
            print("\t\t\t Validation")
            print("--------------------------------------------------------------\n")

            with torch.no_grad():
                for i, (img, label, file_name) in enumerate(test_loader):
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    # print("reloadmodel", img.shape)
                    pred1, pred2, pred3, rgb, confidence = model.predict(img, return_steps=True)
                    loss = model.get_loss(pred3, label).item()
                    batch_size = img.size(0)
                    val_loss.update(loss, batch_size)
                    evaluator.add_error(model.get_loss(pred3, label).item())

                    # if i % 5 == 0:
                    print("[ Epoch: {}/{} - Batch: {}] | Val loss: {:.4f} ]".format(epoch, epochs, i, loss))

                    img_id = file_name[0].split(".")[0]

                    if img_id in TEST_VIS_IMG:
                        model.save_vis({"img": img, "label": label, "pred": pred3, "rgb": rgb, "c": confidence},
                                       os.path.join(path_to_vis, img_id, "epoch_{}.png".format(epoch)))

            print("\n--------------------------------------------------------------\n")
            writer.add_scalar("val_loss", val_loss.avg, epoch)

        val_time = time.time() - start

        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Time ... : {:.4f}".format(train_time))
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        if val_time > 0.1:
            print("....................................................................")
            print(" Val Time ..... : {:.4f}".format(val_time))
            print(" Val Loss ..... : {:.4f}".format(val_loss.avg))
            print("....................................................................")
            print_metrics(metrics, best_metrics)
        print("********************************************************************\n")


        if 0 < val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            best_metrics = evaluator.update_best_metrics()
            print("Saving new best model... \n")
            model.save(path_to_log, epoch, best_metrics, best_val_loss, fold_num)

        log_metrics(train_loss.avg, val_loss.avg, metrics, best_metrics, path_to_metrics_log)


        if epoch % 10 == 0:
            model.save(path_to_checkpoint, epoch, best_metrics, best_val_loss, fold_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_num", type=int, default=FOLD_NUM)
    parser.add_argument("--dataset_num", type=int, default=DATASET_NUM)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    opt = parser.parse_args()
    make_deterministic(opt.random_seed)

    print("\n *** Training configuration ***")
    print("\t Fold num ........ : {}".format(opt.fold_num))
    print("\t Epochs .......... : {}".format(opt.epochs))
    print("\t Batch size ...... : {}".format(opt.batch_size))
    print("\t Learning rate ... : {}".format(opt.lr))
    print("\t Random seed ..... : {}".format(opt.random_seed))

    main(opt)
