"""Evaluates the model"""

import argparse
import logging
import os
import cv2, imageio

import numpy as np
import torch
from torch.autograd import Variable

import dataset.data_loader as data_loader
import model.net as net
from common import utils
from loss.losses import compute_losses, compute_eval_results
from common.manager import Manager

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='experiments/base_model/best_0.5012.pth.tar', help="name of the file in --model_dir containing weights to load")


def evaluate(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    print("eval begin!")

    # loss status and eval status initial
    manager.reset_loss_status()
    manager.reset_metric_status(manager.params.eval_type)
    model.eval()

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []
 
    k = 0
    with torch.no_grad():
        # compute metrics over the dataset

        for data_batch in manager.dataloaders[manager.params.eval_type]:
            # data parse
            imgs_full = data_batch["imgs_ori"]
            video_name = data_batch["video_name"]
            gray_patches = data_batch["imgs_gray_patch"]
            npy_name = data_batch["npy_name"]
            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)
            # compute model output
            output_batch = model(data_batch)
            # compute all metrics on this batch
            eval_results = compute_eval_results(data_batch, output_batch, manager.params)
            img1s_full_warp = eval_results["img1_full_warp"]
            err_avg = eval_results["errs"]

            for j in range(len(err_avg)):
                k += 1
                img2_full = imgs_full[j, 3:, ...].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                img1_full_warp = img1s_full_warp[j].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                gray_patch = gray_patches[j].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                img2_full = cv2.cvtColor(img2_full, cv2.COLOR_BGR2RGB)
                img1_full_warp = cv2.cvtColor(img1_full_warp, cv2.COLOR_BGR2RGB)

                if video_name[j] in RE:
                    MSE_RE.append(err_avg[j])
                elif video_name[j] in LT:
                    MSE_LT.append(err_avg[j])
                elif video_name[j] in LL:
                    MSE_LL.append(err_avg[j])
                elif video_name[j] in SF:
                    MSE_SF.append(err_avg[j])
                elif video_name[j] in LF:
                    MSE_LF.append(err_avg[j])

                if k % 200 == 0:
                    print(k)

                print("{}:{}".format(k, err_avg[j]))
                # eval_save_result([img2_full, img1_full_warp], npy_name[j] + "_" + str(err_avg[j]) + ".gif", manager)

        MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
        MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
        MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
        MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
        MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)
        MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg + MSE_LF_avg) / 5

        Metric = {"MSE_RE_avg":MSE_RE_avg, "MSE_LT_avg":MSE_LT_avg, "MSE_LL_avg":MSE_LL_avg, "MSE_SF_avg":MSE_SF_avg, "MSE_LF_avg":MSE_LF_avg, "AVG":MSE_avg}
        manager.update_metric_status(metrics=Metric, split=manager.params.eval_type, batch_size=1)

        # update data to logger
        manager.logger.info("Loss/valid epoch_{} {}: {:.2f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} ".format(manager.params.eval_type, manager.epoch_val,
                                        MSE_avg, MSE_RE_avg, MSE_LT_avg, MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))
 
        # For each epoch, print the metric
        manager.print_metrics(manager.params.eval_type, title=manager.params.eval_type, color="green")

        # manager.epoch_val += 1

        model.train()

def eval_save_result(save_file, save_name, manager):

    # save dir: model_dir
    save_dir_gif = os.path.join(manager.params.model_dir, 'gif')
    if not os.path.exists(save_dir_gif):
        os.makedirs(save_dir_gif)

    save_dir_gif_epoch = os.path.join(save_dir_gif, str(manager.epoch_val))
    if not os.path.exists(save_dir_gif_epoch):
        os.makedirs(save_dir_gif_epoch)

    if type(save_file)==list: # save gif
        utils.create_gif(save_file, os.path.join(save_dir_gif_epoch, save_name))
    elif type(save_file)==str: # save string information
        f = open(os.path.join(save_dir_gif_epoch, save_name), 'w')
        f.write(save_file)
        f.close()
    elif manager.val_img_save: # save single image
        cv2.imwrite(os.path.join(save_dir_gif_epoch, save_name), save_file)


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.fetch_net(params)

    # Initial status for checkpoint manager
    manager = Manager(model=model, optimizer=None, scheduler=None, params=params, dataloaders=dataloaders, writer=None, logger=logger)

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    evaluate(model, manager)
