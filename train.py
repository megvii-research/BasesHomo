"""Train the model"""

import argparse
import datetime
import os

import torch
import torch.optim as optim
from tqdm import tqdm
# from apex import amp

import dataset.data_loader as data_loader
import model.net as net

from common import utils
from common.manager import Manager
from evaluate import evaluate
from loss.losses import compute_losses

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  #
parser.add_argument('-ow', '--only_weights', action='store_true', help='Only use weights to load or load all train status.')


def train(model, manager):

    # loss status initial
    manager.reset_loss_status()

    # set model to training mode
    torch.cuda.empty_cache()
    model.train()

    # Use tqdm for progress bar
    with tqdm(total=len(manager.dataloaders['train'])) as t:
        for i, data_batch in enumerate(manager.dataloaders['train']):

            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)

            # compute model output and loss
            output_batch = model(data_batch)
            loss = compute_losses(output_batch, data_batch, manager.params)

            # update loss status and print current loss and average loss
            manager.update_loss_status(loss=loss, split="train")

            # clear previous gradients, compute gradients of all variables loss
            manager.optimizer.zero_grad()
            loss['total'].backward()

            # performs updates using calculated gradients
            manager.optimizer.step()
            # manager.logger.info("Loss/train: step {}: {}".format(manager.step, manager.loss_status['total'].val))

            # update step: step += 1
            manager.update_step()

            # infor print
            print_str = manager.print_train_info()

            t.set_description(desc=print_str)
            t.update()

    manager.scheduler.step()

    # update epoch: epoch += 1
    manager.update_epoch()

def train_and_evaluate(model, manager):

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        manager.load_checkpoints()

    for epoch in range(manager.params.num_epochs):

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, manager)

        # Save latest model, or best model weights accroding to the params.major_metric
        manager.check_best_save_last_checkpoints(latest_freq_val=999, latest_freq=1)

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Update args into params
    params.update(vars(args))

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    logger = utils.set_logger(os.path.join(params.model_dir, 'train.log'))

    # Set the tensorboard writer
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create the input data pipeline
    logger.info("Loading the train datasets from {}".format(params.train_data_dir))

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        optimizer = optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.fetch_net(params)
        optimizer = optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)

    # initial status for checkpoint manager
    manager = Manager(model=model, optimizer=optimizer, scheduler=scheduler, params=params, dataloaders=dataloaders, writer=None, logger=logger)

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(model, manager)
