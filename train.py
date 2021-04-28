import torch
import argparse
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from skeleton_h36m import skeleton_H36M
from trainer import Trainer
from h36m_dataset import H36MDataset
from utils import get_config, set_seed, initialize_path
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/H3.6M.yaml', \
                        help='Path to the config file.')
    args = parser.parse_args()

    # initialize
    config = get_config(args.config)
    initialize_path(args, config)
    torch.backends.cudnn.benchmark = True

    # Tensorboard & create model folder
    train_writer = SummaryWriter(os.path.join(config['tb_dir'], 'train'))

    # Set random seed for reproducibility
    print("Random Seed: ", config['manualSeed'])
    set_seed(config['manualSeed'])

    # Load training setting
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    num_workers = config['num_workers']

    # Create dataloader
    dataset_train = H36MDataset('train', config)
    train_loader = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=num_workers)
    if config['valid']:
        dataset_valid = H36MDataset('valid', config)
        valid_loader = DataLoader(dataset_valid, batch_size, shuffle=True, num_workers=num_workers)

    # Setup dataset
    exp_mean = dataset_train.exp_mean
    exp_std = dataset_train.exp_std
    exp_dimTouse = dataset_train.exp_dimTOuse
    pos_mean = dataset_train.pos_mean
    pos_std = dataset_train.pos_std
    pos_dimTouse = dataset_train.pos_dimTouse

    # Setup model
    trainer = Trainer(skeleton_H36M, config)
    tr_info = open(os.path.join(config['info_dir'], "info-network"), "w")
    print(trainer.gen, file=tr_info)
    print(trainer.dis, file=tr_info )
    tr_info.close()

    # Start training
    for epoch in range(n_epochs):
        pbar = tqdm(train_loader)
        for it, data in enumerate(pbar):     # data : (batch_size, seq_len, dim)
            
            losses_dis = trainer.dis_update(data, config)
            losses_gen = trainer.gen_update(data, exp_mean, exp_std, exp_dimTouse, \
                                            pos_mean, pos_std, pos_dimTouse, config)
                                            
            losses_dis_dict = {k:v.item() for k, v in losses_dis.items()}
            losses_gen_dict = {k:v.item() for k, v in losses_gen.items()}

            pbar.set_description("EPOCH[{}][{}/{}]".format(epoch+1, it+1, len(train_loader)))
            pbar.set_postfix(OrderedDict({"Discriminator loss": sum(losses_dis_dict.values()), \
                                          "Generator loss": sum(losses_gen_dict.values())}))

            for k, v in losses_dis_dict.items():
                train_writer.add_scalar(k, v, epoch*len(train_loader)+it)
            for k, v in losses_gen_dict.items():
                train_writer.add_scalar(k, v, epoch*len(train_loader)+it)
            
        # validation
        if config['valid'] and (epoch+1) % config['valid_every'] == 0:
            loss_test = {}
            for t, data_valid in enumerate(valid_loader):
                losses_valid = trainer.test_loss(data_valid, exp_mean, exp_std, exp_dimTouse, \
                                                 pos_mean, pos_std, pos_dimTouse)
                losses_valid_dict = {k:v.item() for k, v in losses_valid.items()}

                for key in losses_valid_dict.keys():
                    loss = losses_valid_dict[key]
                    if key not in loss_test:
                        loss_test[key] = []
                    loss_test[key].append(loss)
            
            log = f'epoch [{epoch+1}], '
            loss_test_avg = dict()
            for key, loss in loss_test.items():
                loss_test_avg[key] = sum(loss) / len(loss)
            log += ' '.join([f'{key:}: [{value:}]' for key, value in loss_test_avg.items()])
            print(log)
            
        trainer.update_learning_rate()

        if (epoch+1) % config['save_every'] == 0:
            trainer.save_checkpoint(epoch+1)
            
        if (epoch+1) >= n_epochs:
            sys.exit('Finish training')


if __name__ == '__main__':
    main()