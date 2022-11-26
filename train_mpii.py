import argparse
import os
import os.path as osp

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
import yaml

from stacked_hourglass.model import _hg
from stacked_hourglass.datasets.mpii import Mpii
from stacked_hourglass.train import do_training_epoch, do_validation_epoch
from stacked_hourglass.utils.logger import Logger
from stacked_hourglass.utils.misc import save_checkpoint, adjust_learning_rate

from LPN.model import get_lpn


def get_optimizer_scheduler(my_model, opt_cfg, sched_cfg):
    for k, v in my_model.named_parameters():
        if not v.requires_grad:
            print(f'Warning: {k} will not be optimized')

    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if opt_cfg['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable,
            lr=opt_cfg['lr'],
            betas=opt_cfg['betas'],
            eps=opt_cfg['eps'],
            weight_decay=opt_cfg['weight_decay'],
        )
    elif opt_cfg['optimizer'] == 'sparse_l1':
        optimizer = OBProxSG(
            trainable,
            lr=opt_cfg['lr'],
            lambda_=opt_cfg['lambda'],
            epochSize=opt_cfg['epochSize'],
            Np=opt_cfg['epochs'] // 10,
        )
    elif opt_cfg['optimizer'] == 'rmsprop':
        optimizer = RMSprop(
            trainable,
            lr=opt_cfg['lr'],
            momentum=opt_cfg['momentum'],
            weight_decay=opt_cfg['weight_decay'],
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        **sched_cfg
    )

    return optimizer, scheduler


def main(args):
    with open(args.cfg, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # Select the hardware device to use for inference.
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Disable gradient calculations by default.
    torch.set_grad_enabled(False)

    # create checkpoint dir
    os.makedirs(osp.join(args.checkpoint, 'tensorrboard'), exist_ok=True)
    board = SummaryWriter(osp.join(args.checkpoint, "tensorboard"))

    if 'hg' in cfg['model']['arch']:
        model = _hg(**cfg['model'])
    if 'lpn' in cfg['model']['arch']:
        model = get_lpn(cfg['model']['resnet_spec'], cfg['model']['use_gcb'])

    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: ", total_params)

    model = DataParallel(model).to(device)

    # Create optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(
        model, cfg['optimizer'], cfg['scheduler']
    )

    best_acc = 0

    # optionally resume from a checkpoint
    if args.resume:
        assert osp.isfile(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        logger = Logger(osp.join(args.checkpoint, 'log.txt'), resume=True)
    else:
        logger = Logger(osp.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    # create data loader
    train_dataset = Mpii(args.image_path, is_train=True, inp_res=args.input_shape)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    cfg['optimizer']['epochSize'] = len(train_loader)

    val_dataset = Mpii(args.image_path, is_train=False, inp_res=args.input_shape)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # train and eval
    args.epochs = cfg['optimizer']['epochs']
    for epoch in trange(args.start_epoch, args.epochs, desc='Overall', ascii=True):
        # train for one epoch
        train_loss, train_acc = do_training_epoch(train_loader, model, device, Mpii.DATA_INFO,
                                                  optimizer,
                                                  scheduler,
                                                  acc_joints=Mpii.ACC_JOINTS)

        # evaluate on validation set
        valid_loss, valid_acc, predictions = do_validation_epoch(val_loader, model, device,
                                                                 Mpii.DATA_INFO, False,
                                                                 acc_joints=Mpii.ACC_JOINTS)

        # print metrics
        lr = scheduler.get_last_lr()[0]
        tqdm.write(f'[{epoch + 1:3d}/{args.epochs:3d}] lr={lr:0.2e} '
                   f'train_loss={train_loss:0.4f} train_acc={100 * train_acc:0.2f} '
                   f'valid_loss={valid_loss:0.4f} valid_acc={100 * valid_acc:0.2f}')

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])
        logger.plot_to_file(osp.join(args.checkpoint, 'log.svg'), ['Train Acc', 'Val Acc'])

        board.add_scalar('Train/acc', train_acc, epoch + 1)
        board.add_scalar('Val/acc', valid_acc, epoch + 1)
        board.add_scalar('Lr', lr, epoch + 1)

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': cfg['model']['arch'],
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, predictions, is_best, checkpoint=args.checkpoint, snapshot=args.snapshot)

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a stacked hourglass model.')
    # Dataset setting
    parser.add_argument('--image-path', default='', type=str,
                        help='path to images')

    # Training strategy
    parser.add_argument('--cfg', default='', type=str,
						help='Config files for model and training')
    parser.add_argument('--input_shape', default=(256, 256), type=int, nargs='+',
                        help='Input shape of the model. Given as: (H, W)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    main(parser.parse_args())
