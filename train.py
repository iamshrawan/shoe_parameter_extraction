import argparse
import json
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, sampler
import numpy as np
import wandb

from SVModel import SVRegressor
from MVModel import MVRegressor
from Dataset import get_dataset, MultiviewImgDataset
from train_utils import train, validate, adjust_learning_rate, save_checkpoint, sample_output, get_predictions

model_names = ['vgg16', 'resnet50', 'resnet18']

parser = argparse.ArgumentParser(description='Shoe Parameter Extraction Training')
parser.add_argument('--arch', '-a', default='vgg16', choices=model_names, 
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16)')
parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
parser.add_argument('-j', '--workers', default=-1, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-d', '--dataset', type=str,
                    help='dataset paath')
parser.add_argument('--epochs', default=90, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    help='mini-batch size (default: 8)')
parser.add_argument('--lr', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--finetune', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--num_shoe_params', type=int, default=22, help='Number of Shoe Parameters to Extract')
parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=2021')
parser.add_argument('--load_checkpoint', type=int, default=-1, help='load checkpoint')
parser.add_argument('--mv', action='store_true', help='train multi-view model?')
parser.add_argument('--num_views', type=int, default=3, help='number of views')
parser.add_argument('--schedule_lr', action='store_true', help='schedule learning rate')
parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay parameter')

#Weights and biases
parser.add_argument('--exp', type=str, default='Dataset500', default='test', 
                    help="Experiment name: To track in weights and biases")
parser.add_argument("--entity", type=str, default='shrawan', help="wandb username")
parser.add_argument("--sample_interval", type=int, default=25, 
                    help="Sample interval to upload validation set prediction")



opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda:0" if opt.cuda else "cpu")
print(device)

split_directory = './manualCropped/dataset_splits'
#mv_directory = './multiview/MVShoeDataset/mutliview_data'
print("<========== Loading Datasets ============>")
if opt.mv:
    single_view = True if opt.num_views == 1 else False
    dataset = MultiviewImgDataset(opt.dataset, shuffle=False, single_view=single_view, num_views=opt.num_views)
    all_indices = list(np.arange(len(dataset)))
    random.seed(opt.seed)
    test_indices = random.sample(all_indices, 16)
    np.save('test_indices.npy', test_indices)
    train_indices = list(np.setdiff1d(all_indices, test_indices))
    dev_indices = random.sample(train_indices, 16)
    train_indices = list(np.setdiff1d(train_indices, dev_indices))
    train_loader = DataLoader(dataset, sampler=sampler.SubsetRandomSampler(train_indices),
                            batch_size=opt.batch_size, drop_last=True)
    dev_loader = DataLoader(dataset, sampler=sampler.SubsetRandomSampler(dev_indices),
                            batch_size=1)
    test_loader = DataLoader(dataset, sampler=sampler.SubsetRandomSampler(test_indices),
                            batch_size=1)
    if single_view:
        model = SVRegressor(arch_name=opt.arch, pre_trained=opt.finetune, num_shoe_params=opt.num_shoe_params)
    else:
        model = MVRegressor(arch_name=opt.arch, in_channel=opt.in_channels, num_views=opt.num_views,
                        pre_trained=opt.finetune, num_shoe_params=opt.num_shoe_params).to(device)
    best_loss = 10**8
    avg_losses = []
    save_losses = False
    
else:
    
    train_dataset = get_dataset(mode='train',n_channel=opt.in_channels, split_path=split_directory)
    dev_dataset = get_dataset(mode='dev', n_channel=opt.in_channels, split_path=split_directory)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    

    #print("<========== Dataset loaded ===============>")

    #print("<========== Loading model ==========>")

    if opt.load_checkpoint != -1:
        print("<========== Loading checkpoint from epoch {} ==========>".format(opt.load_checkpoint))
        checkpoint_state = torch.load("checkpoint/{}/shoe_model_epoch_{}.pth".format(opt.exp, opt.load_checkpoint))
        model = SVRegressor(checkpoint_state['arch'], in_channel=opt.in_channels, pre_trained=opt.finetune)
        model.load_state_dict(checkpoint_state['state_dict'])
        opt.start_epoch = checkpoint_state['epoch']
        opt.lr = checkpoint_state['lr']
        best_loss = checkpoint_state['best_loss']
        with open('metrics/eval_metrics_' + opt.exp + '.json', 'r') as f:
            j = json.load(f)
            avg_losses = j['Dev MSE']
            save_losses = False
    else:
        model = SVRegressor(arch_name=opt.arch, in_channel=opt.in_channels, 
                        pre_trained=opt.finetune, num_shoe_params=opt.num_shoe_params).to(device)
        best_loss = 10**8
        avg_losses = []
        save_losses = False

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=opt.lr, weight_decay=opt.weight_decay)

print("<==========Training begins ==========>")


runs = wandb.init(project="ShoeParameterExt", entity = opt.entity, name=f"{opt.exp}_{opt.arch}", reinit=True)
for epoch in range(opt.start_epoch, opt.epochs + 1):
    if opt.schedule_lr:
        adjust_learning_rate(optimizer, epoch, opt.lr)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, device, epoch, opt.mv)

    # evaluate on dev set
    avg_loss = validate(dev_loader, model, criterion, device, opt.mv)
    avg_losses.append(avg_loss)
    wandb.log({
            "valid_mse": avg_loss
    })
   

    # checkpoint
    if epoch % opt.sample_interval == 0:
        fig1, fig2 = sample_output(model, dev_loader, opt.mv, device)
        wandb.log({"parameter_plots1": wandb.Image(fig1),
                   "parameter_plots2": wandb.Image(fig2)})
        save_losses = True
    if avg_loss < best_loss:
        best_loss = avg_loss

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'lr' : optimizer.param_groups[0]['lr'],
            'arch': model.arch_name,
            'best_loss': best_loss
        }, save_losses, avg_losses, opt.exp)

get_predictions(test_loader, model, device, opt.exp)



