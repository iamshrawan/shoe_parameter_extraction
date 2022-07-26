import torch
import os
import json
import wandb
import numpy as np
import matplotlib.pyplot as plt


def sample_output(model, val_loader, mv, device):
    model.eval()
    batch = next(iter(val_loader))
    if mv:
            N,V,C,H,W = batch[0].size()
            img = batch[0].view(-1,C,H,W).to(device)
            shape_key = batch[1]
            shape_key = shape_key/2 + 0.5
    output = model(img)
    output = output/2 + 0.5
    pred_shk1 = output[0].detach().cpu().numpy()
    true_shk1 = shape_key[0].detach().cpu().numpy()
    pred_shk2 = output[1].detach().cpu().numpy()
    true_shk2 = shape_key[1].detach().cpu().numpy()

    fig1 = plot_shk_bars(pred_shk1, true_shk1, 'Predicted', 'Ground Truth')
    fig2 = plot_shk_bars(pred_shk2, true_shk2, 'Predicted', 'Ground Truth')
    return fig1, fig2

def get_predictions(test_loader, mv_model, device, exp):
    shapekeys = {}
    with torch.no_grad():
        for batch in test_loader:
            N,V,C,H,W = batch[0].size()
            img = batch[0].view(-1,C,H,W).to(device)
            #with torch.autograd.set_detect_anomaly(True):
            pred_shk = (mv_model(img)[0]/2 + 0.5).detach().cpu().tolist()
            true_shk = (batch[1][0]/2 + 0.5).detach().cpu().tolist()
            idx = batch[2].detach().tolist()[0]
            shapekeys[f'{idx}'] = (pred_shk, true_shk)
    with open(f'test_{exp}.json', 'w') as j:
        json.dump(shapekeys, j)
 
def plot_shk_bars(pred_shk, true_shk, plabel, glabel):
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(15,6))
    ax1.bar(np.arange(0,22), pred_shk)
    ax2.bar(np.arange(0,22), true_shk)
    ax1.set_title(plabel)
    ax2.set_title(glabel)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_xticks(np.arange(0,22,1))
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_xticks(np.arange(0,22,1))
    plt.ylabel('Shape Key value')
    plt.xlabel('Parameter')

    return fig

def train(train_loader, model, criterion, optimizer, device, epoch, mv=False):
    model.train()

    for i, batch in enumerate(train_loader):
        if mv:
            N,V,C,H,W = batch[0].size()
            img = batch[0].view(-1,C,H,W).to(device)
            shape_key = batch[1].to(device)
        else:
            img, shape_key = batch[0].to(device), batch[1].to(device)

        # compute output
        output = model(img)
        loss = criterion(output, shape_key)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        
        print("===> Epoch[{}]({}/{}): MSE {:.4f}".format(
            epoch, i, len(train_loader), loss.item()))
        wandb.log({
            "train_mse": loss.item()
        })


def validate(val_loader, model, criterion, device, mv=False):
    model.eval()
    losses = 0
    for batch in val_loader:
        if mv:
            N,V,C,H,W = batch[0].size()
            img = batch[0].view(-1,C,H,W).to(device)
            shape_key = batch[1].to(device)
        else:
            img, shape_key = batch[0].to(device), batch[1].to(device)

        # compute output
        output = model(img)
        loss = criterion(output, shape_key)

        # record loss
        losses += loss.item()
    avg_loss = losses / len(val_loader)
    print("===> Avg. Dev MSE: {:.4f}".format(avg_loss))

    return avg_loss


def save_checkpoint(state, save_losses, avg_losses, exp='Dataset500'):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists("metrics"):
        os.mkdir("metrics")
    if not os.path.exists(os.path.join("checkpoint", exp)):
        os.mkdir(os.path.join("checkpoint", exp))
   
    
    shoe_model_out_path = "checkpoint/{}/shoe_model_{}_epoch_{}.pth".format(exp, state['arch'], state['epoch'])
    torch.save(state, shoe_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint/" + exp))

    if save_losses:
        print("Saving Dev loss values")
        mse = {
            'Dev MSE' : avg_losses
        }
        with open('metrics/eval_metrics_' + exp + '.json', 'w') as e:
            json.dump(mse, e)



def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr
