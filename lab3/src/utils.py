import torch
import matplotlib.pyplot as plt

def dice_score(pred_mask, gt_mask):
    # unify to (batch, H, W)
    gt_mask = gt_mask.view(gt_mask.shape[0], gt_mask.shape[2], gt_mask.shape[3])
    
    epsilon = 1e-6
    # for class only 0,1 dice take 1 part
    common_pixel_1 = torch.sum((pred_mask * gt_mask).type(torch.int), dim=(1,2))
    pred_pixel_1 = torch.sum((pred_mask==1).type(torch.int), dim=(1,2))
    gt_pixel_1 = torch.sum((gt_mask==1).type(torch.int), dim=(1,2))

    dice_score = (2*common_pixel_1 + epsilon) / (pred_pixel_1 + gt_pixel_1 + epsilon)
    return dice_score.mean()

def training_loss_plot(loss_log, filename):
    x = loss_log[0]
    y = loss_log[1]
    plt.clf()
    plt.plot(x, y, marker='o')
    plt.show()
    plt.savefig(f'output/{filename}')


