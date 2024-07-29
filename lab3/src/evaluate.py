from oxford_pet import load_dataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from models.unet import UNet
from utils import dice_score

def evaluate(net, device):
    # implement the evaluation function here
    hyper_parameter = {
        'batch_size': 10, 
    }

    dataset = load_dataset('dataset', 'valid')
    dataloader = DataLoader(dataset, hyper_parameter['batch_size'], shuffle=False)
    model = net
    model.eval()

    epoch_loss = []
    epoch_dice = []
    for idx, samples in enumerate(dataloader):
        img = samples['image'].type(torch.float).to(device)
        mask = samples['mask'].to(device) # class (2 class)
        trimap = samples['trimap'].to(device) # separete image to 3 clss: 1-front-ground, 2-background, 3-unknown
        
        output = model(img)

        loss = model.get_loss(output, mask)
        epoch_loss.append(loss.item())

        # batch, channel, H, W
        class_prob = F.softmax(output, dim=1)
        prediction = torch.argmax(class_prob, dim=1)

        dice = dice_score(prediction, mask)
        epoch_dice.append(dice.item())

    avg_dice = sum(epoch_dice) / len(epoch_dice)
    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    return avg_epoch_loss, avg_dice

if __name__ == '__main__':

    device = torch.device( 'cuda:1' if torch.cuda.is_available() else 'cpu' )
    model = torch.load('saved_models/model_Res34_UNet.pth')
    loss, dice = evaluate(model, device)
    print(f'validation loss: {loss}, dice socre: {dice*100}%')

