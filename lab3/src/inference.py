import argparse
from oxford_pet import load_dataset
from utils import dice_score
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )


def test(args):
    
    hyper_parameter = {
        'batch_size': args.batch_size, 
    }

    dataset = load_dataset(f'{args.data_path}', 'test')
    dataloader = DataLoader(dataset, hyper_parameter['batch_size'], shuffle=False)
    model = torch.load(args.model, map_location=device)
    model.device = device
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

    print(f'test loss: {avg_epoch_loss}, test dice socre: {avg_dice}')

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

# UNet command: python3 src/inference.py --model saved_models/DL_Lab3_UNet_112550069_謝侑哲.pth --data_path dataset/oxford-iiit-pet --batch_size 1
# Res34_UNet command: python3 src/inference.py --model saved_models/DL_Lab3_ResNet34_UNet_112550069_謝侑哲.pth --data_path dataset/oxford-iiit-pet --batch_size 1
if __name__ == '__main__':
    args = get_args()
    print(args)
    test(args)