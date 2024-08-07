import argparse
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from models.unet import UNet
from models.resnet34_unet import ResNet34_Unet
from evaluate import evaluate
import torch
from utils import training_loss_plot
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

def train_UNet(args):
    # implement the training function here
    
    hyper_parameter = {
        'n_epoch': args.epochs,
        'batch_size': args.batch_size, # 3312 samples for train
        'learning_rate': args.learning_rate,
        'regularization': 0,
        'bilinear': True
    }

    dataset = load_dataset(f'{args.data_path}', 'train')
    dataloader = DataLoader(dataset, hyper_parameter['batch_size'], shuffle=True)
    if os.path.exists('saved_models/model_UNet.pth'):
        print('load trained model')
        model = torch.load('saved_models/model_UNet.pth')
    else:
        print('not load model')
        model = UNet(in_channels=3, n_class=2, bilinear=hyper_parameter['bilinear'], device=device).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameter['learning_rate'], weight_decay=hyper_parameter['regularization'])

    loss_log = []
    epoch_log = []
    max_acc = 0
    for epoch in range(hyper_parameter['n_epoch']):
        epoch_loss = []
        for idx, samples in enumerate(dataloader):
            img = samples['image'].type(torch.float).to(device)
            mask = samples['mask'].to(device) # class (2 class)
            trimap = samples['trimap'].to(device) # separete image to 3 clss: 1-front-ground, 2-background, 3-unknown
            output = model(img)
            loss = model.get_loss(output, mask)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        validation_loss, validation_dice = evaluate(model, device)
        with open('output/training_process_record.txt', 'a') as f:
            f.write(f'{epoch},{avg_epoch_loss},{validation_loss},{validation_dice}\n')
        print(f'epoch: {epoch+1}, training_loss: {round(avg_epoch_loss,4)}, validation_loss: {round(validation_loss,4)}, dice score: {round(validation_dice,4)*100}%')
        if max_acc < validation_dice:
            max_acc = validation_dice
            torch.save(model, 'saved_models/model_UNet.pth')
        if (epoch+1)%5 == 0 or epoch==0:
            loss_log.append(avg_epoch_loss)
            epoch_log.append(epoch+1)
    training_loss_plot((loss_log, epoch_log), 'UNet_traing_loss.jpg')

def train_Res34_UNet(args):
    # implement the training function here
    
    hyper_parameter = {
        'n_epoch': args.epochs,
        'batch_size': args.batch_size, # 3312 samples for train
        'learning_rate': args.learning_rate,
        'regularization': 0,
        'bilinear': True,
    }

    dataset = load_dataset(args.data_path, 'train')
    dataloader = DataLoader(dataset, hyper_parameter['batch_size'], shuffle=True)
    if os.path.exists('saved_models/model_Res34_UNet.pth'):
        print('load trained model')
        model = torch.load('saved_models/model_Res34_UNet.pth')
    else:
        print('not load model')
        model = ResNet34_Unet(in_channels=3, n_class=2, bilinear=hyper_parameter['bilinear'], device=device).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameter['learning_rate'], weight_decay=hyper_parameter['regularization'])

    loss_log = []
    epoch_log = []
    max_acc = 0

    for epoch in range(hyper_parameter['n_epoch']):
        epoch_loss = []
        for idx, samples in enumerate(dataloader):
            img = samples['image'].type(torch.float).to(device)
            mask = samples['mask'].to(device) # class (2 class)
            trimap = samples['trimap'].to(device) # separete image to 3 clss: 1-front-ground, 2-background, 3-unknown
            output = model(img)
            loss = model.get_loss(output, mask)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        validation_loss, validation_dice = evaluate(model, device)
        with open('output/training_process_record.txt', 'a') as f:
            f.write(f'{epoch},{avg_epoch_loss},{validation_loss},{validation_dice}\n')
        print(f'epoch: {epoch+1}, training_loss: {round(avg_epoch_loss,4)}, validation_loss: {round(validation_loss,4)}, dice score: {round(validation_dice,4)*100}%')
        
        if max_acc < validation_dice:
            max_acc = validation_dice
            torch.save(model, 'saved_models/model_Res34_UNet.pth')
        
        if (epoch+1)%5 == 0 or epoch==0:
            loss_log.append(avg_epoch_loss)
            epoch_log.append(epoch+1)
    training_loss_plot((epoch_log, loss_log), 'Res34_UNet_traing_loss.jpg')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', type=str, help='choose the model to train')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()

# UNet command: python3 src/train.py --model UNet --data_path dataset/oxford-iiit-pet --epochs 300 --batch_size 20 --learning_rate 0.001
# ResNet34_UNet command: python3 src/train.py --model ResNet34_UNet --data_path dataset/oxford-iiit-pet --epochs 300 --batch_size 60 --learning_rate 0.001

if __name__ == "__main__":
    args = get_args()
    if args.model == 'UNet':
        train_UNet(args)
    elif args.model == 'ResNet34_UNet':
        train_Res34_UNet(args)