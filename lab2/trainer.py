import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from Dataloader import MIBCI2aDataset
from model.SCCNet import SCCNet
import math
import os

device = torch.device( 'cuda:2' if torch.cuda.is_available() else 'cpu' )

if __name__ == '__main__':

    print('training...')
    hyper_paramaters = {
        'batch_size': 10,
        'learning rate': 0.1,

    }
    
    dataset = MIBCI2aDataset('train')
    dataloader = DataLoader(dataset, batch_size=hyper_paramaters['batch_size'], shuffle=True)
    
    if os.path.exists('model_weight/model.pt'):
        model = torch.load('model_weight/model.pt')    
    else:
        model = SCCNet(numClasses=4, device=device, timeSample=0, Nu=22, C=22, Nc=0, Nt=1, dropoutRate=0).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper_paramaters['learning rate'])


    n_epoch = 1000
    n_samples = len(dataset)
    n_iteration = math.ceil(n_samples / hyper_paramaters['batch_size'])
    for epoch in range(n_epoch):
        epoch_loss = []
        for idx, (batch_features, batch_labels) in enumerate(dataloader):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            output = model(batch_features)

            loss = model.get_loss(output, batch_labels)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        if (epoch+1)%100 == 0:
            print(f'epoch: {epoch+1}, loss: {avg_epoch_loss}')
    torch.save(model, 'model_weight/model.pt')