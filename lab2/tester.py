import torch
import torch.nn.functional as F
from Dataloader import MIBCI2aDataset
from model.SCCNet import SCCNet
from torch.utils.data import DataLoader
import os
from trainer import fine_tunning

device = torch.device( 'cuda:2' if torch.cuda.is_available() else 'cpu' )


if __name__ == '__main__':

    dataset = MIBCI2aDataset('test')

    if os.path.exists('model_weight/model_FT_80.2%.pt'):
        print('load model')
        model = torch.load('model_weight/model_FT_80.2%.pt')
    else:
        model = SCCNet(numClasses=4, device=device,Nu=22, C=22, Nc=0, Nt=1).to(device)
    model.eval()

    idx = 0
    all_correct_case = 0
    all_sample = 0
    all_loss = []
    dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
    for i, (features, labels) in enumerate(dataloader):
        idx += 1
        features = features.to(device)
        labels = labels.to(device)

        print(f'subject {idx}')
        output = model.forward(features)
        class_prob = F.softmax(output, dim=1) # dim to provide the dim softmax should process (may be a dim for batch)
        prediction = torch.argmax(class_prob, dim=1)

        loss = model.get_loss(output, labels)
        all_loss.append(loss)

        correct_case = torch.sum( (prediction == labels).type(torch.int) ).item()
        subject_acc = correct_case / labels.shape[0]

        all_correct_case += correct_case
        all_sample += labels.shape[0]
    
    print(f'test accuracy: {all_correct_case/all_sample * 100}%, loss:{sum(all_loss)/len(all_loss)}')