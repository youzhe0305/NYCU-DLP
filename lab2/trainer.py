import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from Dataloader import MIBCI2aDataset
from model.SCCNet import SCCNet
import math

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

if __name__ == '__main__':

    hyper_paramaters = {
        'batch_size': 4,
        'learning rate': 0.1,

    }
    dataset = MIBCI2aDataset('train')
    dataloader = DataLoader(dataset, batch_size=hyper_paramaters['batch_size'], shuffle=True)
    model = SCCNet(numClasses=4, timeSample=0, Nu=22, C=22, Nc=0, Nt=1, dropoutRate=0)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper_paramaters['learning rate'])


    n_epoch = 100
    n_samples = len(dataset)
    n_iteration = math.ceil(n_samples / hyper_paramaters['batch_size'])
    for epoch in range(n_epoch):
        epoch_loss = []
        for idx, (batch_features, batch_labels) in enumerate(dataloader):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            output = model(batch_features)
            loss = model.get_loss(output, batch_labels)
           
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        if (epoch+1)%1 == 0:
            print(f'epoch: {epoch+1}, loss: {avg_epoch_loss}')


    # idx = 0
    # all_correct_case = 0
    # all_sample = 0
    # all_loss = []
    # for subject_features, subject_labels in dataset:
    #     idx += 1
    #     subject_features = subject_features.to(device)
    #     subject_labels = subject_labels.to(device)

    #     print(f'subject {idx}')
    #     output = model.forward(subject_features)
    #     class_prob = F.softmax(output, dim=1) # dim to provide the dim softmax should process (may be a dim for batch)
    #     prediction = torch.argmax(class_prob, dim=1)

    #     loss = model.get_loss(output, subject_labels)
    #     all_loss.append(loss)

    #     correct_case = torch.sum( (prediction == subject_labels).type(torch.int) ).item()
    #     subject_acc = correct_case / subject_labels.shape[0]

    #     all_correct_case += correct_case
    #     all_sample += subject_labels.shape[0]
    
    # print(f'test accuracy: {all_correct_case/all_sample}, loss:{sum(all_loss)/len(all_loss)}')
