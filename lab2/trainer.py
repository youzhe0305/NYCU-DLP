import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from Dataloader import MIBCI2aDataset
from model.SCCNet import SCCNet
import math
import os
import utils

device = torch.device( 'cuda:2' if torch.cuda.is_available() else 'cpu' )

def accuracy_explore_tool(model):

    dataset = MIBCI2aDataset('test')
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
    return all_correct_case/all_sample

def fine_tunning(model, hyper_paramaters):

    for para in model.parameters():
        para.requires_grad = False

    for para in model.fc.parameters():
        para.requires_grad = True
    
    dataset = MIBCI2aDataset('finetune')
    dataloader = DataLoader(dataset, batch_size=hyper_paramaters['batch_size'], shuffle=True)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_paramaters['learning rate'], weight_decay=1e-4)


    n_epoch = hyper_paramaters['n_epoch']
    n_samples = len(dataset)
    loss_log = []
    log_point = []
    min_loss = 1e9
    max_accuracy = 0
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
        if (epoch+1)%1 == 0 or epoch==0:
            # if avg_epoch_loss < min_loss:
            #     torch.save(model, 'model_weight/model.pt')
            print(f'epoch: {epoch+1}, loss: {avg_epoch_loss}')
            acc = accuracy_explore_tool(model)  
            if acc > max_accuracy:
                max_accuracy = acc
                torch.save(model, 'model_weight/model.pt')
        if (epoch+1)%1 == 100 or epoch==0:
            loss_log.append(avg_epoch_loss)
            log_point.append(epoch+1)

        
    utils.training_loss_plot((log_point, loss_log), 'Fine_tune_training_loss.jpg')


if __name__ == '__main__':

    mode = 'finetune'

    print('training...')
    hyper_paramaters = {
        'batch_size': 288,
        'learning rate': 0.001,
        'n_epoch': 200,
    }

    if mode == 'finetune':
        model = torch.load('model_weight/model_LOSO_200_62.5%.pt')
        model.to(device)
        fine_tunning(model, hyper_paramaters)
        os._exit(0)
    elif os.path.exists('model_weight/model.pt'):
        model = torch.load('model_weight/model.pt')    
    else:
        model = SCCNet(numClasses=4, device=device, Nu=22, C=22, Nt=1).to(device)  

    dataset = MIBCI2aDataset('train')
    dataloader = DataLoader(dataset, batch_size=hyper_paramaters['batch_size'], shuffle=True)
    

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_paramaters['learning rate'], weight_decay=1e-4)


    n_epoch = hyper_paramaters['n_epoch']
    n_samples = len(dataset)
    n_iteration = math.ceil(n_samples / hyper_paramaters['batch_size'])
    loss_log = []
    log_point = []
    min_loss = 1e9
    max_accuracy = 0
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
        if (epoch+1)%1 == 0 or epoch==0:
            # if avg_epoch_loss < min_loss:
            #     torch.save(model, 'model_weight/model.pt')
            print(f'epoch: {epoch+1}, loss: {avg_epoch_loss}')
            acc = accuracy_explore_tool(model)  
            if acc > max_accuracy:
                max_accuracy = acc
                torch.save(model, 'model_weight/model.pt')
        if (epoch+1)%1 == 100 or epoch==0:
            loss_log.append(avg_epoch_loss)
            log_point.append(epoch+1)

        
    utils.training_loss_plot((log_point, loss_log), 'SD_training_loss.jpg')
