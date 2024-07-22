import torch
from torch.utils.data.dataloader import DataLoader
from Dataloader import MIBCI2aDataset
from model.SCCNet import SCCNet
import os
import utils
import torch.nn.functional as F

device = torch.device( 'cuda:2' if torch.cuda.is_available() else 'cpu' )

def accuracy_explore_tool(model): # function for test in training process (same as code in tester)

    device = torch.device( 'cuda:2' if torch.cuda.is_available() else 'cpu' )
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

def fine_tunning(model, hyper_paramaters): # function for fine tune train, mostly same as trainer

    # only update the parameters of fully connect layer(classifier), achieve low training cost, high efficiency
    # for para in model.parameters():
    #     para.requires_grad = False

    # for para in model.fc.parameters():
    #     para.requires_grad = True
    
    dataset = MIBCI2aDataset('finetune') # use fine-tune dataset
    dataloader = DataLoader(dataset, batch_size=hyper_paramaters['batch_size'], shuffle=True) 

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_paramaters['learning rate'], weight_decay=1)


    n_epoch = hyper_paramaters['n_epoch']
    loss_log = []
    log_point = []
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
            print(f'epoch: {epoch+1}, loss: {avg_epoch_loss}')
            acc = accuracy_explore_tool(model)  
            if acc > max_accuracy:
                max_accuracy = acc
                torch.save(model, 'model_weight/model.pt')
        if (epoch+1)%100 == 0 or epoch==0:
            loss_log.append(avg_epoch_loss)
            log_point.append(epoch+1)

        
    utils.training_loss_plot((log_point, loss_log), 'Fine_tune_training_loss.jpg')


if __name__ == '__main__':

    mode = None # 'finetune' or None, mode for fine-tune or normal training

    print('training...')
    hyper_paramaters = {
        'batch_size': 288,
        'learning rate': 0.001,
        'n_epoch': 1000,
        'Nu': 22,
        'dropoutRate': 0.5,
    }

    if mode == 'finetune':
        model = torch.load('model_weight/model_FT_79.8%.pt') # load the best LOSO model
        model.to(device)
        fine_tunning(model, hyper_paramaters) # fine tuning training
        os._exit(0) # after fine tuning, directly end the process
    elif os.path.exists('model_weight/model.pt'):
        model = torch.load('model_weight/model.pt') # if model not be trained competely, load model
    else:
        model = SCCNet(numClasses=4, device=device, Nu=hyper_paramaters['Nu'], C=22, Nt=1, dropoutRate=hyper_paramaters['dropoutRate']).to(device)  # build a new model, make parameter tensor on chose device (normaly GPU)

    dataset = MIBCI2aDataset('train') # get the training dataset
    dataloader = DataLoader(dataset, batch_size=hyper_paramaters['batch_size'], shuffle=True) # use dataloader to pratice update after batch rather than epoch
    
    model.train() # open train mode, make drop layer, normalization layer work
    # Adam Optimizer, which decrease learning rate after weight be update 
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_paramaters['learning rate'], weight_decay=1e-4)

    n_epoch = hyper_paramaters['n_epoch']
    loss_log = [] # record training loss for loss plot
    log_point = [] # record time point for loss plot
    max_accuracy = 0 # for model to find the max acc
    for epoch in range(n_epoch):
        epoch_loss = [] # each batch will generate loss, epoch_loss record them for compute average value
        for idx, (batch_features, batch_labels) in enumerate(dataloader):
            batch_features = batch_features.to(device) # make data tensor on chose device (normaly GPU)
            batch_labels = batch_labels.to(device)
            output = model(batch_features) # get output for loss caculate

            loss = model.get_loss(output, batch_labels) # compute cross-entropy loss
            epoch_loss.append(loss.item())
            loss.backward() # backpropagation from loss tensor
            optimizer.step() # update parameter tensor by the .grad attribute get in backpropagation
            optimizer.zero_grad() # clear .grad attribute to avoid accumulation

        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss) # compute average loss of exch batch in a epoch
        if (epoch+1)%1 == 0 or epoch==0: 
            print(f'epoch: {epoch+1}, loss: {avg_epoch_loss}') # log
            acc = accuracy_explore_tool(model)  
            if acc > max_accuracy: # save the best accuracy model
                max_accuracy = acc
                torch.save(model, 'model_weight/model.pt')
        if (epoch+1)%100 == 0 or epoch==0: # record training loss per 100 epoch
            loss_log.append(avg_epoch_loss)
            log_point.append(epoch+1)

        
    utils.training_loss_plot((log_point, loss_log), 'SD_training_loss.jpg')
