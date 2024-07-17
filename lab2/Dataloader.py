import torch
import numpy as np
import os

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method

        X = []
        for p in os.listdir(filePath):
            print(f'{filePath}{p}')
            # load_data shape: (288,22,438) 288 trial, signal: (22,438)
            # 22 -> electrode(channel), 438 -> time point
            load_data = np.load(f'{filePath}{p}') 
            X.append(load_data)
        X = np.array(X).astype(float)
        X = torch.from_numpy(X).type(torch.float32)
        X = X.view(-1, X.shape[2], X.shape[3])
        print(X.shape)
        return X

        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        pass

    def _getLabels(self, filePath):
        # implement the getLabels method
        Y= []
        for p in os.listdir(filePath):
            print(f'{filePath}{p}')
            load_data = np.load(f'{filePath}{p}') # shape: (288)
            Y.append(load_data)
        Y = np.array(Y)
        Y = torch.from_numpy(Y).type(torch.float32)
        Y = Y.view(-1)
        return Y

        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        pass

    def __init__(self, mode):
        # remember to change the file path according to different experiments
        assert mode in ['train', 'test', 'finetune']
        if mode == 'train':
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
        if mode == 'finetune':
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = self._getFeatures(filePath='./dataset/FT/features/')
            self.labels = self._getLabels(filePath='./dataset/FT/labels/')
        if mode == 'test':
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            self.features = self._getFeatures(filePath='./dataset/SD_test/features/')
            self.labels = self._getLabels(filePath='./dataset/SD_test/labels/')

    def __len__(self):
        # implement the len method
        return self.features.shape[0]
        pass

    def __getitem__(self, idx):
        # implement the getitem method
        return self.features[idx], self.labels[idx]
        pass


if __name__ == '__main__':
    dataset = MIBCI2aDataset('train')  
    print(len(dataset))
