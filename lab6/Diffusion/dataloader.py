import torch
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import torchvision.transforms.v2 as transforms

class Diffusion_Dataset(Dataset):
    def __init__(self, root=None, mode='train'):
        super().__init__()

        print('Loading Dataset...')        
        self.root = root
        self.mode = mode
        
        assert root != None, 'Dataset Load Error - root error'
        assert mode in ['train', 'test', 'new_test'], 'Dataset Load Error - mode error'

        with open(f'{mode}.json', 'r') as f:
            self.loaded_json = json.load(f) # 讀json轉dictionary
        if mode == 'train':
            self.img_fnames = list(self.loaded_json.keys())
            self.labels = list(self.loaded_json.values())
            
        elif mode in ['test', 'new_test']:
            self.labels = self.loaded_json

        with open(f'objects.json') as f:
            self.loaded_json = json.load(f)
        self.encoded_labels = torch.zeros(len(self.labels), len(self.loaded_json))
        for idx, label in enumerate(self.labels):
            for object in label:
                self.encoded_labels[idx][ self.loaded_json[object] ] = 1
    
        print('Dataset Done')
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        
        if self.mode == 'train':
            img = Image.open(f'{self.root}/{self.img_fnames[index]}').convert('RGB')
            img = img_transform(img)
            label = self.encoded_labels[index]
            return img, label
        else:
            label = self.encoded_labels[index]
            return label   

def img_transform(img):            
    transform = transforms.Compose([
        transforms.Resize((64,64)), # input classifier is 64*64, use 64*64 as input, output
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True), # PIL to Tensor, range: [0,1]
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #　because input for classifier need normalization, I directly modify origin image, make it learn normalized image
    ])
    return transform(img)

if __name__ == '__main__':
    
    dataset = Diffusion_Dataset(root='../iclevr', mode='train')
    print(len(dataset))
    print(dataset[0][0].shape) # image
    print(dataset[0][1]) # label
    
    dataset = Diffusion_Dataset(root='../iclevr', mode='test')
    print(len(dataset))
    print(dataset[0]) # label
