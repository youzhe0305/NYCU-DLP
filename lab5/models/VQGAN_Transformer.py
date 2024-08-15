import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F

#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors'] # mask_token對應到哪個codebook的embedding，這裡設最後一個是mask
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param']) # 輸出 n_token, code_book_vector，為其屬於某個vector的分數

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path)['transformer_parameter'])

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, q_loss = self.vqgan.encode(x)
        return codebook_indices.view(codebook_mapping.shape[0],-1) # (b*h*w) -> (b,h*w)
        raise Exception('TODO2 step1-1!')
        return None
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        # 要讓gamma(0) = 1, gamma(1) = 0
        if mode == "linear":
            return lambda ratio : 1 - ratio # 輸入ratio，輸出1-ratio
        elif mode == "cosine":
            return lambda ratio : np.cos(ratio * np.pi/2) # 在0~pi/2漸小
        elif mode == "square":
            return lambda ratio : 1 - ratio ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x): # for training
        
        z_indices = self.encode_to_z(x) #ground truth # (b,h*w)
        mask = torch.ones(z_indices.shape).type(torch.LongTensor).to(z_indices.device) * self.mask_token_id # 整張都是mask token的圖
        position_to_mask = torch.randint(0,2, z_indices.shape).type(torch.bool).to(z_indices.device)
        masked_z_indices = position_to_mask * mask + (~position_to_mask) * z_indices
        
        logits = self.transformer(masked_z_indices)  #transformer predict the probability of tokens
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_b, ratio, mask_num): # for inference

        mask = torch.ones(z_indices.shape).type(torch.LongTensor).to(z_indices.device) * self.mask_token_id # 整張都是mask token的圖
        z_indices = mask_b * mask + (~mask_b) * z_indices
        ramaining_mask_num = torch.floor(mask_num * self.gamma(ratio))

        logits = self.transformer(z_indices)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        prob = F.softmax(logits, dim=-1)
        
        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(prob, dim=-1) # 產出機率跟對應的類
        z_indices_predict = mask_b * z_indices_predict + (~mask_b) * z_indices

        #predicted probabilities add temperature annealing gumbel noise as confidence
        g =  torch.distributions.gumbel.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices.device)  # gumbel noise
        temperature = self.choice_temperature * (1 - self.gamma(ratio))
        confidence = z_indices_predict_prob + temperature * g # (1,256)
        confidence[(~mask_b)] = 1e9
        sorted_confidence = torch.sort(confidence, dim=-1)[0] # 由小到大,(batch,n_token)
        threshold = sorted_confidence[:,ramaining_mask_num.long()] # 0~ramaining_mask_num-1, 機率最小的幾個
        ret_mask = confidence < threshold
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens

        return z_indices_predict, ret_mask
        
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
