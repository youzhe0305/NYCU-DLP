import torch
import torch.nn as nn
from .modules.transform import Encoder, Decoder, Codebook

__all__ = [
    "VQGAN"
]

class VQGAN(nn.Module):
    def __init__(self, configs):
        super(VQGAN, self).__init__()
        
        dim = configs['latent_dim']
        self.encoder = Encoder(configs)
        self.decoder = Decoder(configs)
        self.codebook = Codebook(configs)
        self.quant_conv = nn.Conv2d(dim, dim, 1) 
        self.post_quant_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quantized_encoded_images = self.quant_conv(encoded_images) # 把encode出的特徵用1*1捲積，捲成適合轉成embedding的
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        quantized_codebook_mapping = self.post_quant_conv(codebook_mapping) # 把他變成適合decode
        decoded_images = self.decoder(quantized_codebook_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, x):
        encoded_images = self.encoder(x)
        quantized_encoded_images = self.quant_conv(encoded_images)
        #b,c,h,w   
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        #b,c,h,w    indices: (b*h*w,1)
        return codebook_mapping, codebook_indices, q_loss # 得到離散的latent code

    def decode(self, z):
        quantized_codebook_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(quantized_codebook_mapping)
        return decoded_images # 拿離散的latent code做decode

    def calculate_lambda(self, nll_loss, g_loss):
        # nll_loss: E[-logD(G(z))] for Generative model training
        # g_loss: 裡面的VQVAE的loss，也就是||x-x^|| + q_loss(離散化的loss)
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0] # 自動計算兩種loss對於layer最後一層的grad造成多少
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0] # 計算圖會在加減乘除時被建立，算了grad就會被釋放，後面還要用，所以這裡retain_graph

        λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4) 
        # lambda用來平衡loss，乘在nll loss上，當nll_grad大時，代表距離正解越遠，讓nll loss的權重更大，更被重視，梯度也會變更大?
        # 可能可以加快收斂速度?
        λ = torch.clamp(λ, 0, 1e4).detach()
        #discriminator weight=0.8
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.): # 調整discriminator的loss，total loss = Gen Loss + disc_factor * Disc loss
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path), strict=True)
        print("Loaded Checkpoint for VQGAN....")

