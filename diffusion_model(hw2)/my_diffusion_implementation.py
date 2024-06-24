from typing import Dict, Tuple
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import random
import pdb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(0)

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # GELU activation function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out / 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels



class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()

        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)

        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)
        return x


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()

        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [ResidualConvBlock(in_channels, out_channels), ResidualConvBlock(out_channels, out_channels), nn.MaxPool2d(2)]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim

        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]

        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)

def unorm(x):
    # unity norm. results in range of [0,1]
    # assume x (h,w,3)
    xmax = x.max((0,1))
    xmin = x.min((0,1))
    return(x - xmin)/(xmax - xmin)

def norm_all(store, n_t, n_s):
    # runs unity norm on all timesteps of all samples
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t,s] = unorm(store[t,s])
    return nstore

def norm_torch(x_all):
    # runs unity norm on all timesteps of all samples
    # input is (n_samples, 3,h,w), the torch image format
    x = x_all.cpu().numpy()
    xmax = x.max((2,3))
    xmin = x.min((2,3))
    xmax = np.expand_dims(xmax,(2,3))
    xmin = np.expand_dims(xmin,(2,3))
    nstore = (x - xmin)/(xmax - xmin)
    return torch.from_numpy(nstore)

def gen_tst_context(n_cfeat):
    """
    Generate test context vectors
    """
    vec = torch.tensor([
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0],      # human, non-human, food, spell, side-facing
    [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],  [0,0,0,0,0]]      # human, non-human, food, spell, side-facing
    )
    return len(vec), vec

def plot_grid(x,n_sample,n_rows,save_dir,w):
    # x:(n_sample, 3, h, w)
    ncols = n_sample//n_rows
    grid = make_grid(norm_torch(x), nrow=ncols)  # curiously, nrow is number of columns.. or number of items in the row.
    save_image(grid, save_dir + f"run_image_w{w}.png")
    print('saved image at ' + save_dir + f"run_image_w{w}.png")
    return grid

def plot_sample(x_gen_store,n_sample,nrows,save_dir, fn,  w, save=False):
    ncols = n_sample//nrows
    sx_gen_store = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow

    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
    def animate_diff(i, store):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
        return plots
    ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0])
    plt.close()
    if save:
        ani.save(save_dir + f"{fn}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        print('saved gif at ' + save_dir + f"{fn}_w{w}.gif")
    return ani


class CustomDataset(Dataset):
    def __init__(self, surl, lurl, transform, null_context=False):
        sresponse = requests.get(surl)
        sresponse.raise_for_status()
        self.sprites = np.load(BytesIO(sresponse.content))

        lresponse = requests.get(lurl)
        lresponse.raise_for_status()
        self.slabels = np.load(BytesIO(lresponse.content))

        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape

    # Return the number of images in the dataset
    def __len__(self):
        return len(self.sprites)

    # Get the image and label at a given index
    def __getitem__(self, idx):
        # Return the image and label as a tuple
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape

class ContextUnet(nn.Module):
    def __init__(self, in_channels=3, n_feat=64, n_cfeat=5, height=16):
        super(ContextUnet, self).__init__()
        """
        in_channels : number of input channels
        n_feat : number of intermediate feature maps
        n_cfeat : number of classes
        height: height of the image
        """
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  #assume h == w. must be divisible by 4

        # the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True) # [B, 3, 16, 16] -> [B, 64, 16, 16]

        # the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)        # [B, 64, 16, 16] -> [B, 64, 8, 8]
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # [B, 64, 8, 8] -> [B, 128, 4, 4]

        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU()) # [B, 128, 4, 4] -> [B, 128, 1, 1]

        # embedding layers for the timestep and context labels
        self.timeembed1 = EmbedFC(1, 2*n_feat) # [B, 1] -> [B, 128]
        self.timeembed2 = EmbedFC(1, 1*n_feat) # [B, 1] -> [B, 64]
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat) # [B, 5] -> [B, 128]
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat) # [B, 5] -> [B, 64]

        # the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample
            nn.GroupNorm(8, 2 * n_feat), # normalize
            nn.ReLU(),
        ) # [B, 128, 1, 1] -> [B, 128, 4, 4]
        self.up1 = UnetUp(4 * n_feat, n_feat) # [B, 256, 4, 4] -> [B, 64, 8, 8]
        self.up2 = UnetUp(2 * n_feat, n_feat) # [B, 128, 8, 8] -> [B, 64, 16, 16]

        # the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat), # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        ) # [B, 128, 16, 16] -> [B, 3, 16, 16]

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, 1)            : time step
        c : (batch, n_cfeat)      : context label
        """
        # Initial convolution
        init_conv = self.init_conv(x)

        # Downsampling path
        down1 = self.down1(init_conv)
        down2 = self.down2(down1)

        # Reduce to vector
        hiddenvec = self.to_vec(down2)

        # If context is None, create a zero tensor
        if c is None:
            c = torch.zeros(x.size(0), self.n_cfeat, device=x.device)

        # Embed context and timestep
        cemb1 = self.contextembed1(c)
        temb1 = self.timeembed1(t)
        cemb2 = self.contextembed2(c)
        temb2 = self.timeembed2(t)

        # First upsampling block
        up0 = self.up0(hiddenvec)

        # # Incorporating embeddings in upsampling
        # # Reshape embeddings to match spatial dimensions for addition
        cemb1 = cemb1.view(cemb1.size(0), cemb1.size(1), 1, 1).expand_as(up0)
        temb1 = temb1.view(temb1.size(0), temb1.size(1), 1, 1).expand_as(up0)
        # Subsequent upsampling blocks
        up1 = self.up1(cemb1*up0+temb1, down2)  # Incorporate output of down2
        cemb2 = cemb2.view(cemb2.size(0), cemb2.size(1), 1, 1).expand_as(up1)
        temb2 = temb2.view(temb2.size(0), temb2.size(1), 1, 1).expand_as(up1)
        up2 = self.up2(cemb2*up1+temb2, down1)  # Incorporate output of down1
        # Final convolutional layer to map to the same number of channels as input
        out = self.out(torch.cat([up2, init_conv], dim=1))
        #     ###########################################################
        #     # TODO:                                                   #
        #     # Implement the forward pass of the ContextUnet.          #
        #     # If c is None, replace it with an all-zeros tensor       #
        #     # the output tensor should have the same shape as input x #
        #     ###########################################################
        #     ###########################################################
        #     #                    END OF YOUR CODE                     #
        #     ###########################################################
        return out

def construct_ddpm_noise(timesteps, beta1, beta2, device):
    beta_t = torch.linspace(beta1, beta2, steps=timesteps + 1, device=device)
    alpha_t = 1. - beta_t
    alpha_bar_t = torch.cumprod(alpha_t, dim=0)  # cumprod provides the cumulative product
    #######################################################################
    # TODO:                                                               #
    # Implement the noise schedule for the forward diffusion process.     #
    # Should return 3 tensors: beta_t, alpha_t, and alpha_bar_t           #
    # Output shape: [timesteps + 1]                                       #
    # Read the DDPM paper (sections 2 and 4) for how to                   #
    # construct beta_t, and compute alpha_t and alpha_bar_t               #
    # Reference: https://arxiv.org/pdf/2006.11239.pdf                     #
    #######################################################################
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
    alpha_bar_t[0] = 1
    return beta_t, alpha_t, alpha_bar_t

def perturb_input(x, t, noise, alpha_bar_t):
    ################################################################
    # TODO:                                                        #
    # Implement the perturb_input function to perturb an image     #
    # given the timestep, noise, and schedule alpha_bar_t          #
    # Reference: section 3.2 https://arxiv.org/pdf/2006.11239.pdf  #
    ################################################################
    ################################################################
    #                      END OF YOUR CODE                        #
    ################################################################
    # breakpoint()
    # Retrieve alpha_bar for the specific timestep t
    alpha_bar_t_at_t = alpha_bar_t[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # Compute the square root of alpha_bar_t for timestep t
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t_at_t)

    # Compute the square root of (1 - alpha_bar_t) for timestep t
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - alpha_bar_t_at_t)

    # Perturb the input image
    perturbed_x = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
    return perturbed_x

def loss_no_context(x, model, alpha_bar_t, timesteps):
    # Sample random noise
    epsilon = torch.randn_like(x)

    # Sample a random timestep t
    t = torch.randint(1, timesteps + 1, (x.size(0),), device=x.device)
    # Normalize t for model input
    t_normalized = t.float() / timesteps
    # Perturb the input x
    perturbed_x = perturb_input(x, t, epsilon, alpha_bar_t)

    # Predict the noise from the model
    predicted_noise = model(perturbed_x, t_normalized[:, None])

    # Compute the loss as the MSE between predicted noise and true noise
    loss = torch.mean((predicted_noise - epsilon) ** 2)
    #################################################################
    # TODO:                                                         #
    # Implement the loss function                                   #
    # The code should sample a random noise, a random timestep      #
    # (in this order),                                              #
    # perturb the input, predict the noise from the model,          #
    # and compute the loss as the MSE between predicted noise       #
    # and the true noise                                            #
    # Reference: section 3.4 https://arxiv.org/pdf/2006.11239.pdf   #
    # Hint: the random timestep t should be an integer tensor       #
    # between [1, timesteps] and t should be normalized by          #
    # timesteps before passing to model                             #
    #################################################################
    #################################################################
    #                       END OF YOUR CODE                        #
    #################################################################
    return loss

def denoise_add_noise(x, t, pred_noise, alpha_t, beta_t, alpha_bar_t, z=None):
    # breakpoint()
    if z is None:
        # If z is not None, we use the given z value, which should be sampled from N(0, I)
        z = torch.randn_like(x)
    alpha_t_at_t = alpha_t[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    alpha_bar_t_at_t = alpha_bar_t[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    beta_t_at_t = beta_t[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)

    one_minus_alpha_t = 1. - alpha_t_at_t
    sqrt_alpha_t = torch.sqrt(alpha_t_at_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - alpha_bar_t_at_t)
    sqrt_beta_t = torch.sqrt(beta_t_at_t)

    # Remove the predicted noise and add some noise back to avoid collapse
    output = (1/sqrt_alpha_t)*(x-(one_minus_alpha_t/sqrt_one_minus_alpha_bar_t)*pred_noise)+sqrt_beta_t*z
    ##################################################################
    # TODO:                                                          #
    # Implement the function to denoise the input at a               #
    # specified timestep during the reverse diffusion process        #
    # The code should remove the predicted noise                     #
    # but add some noise back (z) in to avoid collapse               #
    # Reference: Algorithm 2 https://arxiv.org/pdf/2006.11239.pdf    #
    ##################################################################
    ##################################################################
    #                        END OF YOUR CODE                        #
    ##################################################################
    return output

# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(model, n_sample, timesteps, alpha_t, beta_t, alpha_bar_t, device='cpu', save_rate=20):
    # Initialize samples from a normal distribution
    height = model.h
    samples = torch.randn(n_sample, 3, height, height, device=device)
    # Array to store intermediate samples for visualization
    intermediate = []

    for i in reversed(range(1, timesteps + 1)):
        print(f'Sampling timestep {i:3d}', end='\r')

        # Normalize timesteps for conditioning
        t = torch.full((n_sample,), i, dtype=torch.float32, device=device) / timesteps

        # For the last timestep, no additional noise is added
        z = torch.randn_like(samples) if i > 1 else torch.zeros_like(samples)

        # Predict the noise with the model
        eps = model(samples, t[:, None])

        # Denoise and add back a scaled amount of noise
        samples = denoise_add_noise(samples, torch.full((n_sample,), i, device=device), eps, alpha_t, beta_t, alpha_bar_t, z=z)
        ################################################################
        # TODO:                                                        #
        # Implement the DDPM sampling process.                         #
        # Iteratively apply denoising to generate samples.             #
        # Note: for i = 1, don't add back in noise (z is all zeros)    #
        ################################################################
        ################################################################
        #                        END OF YOUR CODE                      #
        ################################################################
        if i % save_rate == 0 or i == 1:
            intermediate.append(samples.cpu().numpy())

    intermediate = np.stack(intermediate, axis=0)
    return samples, intermediate

def loss_with_context(x, c, model, alpha_bar_t, timesteps):
    # Randomly mask out c
    device = x.device
    context_mask = torch.bernoulli(torch.zeros(c.shape[0], device=device) + 0.9)
    c = c * context_mask.unsqueeze(-1)

    # Sample random noise
    epsilon = torch.randn_like(x)

    # Sample a random timestep t
    t = torch.randint(1, timesteps + 1, (x.size(0),), device=device)
    # Normalize t for model input
    t_normalized = t.float() / timesteps
    # Perturb the input x
    perturbed_x = perturb_input(x, t, epsilon, alpha_bar_t)

    # Predict the noise from the model, incorporating context c
    predicted_noise = model(perturbed_x, t_normalized[:, None], c)

    # Compute the loss as the MSE between predicted noise and true noise
    loss = torch.mean((predicted_noise - epsilon) ** 2)
    ###############################################################
    # TODO:                                                       #
    # Implement the loss function                                 #
    # The code should sample a random noise, a random timestep,   #
    # perturb the input, predict the noise from the model,        #
    # and compute the loss as the MSE between predicted noise     #
    # and the true noise                                          #
    # Should be very similar to loss_no_context                   #
    ###############################################################
    ###############################################################
    #                      END OF YOUR CODE                       #
    ###############################################################
    return loss

# sample with context using standard algorithm
@torch.no_grad()
def sample_ddpm_context(model, n_sample, context, timesteps, alpha_t, beta_t, alpha_bar_t, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    height = model.h
    device = context.device
    samples = torch.randn(n_sample, 3, height, height).to(device)
    
    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        t = torch.full((n_sample,), i, dtype=torch.float32, device=device) / timesteps
        z = torch.randn_like(samples) if i > 1 else torch.zeros_like(samples)
        eps = model(samples, t[:, None], context)
        samples = denoise_add_noise(samples, torch.full((n_sample,), i, device=device), eps, alpha_t, beta_t, alpha_bar_t, z=z)

        ###########################################################
        # TODO:                                                   #
        # Implement the DDPM sampling process.                    #
        # Iteratively apply denoising to generate samples.        #
        # Note: for i = 1, don't add back in noise                #
        # Should be very similar to sample_ddpm                   #
        ###########################################################
        ###########################################################
        #                    END OF YOUR CODE                     #
        ###########################################################

        if i % save_rate==0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

