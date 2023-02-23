import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def check_dataset_outputs(dataset: Dataset):
    assert len(dataset) == 32600, 'The dataset should contain 32,600 images.'
    index = np.random.randint(len(dataset))
    image = dataset[index]
    assert  image.shape == torch.Size([3, 64, 64]), 'You must reshape the images to be 64x64'
    assert image.min() >= -1 and image.max() <= 1, 'The images should range between -1 and 1.'
    print('Congrats, your dataset implementation passed all the tests')
    
    
def check_discriminator(discriminator: torch.nn.Module):
    images = torch.randn(1, 3, 64, 64)
    score = discriminator(images)
    assert score.shape == torch.Size([1, 1, 1, 1]), 'The discriminator output should be a single score.'
    print('Congrats, your discriminator implementation passed all the tests')
 

def check_generator(generator: torch.nn.Module, latent_dim: int):
    latent_vector = torch.randn(1, latent_dim, 1, 1)
    image = generator(latent_vector)
    assert image.shape == torch.Size([1, 3, 64, 64]), 'The generator should output a 64x64x3 images.'
    print('Congrats, your generator implementation passed all the tests')
    