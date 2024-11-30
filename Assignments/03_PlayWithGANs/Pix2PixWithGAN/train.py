import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from model import Discriminator, Generator
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def main():
    config = {
        'train_list': 'train_list.txt',
        'val_list': 'val_list.txt',
        'batch_size': 50,
        'step_size': 200,
        'num_epochs': 200
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = FacadesDataset(list_file=config['train_list'])
    valid_dataset = FacadesDataset(list_file=config['val_list'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    scheduler_D = StepLR(optimizer_D, step_size=config['step_size'], gamma=0.5)
    scheduler_G = StepLR(optimizer_G, step_size=config['step_size'], gamma=0.5)


    for epoch in range(config['num_epochs']):
        for i, (image, semantic) in enumerate(train_dataloader):
            image, semantic = image.to(device), semantic.to(device)

            optimizer_D.zero_grad()
            fake_semantic = generator(image)
            real_labels = torch.ones(image.size(0), 1).to(device)
            fake_labels = torch.zeros(image.size(0), 1).to(device)

            real_loss = criterion_GAN(discriminator(image, semantic), real_labels)
            fake_loss = criterion_GAN(discriminator(image, fake_semantic.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            g_loss_GAN = criterion_GAN(discriminator(image, fake_semantic), real_labels)
            g_loss_L1 = criterion_L1(fake_semantic, semantic)
            g_loss = g_loss_GAN + 20 * g_loss_L1
            g_loss.backward()
            optimizer_G.step()

            print(f"Epoch [{epoch}/{config['num_epochs']}], Step [{i}/{len(train_dataloader)}], "
                  f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, G Loss GAN: {g_loss_GAN.item()}, G Loss L1: {g_loss_L1.item()}")

            if epoch % 5 == 0 and i == 0:
                save_images(image, semantic, fake_semantic, 'train_results', epoch)

        scheduler_D.step()
        scheduler_G.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                for i, (image, semantic) in enumerate(valid_dataloader):
                    image, semantic = image.to(device), semantic.to(device)
                    fake_semantic = generator(image)
                    save_images(image, semantic, fake_semantic, 'val_results', epoch, num_images=4)


if __name__ == '__main__':
    main()