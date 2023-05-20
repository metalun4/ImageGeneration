import torch

from train import train
from modules import UNet, Diffusion

from torch.nn import MSELoss
from torch.optim import AdamW
from torchvision import transforms
from torchvision import datasets

from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    EPOCHS = 500
    BATCH_SIZE = 128

    IMG_SIZE = 32

    diffusion = Diffusion(img_size=IMG_SIZE, device=device)

    transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE + .25 * IMG_SIZE)),  # image_size + 1/4 * image_size
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR100(root='data/', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(dataset.classes)
    print(num_classes)

    model = UNet(image_size=IMG_SIZE, num_classes=num_classes, device=device).to(device)
    model_name = "DDPM_Conditional_Model"

    LR = 3e-4
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = MSELoss()

    train(epochs=EPOCHS,
          dataloader=dataloader,
          model=model,
          optimizer=optimizer,
          loss_fn=loss_fn,
          diffusion=diffusion,
          model_name=model_name,
          num_classes=num_classes,
          device=device)
