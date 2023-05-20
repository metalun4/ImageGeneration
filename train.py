import logging
from pathlib import Path

from tqdm import tqdm
from copy import deepcopy
from numpy.random import random
from torch.utils.tensorboard import SummaryWriter

from utils import *
from modules import EMA

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(epochs, dataloader, model, optimizer, loss_fn, diffusion, model_name, num_classes, device):
    setup_logging(model_name)
    logger = SummaryWriter(str(Path(f'runs/{model_name}/')))
    length = len(dataloader)
    ema = EMA(0.995)
    ema_model = deepcopy(model).eval().requires_grad_(False)

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = loss_fn(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * length + i)

        if epoch % 10 == 0:
            labels = torch.arange(num_classes).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, Path(f'results/{model_name}/{epoch}.jpg'))
            save_images(ema_sampled_images, Path(f'results/{model_name}/{epoch}_ema.jpg'))
            torch.save(model.state_dict(), Path(f'models/{model_name}/{epoch}_ckpt.pt'))
            torch.save(ema_model.state_dict(), Path(f'models/{model_name}/{epoch}_ema_ckpt.pt'))
            torch.save(optimizer.state_dict(), Path(f'models/{model_name}/{epoch}_optim.pt'))
