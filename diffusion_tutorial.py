# %%
## DiffusionPipeline
# UNet2DConditionModel & PNDMScheduler
from diffusers import DiffusionPipeline
import torch
import os

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda")
image = pipeline("An image of a squirrel in Picasso style").images[0]

print(os.getcwd())
os.chdir("./workspace/images/")
print(os.getcwd())
image.save("image_of_squirrel_painting.png")
# torch.cuda.empty_cache() # memory 초기화

# %%
# Swapping schedulers (PNDMScheduler(default) -> EulerDiscreteScheduler)
from diffusers import EulerDiscreteScheduler
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

# %%
# Models
from diffusers import UNet2DModel
import torch

repo_id = "google/ddpm-cat-256"
model = UNet2DModel.from_pretrained(repo_id)
# print(model.config)

torch.manual_seed(0)
noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
print(noisy_sample.shape)

# for inference
with torch.no_grad():
    noisy_residual = model(sample=noisy_sample, timestep=2).sample

# %%
# Schedulers
from diffusers import DDPMScheduler

repo_id = "google/ddpm-cat-256"
scheduler = DDPMScheduler.from_config(repo_id)
# print(scheduler)

# %%
less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample
print(less_noisy_sample.shape)

# %%
import PIL.Image
import numpy as np

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0,2,3,1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)
# %%
model.to("cuda")
noisy_sample = noisy_sample.to("cuda")

import tqdm
sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = model(sample, t).sample
    
    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i+1)%50==0:
        display_sample(sample, i+1)

# %%

