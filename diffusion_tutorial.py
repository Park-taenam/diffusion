# torch.cuda.empty_cache() # memory 초기화

# %% Simple Inference
from diffusers import DDPMPipeline

image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to("cuda") # trained on a dataset of celebrities images
images = image_pipe(num_inference_steps=25).images
images[0]

# %% Understanding piplines, models and schedulers
# 1. Load the model and scheduler
from diffusers import DDPMScheduler, UNet2DModel

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")

# 2. Set the number of timesteps to run the denoising process for
scheduler.set_timesteps(50)
# print(scheduler.timesteps)

# 3. Create some random noise with the same shape as the desired output:
import torch

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")

# 4. At each timestep, the model does a UNet2DModel.forward() pass and returns the noisy residual. 
# The scheduler’s step() method takes the noisy residual, timestep, and input and it predicts the image at the previous timestep. 
# This output becomes the next input to the model in the denoising loop, and it’ll repeat until it reaches the end of the timesteps array.
input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample

# 5. The last step is to convert the denoised output into an image
from PIL import Image
import numpy as np

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
image

# %%Models
from diffusers import UNet2DModel

repo_id = "google/ddpm-church-256" # trained on church images
model = UNet2DModel.from_pretrained(repo_id)

model_random = UNet2DModel(**model.config) # 이전 꺼와 동일한 config로 랜덤하게 초기화된 모델 생성
model_random.save_pretrained("my_model") # 생성한 모델 save
# !ls my_model

model_random = UNet2DModel.from_pretrained("my_model")
# %%Inference
import torch

torch.manual_seed(0)

noisy_sample = torch.randn(
    1, model.config.in_channels, model.config.sample_size, model.config.sample_size
)
print(noisy_sample.shape)

# 모델은 노이즈가 약간 덜한 이미지 or 노이즈가 약간 덜한 이미지와 입력 이미지의 차이 or 다른 것을 예측
# 이 경우 모델은 잔여 노이즈(노이즈가 약간 덜한 이미지와 입력 이미지 사이의 차이)를 예측
with torch.no_grad():
    noisy_residual = model(sample=noisy_sample, timestep=2).sample

print(noisy_residual.shape) # torch.Size([1, 3, 256, 256])

# %% Schedulers - DDPM
from diffusers import DDPMScheduler

repo_id = "google/ddpm-church-256" 
scheduler = DDPMScheduler.from_config(repo_id)
print(scheduler.config)

# scheduler.save_config("my_scheduler")
# new_scheduler = DDPMScheduler.from_config("my_scheduler")
from IPython.display import display
less_noisy_sample = scheduler.step(
    model_output=noisy_residual, timestep=2, sample=noisy_sample
).prev_sample
print(less_noisy_sample.shape) # torch.Size([1, 3, 256, 256])

import PIL.Image
import numpy as np

def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)

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
  if (i + 1) % 50 == 0:
      display_sample(sample, i + 1)

# %% Scheduler - DDIM
from diffusers import DDIMScheduler

scheduler = DDIMScheduler.from_config(repo_id)
scheduler.set_timesteps(num_inference_steps=50)

import tqdm

sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
  # 1. predict noise residual
  with torch.no_grad():
      residual = model(sample, t).sample

  # 2. compute previous image and set x_t -> x_t-1
  sample = scheduler.step(residual, t, sample).prev_sample

  # 3. optionally look at image
  if (i + 1) % 10 == 0:
      display_sample(sample, i + 1)
# %%