# torch.cuda.empty_cache() # memory 초기화

# %% Simple Inference
from diffusers import DDPMPipeline

image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256") # trained on a dataset of celebrities images
image_pipe.to("cuda")

images = image_pipe().images
images[0]

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
############################################################################################

# %% DiffusionPipeline
# UNet2DConditionModel & PNDMScheduler
from diffusers import DiffusionPipeline
import torch
import os
from IPython.display import display

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda")
image = pipeline("An image of a squirrel in Picasso style").images[0]
image

# print(os.getcwd())
# os.chdir("./workspace") # Lab
# print(os.getcwd())
# image.save("/images/image_of_squirrel_painting.png")

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
##################################
## Effective and Efficient diffusion
## Speed
from diffusers import DiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id)

prompt = "portrait photo of a old warrior chief"
pipeline = pipeline.to("cuda")

generator = torch.Generator("cuda").manual_seed(0)

image = pipeline(prompt, generator=generator).images[0]
image
# %%
# float32 -> float16 (speed up) (10s -> 3s)
# 항상 float16 사용하는 거 추천! 지금까지 output에서 성능 저하 발견 못함
import torch

pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator).images[0]
image
# %%
#  You can find which schedulers are compatible with the current model in the DiffusionPipeline 
pipeline.scheduler.compatibles
# %%
# StableDiffusion model은 PNDMScheduler를 Default로 사용함 (50 inferfence steps)
# DPMSolverMultistepScheduler : 20 or 25 inference steps -> 1s
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator, num_inference_steps=20).images[0]
image
# %%
## Memory
def get_inputs(batch_size=1):
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20

    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}

from PIL import Image

def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

images = pipeline(**get_inputs(batch_size=4)).images
image_grid(images)
# %%
# attention slicing : 순차적으로 수행하면 메모리 아낄 수 있음 -> OOM 방지 가능
pipeline.enable_attention_slicing()
images = pipeline(**get_inputs(batch_size=8)).images
image_grid(images, rows=2, cols=4)
# %%
## Quality - 퀄리티 향상에 집중
# Better checkpoints
# try loading the latest autodecoder from Stability AI into the pipeline
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
pipeline.vae = vae
images = pipeline(**get_inputs(batch_size=8)).images
image_grid(images, rows=2, cols=4)

# %%
# Better prompt engineering
# improve the prompt to include color and higher quality details
prompt += ", tribal panther make up, blue on red, side profile, looking away, serious eyes"
prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"

images = pipeline(**get_inputs(batch_size=8)).images
image_grid(images, rows=2, cols=4)
# %%
prompts = [
    "portrait photo of the oldest warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a young warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
]

generator = [torch.Generator("cuda").manual_seed(1) for _ in range(len(prompts))]
images = pipeline(prompt=prompts, generator=generator, num_inference_steps=25).images
image_grid(images)

# %%
#####################################################################################
### Understanding piplines, models and schedulers
## Deconstruct a basic pipeline
from diffusers import DDPMPipeline

ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")
image = ddpm(num_inference_steps=25).images[0]
image


# %% 
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






















