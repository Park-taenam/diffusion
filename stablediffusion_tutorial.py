# print(os.getcwd())
# os.chdir("./workspace") # Lab
# print(os.getcwd())
# image.save("/images/image_of_squirrel_painting.png")

# %% Simple Inference(Basic Stable Diffusion pipeline)
from diffusers import DiffusionPipeline, StableDiffusionPipeline, EulerDiscreteScheduler
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config) # Swapping schedulers
pipeline.to("cuda")
prompt = "a photo of an astronaut riding a horse on mars"
prompt = "An image of a squirrel in Picasso style"
image = pipeline(prompt).images[0]
image

# %% Effective and Efficient diffusion - Speed
# 1. Using GPU
from diffusers import DiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id)

prompt = "portrait photo of a old warrior chief"
pipeline = pipeline.to("cuda")

generator = torch.Generator("cuda").manual_seed(0)

image = pipeline(prompt, generator=generator).images[0]
image

# 2. float32 -> float16 (speed up) (10s -> 3s)
# 항상 float16 사용하는 거 추천! 지금까지 output에서 성능 저하 발견 못함
import torch

pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator).images[0]
image

# 3. to reduce the number of inference steps
# compatibles method: DiffusionPipeline에서 현재 모델과 호환되는 스케줄러를 찾을 수 있음
print(pipeline.scheduler.compatibles)

# 4. Scheduler
# StableDiffusion model은 PNDMScheduler를 Default로 사용함 (50 inferfence steps)
# DPMSolverMultistepScheduler : 20 or 25 inference steps -> 1s
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

generator = torch.Generator("cuda").manual_seed(0)
image = pipeline(prompt, generator=generator, num_inference_steps=20).images[0]
image


# %% Effective and Efficient diffusion - Memory
# 1. batch size 조정
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

# 2. attention slicing : 순차적으로 수행하면 메모리 아낄 수 있음 -> OOM 방지 가능
pipeline.enable_attention_slicing()
images = pipeline(**get_inputs(batch_size=8)).images
image_grid(images, rows=2, cols=4)

# %% Effective and Efficient diffusion - Quality
# 1. Better checkpoints
# try loading the latest autodecoder from Stability AI into the pipeline
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
pipeline.vae = vae
images = pipeline(**get_inputs(batch_size=8)).images
image_grid(images, rows=2, cols=4)

# 2. Better prompt engineering
# improve the prompt to include color and higher quality details
prompt += ", tribal panther make up, blue on red, side profile, looking away, serious eyes"
prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"

images = pipeline(**get_inputs(batch_size=8)).images
image_grid(images, rows=2, cols=4)

prompts = [
    "portrait photo of the oldest warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a old warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
    "portrait photo of a young warrior chief, tribal panther make up, blue on red, side profile, looking away, serious eyes 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta",
]

generator = [torch.Generator("cuda").manual_seed(1) for _ in range(len(prompts))]
images = pipeline(prompt=prompts, generator=generator, num_inference_steps=25).images
image_grid(images)

# %% Deconstruct the Stable Diffusion pipeline
# (1) from pretrained - vae, tokenizer, text_encoder, unet, scheduler
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler

# VAE, UNet, text encoder models
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae") # 83,653,863 params
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer") # tokenizer이므로 학습 파라미터 없음
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder") # 123,060,480 params
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet") # 859,520,964 params
scheduler = UniPCMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler") # scheduler이므로 학습 파라미터 없음

# (2) to cuda
torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

## Create text embeddings
# (3) Prepare prompt and parameters
prompt = ["a photograph of an astronaut riding a horse"]
# prompt = ["Impressionism, beautiful and colorful tree"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise
batch_size = len(prompt)

# (4) Create text embeddings
# 만약 하나의 prompt에 대해 여러 이미지를 생성하고 싶으면 .to(), .repeat(), .view() 등 추가해야 함. 
# -> pipeline_stable_diffusion.py 참고
# text_embeddings: 입력 문장 임베딩 (prompt embedding)
# uncond_embeddings: ""이면 null이라 uncond이고, 만약 uncond_input에 뭔가 적으면 negative prompt embedding인 것.
text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
) # {'input_ids', 'attention_mask'}. text_input.input_ids = torch.Size([1, 77])
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0] # torch.Size([1, 77, 768])

#  unconditional text embeddings
max_length = text_input.input_ids.shape[-1] # tokenizer.model_max_length (=77)
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt") # {'input_ids', 'attention_mask'}. text_input.input_ids = torch.Size([1, 77])
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] # torch.Size([1, 77, 768])

text_embeddings = torch.cat([uncond_embeddings, text_embeddings]) # torch.Size([2, 77, 768]) # forward pass 두 번 하지 않도록

## Create random noise
# (5) Create Gaussian noise (a starting point for the diffusion process)
# vae model이 3개의 down-sampling layers 가지고 있으므로 8로 나눔
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device) # torch.Size([1, 4, 64, 64])

## Denoise the image
# (6) Denoise the image
latents = latents * scheduler.init_noise_sigma # scheduler.init_noise_sigma = 1

from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)
for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
 
    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

## Decode the image
# (7) Decode the image
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

# (8) Show the image
from diffusers.image_processor import VaeImageProcessor
"""
                                                         # 해당하는 VaeImageProcessor.postprocess() 내부 함수
image = (image / 2 + 0.5).clamp(0, 1)                    # denormalize()
image = image.detach().cpu().permute(0, 2, 3, 1).numpy() # pt_to_numpy()
images = (image * 255).round().astype("uint8")           # numpy_to_pil()
pil_images = [Image.fromarray(img) for img in images]    # numpy_to_pil()
pil_images[0]
"""
img_processor = VaeImageProcessor()
final_image = img_processor.postprocess(image)
final_image[0]

# %%