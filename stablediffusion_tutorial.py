# %% Basic Stable Diffusion pipeline
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image

# %% Deconstruct the Stable Diffusion pipeline
# (1) from pretrained - vae, tokenizer, text_encoder, unet, scheduler
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import UniPCMultistepScheduler

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
text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
) # {'input_ids', 'attention_mask'}. text_input.input_ids = torch.Size([1, 77])
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0] # torch.Size([1, 77, 768])

max_length = text_input.input_ids.shape[-1] # tokenizer.model_max_length (=77)
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt") # {'input_ids', 'attention_mask'}. text_input.input_ids = torch.Size([1, 77])
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] # torch.Size([1, 77, 768])
text_embeddings = torch.cat([uncond_embeddings, text_embeddings]) # torch.Size([2, 77, 768]) # forward pass 두 번 하지 않도록

# text_embeddings: 입력 문장 임베딩 (prompt embedding)
# uncond_embeddings: ""이면 null이라 uncond이고, 만약 uncond_input에 뭔가 적으면 negative prompt embedding인 것.

## Create random noise
# (5) Create Gaussian noise
# vae model이 3개의 down-sampling layers 가지고 있으므로 8로 나눔
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device) # torch.Size([1, 4, 64, 64])

## Denoise the image
# (6) Denoise the image

##### Algorithm
# latents ~ N(0, I) (5번 과정)
# for t=999, 959, ..., 40 do
# -> latents batch를 두 배로 복사.
# -> 복사된 latents를 unet에 태워서 noise_pred 뽑음. 이때 [uncond_embeddings, text_embeddings]의
#    형태로 encoder_hidden_states가 들어가므로 이 순서대로 text 정보가 latents 계산에 반영됨.
# -> CFG를 위해 noise_pred를 noise_pred_uncond, noise_pred_text로 나누고 CFG 식을 통해 
#    두 predicted noise가 하나의 noise_pred로 계산됨.
# -> noise_pred, timestep t, 기존 latents를 이용해 (t-1)의 latents를 계산.
# end for
# return latents
#####

# 위 과정을 scheduler.timesteps에 대해 반복하여 z_0까지 계산.

# cf)
# 1. 즉 positive prompt와 negative(null일 경우엔 unconditional) prompt에 대한 노이즈 예측이 각각 이루어짐.
# 2. 노이즈 예측은 각각 이루어지지만 각 t에 대해 unet의 input으로 들어가는 latents는 두 predicted noise를
#    결합한 noise_pred로부터 만들어짐.

from tqdm.auto import tqdm

latents = latents * scheduler.init_noise_sigma # scheduler.init_noise_sigma = 1
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
# %% Text-guided image-inpainting

