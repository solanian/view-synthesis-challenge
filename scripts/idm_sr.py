import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
# pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipeline = pipeline.to("cuda")



img_path = "rendered_result/006_02.png"
low_res_img = Image.open(img_path)
# low_res_img.save("render_output/006_02_resized.png")
low_res_img = low_res_img.resize((750, 1024))
low_res_img.save("rendered_result/006_02_resized.png")

# prompt = ""

# upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
# upscaled_image.save("upsampled.png")