import torch
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm
from prompt import prompts
from datetime import datetime
import os

if __name__ == '__main__':
    device = 'cuda:5'
    # model = 'sdxl-turbo'
    model = 'sd-2-1'

    if model == 'sdxl-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained('/home/guoshipeng/TrainLLMs/SD/models/sdxl-turbo',
                                                         torch_dtype=torch.float16,
                                                         )
    else:
        # 加载模型
        pipe = StableDiffusionPipeline.from_pretrained('/home/guoshipeng/TrainLLMs/SD/models/stable-diffusion-2-1',
                                                       torch_dtype=torch.float16,
                                                       )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    # default num_inference_steps = 50, bigger num_inference_steps will generate better images
    # guidance_scale = 7.5, bigger guidance_scale will generate more similar images to the prompt
    images = pipe(prompt=prompts, num_inference_steps=120, guidance_scale=8.0).images
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"./output/{time}", exist_ok=True)
    for i,image in enumerate(images):
        # 创建目录
        image.save(f"./output/{time}/{i}.png")

