import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from prompt import prompts
from datetime import datetime
import os

if __name__ == '__main__':
    # 设置模型的路径
    model_id = "/home/guoshipeng/TrainLLMs/SD/models/stable-diffusion-2-1"
    # 加载模型
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda:2")
    images = pipe(prompts).images
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"./output/{time}", exist_ok=True)

    for i,image in enumerate(images):
        # 创建目录
        image.save(f"./output/{time}/{i}.png")

