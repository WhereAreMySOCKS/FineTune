import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# 设置模型的路径
model_id = "/home/guoshipeng/TrainLLMs/SD/models/stable-diffusion-2-1"

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda:4")

# 初始化一个标志变量，用于控制循环
continue_prompting = True

# 使用while循环来提示用户输入，直到输入非空字符串
while continue_prompting:
    prompt = input("请输入你想要生成的图像描述，或者输入'exit'来退出程序：")
    # 检查用户是否输入了'exit'来退出程序
    if prompt.lower() == 'exit':
        print("退出程序。")
        continue_prompting = False  # 设置标志变量以退出循环
    elif prompt.strip():  # 使用strip()来移除字符串首尾的空白字符
        image = pipe(prompt).images[0]
        # 保存图像，文件名可以根据输入的描述动态生成
        image.save(f"./output/{prompt.replace(' ', '_')}.png")
        print(f"图像已保存为：./output/{prompt.replace(' ', '_')}.png")
    else:
        # 用户输入了空字符串，提示他们重新输入
        print("输入不能为空，请重新输入。")

# 循环结束后，如果需要，可以添加其他代码