import os
import os.path as osp
import torch
from typer import Typer

from diffusers.pipelines import AutoPipelineForText2Image
from pipeline import TurboPipeline


# pipe = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

pipe.to("cuda")

# image = pipe(prompt=prompt, num_inference_steps=1,
#              guidance_scale=0.0, generator=generator).images[0]

# image.save("official.png")
# del pipe

work_dirs = 'work_dirs/official-fig4-c-seed42-cfg2'
# os.makedirs('work_dirs', exist_ok=True)

my_pipe = TurboPipeline.build()
# prompt = "A cinematic shot of a baby cat wearing an intricate italian priest robe."
# prompt = "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
# prompt = 'A brain riding a rocketship heading towards the moon.'
prompt = 'A blue colored dog.'
# prompt = 'A bald eagle made of chocolate powder, mango, and whipped cream'

for steps in [1, 2, 3, 4, 8, 25]:

    generator = torch.Generator()
    generator.manual_seed(42)
    # generator.manual_seed(23333)
    image = pipe(prompt=prompt, num_inference_steps=steps,
                 guidance_scale=2, generator=generator).images[0]

    os.makedirs(work_dirs, exist_ok=True)
    image.save(osp.join(work_dirs, f'{steps}.png'))
