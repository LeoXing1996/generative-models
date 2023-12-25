import os
import os.path as osp
import torch
from typer import Typer

from pipeline import TurboPipeline


# pipe = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
# pipe.to("cuda")

# image = pipe(prompt=prompt, num_inference_steps=1,
#              guidance_scale=0.0, generator=generator).images[0]

# image.save("official.png")
# del pipe

os.makedirs('work_dirs', exist_ok=True)

my_pipe = TurboPipeline.build()
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

# for steps in [1, 4, 8, 25]:
for steps in [3]:

    generator = torch.Generator()
    # generator.manual_seed(42)
    generator.manual_seed(2333)
    output = my_pipe(prompt=prompt, num_inference_steps=steps,
                     guidance_scale=0.0, generator=generator,
                     save_pred_x0=True)
    image = output.images[0]
    pred_x0_list = output.pred_x0_list

    save_dir = osp.join('work_dirs', f'{steps}')
    os.makedirs(save_dir, exist_ok=True)
    image.save(osp.join(save_dir, 'output.png'))

    timesteps = my_pipe.scheduler.timesteps

    for idx, pred_x0 in enumerate(pred_x0_list):
        t = timesteps[idx]
        pred_x0[0].save(osp.join(save_dir, f'pred_x0_{t}.png'))
