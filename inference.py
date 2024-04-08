import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_base = "CompVis/stable-diffusion-v1-4"
lora_model_path = "/ocean/projects/cis230017p/zliug/m3c_summer/m3c-diffusers/examples/text_to_image/sd-China-baseline"

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(lora_model_path)
pipe.to("cuda")

generator = [torch.Generator(device="cuda").manual_seed(i) for i in [10551, 8288, 9678, 22969]]

test_prompt ="people with prosthetics at home"

for i in range(4):
    generator_i = generator[i]
    # use the weights from the fully finetuned LoRA model
    image = pipe(test_prompt, generator=generator_i, num_inference_steps=50, guidance_scale=4.5, cross_attention_kwargs={"scale":1.0}).images[0]    
    image.save(lora_model_path+"/test_image"+str(i)+".png")