from pathlib import Path

import gradio as gr

from stable_diffusion_videos import generate_images

class Interface:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.interface_images = gr.Interface(
            self.fn_images,
            inputs=[
                gr.Textbox("blueberry spaghetti", lines=1, label='Prompt 1'),
                gr.Textbox("strawberry spaghetti", lines=1, label='Prompt 2'),

#                 gr.Slider(1, 4, 2, step=1, label='Batch size'),
#                 gr.Slider(1, 16, 1, step=1, label='# Batches'),
#                 gr.Slider(10, 
#                 100, 50, step=1, label='# Inference Steps'),
                #it would be nice to show a grid where you could choose
#                 gr.Slider(5.0, 15.0, 7.5, step=0.5, label='Guidance Scale'),
#                 gr.Slider(64, 512, 64, step=64, label='Height'),
#                 gr.Slider(64, 512, 64, step=64, label='Width'),
#                 gr.Checkbox(False, label='Upsample'),
#                 gr.Textbox("./images", label='Output directory to save results to'),
                # gr.Checkbox(False, label='Push results to Hugging Face Hub'),
                # gr.Textbox("", label='Hugging Face Repo ID to push images to'),
            ],
            outputs= [gr.Gallery().style(grid=[4], height="auto")]
        )

        self.interface_videos = gr.Interface(
            self.fn_videos,
            inputs=[
                gr.Textbox("blueberry spaghetti\nstrawberry spaghetti", lines=2, label='Prompts, separated by new line'),
                gr.Textbox("42\n1337", lines=2, label='Seeds, separated by new line'),
                gr.Slider(3, 1000, 5, step=1, label='# Interpolation Steps between prompts'),
                gr.Slider(3, 60, 5, step=1, label='Output Video FPS'),
                gr.Slider(1, 24, 3, step=1, label='Batch size'),
                gr.Slider(10, 100, 50, step=1, label='# Inference Steps'),
                gr.Slider(5.0, 15.0, 7.5, step=0.5, label='Guidance Scale'),
                gr.Slider(512, 1024, 512, step=64, label='Height'),
                gr.Slider(512, 1024, 512, step=64, label='Width'),
                gr.Checkbox(False, label='Upsample'),
                gr.Textbox("./dreams", label='Output directory to save results to'),
            ],
            outputs=gr.Video(),
        )
        self.interface = gr.TabbedInterface(
            [self.interface_images],
            ['Videos!'],
        )

    def fn_videos(
        self,
        prompts,
        seeds,
        num_interpolation_steps,
        fps,
        batch_size,
        num_inference_steps,
        guidance_scale,
        height,
        width,
        upsample,
        output_dir,
    ):
        prompts = [x.strip() for x in prompts.split('\n') if x.strip()]
        seeds = [int(x.strip()) for x in seeds.split('\n') if x.strip()]

        return self.pipeline.walk(
            prompts=prompts,
            seeds=seeds,
            num_interpolation_steps=num_interpolation_steps,
            fps=fps,
            height=height,
            width=width,
            output_dir=output_dir,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            upsample=upsample,
            batch_size=batch_size
        )

    def fn_images(
        self,
        prompt1,
        prompt2,
        batch_size=1,
        num_batches=1,
        num_inference_steps=50,
        guidance_scale=8,
        height=64,
        width=64,
        upsample=False,
        output_dir="./dreams",
        repo_id=None,
        push_to_hub=False,
    ):
#         prompts = [x.strip() for x in prompts.split('\n') if x.strip()]
        prompts = [prompt1, prompt2]
        all_image_filepaths = []
        
        guidance_scales = [10]
        
        for prompt in prompts:
            print(prompt)
            for guidance_scale in guidance_scales:
                print(guidance_scale)

                image_filepaths = generate_images(
                    self.pipeline,
                    prompt,
                    batch_size=1,
                    num_batches=1,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    output_dir=output_dir,
                    image_file_ext='.jpg',
                    upsample=upsample,
                    height=height,
                    width=width,
                    push_to_hub=push_to_hub,
                    repo_id=repo_id,
                    create_pr=False,
                )
                all_image_filepaths.extend(image_filepaths)
                
        print(all_image_filepaths)
        print(image_filepaths)
            
        return [(x, Path(x).stem) for x in sorted(all_image_filepaths)]

    
    
    def launch(self, *args, **kwargs):
        self.interface.launch(*args, **kwargs)