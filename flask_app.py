import torch
import os
import openai
from stable_diffusion_videos import StableDiffusionWalkPipeline
from stable_diffusion_videos import make_video_pyav
import gradio as gr
from pathlib import Path
from flask import Flask, send_file, request, jsonify, g, render_template, url_for, redirect
from stable_diffusion_videos import generate_images

import base64
from moviepy.editor import *

import PIL
from PIL import Image
from io import BytesIO
import pdb 
import shutil

import re
import json
import numpy as np
# from diffusers import StableDiffusionImg2ImgPipeline
import time
from typing import List, Optional, Tuple, Union
import random

import subprocess

num_generations = 0
app = Flask(__name__)

num_intervals = 0


pipeline = StableDiffusionWalkPipeline.from_pretrained(
    "RunwayML/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    revision="fp16",
    safety_checker=None,
    ).to("cuda")

music = './static/audio/ny_short.wav'
video = './static/audio/ny_short.mp4'

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def prepare_latents(pipeline, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if isinstance(generator, list):
        init_latents = [
            pipeline.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
        ]
        init_latents = torch.cat(init_latents, dim=0)
    else:
        init_latents = pipeline.vae.encode(image).latent_dist.sample(generator)

    # init_latents = pipeline.vae.config.scaling_factor * init_latents

    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        # expand init_latents for batch_size
        deprecation_message = (
            f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
            " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
            " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
            " your script to pass as many initial images as text prompts to suppress this warning."
        )
        deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # get latents
    init_latents = pipeline.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents

    return latents


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
generated_prompts = set()


@app.route('/generate_brainstorm_img', methods=["POST"])
def generate_brainstorm_img():
    print("entered brainstorm")
    global num_generations

    prompt = request.json["start_prompt"]

    
    if request.json["test_seed"]:
        seed = request.json["test_seed"]
    else:
        seed =  random.randint(0,1000)
        
    prompt_and_seed = f"{prompt}_{seed}"

    if request.json["test_seed"]:
    
        curr_seed = seed
        prompt_components = prompt.split(",")
        # main_prompt = prompt_components[0]
        # sub_prompt = prompt_compo
        
        print(f"GENERATING {prompt} with seed {curr_seed}")
        generation_name = f"{prompt}_{curr_seed}"

        if prompt_and_seed not in generated_prompts: 
    
            img=generate_images(prompt,curr_seed)
            img.images[0].save(f"./static/generations/{num_generations-1}_{generation_name}.jpg" , 'JPEG', quality=70)
            generated_prompts.add(f"{prompt}_{seed}")
            num_generations+=1

        if len(prompt_components) >1:

            random.shuffle(prompt_components)
            alternative1 = ",".join(prompt_components )
                
            print(f"GENERATING {alternative1} with seed {curr_seed}")
            alternative1_filename = f"{alternative1}_{curr_seed}"

            if alternative1_filename not in generated_prompts: 

                img=generate_images(alternative1,curr_seed)
                img.images[0].save(f"./static/generations/{num_generations-1}_{alternative1_filename}.jpg" , 'JPEG', quality=70)
                num_generations+=1


            random.shuffle(prompt_components)
            alternative2 = ",".join(prompt_components)

            print(f"GENERATING {alternative2} with seed {curr_seed}")
            alternative2_filename = f"{alternative2}_{curr_seed}"

            if alternative2_filename not in generated_prompts:
                img=generate_images(alternative2,curr_seed)
                img.images[0].save(f"./static/generations/{num_generations-1}_{alternative2_filename}.jpg" , 'JPEG', quality=70)
                num_generations+=1




    else:
        for i in range(3):
            num_generations += 1
            curr_seed = i+seed
            print(f"GENERATING {prompt} with seed {curr_seed}")
            generation_name = f"{prompt}_{curr_seed}"

        
            img=generate_images(prompt,curr_seed)
            img.images[0].save(f"./static/generations/{num_generations-1}_{generation_name}.jpg" , 'JPEG', quality=70)

    return "completed"

@app.route('/receive_img_blob', methods=["POST"])
def receive_img_blob():
    global num_generations

    # file =request.files['image']
    # if file:
    #     file.save("./savedimage.jpg")
    #     print("image saved")
    #this will route to a new 
    interval = "1"
    # interval = request.form["interval_num"]
    prompt = request.form["start_prompt"]

    if request.form["test_seed"]:
        seed = request.form["test_seed"]
    else:
        seed =  random.randint(0,5000)

    prompt2 = request.form["end_prompt"]

    # try:
    #     imgChecked = request.form["imgChecked"]
    # except:
    imgChecked = False

    num_generations += 1
    if prompt:
        print(f"GENERATING {prompt} with seed {seed}")
        generation_name = f"{prompt}_{seed}"

        num_generations += 1
        img=generate_images(prompt,seed)
        img.images[0].save(f"./static/generations/{num_generations-1}_{generation_name}.jpg" , 'JPEG', quality=70)

    if prompt2:
        print(f"GENERATING {prompt2} with seed {seed}")
        generation2_name = f"{prompt2}_{seed}"

        img2 = generate_images(prompt2,seed)
        
        img2.images[0].save(f"./static/generations/{num_generations}_{generation2_name}.jpg" , 'JPEG', quality=70)

    # if imgChecked:
    #     img_output = img2img(prompt,seed)

        
    #     img_output.save(f"./static/generations/{num_generations}_{generation_name}.jpg" , 'JPEG', quality=70)
    #     img_output.save(f"./static/previews/{interval}_start.jpg" , 'JPEG', quality=70)
    # else:
        # pattern = r'(.*)(static)(.*)'
        # output_file_path = re.match(pattern, output_file_path).groups()[-1]
    
    

    # img.images[0].save(f"./static/previews/{interval}_start.jpg" , 'JPEG', quality=70)

    # image_data = base64.b64decode(img_data)
    # with open("./image_prompt.png","wb") as fh:
    #         fh.write(image_data)
    # response = jsonify({})
    # response.headers.add("Access-Control-Allow-Origin", "*")

    return redirect(url_for('audio_test'))

# # @app.route('/img2img', methods=["POST"])
# def img2img(prompt,seed):
#     img_url = "./savedimage.jpg"
#     init_image = Image.open(img_url).convert("RGB")
#     image = preprocess(init_image)
    
#     num_inference_steps = 300
#     strength = 0.8
#     num_images_per_prompt = 1

#     pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
#     pipe = pipe.to(device)
#     init_image = Image.open(img_url).convert("RGB")
#     init_image = init_image.resize((512, 512))
#     image = preprocess(init_image)

#     prompt_embeds = pipe._encode_prompt(
#             prompt,
#             device,
#             num_images_per_prompt,
#             False,
#             ""
#         )

#     generator = torch.Generator(device=device).manual_seed(int(seed) ) 

#     device2 = torch.device(f"cuda:0")
#         # 5. set timesteps
#     pipe.scheduler.set_timesteps(num_inference_steps, device=device)
#     timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps, strength, device2)
#     latent_timestep = timesteps[:1].repeat(1 * num_images_per_prompt)
#     # 6. Prepare latent variables
#     latents = pipe.prepare_latents(
#         image, latent_timestep, 1, num_images_per_prompt, prompt_embeds.dtype, device2
#     )

#     images = pipe(prompt=prompt, image=init_image, strength=0.8, guidance_scale=7.5, generator = generator, num_inference_steps = num_inference_steps).images
#     return images[0]
    


@app.route('/call_generate_end', methods=["POST"])
def call_generate_end():
    json_data = request.json["json_data"]
    print(json_data)
    
    #call json.loads twice because first call returns string
    intervals = json.loads(json.loads(json_data))
    print("entered")
    
    with open('./static/example.json', 'w') as f:
        json.dump(intervals, f)
        f.close()

    interval = request.json["interval_num"]
    prompt = request.json["prompt"]
    seed = request.json["seed"]
    print(seed)
    # seed = request.json["seed"]
    # output_file_path = request.json["outputFilePath"]
    print(f"GENERATING {prompt} with seed {seed}")
    # pattern = r'(.*)(static)(.*)'
    # output_file_path = re.match(pattern, output_file_path).groups()[-1]
    img=generate_images(prompt,seed)
    generation_name = f"{prompt}_{seed}"
    # timestamp = 
    img.images[0].save(f"./static/generations/{generation_name}.jpg" , 'JPEG', quality=70)
    img.images[0].save(f"./static/previews/{interval}_end.jpg" , 'JPEG', quality=70)
    
    return "completed"

@app.route('/call_generate', methods=["POST"])
def call_generate():
    global num_generations
    
    json_data = request.json["json_data"]

    
    #call json.loads twice because first call returns string

    intervals = json.loads(json.loads(json_data))
    print("entered")
    
    with open('./static/example.json', 'w') as f:
        json.dump(intervals, f)
        f.close()

    # interval = request.json["interval_num"]
    interval = 1
    prompt = request.json["start_prompt"]
    seed =  random.randint(0,1000)
    print(seed)
    # seed = request.json["seed"]
    # output_file_path = request.json["outputFilePath"]
    print(f"GENERATING {prompt} with seed {seed}")
    # pattern = r'(.*)(static)(.*)'
    # output_file_path = re.match(pattern, output_file_path).groups()[-1]
    img=generate_images(prompt,seed)
    generation_name = f"{prompt}_{seed}"

    num_generations += 1

    img.images[0].save(f"./static/generations/{num_generations}_{generation_name}.jpg" , 'JPEG', quality=70)
    img.images[0].save(f"./static/previews/{interval}_start.jpg" , 'JPEG', quality=70)
  
    
    return "Completed" 




def find_video_in_folder():
    audioSrc = request.json["audioSrc"]
    print(audioSrc)

    return "Completed"


@app.route("/create_placeholder", methods=["POST"])
def create_placeholder():
    audioSrc = request.json["audioSrc"]
    print(audioSrc)
    return "Completed"


@app.route("/replace_preview", methods=["POST"])
def replace_preview():
    draggedPath = request.json["draggedPath"]
    print(draggedPath)
    interval_num = request.json["intervalNum"]
    start_or_end = request.json["start_or_end"]
    print(interval_num)

    incoming_img = draggedPath.replace("%20", " ")
    regex_pattern = r'(.*)(static.*)(.*)'
    incoming_img = "./" + re.match(regex_pattern, incoming_img).group(2)
    print(incoming_img[:-1], f"./static/previews/{interval_num}_{start_or_end}.jpg")
    shutil.copy(incoming_img[:-1],f"./static/previews/{interval_num}_{start_or_end}.jpg")
    # shutil.move(incoming_img[:-1], f"./static/previews/{interval_num}_{start_or_end}.jpg")
  
    return "Completed"


@app.route("/save_regions", methods=["POST"])
def save_regions():
    json_data = request.json["json_data"]

    #call json.loads twice because first call returns string
    intervals = json.loads(json.loads(json_data))

    with open('./static/example.json', 'w') as f:
        json.dump(intervals, f)
        
        f.close()
    return "Completed"

import string

@app.route("/change_audio", methods=["POST"])
def change_audio():

    global music
    global video

    audio_file = request.json["audio_file"] 
    video_file = request.json["video_file"] 

    music = f'./static/audio/{audio_file}'
    video = f'./static/audio/{video_file}'

    return {"audio_filename":music}


@app.route("/download_audio", methods=["POST"])
def download_audio():

    global music
    global video
     
    # printing lowercase
    letters = string.ascii_lowercase
    filename = ''.join(random.choice(letters) for i in range(10)) 

    audio_file = request.json["audio_file"] 
    print(audio_file)
    music_start = request.json["music_start"] 
    print(music_start)

    music_length = request.json["music_length"] 
    print(music_length)


    subprocess.Popen(f"youtube-dl -f bestaudio  --extract-audio --audio-format mp3 --audio-quality 0 -o 'static/audio/{filename}.%(ext)s' {audio_file}", shell=True, stdout=subprocess.PIPE).stdout.read()
    time.sleep(10)
    subprocess.Popen(f"ffmpeg -i 'static/audio/{filename}.mp3' -acodec pcm_u8 -ar 22050 static/audio/sample.wav", shell=True, stdout=subprocess.PIPE).stdout.read()
    time.sleep(10)
    # ffmpeg -ss 23 -i sample.wav -t 10 -c:a copy static/audio/trimmed_sample.wav


    subprocess.Popen(f"ffmpeg -y -ss {music_start} -i static/audio/sample.wav -t {music_length} -c:a copy ./static/audio/{filename}.wav", shell=True, stdout=subprocess.PIPE).stdout.read()
    # ffmpeg -ss 23 -i sample.wav -t 10 -c:v copy -c:a aac -strict experimental -shortest music.mp4

    # subprocess.Popen(f"ffmpeg -ss {music_start} -t {music_length} -f lavfi -i color=c=blue:s=100x720 -i sample.wav -shortest -fflags +shortest static/audio/music.mp4", shell=True, stdout=subprocess.PIPE).stdout.read()

    music = f'./static/audio/{filename}.mp3'
    video = f'./static/audio/music.mp4'

    return {"audio_filename":f'./static/audio/{filename}.mp3'}

@app.route("/generate_interval", methods=["POST"])
def generate_interval():
    global pipeline
    global num_intervals
    json_data = request.json["json_data"]

    torch.cuda.empty_cache()

    #call json.loads twice because first call returns string
    intervals = json.loads(json.loads(json_data))

    with open('./static/example.json', 'w') as f:
        json.dump(intervals, f)
        
        f.close()

    current_interval_times = [float(request.json["current_interval_start"]),float(request.json["current_interval_end"])]
    curr_interval_number = int(request.json["interval_num"])
    curr_interval_prompt = request.json["start_note"]
    curr_interval_seed = request.json["start_seed"]
    ending_prompt = request.json["end_note"]
    ending_seed = request.json["end_seed"]
    start = request.json["current_interval_start"].replace(".","sec")
    end = request.json["current_interval_end"].replace(".","sec")

    current_interval = request.json["json_data"]

    json_data = request.json["json_data"]

    intervals = json.loads(json.loads(json_data))


    fps = 24
    num_interpolation_steps = [int((b-a) * fps) for a, b in zip(current_interval_times, current_interval_times[1:])]
    
    underscored_curr_interval_prompt = curr_interval_prompt.replace(" ","_")
    underscored_ending_prompt = ending_prompt.replace(" ","_")
    output_name = f"{start}_{end}_{underscored_curr_interval_prompt}_{curr_interval_seed}->{underscored_ending_prompt}_{ending_seed}".replace(",","-")
    print(curr_interval_prompt, ending_prompt)
    video_path = pipeline.walk(
        prompts= [ curr_interval_prompt, ending_prompt ],
        seeds=[ int(curr_interval_seed), int(ending_seed)],
        num_interpolation_steps= num_interpolation_steps,
        height=512,                            # use multiples of 64
        width=512,   
                                # use multiples of 64

        audio_filepath=f'{music}',    # Use your own file
        audio_start_sec=current_interval_times[0],       # Start second of the provided audio
        fps=fps,                               # important to set yourself based on the num_interpolation_steps you defined
        batch_size=4,                          # increase until you go out of memory.
        output_dir='./static/intervals',                 # Where images will be saved
        name=output_name, 
        num_inference_steps=100                            # Subdir of output dir. will be timestamp by default
    )

    shutil.move(f"./static/intervals/{output_name}", f"./static/output/")


    return "completed"

@app.route("/delete_track", methods=["POST"])
def delete_track():
    print("entered")
    path_to_delete = request.json["path_to_delete"]
    regex_pattern = r'(.*)(static.*)\/(.*mp4)'
    relative_path =  "./" + re.match(regex_pattern, path_to_delete).group(2)
    relative_path = relative_path.replace("%3E",">")
    # print(relative_path)
    shutil.rmtree( relative_path)
    return "completed"

@app.route("/delete_stitched_video", methods=["POST"])
def delete_stitched_video():
    os.unlink( "./static/output/stitched_output.mp4")
    return "completed"

@app.route("/delete_file", methods=["POST"])
def delete_file():
    print("entered")
    path_to_delete = request.json["path_to_delete"]
    regex_pattern = r'(.*)(static.*)(.*)'
    relative_path =  "./" + re.match(regex_pattern, path_to_delete).group(2)
    relative_path = relative_path.replace("%20"," ")
    # print(relative_path)
    os.unlink( relative_path[:-1])
    return "completed"

from random import sample

@app.route("/brainstorm", methods=["POST"])
def brainstorm():
    color_transitions = ["blue hour", "light leak", "bloom pass", "gaussian blur", "sepia","color inversion", "neon","saturated", "desaturated", "warm","cool", "pastel", "cyberpunk","pop art", "vintage","film photography", "glitch","vignette"]
    time_transitions = ["sunset", "sunrise", "mosaic", "kaleidoscope", "strobe", "retro", "dreamy", "cartoon", "golden hour", "dramatic lighting", "sunny", "cloudy", "timelapse", "light tunnels","bokeh", "soft lighting","cinematic lighting", "lens flare", "slow motion"]
    # action_transitions = ["motion blur"]
    angle_transitions = ["wide angle", "close-up", "birds eye view","midshot", "fisheye","high angle", "dutch angle", "worm's eye view", "straight angle", "low angle", "vortex", "upside down", "side profile","tilted frame","isometric","medium long" ]

    
    color_trio = sample(color_transitions,5)
    angle_trio = sample(angle_transitions,5)
    time_trio = sample(time_transitions,5)

    return {
        "color_trio": f"{color_trio[0]},{color_trio[1]}, {color_trio[2]}, {color_trio[3]}, {color_trio[4]}",
        "angle_trio":f"{angle_trio[0]}, {angle_trio[1]}, {angle_trio[2]}, {angle_trio[3]}, {angle_trio[4]}",
        "time_trio":f"{time_trio[0]}, {time_trio[1]}, {time_trio[2]}, {time_trio[3]}, {time_trio[4]}"
    }

@app.route("/brainstorm_gpt", methods=["POST"])
def brainstorm_gpt():
    openai.api_key = request.json["openai_api_key"]
    
    goal = request.json["goal"]

    prompt = f"In 5 words or less, describe an image that symbolizes these lyrics '{goal}':"
    print(prompt)

    regex_pattern = r'(.*)\*\*(.*)\*\*(.*)'

    response1 = openai.Completion.create(
        model="text-davinci-003",
        prompt= prompt,
        temperature=0,
        max_tokens=72,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1

        )
    response2 = openai.Completion.create(
        model="text-davinci-003",
        prompt= prompt,
        temperature=1,
        max_tokens=72,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1

        )
    response3 = openai.Completion.create(
        model="text-davinci-003",
        prompt= prompt,
        temperature=2,
        max_tokens=72,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1

        )

    response1 = response1["choices"][0].text.strip()
    response2 = response2["choices"][0].text.strip()
    response3 = response3["choices"][0].text.strip()


    #response1 = openai.ChatCompletion.create(model="gpt-4", max_tokens=100, messages=[{"role": "user", "content": prompt}])["choices"][0]["message"]["content"]
    #response2 = openai.ChatCompletion.create(model="gpt-4", max_tokens=100, messages=[{"role": "user", "content": prompt}])["choices"][0]["message"]["content"]
    #response3 = openai.ChatCompletion.create(model="gpt-4", max_tokens=100, messages=[{"role": "user", "content": prompt}])["choices"][0]["message"]["content"]


    

    return {
        "subject1":response1,
        "subject2": response2,
        "subject3":response3,
       
    }

@app.route("/stitch_videos", methods=["POST"])
def stitch_videos():
    video_paths = request.json["video_paths"].split(',')
    print(video_paths)

    regex_pattern = r'(.*)(static.*)(.*)'
    relative_paths = []
    for path in video_paths:
        relative_path =  (re.match(regex_pattern, path).group(2))
        relative_paths.append(relative_path)
    print(relative_paths)
    video_clips = [VideoFileClip(path.replace("%3E",">")) for path in relative_paths]
    print(video_clips)
    final = concatenate_videoclips(video_clips)
    final.write_videofile("./static/output/stitched_output.mp4", audio_codec='aac', rewrite_audio=False, audio=False)
    return "completed"



@app.route("/generate_video", methods=["POST"])
def generate_video():
    
    shutil.rmtree('./static/output/',ignore_errors=True)

    # start_prompt = request.json["start_prompt"]
    # end_prompt = request.json["end_prompt"]
    # seed = int(request.json["seed"])
    json_data = request.json["json_data"]
    
    #call json.loads twice because first call returns string
    intervals = json.loads(json.loads(json_data))
    print("entered")
    
    with open('./static/example.json', 'w') as f:
        json.dump(intervals, f)
        print("dumped")
        f.close()

    # interpolation_requests = []
    
    # fps = 20
    # interval_counter= 0
    # for curr_interval, next_interval in zip(intervals, intervals[1:]):
    #     interpolation_request = {}


    #     current_prompt = curr_interval["data"]["note"]
    #     end_prompt = next_interval["data"]["note"]

    #     interpolation_request_name = f"interval_{interval_counter}"
    #     interpolation_request["name"] = interpolation_request_name
    #     interval_counter += 1

    #     curr_audio_interval = [curr_interval["start"], next_interval["start"]]

    #     num_interpolation_steps = [(b-a) * fps for a, b in zip(curr_audio_interval, curr_audio_interval[1:])]
    #     interpolation_request["paired_prompts"] = [current_prompt, end_prompt]
    #     interpolation_request["audio_interval"] = curr_audio_interval
    #     interpolation_request["num_interpolation_steps"] = [int(abs(min((b-a) * fps, 30))) for a, b in zip(curr_audio_interval, curr_audio_interval[1:])]

    #     interpolation_requests.append(interpolation_request)
    

    # interpolation_request = {}
    # last_interval = next_interval
    
    # current_prompt = last_interval["data"]["note"]
    # end_prompt = "matte black texture"

    # interpolation_request_name = f"interval_{interval_counter}"
    # interpolation_request["name"] = interpolation_request_name

    # curr_audio_interval = [last_interval["start"], last_interval["end"]]
    # num_interpolation_steps = [(b-a) * fps for a, b in zip(curr_audio_interval, curr_audio_interval[1:])]
    # interpolation_request["paired_prompts"] = [current_prompt, end_prompt]
    # interpolation_request["audio_interval"] = curr_audio_interval
    # interpolation_request["num_interpolation_steps"] = [abs((b-a) * fps) for a, b in zip(curr_audio_interval, curr_audio_interval[1:])]

    # # if interpolation_request["num_interpolation_steps"][0] > 29:
    # #     a = curr_audio_interval[0]
    # #     b =curr_audio_interval[1]
    # #     fps = int(interpolation_request["num_interpolation_steps"][0] / (b-a))

    # interpolation_requests.append(interpolation_request)
    
    # print(interpolation_requests)

    # for interpolation_request in interpolation_requests:
    #     print(interpolation_request['name'])

    #     video_path = pipeline.walk(
    #     prompts=interpolation_request["paired_prompts"],
    #     seeds=[32,32],
    #     num_interpolation_steps=interpolation_request["num_interpolation_steps"],
    #     height=512,                            # use multiples of 64
    #     width=256,   
    #                             # use multiples of 64
    #     audio_filepath='./static/audio/clairdelune.wav',    # Use your own file
    #     audio_start_sec=interpolation_request["audio_interval"][0],       # Start second of the provided audio
    #     fps=fps,                               # important to set yourself based on the num_interpolation_steps you defined
    #     batch_size=1,                          # increase until you go out of memory.
    #     output_dir='./static/',                 # Where images will be saved
    #     name=interpolation_request['name'],                             # Subdir of output dir. will be timestamp by default
    #     )
    #     shutil.move(f"./static/{interpolation_request['name']}", f"./static/output/")

    return "completed"


"""
Helper function for fn serve_img
Src: https://stackoverflow.com/questions/7877282/how-to-send-image-generated-by-pil-to-browser
"""
def serve_pil_image(pil_img):
    img_io =  BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route("/gather_generations", methods=["GET"])
def gather_generations():

    generations_gallery = os.listdir("./static/generations/")
    print(generations_gallery)
    return jsonify(generations_gallery)



def generate_images(prompt,seed):
    global pipeline
    
    torch.cuda.empty_cache()
    

    img = pipeline(prompt,num_inference_steps =50, height=512, width=512, batch_size=1, num_batches=1,seed=int(seed))

    return img


"""
Landing page
"""
@app.route("/hello_world")
def hello_world():
    # frames = sorted(os.listdir("./static/output/output_000000"))
    # frames = ["/output/output_000000/" + frame_filename for frame_filename in frames]
    # print(frames)
    generations = sorted(os.listdir("./static/generations/"))
    generations = ["/generations/" + generation_filename for generation_filename in generations]
    return render_template("videoapp.html", img='generations/generated_img', generations=generations, video = 'dreams/20230202-183918/20230202-183918.mp4')

# @app.route("/start")
# def start_video_creation():
#     torch.cuda.empty_cache()

#     return render_template("start.html")




@app.route("/audio_with_history")
def audio_with_history():
    previous_intervals_path = "./static/example.json"

    # with open(json_file_path, 'r') as j:
    #     previous_intervals = json.loads(j.read())

    torch.cuda.empty_cache()

    interval_folder_names = next(os.walk("./static/output/"))[1]
    
    output_files = next(os.walk("./static/output/"))[2]
    print(interval_folder_names)
    frames = {}
    for folder_name in interval_folder_names:
        frames[folder_name] = {}
        frames[folder_name]["interval_frames"] = sorted(os.listdir("./static/output/" + f"{folder_name}/"))
        frames[folder_name]["interval_frames"] = [f"output/{folder_name}/{frame}" for frame in frames[folder_name]["interval_frames"] ]
        frames[folder_name]["interval_video"] = "./static/output/" + f"{folder_name}/{folder_name}" +".mp4"
    print(frames)
    
    video = "./static/output.mp4"
    # video = [f"./static/output/" + output_file for output_file in output_files if output_file[-4:] == ".mp4"][0]
 
    # interval_folder_frames = [os.listdir("./static/output/"+folder_name) for folder_name in interval_folder_names  ]
    # print(interval_folder_frames)
    # frames = ["/output/interval0/" + frame_filename for frame_filename in interval_folder_frames]
    # print(frames)
    # generations = sorted(os.listdir("./static/generations/"))
    # generations = ["/generations/" + generation_filename for generation_filename in generations]
    
    return render_template("audio_test.html", frames=frames, output_video = video, previous_intervals_path = previous_intervals_path)

from glob import glob
def folder_files_by_ext(folder_path, extension):
    result = [y for x in os.walk(folder_path) for y in glob(os.path.join(x[0], extension))]
    return result


def find_approx_create_time(interval_folder_name):
    # print(interval_folder_name)
    # print(next(os.walk(interval_folder_name))[0])
    internal_folder = next(os.walk("./static/output/" + interval_folder_name))[0]
    print(internal_folder)
    files = next(os.walk(internal_folder))[0]
    interval_video = folder_files_by_ext(files,"*.mp4")
    print(interval_video)
    print(os.path.getmtime(interval_video[0]))
    return os.path.getmtime(interval_video[0])
 

from collections import OrderedDict
@app.route("/", methods=["GET","POST"])
def audio_test():
    global music
    global video

    frames = OrderedDict()
    # video =f"./static/audio/{music}.mp4"
    videos = []

    if os.path.isdir('./static/output'):
        interval_folder_names = sorted(next(os.walk("./static/output/"))[1])
        # key=lambda folder_name: find_approx_create_time(folder_name), reverse=False)
        # interval_folder_names.sort(key=lambda folder_name: find_approx_create_time(folder_name), reverse=True)

        
        
        output_files = next(os.walk("./static/output/"))[2]
      
        
        for folder_name in interval_folder_names:
            frames[folder_name] = OrderedDict()
            folder_path = "./static/output/" + f"{folder_name}/"
            folder_contents = sorted(os.listdir("./static/output/" + f"{folder_name}/"))
            frames[folder_name]["interval_frames"] = sorted(folder_files_by_ext(folder_path,"*.png"))
            frames[folder_name]["interval_frames"] = sorted([frame[9:] for frame in frames[folder_name]["interval_frames"] ])
            frames[folder_name]["interval_video"] = folder_files_by_ext(folder_path,"*.mp4")[0]

        videos = folder_files_by_ext("./static/output/","*.mp4")
        if "./static/output/stitched_output.mp4" in videos:
            video = "./static/output/stitched_output.mp4"
        
        # videos = [f"./static/output/" + output_file for output_file in output_files if output_file[-4:] == ".mp4"]
        print(videos)
        # videos = videos.sort(key=lambda x: os.path.getmtime("./static/generations/" + x), reverse=True)
    
        # interval_folder_frames = [os.listdir("./static/output/"+folder_name) for folder_name in interval_folder_names  ]
        # print(interval_folder_frames)
        # frames = ["/output/interval0/" + frame_filename for frame_filename in interval_folder_frames]

    generations = sorted(os.listdir("./static/generations/"))

    generations.sort(key=lambda x: os.path.getmtime("./static/generations/" + x), reverse=True)

    preview_dict = OrderedDict()
    for filename in os.listdir("./static/previews"):
        interval_preview_number = filename[0]
        preview_dict[interval_preview_number] = filename


    generations = [("generations/" + generation_filename, generation_filename.replace(" ", "_").replace(",","-")) for generation_filename in generations]
    return render_template("audio_test.html", frames=frames, videos = videos, output_video = video, generations = generations)

@app.route("/register_music_change", methods=["POST"])
def register_music_change():
    global music
    global video
    music = request.json["music_choice"]
    print(music)
    
    return {"music":music}


def run():

    app.run(host='0.0.0.0', threaded=True, debug=True, port = 7860)
run()