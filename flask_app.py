# Standard library imports
import os
import re
import json
import time
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union
from io import BytesIO
from glob import glob
from collections import OrderedDict
import random

# Third-party library imports
import torch
import openai
import base64
import numpy as np
from flask import Flask, send_file, request, jsonify, g, render_template, url_for, redirect
from moviepy.editor import *
from PIL import Image

# Project-specific imports
from stable_diffusion_videos import StableDiffusionWalkPipeline, make_video_pyav, generate_images
from random import sample
from openai import OpenAI

# Counter for the number of generations
num_generations = 0

# Counter for the number of intervals
num_intervals = 0 

generated_prompts = set()
# Flask application instance
app = Flask(__name__)

# Stable Diffusion Walk pipeline configuration
pipeline = StableDiffusionWalkPipeline.from_pretrained(
    "RunwayML/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    revision="fp16",
    safety_checker=None,
    ).to("cuda")

# Background music for videos
music = 'clairdelune.wav'


@app.route('/generate_brainstorm_img', methods=["POST"])
def generate_brainstorm_img():
    """
    Generates an image to help interactively preview the generated video. Returns variations with simple logic (shuffling of prompt by comma-delimited phrases). 
    To-do: change the simple logic to a GPT-chain.

    Parameters (of the payload):
    - Prompt
    - Seed

    Returns:
    - Does not return a payload. Generates an image in the backend, saves it

    Example:
    POST /generate_brainstorm_img
    
    """
    global num_generations

    prompt = request.json["start_prompt"]
    test_seed = request.json["test_seed"]

    if test_seed:
        seed = test_seed
    else:
        seed = random.randint(0, 1000)

    prompt_and_seed = f"{prompt}_{seed}"

    if test_seed:
        curr_seed = seed
        prompt_components = prompt.split(",")

        # Generate image for the original prompt
        generate_and_save_image(prompt, curr_seed, prompt_and_seed)

        if len(prompt_components) > 1:
            random.shuffle(prompt_components)

            # Generate images for shuffled prompt components
            for alternative in prompt_components:
                alternative_name = f"{alternative}_{curr_seed}"
                generate_and_save_image(alternative, curr_seed, alternative_name)

    else:
        for i in range(3):
            num_generations += 1
            curr_seed = i + seed
            generation_name = f"{prompt}_{curr_seed}"

            # Generate images for multiple seeds
            generate_and_save_image(prompt, curr_seed, generation_name)

    return ""

def generate_and_save_image(prompt, curr_seed, filename):
    """
    Function that calls the generation pipeline.

    Parameters:
    - Prompt
    - Seed
    - Filename to save as (concatenation of prompt and seed)
    
    """
    global num_generations

    print(f"GENERATING {prompt} with seed {curr_seed}")

    # Check if prompt and seed have been generated before, only generate if they have not been regenerated
    if f"{prompt}_{curr_seed}" not in generated_prompts:
        img = generate_images(prompt, curr_seed)
        img.images[0].save(f"./static/generations/{num_generations-1}_{filename}.jpg", 'JPEG', quality=70)
        generated_prompts.add(f"{prompt}_{curr_seed}")
        num_generations += 1

@app.route('/get_variation', methods=["POST"])
def get_variation():
    """
    GPT chain to get variations of prompts. Permutes the original prompt as per the GPT prompt below. Then generates the image and saves it.
    Logic in varyImage(element) on the Javascript side refreshes the history on the frontend.

    Parameters (of the payload):
    - Prompt
    - OpenAI Key
    """
    prompt = request.json["prompt"]
    openai_key = request.json["openai_api_key"]
    client = OpenAI(api_key=openai_key)

    variation = client.chat.completions.create(
      model="gpt-4-turbo-preview",
      messages=[
        {
          "role": "user",
          "content": "Create a permutation of this prompt.  A permutation is a shuffling of the words around which keeps the general meaning. The words don't have to be the exact same, but the meaning should be similar.\n\nFor example, a permutation of \"A grand ballroom bathed in golden light, adorned with sparkling chandeliers, elaborate decorations\" could be \"An opulent ballroom heavy with spotlights and decor, grand chandelier at the top, wide, sparkling glasses\"\n\nReturn only the permutation."
        },
        {
          "role": "user",
          "content": f"Write one for {prompt}'.\n"
        },

      ],
      temperature=1,
      max_tokens=128,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    ).choices[0].message.content
    variation_sub = variation.replace(" ", "_")
    
    generate_and_save_image(variation, 16, f"{variation_sub}_16")
    return ""

@app.route('/call_generate', methods=["POST"])
def call_generate():
    """
    Utilized in regenerateFrame (which is deprecated on the UI). Generates an image and also saves a preview.
    
    Parameters (of the payload):
    - JSON data representing the intervals. This saves them in the back end to keep state.


    """
    global num_generations
    
    json_data = request.json["json_data"]

    intervals = json.loads(json.loads(json_data))
    
    with open('./static/interval_data.json', 'w') as f:
        json.dump(intervals, f)
        f.close()

    interval = 1
    prompt = request.json["start_prompt"]
    seed =  random.randint(0,1000)

    print(f"GENERATING {prompt} with seed {seed}")

    img=generate_images(prompt,seed)
    generation_name = f"{prompt}_{seed}"

    num_generations += 1

    img.images[0].save(f"./static/generations/{num_generations}_{generation_name}.jpg" , 'JPEG', quality=70)
    img.images[0].save(f"./static/previews/{interval}_start.jpg" , 'JPEG', quality=70)
  
    return 

@app.route("/replace_preview", methods=["POST"])
def replace_preview():
    """
    File is copied over to the previews directory such that we can keep state.

    Parameters (of the payload):
    - draggedPath (image source that was dragged and dropped)
    - intervalNum (interval in the waveform)
    - start_or_end (whether that is a start or end preview)
    
    """
    draggedPath = request.json["draggedPath"]
    interval_num = request.json["intervalNum"]
    start_or_end = request.json["start_or_end"]
    incoming_img = draggedPath.replace("%20", " ")
    regex_pattern = r'(.*)(static.*)(.*)'
    incoming_img = "./" + re.match(regex_pattern, incoming_img).group(2)
    shutil.copy(incoming_img[:-1],f"./static/previews/{interval_num}_{start_or_end}.jpg")
  
    return  {}

# Function saves the regions of the waveform
@app.route("/save_regions", methods=["POST"])
def save_regions():
    """
    Saves regions into the right local file (./static/interval_data.json)
    Parameters (of the payload):
    - json_data (represents intervals)
    """

    json_data = request.json["json_data"]
    intervals = json.loads(json.loads(json_data))
    with open('./static/interval_data.json', 'w') as f:
        json.dump(intervals, f)
        f.close()
    return {}
    
@app.route("/generate_interval", methods=["POST"])
def generate_interval():
    """
    Core function, abstracts over Stable Diffusion Walk Pipeline. Reads from the interval_data.json that collects state information.
    Parses it into interval num, start_seed (seed of start preview image), end_seed (seed of end preview image), start_note (start prompt), end_note (end prompt), and current_interval_start (start time offset in music file) and current_interval_end (ending timestamp in music file)
    Audioreactive flag toggles the smooth flag in the generate_interval function.
    fps is set to be 24. 
    Number of interpolation steps is calculated as a function of interval duration and fps.
    Batch size can be toggled depending on your compute.

    Parameters (of the payload):
    - draggedPath (image source that was dragged and dropped)
    - intervalNum (interval in the waveform)
    - start_or_end (whether that is a start or end preview)
    
    """
    global pipeline
    global num_intervals
    json_data = request.json
    torch.cuda.empty_cache()
    intervals = dict(json_data)


    with open('./static/interval_data.json', 'w') as f:
        json.dump(intervals, f)
        f.close()

    current_interval_times = [float(intervals["current_interval_start"]),float(intervals["current_interval_end"])]
    curr_interval_number = int(intervals["interval_num"])
    curr_interval_prompt = intervals["start_note"]
    curr_interval_seed = intervals["start_seed"]
    ending_prompt = intervals["end_note"]
    ending_seed = intervals["end_seed"]
    start = intervals["current_interval_start"].replace(".","sec")
    end = intervals["current_interval_end"].replace(".","sec")
    audio_reactive = intervals["audio_reactive_flag"]

    fps = 24
    num_interpolation_steps = [int((b-a) * fps) for a, b in zip(current_interval_times, current_interval_times[1:])]
    
    
    underscored_curr_interval_prompt = curr_interval_prompt.replace(" ","_")
    underscored_ending_prompt = ending_prompt.replace(" ","_")
    output_name = f"{start}_{end}_{underscored_curr_interval_prompt}_{curr_interval_seed}->{underscored_ending_prompt}_{ending_seed}".replace(",","-")
    
    # Print for feedback on what interval is being generated
    print(curr_interval_prompt, ending_prompt)
    # Audioreactive flag logic. This controls the alignment to the music. The difference between the two call is that if the smooth_flag is higher, the waveform driving the interpolation is dampened and the visual interpolation is smoother but less audioreactive.
    if (not audio_reactive): 
        video_path = pipeline.walk(
            prompts= [ curr_interval_prompt +" digital painting, 4k", ending_prompt +" digital painting, 4k"],
            seeds=[ int(curr_interval_seed), int(ending_seed)],
            num_interpolation_steps= num_interpolation_steps,
            height=512,                            # use multiples of 64
            width=512,   
            audio_filepath=f'{music}',    # Use your own file
            audio_start_sec=current_interval_times[0],       # Start second of the provided audio
            fps=fps,                               # important to set yourself based on the num_interpolation_steps you defined
            batch_size=4,                          # increase until you go out of memory.
            output_dir='./static/intervals',                 # Where images will be saved
            name=output_name, 
            num_inference_steps=50,                            # Subdir of output dir. will be timestamp by default
            smooth=0.8
        )
    else: 
        video_path = pipeline.walk(
            prompts= [ curr_interval_prompt, ending_prompt ],
            seeds=[ int(curr_interval_seed), int(ending_seed)],
            num_interpolation_steps= num_interpolation_steps,
            height=512,                            # use multiples of 64
            width=512,   
            audio_filepath=f'{music}',    # Use your own file
            audio_start_sec=current_interval_times[0],       # Start second of the provided audio
            fps=fps,                               # important to set yourself based on the num_interpolation_steps you defined
            batch_size=4,                          # increase until you go out of memory.
            output_dir='./static/intervals',                 # Where images will be saved
            name=output_name, 
            num_inference_steps=50,                            # Subdir of output dir. will be timestamp by default
            smooth=0.25
        )
    
    shutil.move(f"./static/intervals/{output_name}", f"./static/output/")
    return "Generated video"



# Deletes track (one generated interval)
@app.route("/delete_track", methods=["POST"])
def delete_track():
    """
    Deletes an interval.
    Parameters (of the payload):
    - path_to_delete (path to delete)
    
    """
    path_to_delete = request.json["path_to_delete"]
    regex_pattern = r'(.*)(static.*)\/(.*mp4)'
    relative_path =  "./" + re.match(regex_pattern, path_to_delete).group(2)
    relative_path = relative_path.replace("%3E",">")
    shutil.rmtree( relative_path)
    return "Deleted track"

# Deletes the video
@app.route("/delete_stitched_video", methods=["POST"])
def delete_stitched_video():
    """
    Deletes the current video output, which is hardcoded to always be `output/stitched_output.mp4`    
    """
    os.unlink( "./static/output/stitched_output.mp4")
    return "Deleted stitched video"

# Deletes the file
@app.route("/delete_file", methods=["POST"])
def delete_file():
    """
    Deletes the image file.

    Parameters (of payload)
    - path_to_delete - Image path
    """
    path_to_delete = request.json["path_to_delete"]
    regex_pattern = r'(.*)(static.*)(.*)'
    relative_path =  "./" + re.match(regex_pattern, path_to_delete).group(2)
    relative_path = relative_path.replace("%20"," ")
    os.unlink( relative_path[:-1])
    return "Deleted image"


@app.route("/brainstorm", methods=["POST"])
def brainstorm():
    """
    Deprecated way of sourcing style keywords, replaced by GPT prompt rewriting.
    """
    color_transitions = ["blue hour", "light leak", "dark", "muted", "psychedelic", "anaglyph", "polaroid", "geometric", "bloom pass", "gaussian blur", "sepia","color inversion", "neon","saturated", "desaturated", "warm","cool", "pastel", "cyberpunk","pop art", "vintage","film photography", "glitch","vignette"]
    time_transitions = ["sunset", "sunrise", "mosaic", "kaleidoscope", "storybook illustration", "street photography", "medieval", "cubism", "anime", "pixel art", "digital painting", "collage", "low poly",  "strobe", "retro", "dreamy", "cartoon", "golden hour", "dramatic lighting", "sunny", "cloudy", "timelapse", "light tunnels","bokeh", "soft lighting","cinematic lighting", "lens flare", "slow motion", "painting", "sketch", "photorealism"]
    angle_transitions = ["wide angle", "close-up", "birds eye view","midshot", "aerial view", "fisheye", "vaporwave", "swirl", "oil painting","Surrealism", "DSLR", "sculpture", "magazine cover", "ink wash painting", "pattern", "map", "Fauvism", "double exposure", "front view", "side view", "high angle", "fractal", "Impressionism", "cubism", "abstract", "dutch angle", "straight angle", "low angle", "vortex", "upside down", "side profile", "3D render", "motion blur", "watercolor", "grunge", "moonlight", "tilted frame","isometric","medium long" ]
    suggestions = color_transitions + time_transitions + angle_transitions
    
    suggestions = sample(suggestions,15)
    

    return {
        "color_trio": f"{suggestions[0]},{suggestions[1]}, {suggestions[2]}, {suggestions[3]}, {suggestions[4]}",
        "angle_trio":f"{suggestions[5]}, {suggestions[6]}, {suggestions[7]}, {suggestions[8]}, {suggestions[9]}",
        "time_trio":f"{suggestions[10]}, {suggestions[11]}, {suggestions[12]}, {suggestions[13]}, {suggestions[14]}"
    }

@app.route("/brainstorm_gpt", methods=["POST"])
def brainstorm_gpt():
    """
    Brainstorming with GPT related subjects for a goal visual that the user wants to express. This goal can be abstract (vibe) or concrete (song names, lyrics, images)

    Parameters (of the payload):
    - goal: Song name, lyric, or image that the user wishes to express. 
    - openai_api_key: API key
    
    """
    goal = request.json["goal"]
    openai_key = request.json["openai_api_key"]
    client = OpenAI(api_key=openai_key)

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system",
          "content": "Consider the song, its genre, and context. Then return images and symbols related with it in one long comma delimited list. Images and symbols should be very visualizable (concrete in nature rather than abstract). "
        },
        {
          "role": "user",
          "content": f"{goal}"
        },
      ],
      temperature=1,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    ).choices[0].message.content
    
    subjects = response.split(",")
    subject_prompts = []

    # GPT will output 10 subjects, pick 5
    for subject in subjects[:5]:
    
        response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {
              "role": "user",
              "content": "Expand each subject into a prompt for a text-to-image generator with rich visual language. Add modifiers like \"4k\" and styles that match the subject. Each prompt should not be more than 20 words. These prompts are not instructional (they should not start with a verb), they are like alt text.\n\nExamples.\nbrain: a colossal human brain is intricately connected to advanced computers housed in the surrounding racks by millions of wires. \n\ncat: Pastel pink baby cat wearing glitter pink jacket sitting in a cafe , pink luxury cafe theme , window open with a london snowy view, realistic art, highly detailed , snow falling\n\ngirl: 1970's dark fantasy book cover paper art of a beautiful girl in a field of sunflowers, symmetric front view, dungeons and dragons"
            },
            {
              "role": "assistant",
              "content": f"Write one for '{subject}'.\n{subject}:"
            },
          ],
          temperature=1,
          max_tokens=20,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        ).choices[0].message.content
        subject_prompts.append(response)

    for prompt in subject_prompts:
        # Generate each subject
        generate_and_save_image(prompt, 16, f"{prompt}_16")
    return { "subjects": subject_prompts
    }

@app.route("/stitch_videos", methods=["POST"])
def stitch_videos():
    """
    Moviepy function that concatenates video in a path. It replaces certain stray characters with regex.
    Audio on concatenated video file seems to have bugs in that it plays on the browser but not in iMovie or on opening the downloaded file
    
    """
    video_paths = request.json["video_paths"].split(',')
    regex_pattern = r'(.*)(static.*)(.*)'
    relative_paths = []
    for path in video_paths:
        relative_path =  (re.match(regex_pattern, path).group(2))
        relative_paths.append(relative_path)
    video_clips = [VideoFileClip(path.replace("%3E",">")) for path in relative_paths]
    final = concatenate_videoclips(video_clips)
#     final.write_videofile("./static/output/stitched_output.mp4")
    final.write_videofile("./static/output/stitched_output.mp4", audio=True, temp_audiofile='./static/output/temp-audio.mp3', audio_codec = 'libmp3lame')
    return "Stitch videos together"


@app.route("/gather_generations", methods=["GET"])
def gather_generations():
    """
    Gathering generations in the static folder.
    
    """
    generations_gallery = os.listdir("./static/generations/")
    return jsonify(generations_gallery)

def generate_images(prompt,seed):
    """
    Generates images with a prompt and seed.

    Parameters (of the payload):
    - Prompt
    - Seed
    """
    global pipeline
    torch.cuda.empty_cache()
    img = pipeline(prompt +" digital painting, 4k" ,num_inference_steps =50, height=512, width=512, batch_size=1, num_batches=1,seed=int(seed))
    return img

def folder_files_by_ext(folder_path, extension):
    """
    Grabs all the files in a folder with a certain extension. In this case, used for getting all .mp4's
    """
    result = [y for x in os.walk(folder_path) for y in glob(os.path.join(x[0], extension))]
    return result

def find_approx_create_time(interval_folder_name):
    """
    Find approximate create time of each video in a folder.
    This function is used to sort the history of the generations container.
    
    Parameters:
    - interval_folder_name (folder to walk through)
    """
    internal_folder = next(os.walk("./static/output/" + interval_folder_name))[0]
    files = next(os.walk(internal_folder))[0]
    interval_video = folder_files_by_ext(files,"*.mp4")
    return os.path.getmtime(interval_video[0])

@app.route("/register_music_change", methods=["POST"])
def register_music_change():
    """
    music, which is a global variable pointer to the music file is changed in value.
    Parameters (of the payload):
    - music_choice (string referring to the audio file path). When we swap the global variable, the music variable, which is what generate_interval passes in as a reference to the audiofile is swapped. 
    """
    global music
    global video
    music = request.json["music_choice"]
    print(music)

    return "completed"


@app.route("/change_audio", methods=["POST"])
def change_audio():
    """
    Alternate change audio_function 
    To-do: Confirm unused, delete
    
    Parameters (of the payload):
    - audio_file
    - video_file
    """

    global music
    global video

    audio_file = request.json["audio_file"] 
    video_file = request.json["video_file"] 

    music = f'./static/audio/{audio_file}'
    video = f'./static/audio/{video_file}'

    return {"audio_filename":music}


@app.route("/download_audio", methods=["POST"])
def download_audio():
    """
    Download music from soundcloud using youtube-dl
    Deprecated because of issues with cutting the music (different extensions like .mp3 and .m4a) have different ffmpeg rules, is where I left off
    
    Parameters (of the payload):
    - audio_file (music source on soundcloud)
    - music_start (start time offset in the music)
    - music_length (duration this value is an int like 10)
    
    """

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


@app.route("/waveform", methods=["GET", "POST"])
def waveform():
    """
    Renders the waveform.html in static/templates/waveform.html
    """
    return render_template("waveform.html")


@app.route("/", methods=["GET", "POST"])
def run_user_interface():
    """
    Renders the /static/templates/run_user_interface.html
    """
    global music
 
    frames = OrderedDict()
    video = f"./static/audio/{music}"
    videos = []

    # Check if the output directory exists
    if os.path.isdir('./static/output'):
        interval_folder_names = sorted(next(os.walk("./static/output/"))[1])
      
        output_files = next(os.walk("./static/output/"))[2]
        
        # Process interval folders
        for folder_name in interval_folder_names:
            if folder_name == '.ipynb_checkpoints':
                pass
            else:
                frames[folder_name] = OrderedDict()
                folder_path = "./static/output/" + f"{folder_name}/"

                # Sort interval frames
                frames[folder_name]["interval_frames"] = sorted(folder_files_by_ext(folder_path, "*.png"))
                frames[folder_name]["interval_frames"] = sorted([frame[9:] for frame in frames[folder_name]["interval_frames"] ])

    #             Get the interval video
                frames[folder_name]["interval_video"] = folder_files_by_ext(folder_path, "*.mp4")[0]

        # Get list of videos
        videos = folder_files_by_ext("./static/output/", "*.mp4")
        if "./static/output/stitched_output.mp4" in videos:
            video = "./static/output/stitched_output.mp4"

    # Get and sort generation files
    generations = sorted(os.listdir("./static/generations/"))
    generations.sort(key=lambda x: os.path.getmtime("./static/generations/" + x), reverse=True)

    # Get interval previews
    preview_dict = OrderedDict()
    for filename in os.listdir("./static/previews"):
        interval_preview_number = filename[0]
        preview_dict[interval_preview_number] = filename

    # Format generation filenames
    generations = [("generations/" + generation_filename, generation_filename.replace(" ", "_").replace(",", "-")) for generation_filename in generations]

    # Render the template with the collected data
    return render_template("user_interface.html", frames=frames, videos=videos, output_video=video, generations=generations)


def run():
    """
    Runs the python app. 
    """
    app.run(host='0.0.0.0', threaded=True, debug=True, port = 7860)
run()
