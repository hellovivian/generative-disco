# 🪩 generative disco

June'23 Update: Hugging Face Spaces demo available here: [vivlavida/generative-disco](https://huggingface.co/spaces/vivlavida/generative-disco) 🌷

![GenDisco](https://user-images.githubusercontent.com/15935546/235154270-3c9d42df-ac39-4472-b0d5-c1e5ae9eb228.gif)



Visuals are a core part of our experience of music. However, creating music visualization is a complex, time-consuming, and resource-intensive process. We introduce Generative Disco, a generative AI system that helps generate music visualizations with large language models and text-to-image models. Users select intervals of music to visualize and then parameterize that visualization by defining start and end prompts. These prompts are warped between and generated according to the beat of the music for audioreactive video. We introduce design patterns for improving generated videos: "transitions", which express shifts in color, time, subject, or style, and "holds", which encourage visual emphasis and consistency.

## How to Use
<a href='https://youtu.be/q22I53jHbuU?t=59'> Demo of System </a> <br><br>
<img width="655" alt="Screenshot 2023-06-25 at 10 44 21 AM" src="https://github.com/hellovivian/generative-disco/assets/15935546/3bf09f07-e1ce-4272-84aa-8e255b44ade6">


The only way the demo differs from the video is that it includes fields for `OpenAI Key` and `Soundcloud URL to Music`. These functions are intended for if you would like to use Disco in a more dedicated way. HuggingFace Spaces allows many people to edit the same space at once, so duplicate / clone the space if you would like to persist your work. <br><br>
The `OpenAI Key` field is only necessary if you want to use the Brainstorming `Describe Interval` function, which retrieves inspiration for different subjects of prompts using GPT-3. We pass the `OpenAI Key` as an argument for an API call. <br><br>
The `Soundcloud URL to Music` allows you to change the music file. <br><br>

### Notes
<ul>
      <li>All fields have to be filled out in the form (Interval #) for the generation to begin.</li>
</ul>

While video intervals are generating you can check the progress by looking at the Logs tab. Generating even a few seconds of content on the community GPU may take awhile, so we recommend generating in very short intervals (like 0.5 seconds at a time).
<img width="655" alt="Screenshot 2023-06-25 at 10 16 19 AM" src="https://github.com/hellovivian/generative-disco/assets/15935546/61d28bce-fb16-499b-8014-179043c642da">



## Links
<a href='https://youtu.be/q22I53jHbuU'> Full Youtube video </a> <br><br>
<a href='https://arxiv.org/abs/2304.08551'> arXiv Preprint </a>

## Examples
https://user-images.githubusercontent.com/15935546/235145415-e30db4a2-8e10-4751-9647-bb55dc189719.mp4

https://user-images.githubusercontent.com/15935546/235147315-9145c8e3-4bf2-430b-8e51-2fe0b8a714be.mov

## System Design
![fig_sysdesign_larger_fin](https://user-images.githubusercontent.com/15935546/235155210-3835148d-72bf-4db7-b8e0-50b0ddfbb147.png)

Generative Disco's system design. Users begin by interacting with the waveform to create intervals within the music (#1). To find prompts that will define the start and end of intervals, users can brainstorm prompts using prompt suggestions from GPT-4 or videography domain knowledge (#4-6) and explore text-to-image generations (#7, #8). Results users like can be dragged and dropped into the start and end areas (#10,#11), after which an interval can be generated. Generated intervals populate in the tracks area (#15) and can be stitched into a video that renders in the Video Area (#9).

### Setup Option 1

Docker hub `docker pull hellovivian/art-ai:disco_local`


### Setup Option 2

*Requires a GPU with a decent amount of VRAM.

Install the package

```bash
pip install -U stable_diffusion_videos
conda env create -f disco_environment.yml
```

Authenticate with Hugging Face

```bash
huggingface-cli login
```

```
conda activate video
python flask_app.py
```

## Repo Structure


The Stable Diffusion checkpoint used was V1-4. The web application was written in Python, Javascript, and Flask. Images were generated with 50 iterations on an NVIDIA V100. The music, assets (images), stylesheets, and Javascript are collected in the static folder. Logic and routing is controlled by `flask_app.py`


## Credits
The system was built on top of two open-source repositories: a) stable-diffusion-videos from Hugging Face, written by Nate Raw and b) wavesurfer.js. stable-diffusion-videos also builds on
[a script](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
) shared by [@karpathy](https://github.com/karpathy). The script was modified to [this gist](https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53), which was then updated/modified to this repo. 

## Citation

BibTex 
```
@misc{liu2023generative,
      title={Generative Disco: Text-to-Video Generation for Music Visualization}, 
      author={Vivian Liu and Tao Long and Nathan Raw and Lydia Chilton},
      year={2023},
      eprint={2304.08551},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```

## Contributing 

You can file any issues/feature requests :)


