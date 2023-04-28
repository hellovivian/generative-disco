# 🪩 generative disco

🏗️ Open-sourcing efforts in progress. Check mid-May 2023. 🌷

![GenDisco](https://user-images.githubusercontent.com/15935546/235154270-3c9d42df-ac39-4472-b0d5-c1e5ae9eb228.gif)

Visuals are a core part of our experience of music. However, creating music visualization is a complex, time-consuming, and resource-intensive process. We introduce Generative Disco, a generative AI system that helps generate music visualizations with large language models and text-to-image models. Users select intervals of music to visualize and then parameterize that visualization by defining start and end prompts. These prompts are warped between and generated according to the beat of the music for audioreactive video. We introduce design patterns for improving generated videos: "transitions", which express shifts in color, time, subject, or style, and "holds", which encourage visual emphasis and consistency.

## Examples
https://user-images.githubusercontent.com/15935546/235145415-e30db4a2-8e10-4751-9647-bb55dc189719.mp4

https://user-images.githubusercontent.com/15935546/235147315-9145c8e3-4bf2-430b-8e51-2fe0b8a714be.mov

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

### Repo Structure



## Credits

This work builds on a repository called stable-diffusion-videos by Nate Raw at Hugging Face.
[a script](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
) shared by [@karpathy](https://github.com/karpathy). The script was modified to [this gist](https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53), which was then updated/modified to this repo. 

## Contributing 

You can file any issues/feature requests 


