import inspect
from typing import Callable, List, Optional, Union
from pathlib import Path
from torchvision.transforms.functional import pil_to_tensor
import librosa
from PIL import Image
from torchvision.io import write_video
import numpy as np
import time
import json

import torch
from packaging import version
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import deprecate, logging
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from torch import nn

from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def get_timesteps_arr(audio_filepath, offset, duration, fps=30, margin=1.0, smooth=0.0):
    y, sr = librosa.load(audio_filepath, offset=offset, duration=duration)

    # librosa.stft hardcoded defaults...
    # n_fft defaults to 2048
    # hop length is win_length // 4
    # win_length defaults to n_fft
    D = librosa.stft(y, n_fft=2048, hop_length=2048 // 4, win_length=2048)

    # Extract percussive elements
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=margin)
    y_percussive = librosa.istft(D_percussive, length=len(y))

    # Get normalized melspectrogram
    spec_raw = librosa.feature.melspectrogram(y=y_percussive, sr=sr)
    spec_max = np.amax(spec_raw, axis=0)
    spec_norm = (spec_max - np.min(spec_max)) / np.ptp(spec_max)

    # Resize cumsum of spec norm to our desired number of interpolation frames
    x_norm = np.linspace(0, spec_norm.shape[-1], spec_norm.shape[-1])
    y_norm = np.cumsum(spec_norm)
    y_norm /= y_norm[-1]
    x_resize = np.linspace(0, y_norm.shape[-1], int(duration*fps))

    T = np.interp(x_resize, x_norm, y_norm)

    # Apply smoothing
    return T * (1 - smooth) + np.linspace(0.0, 1.0, T.shape[0]) * smooth


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def make_video_pyav(
    frames_or_frame_dir: Union[str, Path, torch.Tensor],
    audio_filepath: Union[str, Path] = None,
    fps: int = 30,
    audio_offset: int = 0,
    audio_duration: int = 2,
    sr: int = 22050,
    output_filepath: Union[str, Path] = "output.mp4",
    glob_pattern: str = "*.png",
):
    """
    TODO - docstring here

    frames_or_frame_dir: (Union[str, Path, torch.Tensor]):
        Either a directory of images, or a tensor of shape (T, C, H, W) in range [0, 255].
    """

    # Torchvision write_video doesn't support pathlib paths
    output_filepath = str(output_filepath)

    if isinstance(frames_or_frame_dir, (str, Path)):
        frames = None
        for img in sorted(Path(frames_or_frame_dir).glob(glob_pattern)):
            frame = pil_to_tensor(Image.open(img)).unsqueeze(0)
            frames = frame if frames is None else torch.cat([frames, frame])
    else:
        frames = frames_or_frame_dir

    # TCHW -> THWC
    frames = frames.permute(0, 2, 3, 1)

    if audio_filepath:
        # Read audio, convert to tensor
        audio, sr = librosa.load(audio_filepath, sr=sr, mono=True, offset=audio_offset, duration=audio_duration)
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        write_video(
            output_filepath,
            frames,
            fps=fps,
            audio_array=audio_tensor,
            audio_fps=sr,
            audio_codec="aac",
            options={"crf": "10", "pix_fmt": "yuv420p"},
        )
    else:
        write_video(output_filepath, frames, fps=fps, options={"crf": "10", "pix_fmt": "yuv420p"})

    return output_filepath


class StableDiffusionImg2ImgWalkPipeline():
    def __init__(self):
        super().__init__()
        self.pipe  = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        print(self.pipe)

    def generate_inputs(self, prompt_a, prompt_b, seed_a, seed_b, noise_shape, T, batch_size):

        embeds_a = self.pipe._encode_prompt(
        prompt_a,
        device,
        num_images_per_prompt,
        False,
        "")

        embeds_b = self.pipe._encode_prompt(
        prompt_b,
        device,
        num_images_per_prompt,
        False,
        "")
        latents_dtype = embeds_a.dtype

        # embeds_a = self.embed_text(prompt_a)
        # embeds_b = self.embed_text(prompt_b)
        # latents_dtype = embeds_a.dtype
        latents_a = self.init_noise(seed_a, noise_shape, latents_dtype)
        latents_b = self.init_noise(seed_b, noise_shape, latents_dtype)

        batch_idx = 0
        embeds_batch, noise_batch = None, None
        for i, t in enumerate(T):
            embeds = torch.lerp(embeds_a, embeds_b, t)
            noise = slerp(float(t), latents_a, latents_b)

            embeds_batch = embeds if embeds_batch is None else torch.cat([embeds_batch, embeds])
            noise_batch = noise if noise_batch is None else torch.cat([noise_batch, noise])
            batch_is_ready = embeds_batch.shape[0] == batch_size or i + 1 == T.shape[0]
            if not batch_is_ready:
                continue
            yield batch_idx, embeds_batch, noise_batch
            batch_idx += 1
            del embeds_batch, noise_batch
            torch.cuda.empty_cache()
            embeds_batch, noise_batch = None, None

    def make_clip_frames(
        self,
        prompt_a: str,
        prompt_b: str,
        seed_a: int,
        seed_b: int,
        num_interpolation_steps: int = 5,
        save_path: Union[str, Path] = "outputs/",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        upsample: bool = False,
        batch_size: int = 1,
        image_file_ext: str = ".png",
        T: np.ndarray = None,
        skip: int = 0,
        negative_prompt: str = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        T = T if T is not None else np.linspace(0.0, 1.0, num_interpolation_steps)
        if T.shape[0] != num_interpolation_steps:
            raise ValueError(f"Unexpected T shape, got {T.shape}, expected dim 0 to be {num_interpolation_steps}")

        batch_generator = self.generate_inputs(
            prompt_a,
            prompt_b,
            seed_a,
            seed_b,
            (1, self.unet.in_channels, height // 8, width // 8),
            T[skip:],
            batch_size,
        )

        frame_index = skip
        for _, embeds_batch, noise_batch in batch_generator:
            outputs = self(
                latents=noise_batch,
                text_embeddings=embeds_batch,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                eta=eta,
                num_inference_steps=num_inference_steps,
                output_type="pil" if not upsample else "numpy",
                negative_prompt=negative_prompt,
            )["images"]

            for image in outputs:
                frame_filepath = save_path / (f"frame%06d{image_file_ext}" % frame_index)
                image = image if not upsample else self.upsampler(image)
                image.save(frame_filepath)
                frame_index += 1

    def walk(
        self,
        prompts: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        num_interpolation_steps: Optional[Union[int, List[int]]] = 5,  # int or list of int
        output_dir: Optional[str] = "./dreams",
        name: Optional[str] = None,
        image_file_ext: Optional[str] = ".png",
        fps: Optional[int] = 30,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        upsample: Optional[bool] = False,
        batch_size: Optional[int] = 1,
        resume: Optional[bool] = False,
        audio_filepath: str = None,
        audio_start_sec: Optional[Union[int, float]] = None,
        margin: Optional[float] = 1.0,
        smooth: Optional[float] = 0.0,
        negative_prompt: Optional[str] = None,
    ):
        """Generate a video from a sequence of prompts and seeds. Optionally, add audio to the
        video to interpolate to the intensity of the audio.

        Args:
            prompts (Optional[List[str]], optional):
                list of text prompts. Defaults to None.
            seeds (Optional[List[int]], optional):
                list of random seeds corresponding to prompts. Defaults to None.
            num_interpolation_steps (Union[int, List[int]], *optional*):
                How many interpolation steps between each prompt. Defaults to None.
            output_dir (Optional[str], optional):
                Where to save the video. Defaults to './dreams'.
            name (Optional[str], optional):
                Name of the subdirectory of output_dir. Defaults to None.
            image_file_ext (Optional[str], *optional*, defaults to '.png'):
                The extension to use when writing video frames.
            fps (Optional[int], *optional*, defaults to 30):
                The frames per second in the resulting output videos.
            num_inference_steps (Optional[int], *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (Optional[float], *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (Optional[float], *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            height (Optional[int], *optional*, defaults to None):
                height of the images to generate.
            width (Optional[int], *optional*, defaults to None):
                width of the images to generate.
            upsample (Optional[bool], *optional*, defaults to False):
                When True, upsamples images with realesrgan.
            batch_size (Optional[int], *optional*, defaults to 1):
                Number of images to generate at once.
            resume (Optional[bool], *optional*, defaults to False):
                When True, resumes from the last frame in the output directory based
                on available prompt config. Requires you to provide the `name` argument.
            audio_filepath (str, *optional*, defaults to None):
                Optional path to an audio file to influence the interpolation rate.
            audio_start_sec (Optional[Union[int, float]], *optional*, defaults to 0):
                Global start time of the provided audio_filepath.
            margin (Optional[float], *optional*, defaults to 1.0):
                Margin from librosa hpss to use for audio interpolation.
            smooth (Optional[float], *optional*, defaults to 0.0):
                Smoothness of the audio interpolation. 1.0 means linear interpolation.
            negative_prompt (Optional[str], *optional*, defaults to None):
                Optional negative prompt to use. Same across all prompts.

        This function will create sub directories for each prompt and seed pair.

        For example, if you provide the following prompts and seeds:

        ```
        prompts = ['a dog', 'a cat', 'a bird']
        seeds = [1, 2, 3]
        num_interpolation_steps = 5
        output_dir = 'output_dir'
        name = 'name'
        fps = 5
        ```

        Then the following directories will be created:

        ```
        output_dir
        ├── name
        │   ├── name_000000
        │   │   ├── frame000000.png
        │   │   ├── ...
        │   │   ├── frame000004.png
        │   │   ├── name_000000.mp4
        │   ├── name_000001
        │   │   ├── frame000000.png
        │   │   ├── ...
        │   │   ├── frame000004.png
        │   │   ├── name_000001.mp4
        │   ├── ...
        │   ├── name.mp4
        |   |── prompt_config.json
        ```

        Returns:
            str: The resulting video filepath. This video includes all sub directories' video clips.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        output_path = Path(output_dir)

        name = name or time.strftime("%Y%m%d-%H%M%S")
        save_path_root = output_path / name
        save_path_root.mkdir(parents=True, exist_ok=True)

        # Where the final video of all the clips combined will be saved
        output_filepath = save_path_root / f"{name}.mp4"

        # If using same number of interpolation steps between, we turn into list
        if not resume and isinstance(num_interpolation_steps, int):
            num_interpolation_steps = [num_interpolation_steps] * (len(prompts) - 1)

        if not resume:
            audio_start_sec = audio_start_sec or 0

        # Save/reload prompt config
        prompt_config_path = save_path_root / "prompt_config.json"
        if not resume:
            prompt_config_path.write_text(
                json.dumps(
                    dict(
                        prompts=prompts,
                        seeds=seeds,
                        num_interpolation_steps=num_interpolation_steps,
                        fps=fps,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        upsample=upsample,
                        height=height,
                        width=width,
                        audio_filepath=audio_filepath,
                        audio_start_sec=audio_start_sec,
                        negative_prompt=negative_prompt,
                    ),
                    indent=2,
                    sort_keys=False,
                )
            )
        else:
            data = json.load(open(prompt_config_path))
            prompts = data["prompts"]
            seeds = data["seeds"]
            num_interpolation_steps = data["num_interpolation_steps"]
            fps = data["fps"]
            num_inference_steps = data["num_inference_steps"]
            guidance_scale = data["guidance_scale"]
            eta = data["eta"]
            upsample = data["upsample"]
            height = data["height"]
            width = data["width"]
            audio_filepath = data["audio_filepath"]
            audio_start_sec = data["audio_start_sec"]
            negative_prompt = data.get("negative_prompt", None)

        for i, (prompt_a, prompt_b, seed_a, seed_b, num_step) in enumerate(
            zip(prompts, prompts[1:], seeds, seeds[1:], num_interpolation_steps)
        ):
            # {name}_000000 / {name}_000001 / ...
            save_path = save_path_root / f"{name}_{i:06d}"

            # Where the individual clips will be saved
            step_output_filepath = save_path / f"{name}_{i:06d}.mp4"

            # Determine if we need to resume from a previous run
            skip = 0
            if resume:
                if step_output_filepath.exists():
                    print(f"Skipping {save_path} because frames already exist")
                    continue

                existing_frames = sorted(save_path.glob(f"*{image_file_ext}"))
                if existing_frames:
                    skip = int(existing_frames[-1].stem[-6:]) + 1
                    if skip + 1 >= num_step:
                        print(f"Skipping {save_path} because frames already exist")
                        continue
                    print(f"Resuming {save_path.name} from frame {skip}")

            audio_offset = audio_start_sec + sum(num_interpolation_steps[:i]) / fps
            audio_duration = num_step / fps

            self.make_clip_frames(
                prompt_a,
                prompt_b,
                seed_a,
                seed_b,
                num_interpolation_steps=num_step,
                save_path=save_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                height=height,
                width=width,
                upsample=upsample,
                batch_size=batch_size,
                T=get_timesteps_arr(
                    audio_filepath,
                    offset=audio_offset,
                    duration=audio_duration,
                    fps=fps,
                    margin=margin,
                    smooth=smooth,
                )
                if audio_filepath
                else None,
                skip=skip,
                negative_prompt=negative_prompt,
            )
            make_video_pyav(
                save_path,
                audio_filepath=audio_filepath,
                fps=fps,
                output_filepath=step_output_filepath,
                glob_pattern=f"*{image_file_ext}",
                audio_offset=audio_offset,
                audio_duration=audio_duration,
                sr=44100,
            )

        return make_video_pyav(
            save_path_root,
            audio_filepath=audio_filepath,
            fps=fps,
            audio_offset=audio_start_sec,
            audio_duration=sum(num_interpolation_steps) / fps,
            output_filepath=output_filepath,
            glob_pattern=f"**/*{image_file_ext}",
            sr=44100,
        )

    def embed_text(self, text, negative_prompt=None):
        """Helper to embed some text"""
        text_input = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return embed

    def init_noise(self, seed, noise_shape, dtype):
        """Helper to initialize noise"""
        # randn does not exist on mps, so we create noise on CPU here and move it to the device after initialization
        if self.device.type == "mps":
            noise = torch.randn(
                noise_shape,
                device='cpu',
                generator=torch.Generator(device='cpu').manual_seed(seed),
            ).to(self.device)
        else:
            noise = torch.randn(
                noise_shape,
                device=self.device,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                dtype=dtype,
            )
        return noise
