import torch
import os
from flask_restful import Resource

device = f'cuda:{torch.cuda.device_count() - 1}'
dtype = torch.bfloat16


class CaptionPipeline(Resource):
    def __init__(self, llm_dir, clip_dir):
        self.text_encoder = self.build_llm(llm_dir)
        self.clip = self.build_clip(clip_dir)

    def build_llm(self, model_dir):
        from stepvideo.text_encoder.stepllm import STEP1TextEncoder
        text_encoder = STEP1TextEncoder(model_dir, max_length=320).to(dtype).to(device).eval()
        print("Inintialized text encoder...")
        return text_encoder

    def build_clip(self, model_dir):
        from stepvideo.text_encoder.clip import HunyuanClip
        clip = HunyuanClip(model_dir, max_length=77).to(device).eval()
        print("Inintialized clip encoder...")
        return clip

    def embedding(self, prompts, *args, **kwargs):
        with torch.no_grad():
            try:
                y, y_mask = self.text_encoder(prompts)
                clip_embedding, _ = self.clip(prompts)
                len_clip = clip_embedding.shape[1]
                y_mask = torch.nn.functional.pad(y_mask, (len_clip, 0),
                                                 value=1)  ## pad attention_mask with clip's length
                data = {
                    'y': y.detach().cpu(),
                    'y_mask': y_mask.detach().cpu(),
                    'clip_embedding': clip_embedding.to(torch.bfloat16).detach().cpu()
                }

                return data
            except Exception as err:
                print(f"{err}")
                return None


class StepVaePipeline(Resource):
    def __init__(self, vae_dir, version=2):
        self.vae = self.build_vae(vae_dir, version)
        self.scale_factor = 1.0

    def build_vae(self, vae_dir, version=2):
        from stepvideo.vae.vae import AutoencoderKL
        (model_name, z_channels) = ("vae_v2.safetensors", 64) if version == 2 else ("vae.safetensors", 16)
        model_path = os.path.join(vae_dir, model_name)

        model = AutoencoderKL(
            z_channels=z_channels,
            model_path=model_path,
            version=version,
        ).to(dtype).to(device).eval()
        print("Inintialized vae...")
        return model

    def decode(self, samples, *args, **kwargs):
        with torch.no_grad():
            try:
                dtype = next(self.vae.parameters()).dtype
                device = next(self.vae.parameters()).device
                samples = self.vae.decode(samples.to(dtype).to(device) / self.scale_factor)
                if hasattr(samples, 'sample'):
                    samples = samples.sample
                return samples
            except:
                torch.cuda.empty_cache()
                return None
