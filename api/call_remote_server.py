import torch
import os
from flask import Flask, Response, jsonify, request, Blueprint
from flask_restful import Api, Resource
import pickle
import threading
import argparse

from stepvideo.vae.vae_pipeline import StepVaePipeline, CaptionPipeline

device = f'cuda:{torch.cuda.device_count()-1}'
torch.cuda.set_device(device)
dtype = torch.bfloat16

def parsed_args():
    parser = argparse.ArgumentParser(description="StepVideo API Functions")
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--clip_dir', type=str, default='hunyuan_clip')
    parser.add_argument('--llm_dir', type=str, default='step_llm')
    parser.add_argument('--vae_dir', type=str, default='vae')
    parser.add_argument('--port', type=str, default='8080')
    args = parser.parse_args()
    return args



lock = threading.Lock()
class VAEapi(Resource):
    def __init__(self, vae_pipeline):
        self.vae_pipeline = vae_pipeline
        
    def get(self):
        with lock:
            try:
                feature = pickle.loads(request.get_data())
                feature['api'] = 'vae'
            
                feature = {k:v for k, v in feature.items() if v is not None}
                video_latents = self.vae_pipeline.decode(**feature)
                response = pickle.dumps(video_latents)

            except Exception as e:
                print("Caught Exception: ", e)
                return Response(e)
            
            return Response(response)





lock = threading.Lock()
class Captionapi(Resource):
    def __init__(self, caption_pipeline):
        self.caption_pipeline = caption_pipeline

    def get(self):
        with lock:
            try:
                feature = pickle.loads(request.get_data())
                feature['api'] = 'caption'

                feature = {k:v for k, v in feature.items() if v is not None}
                embeddings = self.caption_pipeline.embedding(**feature)
                response = pickle.dumps(embeddings)

            except Exception as e:
                print("Caught Exception: ", e)
                return Response(e)

            return Response(response)




class RemoteServer(object):
    def __init__(self, args) -> None:
        self.app = Flask(__name__)
        root = Blueprint("root", __name__)
        self.app.register_blueprint(root)
        api = Api(self.app)
        
        self.vae_pipeline = StepVaePipeline(
            vae_dir=os.path.join(args.model_dir, args.vae_dir)
        )
        api.add_resource(
            VAEapi,
            "/vae-api",
            resource_class_args=[self.vae_pipeline],
        )
        
        self.caption_pipeline = CaptionPipeline(
            llm_dir=os.path.join(args.model_dir, args.llm_dir), 
            clip_dir=os.path.join(args.model_dir, args.clip_dir)
        )
        api.add_resource(
            Captionapi,
            "/caption-api",
            resource_class_args=[self.caption_pipeline],
        )


    def run(self, host="0.0.0.0", port=8080):
        self.app.run(host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    args = parsed_args()
    flask_server = RemoteServer(args)
    flask_server.run(host="0.0.0.0", port=args.port)
    