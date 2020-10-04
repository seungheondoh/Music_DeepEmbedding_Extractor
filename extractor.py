from argparse import ArgumentParser, Namespace

import torch
import torchaudio
import numpy as np

from model.net import FCN, FCN05, ShortChunkCNN_Res

def get_audio(mp3_path):
    waveform, sr = torchaudio.load(mp3_path)
    downsample_resample = torchaudio.transforms.Resample(sr, 16000)
    audio_tensor = downsample_resample(waveform)
    return audio_tensor, audio_tensor.shape[1]

def load_model(audio_length, models):
    if models == "FCN05":
        input_length= 8000
        model = FCN05()
        checkpoint_path = (
            f"weights/FCN05-roc_auc=0.8552-pr_auc=0.3344.ckpt"
        )
    elif models == "FCN037":
        input_length= 59049
        model = ShortChunkCNN_Res()
        checkpoint_path = (
            f"weights/ShortChunkCNN037-roc_auc=0.8948-pr_auc=0.4039.ckpt"
        )
    elif models == "FCN29":
        input_length= 464000
        model = FCN()
        checkpoint_path = (
            f"weights/FCN29-roc_auc=0.9025-pr_auc=0.4342.ckpt"
        )
    return input_length, model, checkpoint_path

def make_frames(audio_tensor, input_length):
    num_frame = int(audio_tensor.shape[1] / input_length)
    audio_tensor = torch.reshape(audio_tensor[0,:num_frame*input_length],(num_frame,input_length))
    return audio_tensor

def embedding_extractor(audio_path, model_types):
    audio, audio_length = get_audio(audio_path)
    labels = np.load('dataset/mtat/split/tags.npy')

    input_length, model, checkpoint_path = load_model(audio_length, model_types)

    audio_input = make_frames(audio, input_length)
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
    new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
    model.load_state_dict(new_state_dict)
    
    model.eval() 
    _, embeddings = model(audio_input)
    return embeddings

def main(args) -> None:
    embedding = embedding_extractor(args.audio_path, args.models)
    print(embedding)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--models", default="FCN05", type=str, choices=["FCN05", "FCN037", "FCN29"])
    parser.add_argument("--audio_path", default="dataset/mtat/test_mp3/sample2.mp3", type=str)
    args = parser.parse_args()
    main(args)
