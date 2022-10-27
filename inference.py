import os
import torch

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence, create_symbols_manager

import numpy as np
from scipy.io.wavfile import write

import time

# https://stackoverflow.com/questions/53331247/pytorch-0-4-0-there-are-three-ways-to-create-tensors-on-cuda-device-is-there-s/53332659#53332659
# https://community.esri.com/t5/arcgis-image-analyst-questions/how-force-pytorch-to-use-cpu-instead-of-gpu/td-p/1046738

# torch.cuda.is_available = lambda : False
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')
device = torch.device('cpu')

def get_text(text, hparams, symbol_to_id):
    text_norm = text_to_sequence(text, hparams.data.text_cleaners, symbol_to_id)
    if hparams.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def create_network(hparams, symbols):
    net_g = SynthesizerTrn(
        len(symbols),
        hparams.data.filter_length // 2 + 1,
        hparams.train.segment_size // hparams.data.hop_length,
        **hparams.model).to(device)
    _ = net_g.eval()

    return net_g

def load_checkpoint(network, path):
    _ = utils.load_checkpoint(path, network, None)

# Assume the network has loaded weights and are ready to do inference
def inference(net_with_weights, hparams, symbol_to_id, text):
    stn_tst = get_text(text, hparams, symbol_to_id)

    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = network_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    
    return audio

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
def save_to_wav(audio, path, hparams):
    max = 32767
    audio_int16 = np.floor(((max + 1) * audio)).astype(np.int16)
    write(path, hparams.data.sampling_rate, audio_int16)

hps = utils.get_hparams_from_file("./configs/bb_laptop.json")
symbols_manager = create_symbols_manager(hps.data.language)

network_g = create_network(hps, symbols_manager._symbol_to_id)
load_checkpoint(network_g, "./models/G_bb_19000.pth")

text = "我是御坂妹妹！"
# text = "12345！"
# text = "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired，"

start = time.perf_counter()
audio = inference(network_g, hps, symbols_manager._symbol_to_id, text)
print(f"The inference takes {time.perf_counter() - start} seconds")

print(audio.dtype)

output_dir = './output/'
# python program to check if a path exists
# if it doesn’t exist we create one
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
filename = 'output.wav'
file_path = os.path.join(output_dir, filename)

save_to_wav(audio, file_path, hps)


