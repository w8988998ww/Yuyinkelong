import os
import time

import numpy as np
import torch
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text import create_symbols_manager, text_to_sequence


class AudioGenerator():
    def __init__(self, hparams, device):
        self.hparams = hparams
        self._device = device

        symbols_manager = create_symbols_manager(hparams.data.language)
        self.symbol_to_id = symbols_manager._symbol_to_id

        self.net_g = create_network(hparams, self.symbol_to_id, device)

    def load(self, path):
        load_checkpoint(self.net_g, path)

    def inference(self, text):
        return do_inference(self.net_g, self.hparams, self.symbol_to_id, text, self._device)

def get_text(text, hparams, symbol_to_id):
    text_norm = text_to_sequence(text, hparams.data.text_cleaners, symbol_to_id)
    if hparams.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def create_network(hparams, symbols, device):
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
def do_inference(generator, hparams, symbol_to_id, text, device):
    stn_tst = get_text(text, hparams, symbol_to_id)

    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = generator.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    
    return audio

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
def save_to_wav(audio, sampling_rate, path):
    max = 32767
    audio_int16 = np.floor(((max + 1) * audio)).astype(np.int16)
    write(path, sampling_rate, audio_int16)

if __name__ == "__main__":
    # https://stackoverflow.com/questions/53331247/pytorch-0-4-0-there-are-three-ways-to-create-tensors-on-cuda-device-is-there-s/53332659#53332659
    # https://community.esri.com/t5/arcgis-image-analyst-questions/how-force-pytorch-to-use-cpu-instead-of-gpu/td-p/1046738
    # torch.cuda.is_available = lambda : False
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda')

    device = torch.device('cpu')
    # device = torch.device('cuda')

    config_path = "./configs/bb_laptop.json"
    hps = utils.get_hparams_from_file(config_path)

    audio_generator = AudioGenerator(hps, device)

    checkpoint_path = "./models/G_bb_19000.pth"
    audio_generator.load(checkpoint_path)

    text = "我是御坂妹妹！"
    # text = "12345！"
    # text = "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired，"
    start = time.perf_counter()
    audio = audio_generator.inference(text)
    print(f"The inference takes {time.perf_counter() - start} seconds")

    print(audio.dtype)

    output_dir = './output/'
    # python program to check if a path exists
    # if it doesn’t exist we create one
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = 'output.wav'
    file_path = os.path.join(output_dir, filename)

    save_to_wav(audio, hps.data.sampling_rate, file_path)


