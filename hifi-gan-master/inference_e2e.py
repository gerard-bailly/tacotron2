from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator

import re

h = None
device = None


def load_checkpoint(filepath, device):
    print(filepath)
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a, state_dict_g, h, generator):
    # generator = Generator(h).to(device)

    # state_dict_g = load_checkpoint(a.checkpoint_file, device)
    # generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir("{}".format(a.input_mels_dir))
    # filelist = glob.glob("{}/*.WAVEGLOW".format(a.input_mels_dir))

    os.makedirs(a.output_dir, exist_ok=True)

    # generator.eval()
    # generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            if re.match(".*WAVEGLOW", filname):
                if True:
                    # Processing for generic mel files
                    shape = tuple(np.fromfile(os.path.join(a.input_mels_dir, filname), count = 2, dtype = np.int32))
                    x = np.memmap(os.path.join(a.input_mels_dir, filname),offset=8,dtype=np.float32,shape=shape)
                    x = x.transpose()

                    x = torch.from_numpy(x)
                    x = torch.unsqueeze(x, 0).to(device)
                    # print(x)
                else:
                    x = np.load(os.path.join(a.input_mels_dir, filname))
                    x = torch.FloatTensor(x).to(device)

                y_g_hat = generator(x)
                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')

                output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '.wav')
                write(output_file, h.sampling_rate, audio)
                print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    print(a.checkpoint_file)
    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    inference(a, state_dict_g, h)


if __name__ == '__main__':
    main()

