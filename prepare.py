import os, csv, argparse
import sys
import torch, torchaudio, timm
import numpy as np
from torch.cuda.amp import autocast
import IPython
current_directory = os.path.dirname(os.path.abspath(__file__))  
sys.path.append(current_directory)  
from src.models import ASTModel

# Create a new class that inherits the original ASTModel class
class ASTModelVis(ASTModel):
    def get_att_map(self, block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def forward_visualization(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        # save the attention map of each of 12 Transformer layer
        att_list = []
        for blk in self.v.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        return att_list

def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    # assert sr == 16000, 'input audio sampling rate must be 16kHz'

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels

def ASTpredict():
    # Assume each input spectrogram has 1024 time frames
    input_tdim = 1024
    checkpoint_path = './ast_master/pretrained_models/audio_mdl.pth'
    # now load the visualization model
    ast_mdl = ASTModelVis(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    print(f'[*INFO] load checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()          

    # Load the AudioSet label set
    label_csv = './ast_master/egs/audioset/data/class_labels_indices.csv'       # label and indices for audioset data
    labels = load_label(label_csv)

    feats = make_features("./audio.flac", mel_bins=128)           # shape(1024, 128)
    feats_data = feats.expand(1, input_tdim, 128)           # reshape the feature
    feats_data = feats_data.to(torch.device("cuda:0"))
    # do some masking of the input
    #feats_data[:, :512, :] = 0.

    # Make the prediction
    with torch.no_grad():
        with autocast():
            output = audio_model.forward(feats_data)
            output = torch.sigmoid(output)
    result_output = output.data.cpu().numpy()[0]
    sorted_indexes = np.argsort(result_output)[::-1]

    # Print audio tagging top probabilities
    print('Predice results:')
    for k in range(10):
        print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))
    #return the top 10 labels and their probabilities
    top_labels_probs = {}
    top_labels = {}
    for k in range(10):  
        label = np.array(labels)[sorted_indexes[k]]  
        prob = result_output[sorted_indexes[k]]  
        top_labels[k]= label
        top_labels_probs[k]= prob
    return top_labels, top_labels_probs
