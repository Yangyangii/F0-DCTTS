from config import ConfigArgs as args
import librosa
import numpy as np
import os, sys
from scipy import signal
import copy
import torch
import pysptk
from scipy.io import wavfile
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

def normalize(x, xmin=None, xmax=None):
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    x_norm = (x - xmin) / (xmax - xmin)
    return x_norm

def get_f0(wav, sr, fmin=60, fmax=400, spec_len=None):
    if wav.dtype == np.float32:
        wav = wav * 32768.0
    f0 = pysptk.rapt(wav, fs=sr, hopsize=args.hop_length, min=fmin, max=fmax, otype='f0')
    f0norm = normalize(f0, xmin=0, xmax=fmax)
    if spec_len is not None and spec_len != f0.shape[0]:
        n_pad = spec_len - f0.shape[0]
        f0norm = np.pad(f0norm, [0, n_pad]) # pad into spec length
    f0norm = padding_reduction(f0norm, r=args.r)
    # f0norm = f0norm[::args.r]
    return f0norm

def get_mel_spectrogram(wav, sr):
    # STFT
    linear = librosa.stft(y=wav,
                          n_fft=args.n_fft,
                          hop_length=args.hop_length,
                          win_length=args.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, args.n_fft, args.n_mels, args.fmin, args.fmax)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mel = padding_reduction(mel, r=args.r)
    mel = np.log(np.clip(mel, 1e-5, 1e+5))
    mel_norm = normalize(mel, np.log(1e-5), 3.0)
    # rmel = mel[::args.r, :]
    return mel_norm

def load_audio(fpath, sr=22050):
    wav, sr = librosa.load(fpath, sr=sr)
    ## Pre-processing
    wav = wav / np.abs(wav).max() * 0.99
    wav, _ = librosa.effects.trim(wav)
    return wav, sr

def padding_reduction(x, r=4, pad_mode='constant'):
    # Padding
    t = x.shape[0]
    n_paddings = r - (t % r) if t % r != 0 else 0  # for reduction
    if x.ndim == 2:
        x = np.pad(x, [[0, n_paddings], [0, 0]], mode=pad_mode)
    else:
        x = np.pad(x, [0, n_paddings], mode=pad_mode)
    return x

def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-normalize
    mag = (np.clip(mag, 0, 1) * (args.max_db-args.min_db)) + args.min_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**args.power)

    # de-preemphasis
    # wav = signal.lfilter([1], [1, -args.preemph], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''
    Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(args.gl_iter):
        X_t = librosa.istft(X_best, args.hop_length, win_length=args.win_length, window="hann")
        est = librosa.stft(X_t, args.n_fft, args.hop_length, win_length=args.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = librosa.istft(X_best, args.hop_length, win_length=args.win_length, window="hann")
    y = np.real(X_t)
    return y

def att2img(A):
    '''
    Args:
        A: (1, Tx, Ty) Tensor
    '''
    for i in range(A.shape[-1]):
        att = A[0, :, i]
        local_min, local_max = att.min(), att.max()
        A[0, :, i] = (att-local_min)/local_max
    return A


def plot_att(A, text, global_step, path='.', name=None):
    '''
    Args:
        A: (Tx, Ty) numpy array
        text: (Tx,) list
        global_step: scalar
    '''
    plt.rcParams["font.family"] = 'nanummyeongjo'
    fig, ax = plt.subplots(figsize=(25, 25))
    im = ax.imshow(A)
    fig.colorbar(im, fraction=0.035, pad=0.02)
    fig.suptitle('{} Steps'.format(global_step), fontsize=30)
    plt.ylabel('Text', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plt.yticks(np.arange(len(text)), text)
    if name is not None:
        plt.savefig(os.path.join(path, name), format='png')
    else:
        plt.savefig(os.path.join(
            path, 'A-{}.png'.format(global_step)), format='png')
    plt.close(fig)

def prepro_guided_attention(N, T, g=0.2):
    W = np.zeros([N, T], dtype=np.float32)
    for tx in range(args.max_Tx):
        for ty in range(args.max_Ty):
            if ty <= T:
                W[tx, ty] = 1.0 - np.exp(-0.5 * (ty/T - tx/N)**2 / g**2)
            else:
                W[tx, ty] = 1.0 - np.exp(-0.5 * ((N-1)/N - tx/N)**2 / (g/2)**2) # forcing more at end step
    return W

def get_guided_attention(Tenc, Tdec, Tenc_max, Tdec_max, g=0.2):
    # W = np.zeros([Tenc_max, Tdec_max])
    te, td = np.arange(Tenc_max), np.arange(Tdec_max)
    te_mat = np.expand_dims(te/Tenc, 1).repeat(Tdec_max, 1)
    td_mat = np.expand_dims(td/Tdec, 0).repeat(Tenc_max, 0)
    mat_diag = 1.0 - np.exp(-0.5 * (td_mat - te_mat)**2 / g**2)
    mat_end = 1.0 - np.exp(-0.5*((Tenc-1)/Tenc - te_mat)**2 / (g/2)**2)
    W = np.concatenate([mat_diag[:, :Tdec], mat_end[:, Tdec:]], axis=1)
    return W

def lr_policy(step):
    """
    warm up learning rate function
    :param step:
    Returns:
        :updated learning rate: scalar.
    """
    return args.warm_up_steps**0.5 * np.minimum((step+1) * args.warm_up_steps**-1.5, (step+1)**-0.5)