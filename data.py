import numpy as np
import pandas as pd
import os, sys
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import glob, re
import utils
import codecs, unicodedata
import jamo

from config import ConfigArgs as args

class SpeechDataset(Dataset):
    def __init__(self, data_path, metadata, mem_mode=False):
        '''
        Args:
            data_path (str): path to dataset
            meta_path (str): path to metadata csv file
        '''
        self.data_path = data_path
        self.mem_mode = mem_mode
        self.fpaths, self.texts, self.norms = read_meta(args.speaker, os.path.join(data_path, metadata))
        if self.mem_mode:
            self.mels = [torch.tensor(np.load(os.path.join(
                self.data_path, args.mel_dir, path))) for path in self.fpaths]
        self.f0 = [torch.tensor(np.load(os.path.join(
                self.data_path, args.f0_dir, path))) for path in self.fpaths]
        
    def __getitem__(self, idx):
        text = torch.tensor(self.norms[idx], dtype=torch.long)
        # Memory mode is faster
        if not self.mem_mode:
            mel_path = os.path.join(self.data_path, args.mel_dir, self.fpaths[idx])
            mel = torch.tensor(np.load(mel_path))
        else:
            mel = self.mels[idx]
        pmel = mel
        mel = mel[::args.r]
        f0 = self.f0[idx][::args.r]
        return text, mel, pmel, f0

    def __len__(self):
        return len(self.fpaths)

def get_vocab(lang):
    if lang == 'en':
        vocab = u'''PE !,.?abcdefghijklmnopqrstuvwxyz'''
    elif lang == 'ko':    
        JAMO_LEADS = ''.join([chr(_) for _ in range(0x1100, 0x1113)])
        JAMO_VOWELS = ''.join([chr(_) for _ in range(0x1161, 0x1176)])
        JAMO_TAILS = ''.join([chr(_) for _ in range(0x11A8, 0x11C3)])
        vocab = 'PE !,.?' + JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS
    return vocab

def load_vocab_tool(lang):
    vocab = get_vocab(lang)
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def text_normalize(text, lang):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents
    text = text.lower()
    text = re.sub(u"[^{}]".format(get_vocab(lang)), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def read_meta(speaker, path):
    if speaker.lower() == 'lj':
        return read_lj_meta(path)
    elif speaker.lower() == 'kss':
        return read_kss_meta(path)

def read_lj_meta(path):
    '''
    If we use pandas instead of this function, it may not cover quotes.
    Args:
        path: metadata path
    Returns:
        fpaths, texts, norms
    '''
    char2idx, _ = load_vocab_tool('en')
    lines = codecs.open(path, 'r', 'utf-8').readlines()
    fpaths, texts, norms = [], [], []
    for line in lines:
        fname, text, norm = line.strip().split('|')
        fpath = fname + '.npy'
        text = text_normalize(text, 'en').strip() + u'E'  # ␃: EOS
        text = [char2idx[char] for char in text]
        norm = text_normalize(norm, 'en').strip() + u'E'  # ␃: EOS
        norm = [char2idx[char] for char in norm]
        fpaths.append(fpath)
        texts.append(text)
        norms.append(norm)
    return fpaths, texts, norms

def read_kss_meta(path):
    # Parse
    char2idx, _ = load_vocab_tool('ko')
    meta = pd.read_table(path, sep='|', header=None)
    meta.columns = ['fpath', 'ori', 'expanded', 'decomposed', 'duration', 'en']
    fpaths, texts = [], []
    meta.expanded = 'P' + meta.expanded + 'E'
    for fpath, text in zip(meta.fpath.values, meta.expanded.values):
        t = np.array([char2idx[ch] for ch in jamo.h2j(text)])
        f = os.path.join(os.path.basename(fpath).replace('wav', 'npy'))
        texts.append(t)
        fpaths.append(f)
    return fpaths, texts, texts

def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts, mels, mags).
    Args:
        data: list of tuple (texts, mels, mags).
            - texts: torch tensor of shape (B, Tx).
            - mels: torch tensor of shape (B, Ty/4, n_mels).
            - mags: torch tensor of shape (B, Ty, n_mags).
    Returns:
        texts: torch tensor of shape (batch_size, padded_length).
        mels: torch tensor of shape (batch_size, padded_length, n_mels).
        mels: torch tensor of shape (batch_size, padded_length, n_mags).
    """
    # Sort a data list by text length (descending order).
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    texts, mels, mags = zip(*data)

    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    mel_lengths = [len(mel) for mel in mels]
    mag_lengths = [len(mag) for mag in mags]
    # (number of mels, max_len, feature_dims)
    text_pads = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    mel_pads = torch.zeros(len(mels), max(mel_lengths), mels[0].shape[-1])
    mag_pads = torch.zeros(len(mags), max(mag_lengths), mags[0].shape[-1])
    for idx in range(len(mels)):
        text_end = text_lengths[idx]
        text_pads[idx, :text_end] = texts[idx]
        mel_end = mel_lengths[idx]
        mel_pads[idx, :mel_end] = mels[idx]
        mag_end = mag_lengths[idx]
        mag_pads[idx, :mag_end] = mags[idx]
    return text_pads, mel_pads, mag_pads

def t2m_ga_collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts, mels, mags).
    Args:
        data: list of tuple (texts).
            - texts: torch tensor of shape (B, Tx).
            - mels: torch tensor of shape (B, Ty/4, n_mels).
            - gas: torch tensor of shape (B, max_Tx, max_Ty).
            - f0: (B, Ty/4)
    Returns:
        texts: torch tensor of shape (B, padded_length).
        mels: torch tensor of shape (B, padded_length, n_mels).
        gas: torch tensor of shape (B, Tx, Ty/4)
        f0: (B, Ty/4)
    """
    # Sort a data list by text length (descending order).
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    texts, mels, pmels, f0 = zip(*data)
    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    mel_lengths = [len(mel) for mel in mels]
    pmel_lengths = [len(pmel) for pmel in pmels]
    # (number of mels, max_len, feature_dims)
    B = len(mels)
    text_pads = torch.zeros(B, max(text_lengths), dtype=torch.long)
    mel_pads = torch.zeros(B, max(mel_lengths), mels[0].shape[-1])
    pmel_pads = torch.zeros(B, max(pmel_lengths), pmels[0].shape[-1])
    ga_pads = torch.zeros(B, max(text_lengths), max(mel_lengths))
    f0_pads = torch.zeros(B, max(mel_lengths))
    for idx in range(len(mels)):
        text_end = text_lengths[idx]
        text_pads[idx, :text_end] = texts[idx]
        mel_end = mel_lengths[idx]
        mel_pads[idx, :mel_end] = mels[idx]
        pmel_end = pmel_lengths[idx]
        pmel_pads[idx, :pmel_end] = pmels[idx]
        ga_pads[idx] = torch.tensor(utils.get_guided_attention(text_end, mel_end, ga_pads.size(1), ga_pads.size(2)))
        f0_pads[idx, :mel_end] = f0[idx]
    return text_pads, mel_pads, pmel_pads, ga_pads, f0_pads


class TextDataset(Dataset):
    def __init__(self, text_path, lang, ref_path):
        '''
        Args:
            text path (str): path to text set
        '''
        self.texts = read_text(text_path, lang)
        self.refs = read_f0(ref_path)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        f0 = torch.tensor(self.refs[idx][::args.r])
        return text, f0

    def __len__(self):
        return len(self.texts)


def read_text(path, lang):
    '''
    If we use pandas instead of this function, it may not cover quotes.
    Args:
        path: metadata path
    Returns:
        fpaths, texts, norms
    '''
    char2idx, _ = load_vocab_tool(lang)
    lines = codecs.open(path, 'r', 'utf-8').readlines()
    texts = []
    for line in lines:
        text = 'P' + text_normalize(line, lang).strip() + 'E'  # ␃: EOS
        text = [char2idx[char] for char in jamo.h2j(text)]
        texts.append(text)
    return texts

def read_f0(ref_dir):
    paths = sorted(glob.glob(os.path.join(ref_dir, '*.wav')))
    f0_lst = []
    for path in paths:
        wav, sr = utils.load_audio(path)
        f0 = utils.get_f0(wav, sr, fmin=60, fmax=400)
        f0_lst.append(f0)
    return f0_lst

def synth_collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts, mels, mags).
    Args:
        data: list of tuple (texts,).
            - texts: torch tensor of shape (B, Tx).
    Returns:
        texts: torch tensor of shape (batch_size, padded_length).
    """
    texts, f0 = zip(*data)

    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    f0_lengths = [len(f) for f in f0]
    # (number of mels, max_len, feature_dims)
    text_pads = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    f0_pads = torch.zeros(len(f0), max(f0_lengths))
    for idx in range(len(texts)):
        text_pads[idx, :text_lengths[idx]] = texts[idx]
        f0_pads[idx, :f0_lengths[idx]] = f0[idx]
    return text_pads, f0_pads, None
