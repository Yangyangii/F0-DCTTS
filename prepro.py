# -*- coding: utf-8 -*-
from config import ConfigArgs as args
from utils import prepro_guided_attention
import utils
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import codecs
import data


NUM_JOBS = 8

def job(fpath):
    wav_path = os.path.join(args.data_path, 'wavs', fpath.replace('npy', 'wav'))

    wav, sr = utils.load_audio(wav_path)
    mel = utils.get_mel_spectrogram(wav, sr)
    # ga = prepro_guided_attention(len(text), len(mel), g=args.g)
    f0 = utils.get_f0(wav, sr, fmin=60, fmax=400, spec_len=mel.shape[0])
    np.save(os.path.join(args.data_path, args.mel_dir, fpath), mel)
    np.save(os.path.join(args.data_path, args.f0_dir, fpath), f0)
    return None

def prepro_signal():
    print('Preprocessing signal')
    # Load data
    if args.speaker.lower() == 'lj':
        fpaths, _, _ = data.read_lj_meta(os.path.join(args.data_path, args.meta))
    elif args.speaker.lower() == 'kss':
        fpaths, _, _ = data.read_kss_meta(os.path.join(args.data_path, args.meta))

    # Creates folders
    os.makedirs(os.path.join(args.data_path, args.mel_dir), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, args.f0_dir), exist_ok=True)

    # Creates pool
    p = Pool(NUM_JOBS)

    total_files = len(fpaths)
    with tqdm(total=total_files) as pbar:
        for _ in tqdm(p.imap_unordered(job, fpaths)):
            pbar.update()

def prepro_meta():
    ## train(95%)/test(5%) split for metadata
    print('Preprocessing meta')
    # Parse
    transcript = os.path.join(args.data_path, args.meta)
    train_transcript = os.path.join(args.data_path, 'meta-train.csv')
    test_transcript = os.path.join(args.data_path, 'meta-eval.csv')

    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    train_f = codecs.open(train_transcript, 'w', 'utf-8')
    test_f = codecs.open(test_transcript, 'w', 'utf-8')

    np.random.seed(0)
    n_data = len(lines)
    n_tests = int(n_data*0.01)
    test_indices = np.random.choice(range(n_data), n_tests, replace=False)
    # test_idx = np.load('lj_eval_idx.npy')

    for idx, line in enumerate(lines):
        if idx in test_indices:
            test_f.write(line)
        else:
            train_f.write(line)
    print('# of train set: {}, # of test set: {}'.format(1+idx-n_tests, n_tests))
    print('Complete')

if __name__ == '__main__':
    prepro_signal()
    prepro_meta()
