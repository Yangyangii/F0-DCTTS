from config import ConfigArgs as args
import os, sys, glob
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
from models import DCTTS
from data import TextDataset, synth_collate_fn, load_vocab_tool
import utils
from scipy.io.wavfile import write


def synthesize(model, data_loader, batch_size=100):
    # idx2char = load_vocab_tool(args.lang)[-1]
    with torch.no_grad():
        for step, (texts, f0, _) in tqdm(enumerate(data_loader), total=len(data_loader), ncols=70):
            texts, f0 = texts.to(DEVICE), f0.to(DEVICE)
            f0[f0 > 0.01] = f0[f0 > 0.01] + args.f0_factor
            # prev_mel_hats = torch.zeros([len(texts), args.max_Ty, args.n_mels]).to(DEVICE)
            prev_mel_hats = torch.zeros([len(texts), f0.shape[1], args.n_mels]).to(DEVICE)
            mels, pmels, A = model.synthesize(texts, prev_mel_hats, f0)
            # alignments = A.cpu().detach().numpy()
            # visual_texts = texts.cpu().detach().numpy()
            for idx in range(len(pmels)):
                fname = step*batch_size + idx
                np.save(os.path.join(args.sampledir, 'mel-{:02d}.npy'.format(fname)), pmels[idx].cpu().numpy())
                np.save(os.path.join(args.sampledir, 'f0/{:02d}.npy'.format(fname)), f0[idx].cpu().numpy())
                # text = [idx2char[ch] for ch in visual_texts[idx]]
                # utils.plot_att(alignments[idx], text, args.global_step, path=os.path.join(args.sampledir, 'A'), name='{:02d}.png'.format(fname))
    return None

def main():
    testset = TextDataset(args.testset, args.lang, args.ref_path)
    test_loader = DataLoader(dataset=testset, batch_size=args.test_batch, drop_last=False,
                             shuffle=False, collate_fn=synth_collate_fn, pin_memory=True)

    model = DCTTS(args).to(DEVICE)
    
    ckpt = sorted(glob.glob(os.path.join(args.logdir, args.model_name, '{}-*k.pth'.format(args.model_name))))
    state = torch.load(ckpt[-1])
    model.load_state_dict(state['model'])
    args.global_step = state['global_step']

    print('All of models are loaded.')

    model.eval()
    
    if not os.path.exists(os.path.join(args.sampledir, 'A')):
        os.makedirs(os.path.join(args.sampledir, 'A'))
        os.makedirs(os.path.join(args.sampledir, 'f0'))
    synthesize(model, test_loader, args.test_batch)

if __name__ == '__main__':
    gpu_id = int(sys.argv[1])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
