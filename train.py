from config import ConfigArgs as args
import os, sys, glob
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

import numpy as np
import pandas as pd
from collections import deque
from models import DCTTS
from data import SpeechDataset, t2m_ga_collate_fn, load_vocab_tool
from utils import att2img, plot_att, lr_policy

def main():
    model = DCTTS(args).to(DEVICE)
    print('Model {} is working...'.format(args.model_name))
    print('{} threads are used...'.format(torch.get_num_threads()))
    ckpt_dir = os.path.join(args.logdir, args.model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = MultiStepLR(optimizer, milestones=[50000, 150000, 300000], gamma=0.5) #
    scheduler = LambdaLR(optimizer, lr_policy)

    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.join(ckpt_dir, 'A', 'train'))
        if args.pretrained_path is not None:
            print('Train with pretrained model {}'.format(args.pretrained_path))
            state = torch.load(args.pretrained_path)
            model.custom_load_state_dict(state['model'])
    else:
        print('Already exists. Retrain the model.')
        ckpt = sorted(glob.glob(os.path.join(ckpt_dir, '*k.pth.tar')))[-1]
        state = torch.load(ckpt)
        model.load_state_dict(state['model'])
        args.global_step = state['global_step']
        optimizer.load_state_dict(state['optimizer'])
        # scheduler.load_state_dict(state['scheduler'])

    # model = torch.nn.DataParallel(model, device_ids=list(range(args.no_gpu))).to(DEVICE)

    dataset = SpeechDataset(args.data_path, args.meta_train, mem_mode=args.mem_mode)
    validset = SpeechDataset(args.data_path, args.meta_eval, mem_mode=args.mem_mode)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=t2m_ga_collate_fn,
                             drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=validset, batch_size=args.test_batch,
                              shuffle=False, collate_fn=t2m_ga_collate_fn, pin_memory=True)
    
    writer = SummaryWriter(ckpt_dir)
    train(model, data_loader, valid_loader, optimizer, scheduler,
          batch_size=args.batch_size, ckpt_dir=ckpt_dir, writer=writer)
    return None

def train(model, data_loader, valid_loader, optimizer, scheduler, batch_size=32, ckpt_dir=None, writer=None, mode='1'):
    epochs = 0
    global_step = args.global_step
    l1_criterion = nn.L1Loss() # default average
    bd_criterion = nn.BCELoss()
    GO_frames = torch.zeros([batch_size, 1, args.n_mels]).to(DEVICE) # (N, Ty/r, n_mels)
    idx2char = load_vocab_tool(args.lang)[-1]
    while global_step < args.max_step:
        epoch_loss = 0
        for step, (texts, mels, pmels, gas, f0) in tqdm(enumerate(data_loader), total=len(data_loader), unit='B', ncols=70, leave=False):
            optimizer.zero_grad()
            texts, mels, pmels, gas, f0 = texts.to(DEVICE), mels.to(DEVICE), pmels.to(DEVICE), gas.to(DEVICE), f0.to(DEVICE)
            prev_mels = torch.cat((GO_frames, mels[:, :-1, :]), 1)
            mels_hat, pmels_hat, A = model(texts, prev_mels, f0)  # mels_hat: (N, Ty/r, n_mels), A: (N, Tx, Ty/r)
            
            mel_loss = l1_criterion(mels_hat, mels)
            bd_loss = bd_criterion(mels_hat, mels)
            pmel_loss = l1_criterion(pmels_hat, pmels)
            att_loss = torch.mean(A*gas)
            loss = mel_loss + bd_loss + att_loss + pmel_loss
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            if args.lr_decay:
                scheduler.step()
            if global_step % args.save_term == 0:
                model.eval()
                val_loss = evaluate(model, valid_loader, writer, global_step, args.test_batch)
                save_model(model, optimizer, scheduler, val_loss, global_step, ckpt_dir) # save best 5 models
                model.train()
            global_step += 1
        if args.log_mode:
            # Summary
            writer.add_scalar('train/mel_loss', mel_loss.item(), global_step)
            writer.add_scalar('train/pmel_loss', pmel_loss.item(), global_step)
            writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step)
            alignment = A[0:1].clone().cpu().detach().numpy()
            guided_att = gas[0:1].clone().cpu().detach().numpy()
            writer.add_image('train/alignments', att2img(alignment), global_step) # (Tx, Ty)
            writer.add_image('train/guided_att', att2img(guided_att), global_step) # (Tx, Ty)
            writer.add_scalar('train/ga_loss', att_loss, global_step)
            # text = texts[0].cpu().detach().numpy()
            # text = [idx2char[ch] for ch in text]
            # plot_att(alignment[0], text, global_step, path=os.path.join(args.logdir, type(model).__name__, 'A', 'train'))
            mel_hat = mels_hat[0:1].transpose(1,2)
            mel = mels[0:1].transpose(1, 2)
            writer.add_image('train/mel_hat', mel_hat, global_step)
            writer.add_image('train/mel', mel, global_step)
            # print('Training Loss: {}'.format(avg_loss))
        epochs += 1
    print('Training complete')

def evaluate(model, data_loader, writer, global_step, batch_size=100):
    valid_loss = 0.
    A = None 
    l1_loss = nn.L1Loss()
    with torch.no_grad():
        mel_sum_loss = 0.
        pmel_sum_loss = 0.
        for step, (texts, mels, pmels, gas, f0) in enumerate(data_loader):
            texts, mels, pmels, gas, f0 = texts.to(DEVICE), mels.to(DEVICE), pmels.to(DEVICE), gas.to(DEVICE), f0.to(DEVICE)
            GO_frames = torch.zeros([mels.shape[0], 1, args.n_mels]).to(DEVICE) # (N, Ty/r, n_mels)
            prev_mels = torch.cat((GO_frames, mels[:, :-1, :]), 1)

            mels_hat, pmels_hat, A = model(texts, prev_mels, f0)  # mels_hat: (N, Ty/r, n_mels), A: (N, Tx, Ty/r)
            mel_loss = l1_loss(mels_hat, mels)
            pmel_loss = l1_loss(pmels_hat, pmels)
            att_loss = torch.mean(A*gas)
            mel_sum_loss += mel_loss.item()
            pmel_sum_loss += pmel_loss.item()

        mel_avg_loss = mel_sum_loss / (len(data_loader))
        pmel_avg_loss = pmel_sum_loss / (len(data_loader))
        writer.add_scalar('eval/mel_loss', mel_avg_loss, global_step)
        writer.add_scalar('eval/pmel_loss', pmel_avg_loss, global_step)
        writer.add_scalar('eval/ga_loss', att_loss, global_step)
        alignment = A[0:1].clone().cpu().detach().numpy()
        guided_att = gas[0:1].clone().cpu().detach().numpy()
        writer.add_image('eval/alignments', att2img(alignment), global_step) # (Tx, Ty)
        writer.add_image('eval/guided_att', att2img(guided_att), global_step) # (Tx, Ty)
        # text = texts[0].cpu().detach().numpy()
        # text = [load_vocab_tool(args.lang)[-1][ch] for ch in text]
        # plot_att(alignment[0], text, global_step, path=os.path.join(args.logdir, args.model_name, 'A'))
        writer.add_image('eval/mel_hat', mels_hat[0:1].transpose(1,2), global_step)
        writer.add_image('eval/mel', mels[0:1].transpose(1,2), global_step)
        writer.add_image('eval/pmel_hat', pmels_hat[0:1].transpose(1,2), global_step)
        writer.add_image('eval/pmel', pmels[0:1].transpose(1,2), global_step)
    return mel_avg_loss

def save_model(model, optimizer, scheduler, val_loss, global_step, ckpt_dir):
    fname = '{}-{:03d}k.pth'.format(args.model_name, global_step//1000)
    state = {
        'global_step': global_step,
        'model': model.state_dict(),
        'loss': val_loss,
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
    }
    torch.save(state, os.path.join(ckpt_dir, fname))

if __name__ == '__main__':
    gpu_id = int(sys.argv[1])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set random seem for reproducibility
    seed = 999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    main()
    
