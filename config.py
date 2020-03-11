
class ConfigArgs:
    f0_mode = False
    pretrained_path = 'logs/DCTTS/DCTTS-200k.pth'
    model_name = 'F0-DCTTS' if f0_mode else 'DCTTS'

    speaker = 'kss'
    lang = 'en' if speaker == 'lj' else 'ko'
    data_path = '/home/yangyangii/data/kss'
    mel_dir = 'mels'
    f0_dir = 'f0'
    meta = 'metadata.csv' if speaker == 'lj' else 'transcript.v.1.3.txt'
    meta_train = 'meta-train.csv'
    meta_eval = 'meta-eval.csv'
    testset = 'ko_sents.txt'
    ref_path = 'refs'
    logdir = 'logs'
    sampledir = 'samples'
    testdir = 'tests'
    prepro = True
    mem_mode= True
    log_mode = True
    save_term = 5000
    n_workers = 8
    global_step = 0

    f0_factor = 0.3

    sr = 22050 # sampling rate
    n_fft = 1024
    n_mels = 80
    fmin = 0
    fmax = None
    hop_length = 256
    win_length = 1024
    r = 4  # reduction factor. mel/4
    g = 0.4

    batch_size = 32
    test_batch = 8 # for test
    max_step = 1000000
    begin_gan = 30000
    n_critic = 1
    lr_decay = True
    lr = 0.001
    warm_up_steps = 8000
    # lr_decay_step = 50000 # actually not decayed per this step
    Ce = 128  # for text embedding and encoding
    Cx = 256 # for text embedding and encoding
    Cy = 256 # for audio encoding
    Cs = 512 # for SSRN
    drop_rate = 0.05

    max_Tx = 188
    max_Ty = 250

