# F0-DCTTS (F0 Deep Convolutional TTS)

## Description
- DCTTS with F0

## Prerequisite
- python 3.7
- pytorch 1.3
- pysptk
- librosa, scipy, tqdm, tensorboardX

## Dataset
- [LJ Speech 1.1](https://keithito.com/LJ-Speech-Dataset/)
- [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset), Korean female single speaker speech dataset.

## Samples
- [samples](https://yangyangii.github.io/2020/03/11/F0-DCTTS.html)

## Usage
1. Download the above dataset and modify the path in config.py. And then run the below command.
    ```
    python prepro.py
    ```

2. The baseline DCTTS needs to train 100k+ steps
    ```
    python train.py <gpu_id>
    ```

3. After training the baseline, you can train F0-DCTTS. Change "f0_mode=True" and "pretrained_path=..." in config.py. And then run the below command one more.
    ```
    python train.py <gpu_id>
    ```

4. You can synthesize some speech with f0. You can control using "f0_factor=..." in config.py. 
    ```
    python synthesize.py <gpu_id>
    ```


## Notes
- This method is easy and simple, but verrrrrrrrrry naive approach.

