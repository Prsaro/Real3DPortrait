import numpy as np
import torch
import glob
import os
import random
import tqdm
import librosa
import parselmouth
import pickle
import sys
sys.path.append("/zjs/")

from utils.commons.pitch_utils import f0_to_coarse
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.commons.os_utils import multiprocess_glob
from utils.audio.io import save_wav

from moviepy.editor import VideoFileClip
# from moviepy import VideoFileClip
from utils.commons.hparams import hparams, set_hparams

def resample_wav(wav_name, out_name, sr=16000):
    wav_raw, sr = librosa.core.load(wav_name, sr=sr)
    save_wav(wav_raw, out_name, sr)
    
def split_wav(mp4_name, wav_name=None):
    if wav_name is None:
        wav_name = mp4_name.replace(".mp4", ".wav").replace("/video/", "/audio/")
    if os.path.exists(wav_name):
        return wav_name
    os.makedirs(os.path.dirname(wav_name), exist_ok=True)
    
    video = VideoFileClip(mp4_name,verbose=False)
    dur = video.duration
    audio = video.audio 
    assert audio is not None
    audio.write_audiofile(wav_name,fps=16000,verbose=False,logger=None)
    return wav_name

def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2

def extract_mel_from_fname(wav_path,
                      fft_size=512,
                      hop_size=320,
                      win_length=512,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-6,
                      sample_rate=16000,
                      min_level_db=-100):
    if isinstance(wav_path, str):
        wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, center=False)
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel = mel_basis @ spc

    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    mel = mel.T

    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)

    return wav.T, mel

def extract_f0_from_wav_and_mel(wav, mel,
                        hop_size=320,
                        audio_sample_rate=16000,
                        ):
    time_step = hop_size / audio_sample_rate * 1000
    f0_min = 80
    f0_max = 750
    f0 = parselmouth.Sound(wav, audio_sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 8
    if delta_l > 0:
        f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0)
    return f0, pitch_coarse


def extract_mel_f0_from_fname(wav_name=None, out_name=None):
    try:
        out_name = wav_name.replace(".wav", "_mel_f0.npy").replace("/audio/", "/mel_f0/")
        os.makedirs(os.path.dirname(out_name), exist_ok=True)

        wav, mel = extract_mel_from_fname(wav_name)
        f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
        out_dict = {
            "mel": mel, # [T, 80]
            "f0": f0,
        }
        np.save(out_name, out_dict)
    except Exception as e:
        print(e)

def extract_mel_f0_from_video_name(mp4_name, wav_name=None, out_name=None):
    if mp4_name.endswith(".mp4"):
        wav_name = split_wav(mp4_name, wav_name)
        if out_name is None:
            out_name = mp4_name.replace(".mp4", "_mel_f0.npy").replace("/video/", "/mel_f0/")
    elif mp4_name.endswith(".wav"):
        wav_name = mp4_name
        if out_name is None:
            out_name = mp4_name.replace(".wav", "_mel_f0.npy").replace("/audio/", "/mel_f0/")

    os.makedirs(os.path.dirname(out_name), exist_ok=True)

    wav, mel = extract_mel_from_fname(wav_name)

    f0, f0_coarse = extract_f0_from_wav_and_mel(wav, mel)
    out_dict = {
        "mel": mel, # [T, 80]
        "f0": f0,
    }
    np.save(out_name, out_dict)

def load_file(name):
    with open(name, "rb") as f:
        content = pickle.load(f)
    return content

def save_file(name, content):
    with open(name, "wb") as f:
        pickle.dump(content, f) 

def out_exist_job(vid_name):
    out_name = vid_name.replace("/video/", "/audio/").replace(".mp4",".wav") 
    lms_name = vid_name.replace("/video/", "/mel_f0/").replace(".mp4","_mel_f0.npy") 
    if os.path.exists(out_name) and os.path.exists(lms_name):
        return None
    else:
        return vid_name

def get_todo_vid_names(vid_names):
    if len(vid_names) == 1:
        return vid_names
    todo_vid_names = []
    for i, res in multiprocess_run_tqdm(out_exist_job, vid_names, num_workers=16):
        if res is not None:
            todo_vid_names.append(res)
    return todo_vid_names


if __name__ == '__main__':
    from argparse import ArgumentParser
    import glob, tqdm
    parser = ArgumentParser()
    parser.add_argument('--video_id', type=str, default='May', help='')
    parser.add_argument("--vid_dir", default='/data/cleaned_data/video/')
    parser.add_argument("--ds_name", default='cleaned_data') # 'nerf' | 'CelebV-HQ' | 'TH1KH_512' | etc
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--load_names", action="store_true")
    parser.add_argument("--reset", action="store_true")

    args = parser.parse_args()
    ds_name = args.ds_name
    vid_dir = args.vid_dir
    load_names = args.load_names
    
    ### Process Single Long Audio for NeRF dataset
    # person_id = args.video_id

    # wav_16k_name = f"data/processed/videos/{person_id}/aud.wav"
    # out_name = f"data/processed/videos/{person_id}/aud_mel_f0.npy"
    # extract_mel_f0_from_video_name(wav_16k_name, out_name)
    # print(f"Saved at {out_name}")
    ## Process dataset videos
    print(f"args {args}")
    if ds_name.lower() == 'nerf':
        person_id = args.video_id

        wav_16k_name = f"data/processed/videos/{person_id}/aud.wav"
        out_name = f"data/processed/videos/{person_id}/aud_mel_f0.npy"
        extract_mel_f0_from_video_name(wav_16k_name, out_name)
        print(f"Saved at {out_name}")
    else:
        if ds_name in ['lrs3_trainval']:
            vid_name_pattern = os.path.join(vid_dir, "*/*.mp4")
        elif ds_name in ['TH1KH_512', 'CelebV-HQ', 'cleaned_data']:
            vid_name_pattern = os.path.join(vid_dir, "*.mp4")
        elif ds_name in ['lrs2', 'lrs3', 'voxceleb2', 'CMLR']:
            vid_name_pattern = os.path.join(vid_dir, "*/*/*.mp4")
        elif ds_name in ["RAVDESS", 'VFHQ']:
            vid_name_pattern = os.path.join(vid_dir, "*/*/*/*.mp4")
        else:
            raise NotImplementedError()
        
        vid_names_path = os.path.join(vid_dir, "vid_names.pkl")
        if os.path.exists(vid_names_path) and load_names:
            print(f"loading vid names from {vid_names_path}")
            vid_names = load_file(vid_names_path)
        else:
            vid_names = multiprocess_glob(vid_name_pattern)
        vid_names = sorted(vid_names)
        print(f"saving vid names to {vid_names_path}")
        save_file(vid_names_path, vid_names)
    
    print(vid_names[:10])
    random.seed(args.seed)
    random.shuffle(vid_names)

    process_id = args.process_id
    total_process = args.total_process
    ## Split the dataset into total_process parts
    if total_process > 1:
        assert process_id <= total_process -1
        num_samples_per_process = len(vid_names) // total_process
        if process_id == total_process:
            vid_names = vid_names[process_id * num_samples_per_process : ]
        else:
            vid_names = vid_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
    
    if not args.reset:
        vid_names = get_todo_vid_names(vid_names)

    failed_img_names = []
    for i in tqdm.trange(len(vid_names), desc=f"process {process_id}: extracting mel f0 ..."):
        try:
            img_name = vid_names[i]
            extract_mel_f0_from_video_name(img_name)
        except Exception as e:
            print(f"Failed {vid_names[i]}", e)
            failed_img_names.append(vid_names[i])
        print(f"finished {i + 1} / {len(vid_names)} = {(i + 1) / len(vid_names):.4f}, failed {len(failed_img_names)} / {i + 1} = {len(failed_img_names) / (i + 1):.4f}")
        sys.stdout.flush()
    # print(f"all failed image names: {failed_img_names}")
    print(f"All finished!")
