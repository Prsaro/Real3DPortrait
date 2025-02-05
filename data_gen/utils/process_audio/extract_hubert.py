from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import numpy as np
import torch
import os
import pickle
import sys
import tqdm
sys.path.append("/zjs/")
from utils.commons.hparams import set_hparams, hparams
from utils.commons.os_utils import multiprocess_glob
from utils.commons.multiprocess_utils import multiprocess_run_tqdm


wav2vec2_processor = None
hubert_model = None


def get_hubert_from_16k_wav(wav_16k_name):
    speech_16k, _ = sf.read(wav_16k_name)
    hubert = get_hubert_from_16k_speech(speech_16k)
    return hubert

@torch.no_grad()
def get_hubert_from_16k_speech(speech, device="cuda"):
    global hubert_model, wav2vec2_processor
    local_path = '/zjs/models--facebook--hubert-large-ls960-ft/snapshots/ece5fabbf034c1073acae96d5401b25be96709d8'
    if hubert_model is None:
        print("Loading the HuBERT Model...")
        if os.path.exists(local_path):
            hubert_model = HubertModel.from_pretrained(local_path)
        else:
            hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = hubert_model.to(device)
    if wav2vec2_processor is None:
        print("Loading the Wav2Vec2 Processor...")
        if os.path.exists(local_path):
            wav2vec2_processor = Wav2Vec2Processor.from_pretrained(local_path)
        else:
            wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

    if speech.ndim ==2:
        speech = speech[:, 0] # [T, 2] ==> [T,]
    
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all

    if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
        hidden_states = hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]

    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T: # if skipping the last short 
        ret = torch.cat([ret, ret[:, -1:, :].repeat([1,expected_T-ret.shape[0],1])], dim=1)
    else:
        ret = ret[:expected_T]

    return ret

def out_exist_job(vid_name):
    out_name = vid_name.replace("/video/", "/hubert/").replace(".mp4","_hubert.npy") 
    if os.path.exists(out_name):
        return None
    else:
        return vid_name
    
def get_todo_vid_names(vid_names):
    if len(vid_names) == 1: # nerf
        return vid_names
    todo_vid_names = []
    for i, res in multiprocess_run_tqdm(out_exist_job, vid_names, num_workers=128):
        if res is not None:
            todo_vid_names.append(res)
    return todo_vid_names

def save_file(name, content):
    with open(name, "wb") as f:
        pickle.dump(content, f) 
        
def load_file(name):
    with open(name, "rb") as f:
        content = pickle.load(f)
    return content


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--video_id', type=str, default='May', help='')
    parser.add_argument("--vid_dir", default='/data/cleaned_data/video')
    parser.add_argument("--ds_name", default='cleaned_data')
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--reset", action="store_true")
    # parser.add_argument("--load_names", action="store_true")
    parser.add_argument("--load_names", default=True, type=bool)
    args = parser.parse_args()
    ### Process Single Long Audio for NeRF dataset
    person_id = args.video_id
    vid_dir = args.vid_dir
    ds_name = args.ds_name
    load_names = args.load_names
    if ds_name.lower() == 'nerf':
        wav_16k_name = f"data/processed/videos/{person_id}/aud.wav"
        hubert_npy_name = f"data/processed/videos/{person_id}/aud_hubert.npy"
        speech_16k, _ = sf.read(wav_16k_name)
        hubert_hidden = get_hubert_from_16k_speech(speech_16k)
        np.save(hubert_npy_name, hubert_hidden.detach().numpy())
        print(f"Saved at {hubert_npy_name}")
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
        if not load_names:
            print(f"saving vid names to {vid_names_path}")
            save_file(vid_names_path, vid_names)
    
    aud_names = [video_name.replace("/video/", "/audio/").replace(".mp4",".wav") for video_name in vid_names]
    
    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process -1
        num_samples_per_process = len(vid_names) // total_process
        if process_id == total_process:
            aud_names = aud_names[process_id * num_samples_per_process : ]
            vid_names = vid_names[process_id * num_samples_per_process : ]
        else:
            aud_names = aud_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
            vid_names = vid_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]
    
    if not args.reset:
        vid_names = get_todo_vid_names(vid_names)
    print(f"todo videos number: {len(vid_names)}")

    failed_img_names = []
    for i in tqdm.trange(len(vid_names), desc=f"process {process_id}: extracting hubert ..."):
        vid_name = vid_names[i]
        aud_name = aud_names[i]
        try:
            hubert_hidden = get_hubert_from_16k_wav(aud_name)
            out_name = vid_name.replace("/video/", "/hubert/").replace(".mp4","_hubert.npy")
            os.makedirs(os.path.dirname(out_name), exist_ok=True)
            np.save(out_name, hubert_hidden.detach().numpy())
        except Exception as e:
            failed_img_names.append(vid_name)
            print(f"Failed to process {vid_name}, {e}")
        print(f"finished {i + 1} / {len(vid_names)} = {(i + 1) / len(vid_names):.4f}, failed {len(failed_img_names)} / {i + 1} = {len(failed_img_names) / (i + 1):.4f}")
        sys.stdout.flush()
    print(f"all failed image names: {failed_img_names}")
    print(f"All finished!")


