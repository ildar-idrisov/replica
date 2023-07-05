#
# python3 run.py --input_file shared/orlov.mp4 --output_file res.mp4
#

import numpy as np
import torch
import sys
import requests
import argparse
import subprocess
import os
# voice clonning
#sys.path.insert(0, "Wav2Lip") ### TODO: move to Dockerfile
sys.path.insert(0, "bark-with-voice-clone") ### TODO: move to Dockerfile
from bark.generation import load_codec_model, generate_text_semantic, SAMPLE_RATE
from hubert.hubert_manager import HuBERTManager
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
from encodec.utils import convert_audio
# voice synthesis
from bark.api import generate_audio
#from transformers import BertTokenizer
from bark.generation import preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
# voice cleaning
from df.enhance import enhance, init_df, load_audio, save_audio
# automatic speach recognition
import whisper
# wirking with audio
import torchaudio
import soundfile as sf
from scipy.io.wavfile import write as write_wav
import librosa

if not os.path.exists("temp"):
    os.makedirs("temp")

def voice_clonning_setup_bark(device):
    codec_model = load_codec_model(use_gpu=True if device == "cuda" else False)
    return codec_model

def voice_clonning_download_hubert(lang):
    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed()

    if (lang == "eng"):
        tokenizer_file = "tokenizer_eng.pth"
        hubert_manager.make_sure_tokenizer_installed(model = "quantifier_hubert_base_ls960_14.pth", repo = "GitMylo/bark-voice-cloning", local_file = tokenizer_file)
    elif (lang == "pol"):
        tokenizer_file = "tokenizer_pol.pth"
        hubert_manager.make_sure_tokenizer_installed(model = "polish-HuBERT-quantizer_8_epoch.pth", repo = "Hobis/bark-voice-cloning-polish-HuBERT-quantizer", local_file = tokenizer_file)
    elif (lang == "ger"):
        tokenizer_file = "tokenizer_ger.pth"
        hubert_manager.make_sure_tokenizer_installed(model = "german-HuBERT-quantizer_14_epoch.pth", repo = "CountFloyd/bark-voice-cloning-german-HuBERT-quantizer", local_file = tokenizer_file)
    return tokenizer_file

### TODO: change data/models/hubert/ to models/hubert/ here and in bark-with-voice-clone/hubert/hubert_manager.py
def voice_clonning_setup_hubert(device):
    hubert_model = CustomHubert(checkpoint_path="data/models/hubert/hubert.pt").to(device)
    return hubert_model

def voice_clonning_setup_tokenizer(device, tokenizer_file):
    tokenizer = CustomTokenizer.load_from_checkpoint("data/models/hubert/" + tokenizer_file).to(device)
    return tokenizer

def voice_cleaning_setup():
    df_model, df_state, _ = init_df()
    return df_model, df_state

### TODO: убрать запись в файл
def clean_audio(df_model, df_state, audio_noise_wav_file, audio_clean_wav_file):
    noisy_audio, _ = load_audio(audio_noise_wav_file, sr=df_state.sr())
    audio = enhance(df_model, df_state, noisy_audio)
    save_audio(audio_clean_wav_file, audio, df_state.sr())

### TODO: посмотреть необходимость torchaudio, файл уже wav
### TODO: передавать аудио файл в виде буфера
### TODO: сохранять клонированный голос в спец. папке
def clone_voice(hubert_model, tokenizer, codec_model, voice_to_clone_file, voice_fingerprint_file): # the audio you want to clone (under 13 seconds)
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(voice_to_clone_file)
    wav = convert_audio(wav, sr, codec_model.sample_rate, codec_model.channels)
    wav = wav.to(device)
    
    semantic_vectors = hubert_model.forward(wav, input_sample_hz=codec_model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)
    
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = codec_model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
    
    # move codes to cpu
    codes = codes.cpu().numpy()
    # move semantic tokens to cpu
    semantic_tokens = semantic_tokens.cpu().numpy()
    
    np.savez(voice_fingerprint_file, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

def transcribe_audio_setup(whisper_size = "small"):
    whisper_model = whisper.load_model(whisper_size)
    return whisper_model

def transcribe_audio(whisper_model, audio_clean_wav_file):
    result = whisper_model.transcribe(audio_clean_wav_file)
    text_output = result["text"]
    return text_output

def translate_text_setup(device, lang_pair = "ru-en"):
    if (lang_pair == "ru-en"):
        translate_model = torch.hub.load("pytorch/fairseq", "transformer.wmt19.ru-en", checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt", tokenizer="moses", bpe="fastbpe")
    elif (lang_pair == "en-ru"):
        translate_model = torch.hub.load("pytorch/fairseq", "transformer.wmt19.en-ru", checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt", tokenizer="moses", bpe="fastbpe")
    translate_model.eval()
    if (device == "cuda"):
        translate_model.cuda()
    return translate_model

def translate_text(translate_model, text_input):
    text_output = translate_model.translate(text_input)
    return text_output

def voice_synthesis_setup():
    # download and load all models
    preload_models(
        text_use_gpu=True,
        text_use_small=False,
        coarse_use_gpu=True,
        coarse_use_small=False,
        fine_use_gpu=True,
        fine_use_small=False,
        codec_use_gpu=True,
        force_reload=False,
        path="models"
    )

### TODO: voice_name - абсолютный путь
def synthesize_voice(text_prompt, voice_name, mode = "simple"):
    if (mode == "simple"):
        audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)
    elif (mode == "full"):
        x_semantic = generate_text_semantic(
            text_prompt,
            history_prompt=voice_name,
            temp=0.7,
            top_k=50,
            top_p=0.95,
        )
        
        x_coarse_gen = generate_coarse(
            x_semantic,
            history_prompt=voice_name,
            temp=0.7,
            top_k=50,
            top_p=0.95,
        )
        
        x_fine_gen = generate_fine(
            x_coarse_gen,
            history_prompt=voice_name,
            temp=0.5,
        )
        audio_array = codec_decode(x_fine_gen)
    return audio_array

def video_synchronization_setup():
    url = "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA"
    response = requests.get(url)
    
    with open("Wav2Lip/checkpoints/wav2lip_gan.pth", "wb") as f:
        f.write(response.content)
    
    # Download pretrained model for face detection
    url = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
    response = requests.get(url)
    
    with open("Wav2Lip/face_detection/detection/sfd/s3fd.pth", "wb") as f:
        f.write(response.content)

def sync_video(input_video_file, input_audio_file, output_video_file):
    audio, sr = librosa.load(input_audio_file, sr=None)
    sf.write("temp/voice_sync.wav", audio, sr, format="wav")
    pad_top = 0
    pad_bottom = 10
    pad_left = 0
    pad_right = 0
    rescaleFactor = 1
    nosmooth = False
    
    # Set the path to the Wav2Lip model and input files
    checkpoint_path = "Wav2Lip/checkpoints/wav2lip_gan.pth"

    ### TODO: переписать вызов через внутреннее API
    # Run the Wav2Lip model
    cmd = f"python3 Wav2Lip/inference.py --checkpoint_path {checkpoint_path} --face {input_video_file} --audio temp/voice_sync.wav --pads {pad_top} {pad_bottom} {pad_left} {pad_right} --resize_factor {rescaleFactor} {'--nosmooth' if nosmooth else ''} --outfile {output_video_file}"
    subprocess.run(cmd.split())
    #subprocess.run(["python", "Wav2Lip/inference.py", "--checkpoint_path", str(checkpoint_path), "--face", str("../" + input_video_file), "--audio", str("../" + input_audio_file), "--pads", str(pad_top), str(pad_bottom), str(pad_left), str(pad_right), "--resize_factor", str(rescaleFactor), str("--nosmooth" if nosmooth else ""), "--outfile", str("../" + output_video_file)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" # or "cpu"

    ### TODO: добавить аргумент verbose и выводить логи поэтапно
    cmd = f"ffmpeg -y -i {args.input_file} temp/input_audio.wav"
    subprocess.run(cmd.split())
    #subprocess.run(["ffmpeg", "-y", "-i", str(args.input_file), "temp/input_audio.wav"])

    ### TODO: пройтись по записи и проанализировать качество аудио и количество голосов/роли
    ### определить правильность выбранного голоса (голос для клонирования должен быть один и достаточно чистый)
    ### если несколько человек, то по тембру выбрать голос каждого и склонировать каждого с отметкой его тембра для соответствующего синтеза
    ### по спектру смотреть присутствуют ли в аудио отрезке другие звуки, если есть только голосовые частоты, то брать для клонирования. проходиться окном по аудио и искать отрезок с голосом и минимумом посторонних звуков
    df_model, df_state = voice_cleaning_setup()
    clean_audio(df_model, df_state, "temp/input_audio.wav", "temp/clean_audio.wav")

    codec_model = voice_clonning_setup_bark(device)
    tokenizer_file = voice_clonning_download_hubert("eng")
    hubert_model = voice_clonning_setup_hubert(device)
    tokenizer_model = voice_clonning_setup_tokenizer(device, tokenizer_file)
    clone_voice(hubert_model, tokenizer_model, codec_model, "temp/clean_audio.wav", "temp/voice_clone.npz")
    
    whisper_model = transcribe_audio_setup("small")
    text_transcribed = transcribe_audio(whisper_model, "temp/clean_audio.wav")

    translate_model = translate_text_setup(device, "ru-en")
    text_translated = translate_text(translate_model, text_transcribed)

    del codec_model, hubert_model, tokenizer_model, whisper_model, translate_model
    
    voice_synthesis_setup()
    audio_array = synthesize_voice(text_translated, "temp/voice_clone.npz", "simple")

    ### TODO: разобраться с этой файловой херью
    write_wav("temp/voice_synt_noise.wav", SAMPLE_RATE, audio_array)
    noisy_audio, _ = load_audio("temp/voice_synt_noise.wav", sr=df_state.sr())
    #df_model, df_state = voice_cleaning_setup()
    audio = enhance(df_model, df_state, noisy_audio)
    save_audio("temp/voice_synt.wav", audio, df_state.sr())

    video_synchronization_setup()
    sync_video(args.input_file, "temp/voice_synt.wav", args.output_file)
