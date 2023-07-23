# https://github.com/serp-ai/bark-with-voice-clone
# https://github.com/Datasciensyash/ResemblyzerSlim.git
# https://github.com/justinjohn0306/Wav2Lip

import numpy as np
import torch
import requests
import subprocess
import os
import sys
import gc
#sys.path.insert(0, "Wav2Lip") ### TODO: move to Dockerfile
sys.path.insert(0, "bark-with-voice-clone") ### TODO: move to Dockerfile"
sys.path.insert(0, "ResemblyzerSlim") ### TODO: move to Dockerfile
# voice clonning
from bark.generation import load_codec_model, SAMPLE_RATE
from hubert.hubert_manager import HuBERTManager
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
from encodec.utils import convert_audio
# voice synthesis
from bark.api import generate_audio
# voice conversion
from TTS.api import TTS
#from transformers import BertTokenizer
from bark.generation import preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
# voice cleaning
from df.enhance import enhance, init_df, load_audio, save_audio
# automatic speach recognition
import whisper
# voice similaruity
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
# text similaruity
from simphile import jaccard_similarity, euclidian_similarity, compression_similarity
import statistics
# wirking with audio
import torchaudio
import soundfile as sf
import librosa ### заменить на pydub
from scipy.io.wavfile import write as write_wav

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
    hubert_model = CustomHubert(checkpoint_path="models/hubert/hubert.pt").to(device)
    return hubert_model

def voice_clonning_setup_tokenizer(device, tokenizer_file):
    tokenizer = CustomTokenizer.load_from_checkpoint("models/hubert/" + tokenizer_file).to(device)
    return tokenizer

def voice_cleaning_setup():
    df_model, df_state, _ = init_df()
    return df_model, df_state

def resemblyzer_setup():
    resemblyzer_encoder = VoiceEncoder()
    return resemblyzer_encoder

### TODO: убрать запись в файл
### TODO: разобраться с этой файловой херью
def clean_audio(df_model, df_state, audio_noise_wav_file, audio_clean_wav_file):
    noisy_audio, _ = load_audio(audio_noise_wav_file, sr=df_state.sr())
    audio = enhance(df_model, df_state, noisy_audio)
    save_audio(audio_clean_wav_file, audio, df_state.sr())

### TODO: посмотреть необходимость torchaudio, файл уже wav
### TODO: передавать аудио файл в виде буфера
### TODO: сохранять клонированный голос в спец. папке
def clone_voice(device, hubert_model, tokenizer_model, codec_model, voice_to_clone_file, voice_fingerprint_file): # the audio you want to clone (under 13 seconds)
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(voice_to_clone_file)
    wav = convert_audio(wav, sr, codec_model.sample_rate, codec_model.channels)
    wav = wav.to(device)
    
    semantic_vectors = hubert_model.forward(wav, input_sample_hz=codec_model.sample_rate)
    semantic_tokens = tokenizer_model.get_token(semantic_vectors)
    
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = codec_model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
    
    # move codes to cpu
    codes = codes.cpu().numpy()
    # move semantic tokens to cpu
    semantic_tokens = semantic_tokens.cpu().numpy()
    
    np.savez(voice_fingerprint_file, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

def clone_voice_find_best(device, hubert_model, tokenizer_model, codec_model, input_file, resemblyzer_encoder, mode):
    ### TODO: embed_utterance - работает с одним wav файлом, embed_speaker - с несколькими высказываниями одного спикера.
    ### Можно взять ряд аудио файлов на вход для расчета эмбеддинга авторского голоса, и т ак же нагенерить 10 голосов для оценки похожести сгенеренных семплов на оригинал
    audio_duration = librosa.get_duration(filename=input_file)
    text = "Hello! I am currently in Europe on tour, look what beauty is behind me. But in general I am in Vienna now, it is very beautiful here"
    clone_scores = []
    for i in range(int(audio_duration // 10)):
        ### TODO: если конец аудио файла, то break
        cmd = f"ffmpeg -y -i {input_file} -ss {i*10} -t 10 temp/input_audio_{i}.wav"
        subprocess.run(cmd.split())
        ### TODO: удалить паузы
        clone_voice(device, hubert_model, tokenizer_model, codec_model, f"temp/input_audio_{i}.wav", f"temp/voice_clone_{i}.npz")
        
        fpath = Path(f"temp/input_audio_{i}.wav")
        wav = preprocess_wav(fpath)
        embeds_a = resemblyzer_encoder.embed_utterance(wav)
        np.set_printoptions(precision=3, suppress=True)
    
        voice_synt_file = "temp/voice_synt_noise.wav"
        sim_audio_lst = []
        for j in range(10):
            
            audio_array = synthesize_voice(text, f"temp/voice_clone_{i}.npz", mode)
            write_wav(voice_synt_file, SAMPLE_RATE, audio_array)
    
            fpath = Path(voice_synt_file)
            wav = preprocess_wav(fpath)
            embeds_b = resemblyzer_encoder.embed_utterance(wav)
            np.set_printoptions(precision=3, suppress=True)
            sim_audio = np.inner(embeds_a, embeds_b)
            
            sim_audio_lst.append(sim_audio)
            print(j, "sim_audio:", sim_audio)

        clone_score = statistics.median(sim_audio_lst)
        clone_scores.append((i, clone_score, f"temp/voice_clone_{i}.npz"))

    clone_score = sorted(clone_scores, reverse=True, key=lambda item: item[1])[0] ### TODO: sorted()[1] для исключения выбросов
    print("best voice:", clone_score)
    clone_score_file = clone_score[2]
    return clone_score_file

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

def synthesize_voice_list(text, voice_name, resemblyzer_encoder, mode, original_voice, search_iter = 10):
    fpath = Path(original_voice)
    wav = preprocess_wav(fpath)
    embeds_a = resemblyzer_encoder.embed_utterance(wav)
    np.set_printoptions(precision=3, suppress=True)
    
    samples = []
    for i in range(search_iter):
        audio_array = synthesize_voice(text, voice_name, mode)
        write_wav(f"temp/voice_synt_noise_{i}.wav", SAMPLE_RATE, audio_array)
        
        fpath = Path(f"temp/voice_synt_noise_{i}.wav")
        wav = preprocess_wav(fpath)
        embeds_b = resemblyzer_encoder.embed_utterance(wav)
        np.set_printoptions(precision=3, suppress=True)
    
        sim = np.inner(embeds_a, embeds_b)
        samples.append((i, sim, f"temp/voice_synt_noise_{i}.wav"))
        print(i, sim)
        #if (sim > 0.9):
        #    break
    samples = list(filter(lambda item: item[1] > 0.75, samples))
    samples = sorted(samples, reverse=True, key=lambda item: item[1])
    print(samples)
    return samples

def compare_text(text_a, text_b):
    print(text_a)
    print(text_b)
    print(f"Jaccard Similarity: {jaccard_similarity(text_a, text_b)}")
    #print(f"Euclidian Similarity: {euclidian_similarity(text_a, text_b)}")
    print(f"Compression Similarity: {compression_similarity(text_a, text_b)}")
    
    return jaccard_similarity(text_a, text_b)

def find_best_sample(original_text, text_samples, whisper_model):
    best_speech_sample = None
    for sample in text_samples:
        text_transcribed = transcribe_audio(whisper_model, sample[2])
        if (compare_text(original_text, text_transcribed) > 0.72):
            best_speech_sample = sample
            break
    ### TODO: перевести сгенеренное аудио в текст, если фраза присутствует, то вырезать ненужное в начале и в конце
    if (best_speech_sample != None):
        best_speech_file = best_speech_sample[2]
    else:
        best_speech_file = text_samples[0][2]
    return best_speech_file

def synthesize_voice_find_best(text, voice_name, resemblyzer_encoder, whisper_model, mode, original_voice, search_iter = 30):
    fpath = Path(original_voice)
    wav = preprocess_wav(fpath)
    embeds_a = resemblyzer_encoder.embed_utterance(wav)
    np.set_printoptions(precision=3, suppress=True)

    voice_synts = []
    for i in range(search_iter):
        audio_array = synthesize_voice(text, voice_name, mode)
        write_wav(f"temp/voice_synt_noise_{i}.wav", SAMPLE_RATE, audio_array)

        fpath = Path(f"temp/voice_synt_noise_{i}.wav")
        wav = preprocess_wav(fpath)
        embeds_b = resemblyzer_encoder.embed_utterance(wav)
        np.set_printoptions(precision=3, suppress=True)
        sim_audio = np.inner(embeds_a, embeds_b)

        text_transcribed = transcribe_audio(whisper_model, f"temp/voice_synt_noise_{i}.wav")
        sim_text = compare_text(text, text_transcribed)
        print(i, sim_audio, sim_text)

        voice_synts.append((i, sim_audio, sim_text, f"temp/voice_synt_noise_{i}.wav"))
        if (sim_audio > 0.70 and sim_text > 0.72):
            break

    print(voice_synts)
    voice_synts = sorted(voice_synts, reverse=True, key=lambda item: item[1])[:10]
    print(voice_synts)
    voice_synts = sorted(voice_synts, reverse=True, key=lambda item: item[2])[0]
    print(voice_synts)
    voice_synt_file = voice_synts[3]
    return voice_synt_file

def voice_conversion_setup(language = "eng"):
    #For these models use the following name format: `tts_models/<lang-iso_code>/fairseq/vits`.
    #You can find the list of language ISO codes [here](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html) and learn about the Fairseq models [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms).
    tts = TTS(f"tts_models/{language}/fairseq/vits")
    return tts

def voice_conversion(tts, text, speaker_voice, output_file):
    tts.tts_with_vc_to_file(
        text,
        speaker_wav=speaker_voice,
        file_path=output_file
    )

def video_synchronization_setup():
    ### TODO: перенести в Dockerfile
    url = "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA"
    response = requests.get(url)
    
    with open("models/wav2lip_gan.pth", "wb") as f:
        f.write(response.content)
    
    # Download pretrained model for face detection
    url = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
    response = requests.get(url)
    
    if not os.path.exists("models/face_detection"):
        os.makedirs("models/face_detection")
    with open("models/face_detection/s3fd.pth", "wb") as f:
        f.write(response.content)

def sync_video(input_video_file, input_audio_file, output_video_file):
    audio, sr = librosa.load(input_audio_file, sr=None)      ### TODO: можно удалить
    sf.write("temp/voice_sync.wav", audio, sr, format="wav") ### TODO: можно удалить
    pad_top = 0
    pad_bottom = 10
    pad_left = 0
    pad_right = 0
    rescaleFactor = 1
    nosmooth = True ### TODO: не уверен, что нужно сглаживание, но оно выключено, чтоб не падал код при отсутствии лица
    
    # Set the path to the Wav2Lip model and input files
    checkpoint_path = "models/wav2lip_gan.pth"

    ### TODO: переписать вызов через внутреннее API
    # Run the Wav2Lip model
    cmd = f"python3 wav2lip/inference.py --checkpoint_path {checkpoint_path} --face {input_video_file} --audio temp/voice_sync.wav --pads {pad_top} {pad_bottom} {pad_left} {pad_right} --resize_factor {rescaleFactor} {'--nosmooth' if nosmooth else ''} --outfile {output_video_file}"
    subprocess.run(cmd.split())

def clear_memory():
    ### TODO: добавить удаление всех неиспользуемых моделей
    torch.cuda.empty_cache()
    gc.collect()