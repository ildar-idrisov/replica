#
# python3 run.py --input_file shared/orlov.mp4 --output_file res.mp4
#

import sys
import argparse
import subprocess
import os
# voice clonning
#sys.path.insert(0, "Wav2Lip") ### TODO: move to Dockerfile
sys.path.insert(0, "bark-with-voice-clone") ### TODO: move to Dockerfile
import utils
### TODO: убрать лишние зависимости
from bark.generation import SAMPLE_RATE
from df.enhance import enhance, load_audio, save_audio
from scipy.io.wavfile import write as write_wav

if not os.path.exists("temp"):
    os.makedirs("temp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" # or "cpu"

    ### TODO: добавить аргумент verbose и выводить логи поэтапно
    cmd = f"ffmpeg -y -i {args.input_file} temp/input_audio.wav"
    subprocess.run(cmd.split())

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
