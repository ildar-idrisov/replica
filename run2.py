#
# python run2.py --input_file shared/orlov.mp4 --output_file temp/res.mp4
#

import sys
import argparse
import subprocess
import os
import glob
import random
import torch
# voice clonning
from replica import utils as replica
### TODO: убрать лишние зависимости
from bark.generation import SAMPLE_RATE
from df.enhance import enhance, load_audio, save_audio
from pydub import AudioSegment
from pydub.silence import split_on_silence

if not os.path.exists("temp"):
    os.makedirs("temp")
else:
    files = glob.glob('temp/*')
    for f in files:
        os.remove(f)

### TODO: перенести внутренние переменные внутрь библиотеки, такие как temp/input_audio.wav, hubert_model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="shared/orlov_in.mp4")
    parser.add_argument("--output_file", type=str, default="temp/res.mp4")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ### TODO: добавить аргумент verbose и выводить логи поэтапно
    ### TODO: найти более подходящий отрывок голоса

    ### TODO: пройтись по записи и проанализировать качество аудио и количество голосов/роли
    ### определить правильность выбранного голоса (голос для клонирования должен быть один и достаточно чистый)
    ### если несколько человек, то по тембру выбрать голос каждого и склонировать каждого с отметкой его тембра для соответствующего синтеза
    ### по спектру смотреть присутствуют ли в аудио отрезке другие звуки, если есть только голосовые частоты, то брать для клонирования. проходиться окном по аудио и искать отрезок с голосом и минимумом посторонних звуков
    input_file = args.input_file
    audio = AudioSegment.from_file(input_file)
    duration_in_s = len(audio) / 1000

    DURATION_LIMIT = 30
    if (duration_in_s > DURATION_LIMIT):
        START_POSITION = random.randint(0, int(duration_in_s - DURATION_LIMIT))
    else:
        START_POSITION = 0

    cmd = f"ffmpeg -y -i {input_file} -ss {START_POSITION} -t {DURATION_LIMIT} temp/input_video.mp4 -ss {START_POSITION} -t {DURATION_LIMIT} temp/input_audio.wav"
    subprocess.run(cmd.split())

    replica.video_synchronization_setup() ### TODO: тут только скачивание моделей
    
    audio = AudioSegment.from_file("temp/input_audio.wav")
    duration_in_s = len(audio) / 1000
    
    for i in range(int(duration_in_s // 10)):
        cmd = f"ffmpeg -y -i temp/input_video.mp4 -ss {i*10} -t 10 temp/input_video_cut.mp4"
        subprocess.run(cmd.split())
        cmd = f"ffmpeg -y -i temp/input_audio.wav -ss {i*10} -t 10 temp/input_audio_cut.wav"
        subprocess.run(cmd.split())
        
        df_model, df_state = replica.voice_cleaning_setup()
        replica.clean_audio(df_model, df_state, "temp/input_audio_cut.wav", "temp/clean_audio_cut.wav")
        del df_model, df_state
        replica.clear_memory()
    
        audio = AudioSegment.from_file("temp/input_audio_cut.wav")
        if audio.channels == 2:
            channels = audio.split_to_mono()
            result = channels[1].overlay(channels[0].invert_phase())
            result = result.set_channels(2)
        else:
            audio_clean = AudioSegment.from_file("temp/clean_audio_cut.wav")
            inverted_audio = audio_clean.invert_phase()
            result = audio1.overlay(inverted_audio)
        result.export("temp/bg_audio_cut.wav", format='wav')
        
        whisper_model = replica.transcribe_audio_setup("medium")
        text_transcribed = replica.transcribe_audio(whisper_model, "temp/clean_audio_cut.wav")
        
        translate_model = replica.translate_text_setup(device, "ru-en")
        text_translated = replica.translate_text(translate_model, text_transcribed)

        del whisper_model, translate_model
        replica.clear_memory()

        # Alternative cloning. TTS - VITS, conversion - FreeVC
        tts = replica.voice_conversion_setup("eng")
        replica.voice_conversion(tts, text_translated, "temp/clean_audio_cut.wav", "temp/converted_voice.wav")
        del tts.synthesizer
        del tts.voice_converter
        del tts
        replica.clear_memory()

        converted_voice = AudioSegment.from_file("temp/converted_voice.wav")
        chunks = split_on_silence(converted_voice, 
            min_silence_len = 200,
            silence_thresh = -40,
            keep_silence=200,
        )
        joined_chunks = sum(chunks)
        joined_chunks.export("temp/converted_voice_cut.wav", format="wav")

        bg_audio = AudioSegment.from_file("temp/bg_audio_cut.wav")
        converted_voice = AudioSegment.from_file("temp/converted_voice_cut.wav")
        speed_rate = len(converted_voice)/len(bg_audio)
        if (speed_rate > 1.0):
            converted_voice = converted_voice.speedup(playback_speed=speed_rate)
        combined_voice_bg = bg_audio.overlay(converted_voice)
        combined_voice_bg.export("temp/combined_voice_bg.wav", format='wav')
        
        video_file = f"output_video_{i:03n}.mp4"
        replica.sync_video(device, "temp/input_video_cut.mp4", "temp/combined_voice_bg.wav", f"temp/{video_file}")

        with open('temp/filelist.txt', 'a') as file:
            file.write(f"file '{video_file}'\n")
    
    cmd = f"ffmpeg -y -f concat -safe 0 -i temp/filelist.txt {args.output_file}"
    subprocess.run(cmd.split())
