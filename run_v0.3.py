#
# python run_v0.3.py --input_file shared/orlov.mp4 --output_file temp/res.mp4
#

import sys
import argparse
import subprocess
import os
import glob
import random
import torch
from pathlib import Path
# voice clonning
from replica import utils as replica_utils
from replica import slicer
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
    START_POSITION = 898 # debug
    print("START_POSITION =", START_POSITION)

    cmd = f"ffmpeg -y -hide_banner -loglevel error -i {input_file} -ss {START_POSITION} -t {DURATION_LIMIT} temp/input_video.mp4 -ss {START_POSITION} -t {DURATION_LIMIT} temp/input_audio.wav"
    subprocess.run(cmd.split())

    slicer = slicer.Slicer()
    sliced_data = slicer.slice("temp/input_video.mp4", output_language="english")
    del slicer
    replica_utils.clear_memory()

    replica_utils.video_synchronization_setup() ### TODO: тут только скачивание моделей

    for sample in sliced_data:
        sample_chunk_number = sample["chunk_number"]
        input_video_sample_file = f"temp/video_chunk_{sample_chunk_number}.mp4"
        input_audio_sample_file = f"temp/audio_chunk_{sample_chunk_number}.wav"
        if (sample["text_presents"] == True):
            df_model, df_state = replica_utils.voice_cleaning_setup()
            replica_utils.clean_audio(df_model, df_state, input_audio_sample_file, "temp/clean_audio_cut.wav")
            del df_model, df_state
            replica_utils.clear_memory()
    
            ### TODO: все улучшения/изменения аудио переместить в audio.py
            audio = AudioSegment.from_file(input_audio_sample_file)
            if audio.channels == 2:
                channels = audio.split_to_mono()
                result = channels[1].overlay(channels[0].invert_phase())
                result = result.set_channels(2)
            else:
                audio_clean = AudioSegment.from_file("temp/clean_audio_cut.wav")
                inverted_audio = audio_clean.invert_phase()
                result = audio1.overlay(inverted_audio)
            result.export("temp/bg_audio_cut.wav", format='wav')

            # Alternative cloning. TTS - VITS, conversion - FreeVC
            tts = replica_utils.voice_conversion_setup("eng")
            ### TODO: синхронизировать синтез речи с оригинальный аудио по таймстемпам для каждого предложения
            converted_voices = []
            for chunk in sample["content"]:
                replica_utils.voice_conversion(tts, chunk["text"], "temp/clean_audio_cut.wav", "temp/converted_voice.wav")

                ### TODO: перенести в модель audio.py и сделать правилое изменение длительности синтезированного текста, чтоб не только было ускорение аудио, но и удаление/добавление тишины между словами
                converted_voice = AudioSegment.from_file("temp/converted_voice.wav")
                chunks = split_on_silence(converted_voice, 
                    min_silence_len = 200,
                    silence_thresh = -40,
                    keep_silence=200,
                )
                joined_chunks = sum(chunks)
                joined_chunks.export("temp/converted_voice_cut.wav", format="wav")

                converted_voice = AudioSegment.from_file("temp/converted_voice_cut.wav")
                speed_rate = (len(converted_voice)/1000) / (chunk["end"]-chunk["start"])
                if (speed_rate > 1.0):
                    converted_voice = converted_voice.speedup(playback_speed=speed_rate)
                converted_voices.append({"segment" : converted_voice, "start" : chunk["start"], "end" : chunk["end"]})
            del tts.synthesizer
            del tts.voice_converter
            del tts
            replica_utils.clear_memory()

            start = sample["start"]
            combined_audio = AudioSegment.empty()
            for voice in converted_voices:
                if (voice["start"] > start):
                    silence_duration = voice["start"] - start
                    combined_audio += AudioSegment.silent(duration=silence_duration)
                combined_audio += voice["segment"]
                start = voice["end"]
            combined_audio.export("temp/combined_voice.wav", format='wav')

            bg_audio = AudioSegment.from_file("temp/bg_audio_cut.wav")
            combined_voice_bg = bg_audio.overlay(combined_audio)
            combined_voice_bg.export("temp/combined_voice_bg.wav", format='wav')
            
            video_file = f"output_video_{sample_chunk_number:03n}.mp4"
            ### TODO: нельзя отдавать аудио с фоном, нужно отдать чистое аудио голоса, сгенерить видео, а потом наложить звук фона
            replica_utils.sync_video(device, input_video_sample_file, "temp/combined_voice_bg.wav", f"temp/{video_file}")
        else:
            video_file = Path(input_video_sample_file).name

        with open('temp/filelist.txt', 'a') as file:
            file.write(f"file '{video_file}'\n")
    
    cmd = f"ffmpeg -y -f concat -safe 0 -i temp/filelist.txt {args.output_file}"
    subprocess.run(cmd.split())
