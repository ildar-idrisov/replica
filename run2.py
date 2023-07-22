#
# python run2.py --input_file shared/orlov.mp4 --output_file temp/res.mp4
#

import sys
import argparse
import subprocess
import os
import glob
import random
# voice clonning
from replica import utils as replica
### TODO: убрать лишние зависимости
from bark.generation import SAMPLE_RATE
from df.enhance import enhance, load_audio, save_audio
from pydub import AudioSegment

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
    
    device = "cuda" # "cuda" or "cpu"
    
    ### TODO: добавить аргумент verbose и выводить логи поэтапно
    ### TODO: найти более подходящий отрывок голоса

    ### TODO: пройтись по записи и проанализировать качество аудио и количество голосов/роли
    ### определить правильность выбранного голоса (голос для клонирования должен быть один и достаточно чистый)
    ### если несколько человек, то по тембру выбрать голос каждого и склонировать каждого с отметкой его тембра для соответствующего синтеза
    ### по спектру смотреть присутствуют ли в аудио отрезке другие звуки, если есть только голосовые частоты, то брать для клонирования. проходиться окном по аудио и искать отрезок с голосом и минимумом посторонних звуков
    input_file = args.input_file
    audio = AudioSegment.from_file(input_file)
    duration_in_s = len(audio) / 1000

    DURATION_LIMIT = 20
    START_POSITION = random.randint(0, int(duration_in_s - DURATION_LIMIT))
    print("===== START_POSITION:", START_POSITION, "=====")
    if (duration_in_s > DURATION_LIMIT):
        cmd = f"ffmpeg -y -i {input_file} -c copy -ss {START_POSITION} -t {DURATION_LIMIT} temp/input_cut.mp4"
        subprocess.run(cmd.split())
        input_file = "temp/input_cut.mp4"

    cmd = f"ffmpeg -y -i {input_file} -c copy temp/input_video.mp4 temp/input_audio.wav"
    subprocess.run(cmd.split())
    
    df_model, df_state = replica.voice_cleaning_setup()
    replica.clean_audio(df_model, df_state, "temp/input_audio.wav", "temp/clean_audio.wav")
    del df_model, df_state
    replica.clear_memory()

    audio = AudioSegment.from_file("temp/input_audio.wav")
    if audio.channels == 2:
        channels = audio.split_to_mono()
        result = channels[1].overlay(channels[0].invert_phase())
        result = result.set_channels(2)
    else:
        audio_clean = AudioSegment.from_file("temp/clean_audio.wav")
        inverted_audio = audio_clean.invert_phase()
        result = audio1.overlay(inverted_audio)
    result.export("temp/bg_audio.wav", format='wav')

    audio = AudioSegment.from_file("temp/clean_audio.wav")
    duration_in_s = len(audio) / 1000
    for i in range(int(duration_in_s // 10)):
        cmd = f"ffmpeg -y -i temp/input_video.mp4 -c copy -ss {i*10} -t 10 temp/input_video_cut.wav"
        subprocess.run(cmd.split())
        cmd = f"ffmpeg -y -i temp/clean_audio.wav -c copy -ss {i*10} -t 10 temp/clean_audio_cut.wav"
        subprocess.run(cmd.split())
        cmd = f"ffmpeg -y -i temp/bg_audio.wav -c copy -ss {i*10} -t 10 temp/bg_audio_cut.wav"
        subprocess.run(cmd.split())
        
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

        bg_audio = AudioSegment.from_file("temp/bg_audio_cut.wav")
        converted_voice = AudioSegment.from_file("temp/converted_voice.wav")
        converted_voice_speed = converted_voice.speedup(playback_speed=len(converted_voice)/len(bg_audio))
        combined_voice_bg = bg_audio.overlay(converted_voice_speed)
        combined_voice_bg.export("temp/combined_voice_bg.wav", format='wav')
        
        replica.video_synchronization_setup()
        replica.sync_video("temp/input_video_cut.mp4", "temp/combined_voice_bg.wav", f"temp/output_video_{i:03n}.mp4")

    video_files = sorted([f for f in os.listdir("temp") if f.startswith('output_video_') and f.endswith('.mp4')])
    with open('temp/filelist.txt', 'w') as file:
        for video_file in video_files:
            file.write(f"file '{video_file}'\n")
    
    cmd = f"ffmpeg -y -f concat -safe 0 -i temp/filelist.txt -c copy {args.output_file}"
    subprocess.run(cmd.split())

    #for i in range(int(audio_duration // 10)):
    #    cmd = f"ffmpeg -y -i {input_file} -c copy -ss {i*10} -t 10 temp/input_video.mp4 -ss {i*10} -t 10 temp/input_audio.wav"
    #    subprocess.run(cmd.split())
    #    df_model, df_state = replica.voice_cleaning_setup()
    #    replica.clean_audio(df_model, df_state, "temp/input_audio.wav", "temp/clean_audio.wav")
    #    
    #    whisper_model = replica.transcribe_audio_setup("medium")
    #    text_transcribed = replica.transcribe_audio(whisper_model, "temp/clean_audio.wav")
    #    
    #    translate_model = replica.translate_text_setup(device, "ru-en")
    #    text_translated = replica.translate_text(translate_model, text_transcribed)
    #    
    #    del whisper_model, translate_model
    #    del df_model, df_state
    #    replica.clear_memory()
    #
    #    # Alternative cloning. TTS - VITS, conversion - FreeVC
    #    tts = replica.voice_conversion_setup("eng")
    #    replica.voice_conversion(tts, text_translated, "temp/clean_audio.wav", "temp/converted_voice.wav")
    #    del tts.synthesizer
    #    del tts.voice_converter
    #    del tts
    #    replica.clear_memory()
    #    
    #    replica.video_synchronization_setup()
    #    replica.sync_video("temp/input_video.mp4", "temp/converted_voice.wav", f"temp/output_video_{i:03n}.mp4")
#
    #cmd = f"ffmpeg -y -f concat -i temp/output_video_%03d.mp4 -c copy {args.output_file}"
    #subprocess.run(cmd.split())
