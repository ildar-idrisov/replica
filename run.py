#
# python run.py --input_file shared/orlov.mp4 --output_file temp/res.mp4
#

import sys
import argparse
import subprocess
import os
# voice clonning
from replica import utils as replica
### TODO: убрать лишние зависимости
from bark.generation import SAMPLE_RATE
from df.enhance import enhance, load_audio, save_audio

if not os.path.exists("temp"):
    os.makedirs("temp")

### TODO: перенести внутренние переменные внутрь библиотеки, такие как temp/input_audio.wav, hubert_model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="shared/orlov_in.mp4")
    parser.add_argument("--output_file", type=str, default="temp/res.mp4")
    args = parser.parse_args()
    
    device = "cuda" # or "cpu"

    # Clone
    
    ### TODO: добавить аргумент verbose и выводить логи поэтапно
    ### TODO: найти более подходящий отрывок голоса
    cmd = f"ffmpeg -y -i {args.input_file} -ss 0 -t 50 temp/input_audio.wav"
    subprocess.run(cmd.split())

    ### TODO: пройтись по записи и проанализировать качество аудио и количество голосов/роли
    ### определить правильность выбранного голоса (голос для клонирования должен быть один и достаточно чистый)
    ### если несколько человек, то по тембру выбрать голос каждого и склонировать каждого с отметкой его тембра для соответствующего синтеза
    ### по спектру смотреть присутствуют ли в аудио отрезке другие звуки, если есть только голосовые частоты, то брать для клонирования. проходиться окном по аудио и искать отрезок с голосом и минимумом посторонних звуков
    df_model, df_state = replica.voice_cleaning_setup()
    replica.clean_audio(df_model, df_state, "temp/input_audio.wav", "temp/clean_audio.wav") ### TODO: попробовать убрать чистку, возможно чистка удаляет нужную информацию для клонирования
    cmd = f"ffmpeg -y -i temp/clean_audio.wav -af silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-15dB temp/input_audio_wo_silence.wav"
    subprocess.run(cmd.split())

    codec_model = replica.voice_clonning_setup_bark(device)
    tokenizer_file = replica.voice_clonning_download_hubert("eng")
    hubert_model = replica.voice_clonning_setup_hubert(device)
    tokenizer_model = replica.voice_clonning_setup_tokenizer(device, tokenizer_file)
    #replica.clone_voice(device, hubert_model, tokenizer_model, codec_model, "temp/clean_audio.wav", "temp/voice_clone.npz")
    resemblyzer_encoder = replica.resemblyzer_setup()
    replica.voice_synthesis_setup()
    voice_clone_file = replica.clone_voice_find_best(device, hubert_model, tokenizer_model, codec_model, "temp/clean_audio.wav", resemblyzer_encoder, "simple")
    del codec_model, hubert_model, tokenizer_model, resemblyzer_encoder

    # Synthesis

    cmd = f"ffmpeg -y -i {args.input_file} -ss 0 -t 10 temp/input_audio.wav"
    subprocess.run(cmd.split())
    df_model, df_state = replica.voice_cleaning_setup()
    replica.clean_audio(df_model, df_state, "temp/input_audio.wav", "temp/clean_audio.wav")
    
    whisper_model = replica.transcribe_audio_setup("medium")
    text_transcribed = replica.transcribe_audio(whisper_model, "temp/clean_audio.wav")
    
    translate_model = replica.translate_text_setup(device, "ru-en")
    text_translated = replica.translate_text(translate_model, text_transcribed)
    
    del whisper_model, translate_model
    
    replica.voice_synthesis_setup()
    resemblyzer_encoder = replica.resemblyzer_setup()
    whisper_model = replica.transcribe_audio_setup("small.en")
    #speech_list = replica.synthesize_voice_list(text_translated, "temp/voice_clone.npz", resemblyzer_encoder, "simple", "temp/clean_audio.wav", 10)
    #best_speech_file = replica.find_best_sample(text_translated, speech_list, whisper_model)
    best_speech_file = replica.synthesize_voice_find_best(text_translated, voice_clone_file, resemblyzer_encoder, whisper_model, "simple", "temp/clean_audio.wav", 30)
    del whisper_model, resemblyzer_encoder

    replica.clean_audio(df_model, df_state, best_speech_file, "temp/voice_synt.wav")

    replica.video_synchronization_setup()
    replica.sync_video(args.input_file, "temp/voice_synt.wav", args.output_file)
