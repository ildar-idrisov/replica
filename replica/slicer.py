import math
import string
import librosa
import subprocess
from stable_whisper import load_model, WhisperResult
from .text import TextProcessor, languages_abbrev

CHUNK_DURATION = 10

class Slicer:
    def __init__(self, model = "medium"):
        self.stw_model = load_model(model)
        self.text_proc = TextProcessor()

    def slice(self, video_file, chunk_dur = CHUNK_DURATION, output_language="english"):
        audio_file = "temp/input_audio_full.wav"
        self.media_slicer(video_file, output_audio_file = audio_file)

        transcript = self.transcribe(audio_file)
        transcript_corrected = self.correct_original_text(transcript)
        transcript_corrected_sentences = self.text_proc.split_into_sentences(transcript_corrected["text"], transcript_corrected["language"])

        translated_text = self.text_proc.translate_text(transcript_corrected["text"], output_language)
        translated_sentences = self.text_proc.split_into_sentences(translated_text, output_language)

        self.matches = []
        start = 0
        for sent_pair in zip(transcript_corrected_sentences, translated_sentences):
            end = start + len(sent_pair[0].split()) - 1
            start_of_sentence = transcript_corrected["words"][start]["start"]
            end_of_sentence = transcript_corrected["words"][end]["end"]

            self.matches.append({"original_text" : sent_pair[0], "text" : sent_pair[1], "start" : start_of_sentence, "end" : end_of_sentence})
            start = end + 1

        res = []
        start = 0
        audio_duration = int(librosa.get_duration(path=audio_file))
        while start < audio_duration:
            chunk = self.get_chunk_by_time(start, chunk_dur)
            if chunk:
                text_presents = True
                end = chunk[-1]["end"]
            else:
                text_presents = False
                chunk = self.get_next_single_chunk(start)
                if chunk:
                    end = min(chunk["start"], start + chunk_dur)
                else:
                    end = audio_duration

            chunk_number = len(res)
            video_cut_file = f"temp/video_chunk_{chunk_number}.mp4"
            audio_cut_file = f"temp/audio_chunk_{chunk_number}.wav"
            res.append({"chunk_number" : chunk_number,
                "video_file" : video_cut_file,
                "audio_file" : audio_cut_file,
                "text_presents" : text_presents,
                "start" : start,
                "end" : end,
                "content" : chunk})
            self.media_slicer(video_file, video_cut_file, audio_cut_file, start, end)
            start = end

        return res

    def transcribe(self, audio_file):
        transcript = self.stw_model.transcribe(audio_file)
        return transcript

    def correct_original_text(self, transcript):
        transcript_dict = transcript.to_dict()
        text_corr = self.text_proc.add_punctuation(transcript_dict["text"])
        text_corr_splitted = text_corr.split()
### TODO: !!! ходить по text_corr_splitted и искать соответствующие таймстемпы, забирать их в новый список
### TODO: отдать в ChatGPT список кортежей, где каждый кортеж (слово, start, end) от whisper, а также исправленное предложение. Попросить сопоставить слова и заменить в списке все слова с пунктуацией
### TODO: пройтись по списку и заменять только те слова, в которых было сделано изменение
        words_lst = []
        temp_word = ""
        for seg in transcript_dict["segments"]:
            for wseg in seg["words"]:
                word = wseg["word"].strip()
                word = word.translate(str.maketrans("", "", string.punctuation)).lower()
                word_corr = text_corr_splitted[len(words_lst)].strip()
                word_corr = word_corr.translate(str.maketrans("", "", string.punctuation)).lower()
                if (word == word_corr):
                    words_lst.append({"word" : text_corr_splitted[len(words_lst)], "start" : wseg["start"], "end" : wseg["end"]})
                    temp_word = ""
                else:
                    print(word, word_corr)
                    pos = word_corr.find(word)
                    if (pos == 0):
                        start = wseg["start"]
                        temp_word = word
                    elif (pos > 0):
                        end = wseg["end"]
                        temp_word += word
                        if (temp_word == word_corr):
                            words_lst.append({"word" : text_corr_splitted[len(words_lst)], "start" : start, "end" : end})
                            temp_word = ""
                        else:
                            assert("Something goes wrong" == False)
                    else:
                        assert("Something goes wrong" == False)

        lang = languages_abbrev[transcript_dict["language"]]
        result_dict = {"text" : text_corr, "language" : lang, "words" : words_lst}
        return result_dict

    def get_chunk_by_time(self, start = 0, duration = CHUNK_DURATION):
        result = []
        for match in self.matches:
            if (match["start"] >= start and match["end"] <= start + duration):
                result.append({"text" : match["text"],
                              "start" : match["start"],
                              "end" : match["end"]})
            elif (match["end"] > start + duration):
                break
        return result

    def get_next_single_chunk(self, start = 0):
        result = None
        for match in self.matches:
            if (match["start"] >= start):
                result = {"text" : match["text"],
                          "start" : match["start"],
                          "end" : match["end"]}
                break
        return result

    def get_last_timestamp():
        return self.matches[-1].end

    def media_slicer(self, input_video_file, output_video_file = None, output_audio_file = None, start = None, end = None):
        cmd = f"ffmpeg -y -hide_banner -loglevel error -i {input_video_file}"
        if (output_video_file != None):
            if (start != None and end != None):
                cmd += f" -ss {start} -to {end}"
            cmd += f" {output_video_file}"
            
        if (output_audio_file != None):
            if (start != None and end != None):
                cmd += f" -ss {start} -to {end}"
            cmd += f" {output_audio_file}"
            
        assert (output_video_file != None or output_audio_file != None)
        subprocess.run(cmd.split())