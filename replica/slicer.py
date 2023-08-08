import math
import librosa
import subprocess
from stable_whisper import load_model, WhisperResult

CHUNK_DURATION = 10

class Slicer:
    def __init__(self, model = "medium", chunk_dur = CHUNK_DURATION):
        self.model = load_model(model)
        self.chunk_dur = chunk_dur

    def slice(self, video_file):
        audio_file = "temp/input_audio_full.wav"
        self.media_slicer(video_file, output_audio_file = audio_file)

        self.transcribe(audio_file)
        self.save_json("temp/input_audio_full.json")

        res = []
        start = 0
        audio_duration = int(librosa.get_duration(path=audio_file))
        while start < audio_duration:
            chunk = self.get_chunk_by_time(start, self.chunk_dur)
            if chunk:
                text_presents = True
                end = math.ceil(chunk[-1]["end"])
            else:
                text_presents = False
                chunk = self.get_next_single_chunk(start)
                if chunk:
                    end = math.floor(chunk["start"])
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
        self.transcript = self.model.transcribe(audio_file)
        self.matches = self.transcript.find(r'[^.]+\.')

    def get_chunk_by_sentances(self, start = 0, sents = None):
        result = []
        matches_len = len(self.matches)
        if (sents == None):
            end = matches_len
        elif (sents > matches_len - start):
            sents = matches_len
            end = start + sents
        else:
            end = start + sents

        for match in self.matches[start:end]:
            result.append({"text" : match.text_match,
                          "start" : match.start,
                          "end" : match.end})
        return result

    def get_chunk_by_time(self, start = 0, duration = CHUNK_DURATION):
        result = []
        for match in self.matches:
            if (match.start >= start and match.end <= start + duration):
                result.append({"text" : match.text_match,
                              "start" : match.start,
                              "end" : match.end})
            elif (match.end > start + duration):
                break
        return result

    def get_next_single_chunk(self, start = 0):
        result = None
        for match in self.matches:
            if (match.start >= start):
                result = {"text" : match.text_match,
                          "start" : match.start,
                          "end" : match.end}
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

    def save_json(self, json_file):
        self.transcript.save_as_json(json_file)

    def load_json(self, json_file):
        self.transcript = WhisperResult(json_file)
        self.matches = self.transcript.find(r'[^.]+\.')