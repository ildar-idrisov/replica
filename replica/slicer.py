from stable_whisper import load_model

class Slicer:
    def __init__(self, model = "medium"):
        self.model = load_model(model)

    def transcribe(self, audio_file):
        self.transcript = self.model.transcribe(audio_file)

    def get_result_by_sentence(self, start = 0, steps = None):
        matches = self.transcript.find(r'[^.]+\.')
        
        result = []
        matches_len = len(matches)
        if (steps == None):
            end = matches_len
        elif (steps > matches_len - start):
            steps = matches_len
            end = start + steps
        else:
            end = start + steps

        for match in matches[start:end]:
            result.append({"text" : match.text_match,
                          "start" : match.start,
                          "end" : match.end})
        return result

    def get_result_by_time(self, start = 0, duration = 10):
        matches = self.transcript.find(r'[^.]+\.')
        
        result = []
        if (duration > matches[-1].end - start):
            duration = matches[-1].end - start

        for match in matches:
            if (match.start >= start and match.end <= start + duration):
                result.append({"text" : match.text_match,
                              "start" : match.start,
                              "end" : match.end})
            elif (match.end > start + duration):
                break
        return result
    
    def save_json(self, json_file):
        self.transcript.save_as_json(json_file)

    def load_json(self, json_file):
        self.transcript = stable_whisper.WhisperResult(json_file)