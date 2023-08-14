import os
import openai
import torch
import nltk
from nltk.tokenize import sent_tokenize

OPENAI_API_KEY = "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"
languages_abbrev = {
    "ru" : "russian",
    "rus" : "russian",
    "russian" : "russian",
    "en" : "english",
    "eng" : "english",
    "english" : "english",
}

class TextProcessor:    
    def __init__(self, openai_api_key=OPENAI_API_KEY):
        if (openai_api_key):
            openai.api_key = openai_api_key
        nltk.download('punkt')

    def add_punctuation(self, transcript):
        ### TODO: отдавать в openai текст по частям, так чтобы сохранялась целостность текста
        prompt = f"Place punctuation marks in the text: {transcript}. Save original text language and words order. Don't say anything else except the result text.  Write the result only, without additional information."
        #prompt = f"Place punctuation marks in the text: {transcript}. Save original text language and words order. Write the answer in the form of a python list of tuples, where each tuple will contain the word before the correction and the word after the correction with punctuation marks. Don't say anything else except the result list.  Write the result only, without additional information."

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a professional text editor"},
                #{"role": "system", "content": "You're a professional text editor who knows Python. Write python list only"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=256
        )
        response = completion.choices[0].message.content
        #print("TextProcessor add_punctuation() in:", transcript)
        #print("TextProcessor add_punctuation() out:", response)
        return response

    def translate_text(self, transcript, language="english"):
        ### TODO: отдавать в openai текст по частям, так чтобы сохранялась целостность текста
        prompt = f"Correct the mistakes in the text and translate: {transcript} to {language}. Don't say anything else except the result text.  Write the result only, without additional information."
    
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "You're a professional language translator"},
                    {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=256
        )
        response = completion.choices[0].message.content
        #print("TextProcessor translate_text() in:", transcript)
        #print("TextProcessor translate_text() out:", response)
        return response

    def split_into_sentences(self, text, language):
        sentences = sent_tokenize(text, language=language)
        return sentences
    
    def translate_text_setup_local(self, device, lang_pair = "ru-en"):
        if (lang_pair == "ru-en"):
            translate_model = torch.hub.load("pytorch/fairseq", "transformer.wmt19.ru-en", checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt", tokenizer="moses", bpe="fastbpe")
        elif (lang_pair == "en-ru"):
            translate_model = torch.hub.load("pytorch/fairseq", "transformer.wmt19.en-ru", checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt", tokenizer="moses", bpe="fastbpe")
        translate_model.eval()
        if (device == "cuda"):
            translate_model.cuda()
        return translate_model

    def translate_text_local(self, translate_model, text_input):
        text_output = translate_model.translate(text_input)
        return text_output