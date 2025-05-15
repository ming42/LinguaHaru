import logging
import os
import re
import unicodedata
from copy import copy
from .cache import TranslationCache
from . import shared_constants


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class BaseTranslator:
    name = "base"
    envs = {}
    lang_map = {}
    CustomPrompt = False
    ignore_cache = False

    def __init__(self, lang_in, lang_out, model):
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.model = model

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
                "model": model,
            },
        )
        temp_folder = os.path.join("temp", shared_constants.PDF_FILE_NAME)
        translation_json_path = os.path.join(temp_folder, "dst_translated.json")
        if os.path.exists(translation_json_path):
            self.cache.update_translations_from_json(translation_json_path)

    def set_envs(self, envs):
        # Detach from self.__class__.envs
        # Cannot use self.envs = copy(self.__class__.envs)
        # because if set_envs called twice, the second call will override the first call
        self.envs = copy(self.envs)
        for key in self.envs:
            if key in os.environ:
                self.envs[key] = os.environ[key]
        if envs is not None:
            for key in envs:
                self.envs[key] = envs[key]

    def add_cache_impact_parameters(self, k: str, v):
        """
        Add parameters that affect the translation quality to distinguish the translation effects under different parameters.
        :param k: key
        :param v: value
        """
        self.cache.add_params(k, v)

    def translate(self, text, ignore_cache=False):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                return cache

        translation = text
        self.cache.set(text, translation)
        return translation

    def do_translate(self, text):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError
    
    def process_translation_cache(self):
        temp_folder = os.path.join("temp", shared_constants.PDF_FILE_NAME)
        text_json_path = os.path.join(temp_folder, "src.json")
        self.cache.export_translation_to_json(text_json_path)

    def prompt(self, text, prompt):
        if prompt:
            context = {
                "lang_in": self.lang_in,
                "lang_out": self.lang_out,
                "text": text,
            }
            return eval(prompt.safe_substitute(context))
        else:
            return [
                {
                    "role": "system",
                    "content": "You are a professional,authentic machine translation engine.",
                },
                {
                    "role": "user",
                    "content": f"Translate the following markdown source text to {self.lang_out}. Keep the formula notation {{v*}} unchanged. Output translation directly without any additional text.\nSource Text: {text}\nTranslated Text:",  # noqa: E501
                },
            ]

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"