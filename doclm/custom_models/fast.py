
import os
import re
from abc import ABC, abstractmethod

import logging

import requests
import fasttext

from .model_config import model_config_dict

log = logging.getLogger("doclogger")
log.disabled = False


class CustomModel(ABC):
    def __init__(self, model_directory="./downloaded_models/", model_name=None, download_url=None):
        self.model_directory = model_directory
        os.makedirs(self.model_directory, exist_ok=True)  # Ensure the directory exists
        self.download_url = download_url
        self.model_name = model_name

    def get_model_path(self):
        """ Return the path of the model if it's available, otherwise try to download it and return the path. Return
        None if not found and cannot be downloaded."""
        model_path = os.path.join(self.model_directory, self.model_name)
        if os.path.exists(model_path):
            log.info(f"Model is already available at {model_path}")
            return model_path
        return None


    def download_model(self):
        """ Download a model file from the specified URL to a path and return the path if successful, None otherwise."""
        log.info(f"Attempting to download the model from {self.download_url}")
        model_path = os.path.join(self.model_directory, self.model_name)
        response = requests.get(self.download_url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            log.info(f"Model successfully downloaded to {model_path}")
            return model_path
        raise RuntimeError(f"Failed to download the file from {self.download_url} with status code {response.status_code}")

    @abstractmethod
    def load_model(self):
        """Load the model from the given path into memory. This method must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_data):
        """Make predictions with the model based on input_data. This method must be implemented by subclasses."""
        raise NotImplementedError


class LanguagePredictionFasttext(CustomModel):
    def __init__(self, model_directory, model_name, download_url):
        super().__init__(model_directory, model_name, download_url)
        self.model = self.load_model()  # Initialize the model attribute

    def load_model(self):
        """Load the FastText model from the given path"""
        model_path = self.get_model_path()
        if not model_path:
            log.info("Downloading model...")
            model_path = self.download_model()
        model = fasttext.load_model(model_path)
        log.info("Model loaded successfully.")
        return model


    def predict(self, input_text, **kwargs):
        """Use the loaded FastText model to predict the language of the provided text and return the language code."""
        # remove backspaces from the text if any
        input_text = re.sub(r'\s+', ' ', input_text)
        labels, probabilities = self.model.predict(input_text)
        if labels:
                # Extract language code from the label
            language_code = labels[0].split('_')[-1]  # Skip the '__label__' prefix
                # language = pycountry.languages.get(alpha_2=language_code)
                # log.info(f"Predicted language: {language} with confidence: {probabilities[0]}")
            return language_code
        log.error("No prediction was made.")

        raise RuntimeError("No prediction can be madewas made.")


# class ModelInstantiator:

    # @staticmethod
def get_fast_text_model():
    kwargs = model_config_dict['fasttext']
    return LanguagePredictionFasttext(model_directory=kwargs['model_directory'],
                                      model_name=kwargs['model_name'],
                                      download_url=kwargs['model_download_url']
                                      )

FAST_LANG_MODEL = get_fast_text_model()

