
import argparse
import logging
import re
import os
import csv
import random

import apache_beam as beam
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.coders.coders import Coder
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions, DirectOptions

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords")

# CLEANING
STOP_WORDS = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

class CustomCoder(Coder):
    """Custom coder utilizado para ller y escribir strings. Realiza una serie de tranformaciones entre codificaciones"""

    def __init__(self, encoding: str):
        self.enconding = encoding

    def encode(self, value):
        return value.encode(self.enconding)

    def decode(self, value):
        return value.decode(self.enconding)

    def is_deterministic(self):
        return True

class PreprocessColumnsTrainFn(beam.DoFn):
    """Realiza el preprocesamiento propio de NLP"""

    def process_text(self, text):
        text = re.sub(TEXT_CLEANING_RE, " ", str(text).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in STOP_WORDS:
                # Si el token es un número, conviértelo a cadena de texto
                if token.replace('.', '', 1).isdigit():  # Verifica si el token es un número (incluyendo números decimales)
                    tokens.append("number")
                else:
                    tokens.append(STEMMER.stem(token))
        return " ".join(tokens)

    def process(self, element):
        if isinstance(element, str):
            try:
                text, sentiment = element.split(",")
            except ValueError:
                logging.warning("No se pudo dividir la línea por la coma. Usando la línea como está.")
                text = element.strip()
                sentiment = "1"  # Asignamos un valor por defecto
            processed_text = self.process_text(text)
            processed_sentiment = sentiment  # No necesitas procesar el sentimiento aquí
            yield f"{processed_text}, {processed_sentiment}"


def run(argv=None, save_main_session=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", dest="work_dir", required=True, help="Directorio de trabajo")
    parser.add_argument("--input", dest="input", required=True, help="Dataset de entrada en el directorio de trabajo")
    parser.add_argument("--output", dest="output", required=True, help="Ruta de salida para almacenar los datos transformados")
    parser.add_argument("--mode", dest="mode", required=True, choices=["train", "test"], help="Tipo de salida para almacenar los datos transformados")

    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    pipeline_options.view_as(DirectOptions).direct_num_workers = 0

    with beam.Pipeline(options=pipeline_options) as p:
        raw_data = p | "ReadTrollData" >> ReadFromText(known_args.input, coder=CustomCoder("latin-1"))

        if known_args.mode == "train":
            transformed_data = (raw_data
                                | "Preprocess" >> beam.ParDo(PreprocessColumnsTrainFn()))

            eval_percent = 20
            assert 0 < eval_percent < 100, "eval_percent must be in the range (0-100)"
            train_dataset, eval_dataset = (transformed_data
                                           | "Split dataset"
                                           >> beam.Partition(lambda elem, _: int(random.uniform(0, 100) < eval_percent), 2))

            train_dataset | "TrainWriteToCSV" >> WriteToText(os.path.join(known_args.output, "train", "part"), file_name_suffix=".csv")
            eval_dataset | "ValWriteToCSV" >> WriteToText(os.path.join(known_args.output, "val", "part"), file_name_suffix=".csv")

        else:
            transformed_data = (raw_data
                                | "Preprocess" >> beam.ParDo(PreprocessColumnsTrainFn()))

            transformed_data | "TestWriteToCSV" >> WriteToText(os.path.join(known_args.output, "test", "part"), file_name_suffix=".csv")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
