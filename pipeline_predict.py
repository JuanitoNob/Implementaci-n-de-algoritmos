
import os
import argparse
import logging
import pickle
import pandas as pd
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam as beam

class Predict(beam.DoFn):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None

    def setup(self):
        # Load tokenizer
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.pkl')
        with open(tokenizer_path, 'rb') as tokenizer_file:
            self.tokenizer = pickle.load(tokenizer_file)

        # Load model
        self.model = tf.keras.models.load_model(self.model_dir)

    def process(self, element):
        if self.model is None or self.tokenizer is None:
            logging.error("Model or tokenizer not loaded properly.")
            return
        
        test_csv_path = element
        test_df = pd.read_csv(test_csv_path)
        test_sequences = self.tokenizer.texts_to_sequences(test_df.iloc[:, 0])
        max_sequence_length = max([len(seq) for seq in test_sequences])
        test_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')
        predictions = self.model.predict(test_sequences)

        # Calculate loss and accuracy
        loss, accuracy = self.model.evaluate(test_sequences, test_df.iloc[:, 1], verbose=0)

        yield {
            'loss': loss,
            'accuracy': accuracy
        }

def run_inference_pipeline(test_csv_paths, model_dir):
    options = PipelineOptions()
    with beam.Pipeline(options=options) as pipeline:
        _ = (
            pipeline
            | 'Create' >> beam.Create(test_csv_paths)
            | 'Predict' >> beam.ParDo(Predict(model_dir))
            | 'Print' >> beam.Map(print)
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv-paths', dest='test_csv_paths', nargs='+', required=True,
                        help='Paths to the CSV files containing test data')
    parser.add_argument('--model-dir', dest='model_dir', required=True,
                        help='Directory containing the trained model')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_inference_pipeline(args.test_csv_paths, args.model_dir)

