# Parses the results from the llama model to make them compatible with the rest of the codebase.

import os
import pandas as pd

from config import Config


class LlamaParser:
    @classmethod
    def parse_caption(cls, caption):
        caption = caption.lower()

        # Location
        location_results = {
            'shoulder': 0,
            'back': 0,
            'knee': 0,
            'other': 0
        }

        for word in caption.split():
            if word.lower() in location_results:
                location_results[word] = 1

        # Chronicity
        chronicity_results = None

        # Mutually exclusive
        if 'acute on chronic' in caption:
            chronicity_results = 2
        elif 'chronic' in caption:
            chronicity_results = 1
        elif 'acute' in caption:
            chronicity_results = 0

        return location_results, chronicity_results

    @classmethod
    def hammer_time(cls, model_name):
        df_results = pd.read_pickle(f'Results/Generated{model_name}.pickle')
        
        df_gen = df_results['generated'].apply(cls.parse_caption)
        df_true = df_results['ground_truth'].apply(cls.parse_caption)

        # Process separately for location and chronicity
        df_gen_location = df_gen.str[0]
        df_gen_location = pd.DataFrame.from_records(df_gen_location).values
        df_true_location = df_true.str[0]
        df_true_location = pd.DataFrame.from_records(df_true_location).values
        df_location = pd.DataFrame.from_records((df_gen_location, df_true_location)).T
        df_location.columns = ['PRED', 'TRUE']

        # Chronicity - Does not require any further processing
        df_gen_chronicity = df_gen.str[1]
        df_true_chronicity = df_true.str[1]
        df_chronicity = pd.concat([df_gen_chronicity, df_true_chronicity], axis=1)
        df_chronicity.columns = ['PRED', 'TRUE']
        df_chronicity = df_chronicity.dropna(subset=['TRUE'])
        
        # Save these data
        df_location.to_pickle(f'Results/LOCATION/Generated{model_name}.pickle')
        df_chronicity.to_pickle(f'Results/CHRONICITY/Generated{model_name}.pickle')
