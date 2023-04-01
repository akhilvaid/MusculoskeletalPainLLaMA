# Config

class Config:
    # Paths
    notes = 'Data/Notes.xlsx'
    training_data = 'Data/Train.pickle'
    testing_data = 'Data/Test.pickle'

    result_dir = 'Results'
    model_dir = 'Models'
    cache_dir = 'ModelCache'

    # Models
    models = {
        'BERT': {
            'model_identifier': 'bert-base-uncased',
            'max_length': 512,
        },
        'Longformer': {
            'model_identifier': 'allenai/longformer-base-4096',
            'max_length': 4096,
        },
    }

    location_mapping = {
        1: 'SHOULDER_PAIN',
        2: 'LOWER_BACK_PAIN',
        3: 'KNEE_PAIN',
        4: 'OTHER_PAIN'
    }

    location_to_text = {
        'SHOULDER_PAIN': 'A  Shoulder Pain',
        'LOWER_BACK_PAIN': 'B  Lower Back Pain',
        'KNEE_PAIN': 'C  Knee Pain',
        'OTHER_PAIN': 'D  Other Pain'
    }

    chronicity_mapping = {
        0: 'ACUTE',
        1: 'CHRONIC',
        2: 'ACUTE_ON_CHRONIC'
    }

    chronicity_to_text = {
        'ACUTE': 'B  Acute',
        'CHRONIC': 'C  Chronic',
        'ACUTE_ON_CHRONIC': 'D  Acute on Chronic'
    }

    llama_name_mapping = {
        'FTNotesOnly': 'LLaMA-7B',
        'FTNotesAndAlpaca': 'LLaMA-7B (Notes + Alpaca)',
    }

    # Transformer
    batch_size = 16  # For 4x A100s

    # Fine Tuning
    n_epochs = 25
    patience = 8

    # Resourcing
    preprocess_workers = 40
    gpus = 4

    # General
    debug = False
    random_state = 42
