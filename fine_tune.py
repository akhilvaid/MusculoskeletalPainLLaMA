#!/bin/python

import os

import tqdm
import torch
import transformers
import pandas as pd
import numpy as np

from config import Config
from sklearn import metrics
from torch.cuda.amp import autocast, GradScaler

torch.manual_seed(Config.random_state)
torch.cuda.manual_seed(Config.random_state)
transformers.set_seed(Config.random_state)
torch.backends.cudnn.benchmark = True


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df_text, tokenizer, outcome, max_length):
        self.df = df_text
        self.tokenizer = tokenizer
        self.outcome = outcome
        self.max_length = max_length

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if self.outcome == 'LOCATION':
            label = self.df.iloc[index]\
                [['SHOULDER_PAIN', 'LOWER_BACK_PAIN', 'KNEE_PAIN', 'OTHER_PAIN']].values.astype(float)
            label_tensor = torch.tensor(label, dtype=torch.float)  # BCE loss

        elif self.outcome == 'CHRONICITY':
            label = self.df.iloc[index]['CHRONICITY'] - 1
            label_tensor = torch.tensor(label, dtype=torch.long)  # Cross entropy loss

        text = self.df.iloc[index]['NOTES'].lower()
        encoded = self.tokenizer.encode_plus(
            text, max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_length=True)

        ids = encoded['input_ids'].squeeze()  # Not sure why an extra dimension is added
        attention_mask = encoded['attention_mask'].squeeze()

        # TODO See if attention mask adds anything
        return ids, attention_mask, label_tensor


class FineTune:
    def __init__(self, model_name, outcome=None):
        self.model_name = model_name
        self.model_identifier = Config.models[model_name]['model_identifier']
        self.max_length = Config.models[model_name]['max_length']
        self.outcome = outcome

        print(f'Fine-tuning {self.model_name} for {self.outcome}...')

        # Just in case
        assert self.model_name in Config.models.keys()
        assert self.outcome in ('CHRONICITY', 'LOCATION')

        # Dirs
        self.result_dir = os.path.join(Config.result_dir, self.outcome, model_name)
        self.model_dir = os.path.join(Config.model_dir, self.outcome, model_name)

        # Housekeeping
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def eval_model(self, model, dataloader, epoch):
        model.eval()

        if self.outcome == 'LOCATION':
            criterion = torch.nn.BCEWithLogitsLoss()  # Multi-label classification
        else:
            criterion = torch.nn.CrossEntropyLoss()

        epoch_loss = 0
        all_labels = []
        all_pred = []

        for batch in tqdm.tqdm(dataloader):
            with torch.no_grad():
                with autocast():
                    ids = batch[0].cuda()
                    attention_mask = batch[1].cuda()
                    labels = batch[2].cuda()

                    # Does NOT return a conventional tensor
                    output = model(ids, attention_mask=attention_mask)
                    loss = criterion(output.logits, labels)

                    epoch_loss += loss.item() * ids.shape[0]

            all_labels.extend(labels.squeeze().tolist())
            
            if self.outcome == 'LOCATION':
                all_pred.extend(torch.sigmoid(output.logits.squeeze()).tolist())
            else:
                all_pred.extend(torch.softmax(output.logits, dim=1).tolist())  # CrossEntropy loss

        eval_loss = epoch_loss / len(dataloader)

        # Save preds
        df_y = pd.DataFrame([all_labels, all_pred]).T
        df_y.columns = ['TRUE', 'PRED']

        outfile_path = os.path.join(self.result_dir, f'epoch_{epoch}.pickle')
        df_y.to_pickle(outfile_path)

        return eval_loss

    def load_and_ft_checkpoint(self, train_dataloader, test_dataloader):
        performance_track = []  # For all performance
        metric_track = []  # For patience

        if self.outcome == 'LOCATION':
            num_labels = 4
            criterion = torch.nn.BCEWithLogitsLoss()  # Multi-label classification
        else:
            num_labels = 3
            criterion = torch.nn.CrossEntropyLoss()

        # Same thing but with a pretrained Longformer model
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_identifier,
            cache_dir=Config.cache_dir,
            num_labels=num_labels,
            output_attentions=False,
            local_files_only=False)

        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model.cuda()

        optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scaler = GradScaler()

        model.train()
        for epoch in range(Config.n_epochs):
            epoch_loss = 0

            for batch in tqdm.tqdm(train_dataloader):

                # Same as optim.zero_grad()
                for param in model.parameters():
                    param.grad = None

                ids = batch[0].cuda()
                attention_mask = batch[1].cuda()
                labels = batch[2].cuda()

                # Forward pass
                output = model(ids, attention_mask=attention_mask)
                loss = criterion(output.logits, labels)
                epoch_loss += loss.item() * ids.shape[0]

                # Gradient scaling for AMP
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

            # Eval on test set
            eval_loss = self.eval_model(model, test_dataloader, epoch)
            eval_loss_metric = 1 - eval_loss

            # Overall epoch loss
            training_loss = epoch_loss / len(train_dataloader)
            print('Epoch:', epoch, 'Tr loss:', training_loss, 'Te Loss:', eval_loss)

            performance_track.append((self.outcome, self.model, epoch, training_loss, eval_loss))
            df_perf = pd.DataFrame(
                performance_track,
                columns=['outcome', 'model', 'epoch', 'training_loss', 'eval_loss'])

            # Keep track of performance
            metric_track.append(eval_loss_metric)

            # Save model
            if eval_loss_metric >= max(metric_track):
                print('Saving model')
                outfile_path = os.path.join(self.model_dir, f'epoch_{epoch}.pth')
                torch.save(model.state_dict(), outfile_path)

            # Patience
            if len(metric_track) >= Config.patience:
                perf_eval = metric_track[(len(metric_track) - Config.patience):]

                _yardstick = max(perf_eval)  # Tracking loss
                if _yardstick == perf_eval[0]:
                    print(f'Patience threshold exceeded @E {epoch} @TP {perf_eval[0]} > {perf_eval[-1]}')

                    write_header = True
                    perf_outfile = f'Performance_{self.outcome}.csv'
                    if os.path.exists(perf_outfile):
                        write_header = False
                    df_perf.to_csv(
                        perf_outfile,
                        mode='a', index=False, header=write_header)
                    return

    def hammer_time(self):
        # Load tokenizer
        tokenizer_base_ref = (
            transformers.LongformerTokenizer if self.model_name == 'Longformer'
            else transformers.BertTokenizer)
        tokenizer = tokenizer_base_ref.from_pretrained(
            self.model_identifier,
            cache_dir=Config.cache_dir,
            local_files_only=False)

        # Load dataset
        df_train = pd.read_pickle(Config.training_data)
        df_test = pd.read_pickle(Config.testing_data)

        if Config.debug:
            print('DEBUG MODE')
            df_train = df_train.sample(100)
            setattr(Config, 'n_epochs', 2)

        if self.outcome == 'CHRONICITY':
            start_len = df_train.shape[0]
            df_train = df_train[df_train['CHRONICITY'] != 0]
            assert df_train.shape[0] < start_len

            df_test = df_test[df_test['CHRONICITY'] != 0]

        train_dataset = TextDataset(df_train, tokenizer, self.outcome, self.max_length)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=Config.batch_size, shuffle=True,
            num_workers=Config.preprocess_workers, pin_memory=True)

        test_dataset = TextDataset(df_test, tokenizer, self.outcome, self.max_length)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=Config.batch_size, shuffle=False,
            num_workers=Config.preprocess_workers, pin_memory=True)

        # TTE
        self.load_and_ft_checkpoint(train_dataloader, test_dataloader)
