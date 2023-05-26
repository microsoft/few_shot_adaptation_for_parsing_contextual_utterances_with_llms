'''
This script is used to train a classifier to predict whether a turn part needs to be revised or not.
'''
import argparse
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import Dataset, random_split
import jsonlines
from dataclasses import dataclass
from random import sample
import random
random.seed(711)

@dataclass
class Example:
    utterance: str
    label: str
    dialogue_id: str
    turn_part_id: int

class RevisionClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, full_examples):
        self.encodings = encodings
        self.labels = labels
        self.full_examples = full_examples

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['full_example'] = self.full_examples[idx]
        return item

    def __len__(self):
        return len(self.labels)

def train_classifier(find_event_revise_train_file, find_event_revise_valid_file, find_event_train_file, find_event_valid_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    all_train_examples = []
    find_event_revise_train_data = []
    with jsonlines.open(find_event_revise_train_file) as reader:
        for line in reader:
            find_event_revise_train_data.append(Example(utterance=line['utterance'], label=1, dialogue_id=line['dialogue_id'], turn_part_id=line['turn_part_id']))

    find_event_train_examples = []
    with jsonlines.open(find_event_train_file) as reader:
        for line in reader:
            find_event_train_examples.append(Example(utterance=line['utterance'], label=0, dialogue_id=line['dialogue_id'], turn_part_id=line['turn_part_id']))
    all_train_examples.extend(find_event_revise_train_data)
    all_train_examples.extend(sample(find_event_train_examples, len(find_event_revise_train_data)))
    print(len(all_train_examples))


    # Do the same thing for the validation data
    find_event_revise_val_data = []
    with jsonlines.open(find_event_revise_valid_file) as reader:
        for line in reader:
            find_event_revise_val_data.append(Example(utterance=line['utterance'], label=1, dialogue_id=line['dialogue_id'], turn_part_id=line['turn_part_id']))
            
    find_event_val_examples = []
    with jsonlines.open(find_event_valid_file) as reader:
        for line in reader:
            find_event_val_examples.append(Example(utterance=line['utterance'], label=0, dialogue_id=line['dialogue_id'], turn_part_id=line['turn_part_id']))
            
    all_val_examples = []
    all_val_examples.extend(find_event_revise_val_data)
    all_val_examples.extend(sample(find_event_val_examples, len(find_event_revise_val_data)))
    print(len(all_val_examples))



    train_labels = [example.label for example in all_train_examples]
    val_labels = [example.label for example in all_val_examples]
    train_encodings = tokenizer([example.utterance for example in all_train_examples], truncation=True, padding=True)
    val_encodings = tokenizer([example.utterance for example in all_val_examples], truncation=True, padding=True)

    revision_dataset_train =  RevisionClassificationDataset(train_encodings, train_labels, all_train_examples)
    revision_dataset_val =  RevisionClassificationDataset(val_encodings, val_labels, all_val_examples)

    model.to(device)
    training_args = TrainingArguments(output_dir='./revision_results_roberta', num_train_epochs=100, logging_steps=100, save_steps=100,
                                    per_device_train_batch_size=2, gradient_accumulation_steps=8, per_device_eval_batch_size=8,
                                    warmup_steps=100, weight_decay=0.05, logging_dir='./logs', report_to = 'none', fp16=True)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    Trainer(model=model,  
            args=training_args, 
            train_dataset=revision_dataset_train, 
            eval_dataset=revision_dataset_val).train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--find_event_revise_train_file', type=str, required=True)
    parser.add_argument('--find_event_revise_valid_file', type=str, required=True)
    parser.add_arguemnt('--find_event_train_file', type=str, required=True)
    parser.add_arguemnt('--find_event_valid_file', type=str, required=True)
    parser.add_arguemnt('--output_dir', type=str, required=True)

    args = parser.parse_args()
    train_classifier(args.find_event_revise_train_file, args.find_event_revise_valid_file, args.find_event_train_file, args.find_event_valid_file, args.output_dir)