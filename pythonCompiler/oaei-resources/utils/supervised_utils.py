from tqdm.auto import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import random
import math
from collections import defaultdict
from utils.extracts import get_abstract_single
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np


def create_train_examples(pos_alignments, neg_alignments):
    train_examples = list()
    for element in pos_alignments:
        train_examples.append([1, element[0], element[1]])
    for element in neg_alignments:
        train_examples.append([0, element[0], element[1]])
    return train_examples


def filter_non_abstracts(abstracts, entities):
    final_entities = list()
    final_abstracts = list()
    for pair, ent in zip(abstracts, entities):
        if len(pair[0]) > 1 and len(pair[1]) > 1:
            final_abstracts.append(pair)
            final_entities.append(ent)
    return final_entities, final_abstracts


def train_hugginface_model(model, datasets, device, optimizer, lr_scheduler):
    progress_bar = tqdm(range(sum([len(item) for item in datasets])))
    for epoch, ds in enumerate(datasets):
        for batch in ds:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


def eval_hugginface_model(model, dataset, device):
    pred_scores = list()
    with torch.no_grad():
        for batch in tqdm(dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            scores = outputs.logits
            probs = scores.softmax(dim=1)
            pred_scores.append(probs)
    pred_scores = torch.cat(pred_scores, dim=0)
    return pred_scores


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)


def create_dataset_from_pandas(df, tokenizer, device="cpu", batch_size: int = 4):
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'], device=device)
    dl = DataLoader(tokenized_dataset, shuffle=False, batch_size=batch_size)
    return dl


def get_train_val_data_pd(source_graph, target_graph, pos_pairs, neg_pairs, switched: bool, query_func, val_ratio: float = 0.2):
    val_pos_pair_ids = random.sample(range(len(pos_pairs)), math.floor(len(pos_pairs) * val_ratio))

    neg_dict = defaultdict(list)
    if switched:
        for pair in neg_pairs:
            neg_dict[pair[1]].append(pair[0])
    else:
        for pair in neg_pairs:
            neg_dict[pair[0]].append(pair[1])

    val_pairs = list()
    train_pairs = list()

    for item in range(len(pos_pairs)):
        if item in val_pos_pair_ids:
            val_pairs.append([pos_pairs[item][0], pos_pairs[item][1], 1])
        else:
            train_pairs.append([pos_pairs[item][0], pos_pairs[item][1], 1])

    for pair in list(val_pairs):
        if switched:
            val_pairs.extend([[item, pair[1], 0] for item in neg_dict[pair[1]]])
        else:
            val_pairs.extend([[pair[0], item, 0] for item in neg_dict[pair[0]]])

    for pair in list(train_pairs):
        if switched:
            train_pairs.extend([[item, pair[1], 0] for item in neg_dict[pair[1]]])
        else:
            train_pairs.extend([[pair[0], item, 0] for item in neg_dict[pair[0]]])

    train_df = pd.DataFrame(train_pairs, columns=["entity1", "entity2", "label"])
    val_df = pd.DataFrame(val_pairs, columns=["entity1", "entity2", "label"])

    train_df["sentence1"] = train_df["entity1"].apply(lambda x: get_abstract_single(source_graph, x))
    train_df["sentence2"] = train_df["entity2"].apply(lambda x: get_abstract_single(target_graph, x))
    val_df["sentence1"] = val_df["entity1"].apply(lambda x: get_abstract_single(source_graph, x))
    val_df["sentence2"] = val_df["entity2"].apply(lambda x: get_abstract_single(target_graph, x))
    train_df = train_df[((train_df["sentence1"] != "") & (train_df["sentence2"] != ""))]
    val_df = val_df[((val_df["sentence1"] != "") & (val_df["sentence2"] != ""))]

    return train_df, val_df


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="weighted")
    precision = precision_score(y_true=labels, y_pred=pred, average="weighted")
    f1 = f1_score(y_true=labels, y_pred=pred, average="weighted")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

