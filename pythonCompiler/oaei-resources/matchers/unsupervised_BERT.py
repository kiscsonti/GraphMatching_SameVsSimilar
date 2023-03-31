import os
from utils.bert_common import generate_neg_pairs
from utils.extracts import (type_force_filter, get_abstracts, get_abstract_single)
from utils.supervised_utils import tokenize_function, compute_metrics
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from utils.utilities import cut_sentences_spacy, cut_sentences_textblob
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import softmax
from matchers.sentenceBERT import SentenceBERT
# from base.io_services import load_result_alignment
import torch
import logging


class BERTUnSupervisedMatcher:

    def __init__(self, model_name_sentencebert: str, bert_treshold: float, bert_topk, model_name_trained_bert: str,
                 negative_pairs: int, batch_size=1, epochs=100, abstract_cut=None, node_fields=None,
                 learning_rate=1e-5, device="cuda"):
        if node_fields is None:
            node_fields = "label,altlabel"
        self.name = "BERTUnSupervisedMatcher"

        self.model_name_bert = model_name_sentencebert
        self.bert_topk = int(bert_topk)
        self.bert_treshold = int(bert_treshold)

        self.model_name = model_name_trained_bert
        self.negative_pairs = negative_pairs
        self.node_fields = node_fields

        self.batch_size = int(batch_size)
        self.epochs = epochs
        self.abstract_cut = abstract_cut
        self.lr = learning_rate

        self.device = device
        if self.device == "cuda" and not torch.cuda.is_available():
            logging.warning("No cuda available! Running on CPU!")
            self.device = "cpu"

        assert (self.model_name == 'distilbert-base-uncased' or
                self.model_name == "bert-base-uncased" or
                self.model_name == "bert-large-uncased" or
                self.model_name == "roberta-base" or
                self.model_name == "roberta-large" or
                self.model_name == "albert-base-v2" or
                self.model_name == "facebook/bart-base" or
                self.model_name == "facebook/bart-large" or
                self.model_name == "textattack/albert-base-v2-MRPC" or
                self.model_name == "prajjwal1/albert-base-v2-mnli" or
                self.model_name == "microsoft/deberta-v3-large" or
                self.model_name == "microsoft/deberta-v3-base"
                )

    def argmax_decision(self, raw_pred):
        argmaxes = np.argmax(raw_pred, axis=1)
        softmaxed_preds = softmax(raw_pred)
        return argmaxes, [softmaxed_preds[i][item] for i, item in enumerate(argmaxes)]

    def get_unioned_df(self, entity_pairs, values):
        df = pd.DataFrame({
            "entity1": [item[0] for item in entity_pairs],
            "entity2": [item[1] for item in entity_pairs],
            "sentence1": values[0],
            "sentence2": values[1],
        })
        return df

    @staticmethod
    def type_category_abstract_filter(alignment_pair, source_graph, target_graph):
        if not type_force_filter(alignment_pair, source_graph, target_graph):
            return True
        if ("" == get_abstract_single(source_graph, alignment_pair[0]) or
                "" == get_abstract_single(target_graph, alignment_pair[1])):
            return True
        return False

    def match_compute(self, source_graph, target_graph, training_alignment, prev_step_alignment_pool):

        # gold_alignment = load_result_alignment(self.positive_pool_path, self.task_name)

        #Generate train data
        bert_matcher = SentenceBERT(model_name=self.model_name_bert,
                                    node_fields=self.node_fields,
                                    treshold=self.bert_treshold,
                                    topk=self.bert_topk,
                                    cutabstract=self.abstract_cut,
                                    device=self.device)
        source_nodes, target_nodes, searched_space, switched = bert_matcher.get_pre_searched_space(
            source_graph, target_graph, None, [self.model_name_bert, self.device, self.bert_topk]
        )

        train_pos_input_alignment, test_pos_input_alignment = train_test_split(training_alignment, test_size=0.15)

        train_neg_input_alignment = generate_neg_pairs(source_graph, target_graph, source_nodes, target_nodes,
                                                       searched_space, train_pos_input_alignment, switched,
                                                       self.negative_pairs,
                                                       BERTUnSupervisedMatcher.type_category_abstract_filter)
        test_neg_input_alignment = generate_neg_pairs(source_graph, target_graph, source_nodes, target_nodes,
                                                      searched_space, test_pos_input_alignment, switched,
                                                      self.negative_pairs,
                                                      BERTUnSupervisedMatcher.type_category_abstract_filter)

        train_pos_abstracts = get_abstracts(source_graph, target_graph, train_pos_input_alignment)
        test_pos_abstracts = get_abstracts(source_graph, target_graph, test_pos_input_alignment)
        train_neg_abstracts = get_abstracts(source_graph, target_graph, train_neg_input_alignment)
        test_neg_abstracts = get_abstracts(source_graph, target_graph, test_neg_input_alignment)
        pool_abstracts = get_abstracts(source_graph, target_graph, prev_step_alignment_pool)

        if self.abstract_cut:
            train_pos_abstracts = (cut_sentences_textblob(train_pos_abstracts[0], self.abstract_cut),
                                   cut_sentences_textblob(train_pos_abstracts[1], self.abstract_cut))

            test_pos_abstracts = (cut_sentences_textblob(test_pos_abstracts[0], self.abstract_cut),
                                  cut_sentences_textblob(test_pos_abstracts[1], self.abstract_cut))
            train_neg_abstracts = (cut_sentences_textblob(train_neg_abstracts[0], self.abstract_cut),
                                   cut_sentences_textblob(train_neg_abstracts[1], self.abstract_cut))
            test_neg_abstracts = (cut_sentences_textblob(test_neg_abstracts[0], self.abstract_cut),
                                  cut_sentences_textblob(test_neg_abstracts[1], self.abstract_cut))
            pool_abstracts = (cut_sentences_textblob(pool_abstracts[0], self.abstract_cut),
                              cut_sentences_textblob(pool_abstracts[1], self.abstract_cut))

        pool_df = pd.DataFrame({
            "entity1": [item[0] for item in prev_step_alignment_pool],
            "entity2": [item[1] for item in prev_step_alignment_pool],
            "sentence1": pool_abstracts[0],
            "sentence2": pool_abstracts[1],
        })
        print("Pool initial length:", len(pool_df))
        pool_df = pool_df[(pool_df["sentence1"] != "") | (pool_df["sentence2"] != "")]
        print("Pool valids length:", len(pool_df))


        pos_label = 1
        neg_label = 0
        train_pos_df = self.get_unioned_df(train_pos_input_alignment, train_pos_abstracts)
        train_pos_df["label"] = pos_label
        train_neg_df = self.get_unioned_df(train_neg_input_alignment, train_neg_abstracts)
        train_neg_df["label"] = neg_label
        train_df = pd.concat([train_pos_df,
                              train_neg_df])
        train_df = train_df[(train_df["sentence1"] != "") | (train_df["sentence2"] != "")]

        test_pos_df = self.get_unioned_df(test_pos_input_alignment, test_pos_abstracts)
        test_pos_df["label"] = pos_label
        test_neg_df = self.get_unioned_df(test_neg_input_alignment, test_neg_abstracts)
        test_neg_df["label"] = neg_label
        test_df = pd.concat([test_pos_df,
                             test_neg_df])
        test_df = test_df[(test_df["sentence1"] != "") | (test_df["sentence2"] != "")]

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(test_df)

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

        # Train model
        training_args = TrainingArguments(output_dir="test_trainer",
                                          evaluation_strategy="epoch",
                                          save_strategy="epoch",
                                          num_train_epochs=self.epochs,
                                          load_best_model_at_end=True,
                                          eval_steps=500,
                                          logging_steps=10,
                                          per_device_train_batch_size=self.batch_size,
                                          per_device_eval_batch_size=self.batch_size,
                                          learning_rate=self.lr,
                                          )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        trainer.train()

        #Predict over pool
        pred_df = [[0, row["sentence1"], row["sentence2"]] for i, row in pool_df.iterrows()]
        pred_df = pd.DataFrame(pred_df, columns=["label", "sentence1", "sentence2"])

        pred_dataset = Dataset.from_pandas(pred_df)
        pred_dataset = pred_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        raw_pred, _, _ = trainer.predict(pred_dataset)

        y_pred, confidence = self.argmax_decision(raw_pred)

        resulting_alignment = list()
        for i, pred in enumerate(y_pred):
            if pred == 1:
                resulting_alignment.append([pool_df.iloc[i]["entity1"],
                                            pool_df.iloc[i]["entity2"],
                                            "=", confidence[i]])

        print(self.name, "alignment length:", len(resulting_alignment))
        return resulting_alignment
