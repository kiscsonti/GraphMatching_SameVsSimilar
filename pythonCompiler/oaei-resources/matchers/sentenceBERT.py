import logging

from rdflib import URIRef, RDFS
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from datetime import datetime
from utils.extracts import type_force_filter, skos_NS, get_abstract_single, get_label, get_altlabel
from utils.bert_common import semantic_seach_generate_pairs


class SentenceBERT:
    """
    Class that is a matcher
    Given a sentenceBert model name it loads the model and converts all the nodes with the given properties to an
    embedding representation. From these embeddings it is going to create TopX most similar pairs based on
    cosine similarity. There is also a treshold that discards pairs below it.
    """

    def __init__(self, model_name: str, node_fields, treshold: float = 0.6, topk: int = 6,
                 cutabstract=None, device="cuda"):
        self.name = "SentenceBERT"
        self.model_name = model_name
        self.node_fields = node_fields
        self.cutabs = cutabstract
        self.topk = topk
        self.treshold = treshold
        self.device = device
        if self.device == "cuda" and not torch.cuda.is_available():
            logging.warning("No cuda available! Running on CPU!")
            self.device = "cpu"

    def match_compute(self, source_graph, target_graph, input_alignment):

        (source_nodes, target_nodes,
         searched_space, switched) = self.get_pre_searched_space(source_graph, target_graph, input_alignment,
                                                                 [self.model_name, self.device, self.topk])

        alignment, extra_info = semantic_seach_generate_pairs(source_nodes, target_nodes, searched_space,
                                                              float(self.treshold), switched)
        alignment = list(filter(lambda x: type_force_filter(x, source_graph, target_graph), alignment))
        return alignment

    def get_pre_searched_space(self, source_graph, target_graph, input_alignment, params):
        """

        :param source_graph:
        :param target_graph:
        :param input_alignment:
        :param params: [model_name, device, topk]
        :return:
        """

        model = SentenceTransformer(params[0], device=params[1])
        print(datetime.now(), "Gather nodes...")

        source_nodes, source_values = self.get_node_values(source_graph)
        target_nodes, target_values = self.get_node_values(target_graph)

        print(datetime.now(), "BERT Embedding started...")
        source_embeddings = model.encode(source_values)
        src_emb_tensor = np.stack(source_embeddings)
        src_emb_tensor = torch.from_numpy(src_emb_tensor)

        target_embeddings = model.encode(target_values)
        trg_emb_tensor = np.stack(target_embeddings)
        trg_emb_tensor = torch.from_numpy(trg_emb_tensor)

        print(datetime.now(), "Generating pairs...")
        switched = True if len(source_nodes) > len(target_nodes) else False
        searched_space = self.get_top_pairs(src_emb_tensor, trg_emb_tensor, params[2], switched)
        return source_nodes, target_nodes, searched_space, switched

    def get_top_pairs(self, src_emb_tensor, trg_emb_tensor, topk, switched):

        print(datetime.now(), "Matching to target nodes...")

        if switched:
            searched_space = util.semantic_search(trg_emb_tensor, src_emb_tensor, top_k=topk)
        else:
            searched_space = util.semantic_search(src_emb_tensor, trg_emb_tensor, top_k=topk)

        return searched_space

    def get_value_list(self, graph, node):
        node_values = self.node_fields.split(",")
        node_representation = list()
        for val in node_values:
            val = val.strip()
            if val == "label":
                node_representation.append(get_label(graph, node))
            elif val == "altlabel":
                node_representation.append(get_altlabel(graph, node))
            elif val == "abstract":
                node_representation.append(get_abstract_single(graph, node, self.cutabs))
        return node_representation

    def get_value_string(self, graph, node):
        return " ".join(self.get_value_list(graph, node))

    def get_node_values(self, graph):
        nodes = set()
        values = list()

        for s, p, o in graph.triples((None, RDFS.label, None)):
            if isinstance(s, URIRef):
                nodes.add(s)
        nodes = list(nodes)

        for node in nodes:
            values.append(self.get_value_string(graph, node))
        return nodes, values
