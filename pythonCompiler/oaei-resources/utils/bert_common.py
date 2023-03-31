from rdflib import URIRef, RDFS
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime
import numpy as np
from collections import defaultdict
from utils.extracts import ontology_NS
from itertools import compress


def semantic_seach_generate_pairs(source_nodes, target_nodes, semantic_search_result, treshold: float, switched: bool):
    should_be_cnt = 0
    alignment = list()
    for i, row in enumerate(semantic_search_result):
        for element in row:
            if element["score"] >= treshold:
                if switched:
                    alignment.append([str(source_nodes[element["corpus_id"]]),
                                      str(target_nodes[i]),
                                      "=",
                                      element["score"]])
                    should_be_cnt += 1
                else:
                    alignment.append([str(source_nodes[i]),
                                      str(target_nodes[element["corpus_id"]]),
                                      "=",
                                      element["score"]])
                    should_be_cnt += 1
    extra_info = "_" + str(should_be_cnt) + "_" + str(len(alignment))
    return alignment, extra_info


def generate_pos_and_neg(source_nodes, target_nodes, semantic_search_result, treshold: float, switched: bool, neg_cnt,
                         pos_topk):
    should_be_cnt = 0
    pos_alignment = list()
    neg_alignment = list()
    for i, row in enumerate(semantic_search_result):
        for element in row[0:pos_topk]:
            if element["score"] >= treshold:
                if switched:
                    pos_alignment.append([str(source_nodes[element["corpus_id"]]),
                                          str(target_nodes[i]),
                                          "=",
                                          element["score"]])
                    should_be_cnt += 1
                else:
                    pos_alignment.append([str(source_nodes[i]),
                                          str(target_nodes[element["corpus_id"]]),
                                          "=",
                                          element["score"]])
                    should_be_cnt += 1
        # print(len(row[len(row) - neg_cnt:len(row)]))
        if row[0]["score"] > treshold:
            for negatives in row[len(row) - neg_cnt:len(row)]:
                if switched:

                    neg_alignment.append([str(source_nodes[negatives["corpus_id"]]),
                                          str(target_nodes[i]),
                                          "=",
                                          negatives["score"]])
                else:
                    neg_alignment.append([str(source_nodes[i]),
                                          str(target_nodes[negatives["corpus_id"]]),
                                          "=",
                                          negatives["score"]])

    extra_info = "_" + str(should_be_cnt) + "_" + str(len(pos_alignment))
    return pos_alignment, neg_alignment, extra_info


def generate_neg_inputal(source_nodes, target_nodes, semantic_search_result, input_alignment, neg_cnt,
                         treshold: float, switched: bool, skip_size: int = 0):
    neg_alignment = list()
    if switched:
        nodes_tmp = set([str(item[1]) for item in input_alignment])
        nodes_ids = [i for i, item in enumerate(target_nodes) if str(item) in nodes_tmp]
    else:
        nodes_tmp = set([str(item[0]) for item in input_alignment])
        nodes_ids = [i for i, item in enumerate(source_nodes) if str(item) in nodes_tmp]

    inputal_dict = defaultdict(set)
    for item in input_alignment:
        inputal_dict[str(item[0])].add(str(item[1]))

    for item in nodes_ids:
        elem_cnt = 0
        # found = False
        for i, it in enumerate(semantic_search_result[item]):
            if i < skip_size:
                continue
            if switched:
                src_node = str(source_nodes[it["corpus_id"]])
                target_node = str(target_nodes[item])
            else:
                src_node = str(source_nodes[item])
                target_node = str(target_nodes[it["corpus_id"]])
            if target_node in inputal_dict[src_node]:
                # found = True
                continue
            if it["score"] < treshold:
                # print("Score:", item, elem_cnt, found)
                break
            if elem_cnt >= neg_cnt:
                # print("Count:", item, elem_cnt, found)
                break

            elem_cnt += 1
            neg_alignment.append([src_node,
                                  target_node,
                                  "=",
                                  it["score"]])
    return neg_alignment


def generate_neg_inputal_old(source_nodes, target_nodes, semantic_search_result, input_alignment, neg_cnt,
                         treshold: float, switched: bool):
    inputal_dict = defaultdict(set)
    for item in input_alignment:
        inputal_dict[str(item[0])].add(str(item[1]))

    should_be_cnt = 0
    neg_alignment = list()
    for i, row in enumerate(semantic_search_result):
        counted = 0
        for negatives in row:
            if counted >= neg_cnt:
                break
            if negatives["score"] > treshold:

                if switched:
                    src_node = str(source_nodes[negatives["corpus_id"]])
                    target_node = str(target_nodes[i])
                else:
                    src_node = str(source_nodes[i])
                    target_node = str(target_nodes[negatives["corpus_id"]])

                if len(inputal_dict[src_node]) == 0:
                    break

                if target_node in inputal_dict[src_node]:
                    continue
                else:
                    counted += 1
                    neg_alignment.append([src_node,
                                          target_node,
                                          "=",
                                          negatives["score"]])

    return neg_alignment


def generate_alignment_pool(source_nodes, target_nodes, semantic_search_result, pos_cnt,
                            treshold: float, switched: bool):
    pool_alignment = list()
    for i, row in enumerate(semantic_search_result):
        counted = 0
        for pos_row in row[:pos_cnt]:
            if pos_row["score"] > treshold:

                if switched:
                    src_node = str(source_nodes[pos_row["corpus_id"]])
                    target_node = str(target_nodes[i])
                else:
                    src_node = str(source_nodes[i])
                    target_node = str(target_nodes[pos_row["corpus_id"]])

                counted += 1
                pool_alignment.append([src_node,
                                       target_node,
                                       "=",
                                       pos_row["score"]])

    return pool_alignment


def get_abstracts(source_graph, target_graph, input_alignment):
    src_abstracts = list()
    trg_abstracts = list()
    for alignment_pair in input_alignment:
        try:
            _, _, src_abstract = list(source_graph.triples((URIRef(alignment_pair[0]), ontology_NS.abstract, None)))[0]
        except Exception as e:
            src_abstract = ""
        try:
            _, _, trg_abstract = list(target_graph.triples((URIRef(alignment_pair[1]), ontology_NS.abstract, None)))[0]
        except Exception as e:
            trg_abstract = ""
        src_abstract = str(src_abstract).replace("\n", "")
        trg_abstract = str(trg_abstract).replace("\n", "")
        src_abstracts.append(src_abstract)
        trg_abstracts.append(trg_abstract)
    return src_abstracts, trg_abstracts


def filter_by_nli(alignments, abstracts, probs, treshold):
    selected_elements = list()
    for i, item in enumerate(alignments):
        if abstracts[i][0] == "" or abstracts[i][1] == "":
            selected_elements.append(False)
        else:
            softmax_score = probs[i, :]
            print(softmax_score[0])
            if softmax_score[0] > treshold:
                selected_elements.append(True)
            else:
                selected_elements.append(False)
    return list(compress(alignments, selected_elements))


def generate_neg_pairs(source_graph, target_graph, source_nodes, target_nodes, searched_space, pos_pairs, switched, size: int,
                       exclude_filter=None):
    """
    This function generates negative pairs given a list of positive pairs. Given the BERT models' most similar k pairs
    it will select the top [size] elements. If this top [size] contains the positive pair that pair will be skipped
    and not be considered as a negative
    :param source_nodes: List of nodes
    :param target_nodes: List of nodes
    :param searched_space: BERT semantic search result
    :param pos_pairs: Positive pairs from gold data
    :param switched: Whether the semantic search result is transposed or not
    :param size: How many examples to generate for each positive example
    :param exclude_filter: Function that excludes some pairs
    :return:
    """
    neg_alignment = list()
    if switched:
        nodes_tmp = set([str(item[1]) for item in pos_pairs])
        nodes_ids = [i for i, item in enumerate(target_nodes) if str(item) in nodes_tmp]
    else:
        nodes_tmp = set([str(item[0]) for item in pos_pairs])
        nodes_ids = [i for i, item in enumerate(source_nodes) if str(item) in nodes_tmp]

    inputal_dict = defaultdict(set)
    for item in pos_pairs:
        inputal_dict[str(item[0])].add(str(item[1]))

    for item in nodes_ids:
        elem_cnt = 0
        for i, top_ith_element in enumerate(searched_space[item]):
            if elem_cnt >= size:
                break
            if switched:
                src_node = str(source_nodes[top_ith_element["corpus_id"]])
                target_node = str(target_nodes[item])
            else:
                src_node = str(source_nodes[item])
                target_node = str(target_nodes[top_ith_element["corpus_id"]])

            if target_node in inputal_dict[src_node]:
                continue
            if exclude_filter is not None and exclude_filter([src_node, target_node], source_graph, target_graph):
                continue

            elem_cnt += 1
            neg_alignment.append([src_node,
                                  target_node,
                                  "=",
                                  top_ith_element["score"]])
    return neg_alignment
