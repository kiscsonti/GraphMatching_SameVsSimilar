from rdflib import URIRef, RDFS, Namespace
from typing import List
from utils.utilities import cut_sentences_textblob, cut_sentences_textblob_single

NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
owl_NS = Namespace("http://www.w3.org/2002/07/owl#")
skos_NS = Namespace("http://www.w3.org/2004/02/skos/core#")
ontology_NS = Namespace("http://dbkwik.webdatacommons.org/ontology/")
log_str = ""


def get_field_relation(field):

    if field == "abstract":
        return ontology_NS.abstract
    elif field == "label":
        return RDFS.label
    elif field == "altlabel":
        return skos_NS.altLabel
    else:
        raise NotImplementedError


def get_abstracts_allnodes(source_graph, target_graph):
    src_abstracts = dict()
    trg_abstracts = dict()

    for s, p, o in source_graph.triples((None, ontology_NS.abstract, None)):
        o = clean_abstract(str(o))
        src_abstracts[str(s)] = o

    for s, p, o in target_graph.triples((None, ontology_NS.abstract, None)):
        o = clean_abstract(str(o))
        trg_abstracts[str(s)] = o

    return src_abstracts, trg_abstracts


def get_abstracts(source_graph, target_graph, input_alignment=None):
    if input_alignment is None:
        return get_abstracts_allnodes(source_graph, target_graph)
    else:
        return get_abstracts_alignment(source_graph, target_graph, input_alignment)


def get_abstract_single(graph, element, cutabs=None):
    try:
        _, _, value = list(graph.triples((URIRef(element), ontology_NS.abstract, None)))[0]
    except Exception as e:
        value = ""
    if cutabs and value != "":
        value = cut_sentences_textblob_single(value, cutabs)
    return value


def get_label(graph, element):
    try:
        _, _, value = list(graph.triples((URIRef(element), RDFS.label, None)))[0]
    except Exception as e:
        value = ""
    return value


def get_altlabel(graph, element):
    try:
        _, _, value = list(graph.triples((URIRef(element), skos_NS.altLabel, None)))[0]
    except Exception as e:
        value = ""
    return value


def get_xy_alignment(source_graph, target_graph, input_alignment, getter_func):
    src_values = list()
    trg_values = list()
    for alignment_pair in input_alignment:
        src_val = getter_func(source_graph, alignment_pair[0])
        target_val = getter_func(target_graph, alignment_pair[1])
        src_val = str(src_val).replace("\n", "")
        target_val = str(target_val).replace("\n", "")
        src_values.append(src_val)
        trg_values.append(target_val)
    return src_values, trg_values


def clean_abstract(abstract_str):
    return str(abstract_str).replace("\n", " ").replace("WtXmlEmptyTag", "").strip()


def get_abstracts_alignment(source_graph, target_graph, input_alignment):
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
        src_abstract = clean_abstract(str(src_abstract))
        trg_abstract = clean_abstract(str(trg_abstract))
        src_abstracts.append(src_abstract)
        trg_abstracts.append(trg_abstract)
    return src_abstracts, trg_abstracts


def type_force_filter(alignment_row, source_graph, target_graph):
    if ("Category" in str(alignment_row[0])) != ("Category" in str(alignment_row[1])):
        return False
    node_a = URIRef(alignment_row[0])
    node_b = URIRef(alignment_row[1])

    node_a_type = get_type(source_graph, node_a)
    node_b_type = get_type(target_graph, node_b)
    if (is_class_tpye(node_b_type) != is_class_tpye(node_a_type) or
            is_property_tpye(node_b_type) != is_property_tpye(node_a_type)):
        return False
    return True


def get_type(graph, node):
    try:
        return list(graph.triples((node, NS.type, None)))[0][2]
    except Exception as e:
        return None


def is_class_tpye(node):
    return node == owl_NS.Class


def is_property_tpye(node):
    return node == NS.Property


def get_first_connection(graph, node_name, conn_type):
    connections = list(graph.triples((URIRef(node_name), conn_type, None)))
    if len(connections) == 0:
        ""
    else:
        return connections[0][-1]


def get_first_connection_list(graph, nodes: List[str], conn_type):
    return [get_first_connection(graph, node, conn_type) for node in nodes]


def type_category_abstract_filter(alignment_pair, source_graph, target_graph):
    if not type_force_filter(alignment_pair, source_graph, target_graph):
        return True
    if ("" == get_abstract_single(source_graph, alignment_pair[0]) or
            "" == get_abstract_single(target_graph, alignment_pair[1])):
        return True
    return False
