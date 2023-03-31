from rdflib import URIRef, RDFS
from utils.extracts import skos_NS, get_type, is_class_tpye, is_property_tpye, get_field_relation
from collections import defaultdict


class ExactMatchDynamicFields:

    def __init__(self, fields: str):
        self.name = "ExactMatchDynamicFields"
        self.fields = fields
        self.resulting_alignment = list()

    def init_label2uri(self, source_graph):

        label2uri = defaultdict(set)

        for field in self.fields.split(","):
            field_relation = get_field_relation(field)

            for s, p, o in source_graph.triples((None, field_relation, None)):
                if isinstance(s, URIRef):
                    label2uri[str(o).lower()].add(s)
        return label2uri

    def match_compute(self, source_graph, target_graph, input_alignment):
        """
        Here i am matchin on both altLabel and label strings
        :param source_graph:
        :param target_graph:
        :param input_alignment:
        :param params:
        :return:
        """
        alignment = []

        label2uri = self.init_label2uri(source_graph)

        valid_target_nodes = set()
        for s, p, o in target_graph.triples((None, RDFS.label, None)):
            if isinstance(s, URIRef):
                valid_target_nodes.add(s)

        for node in valid_target_nodes:
            # match labels

            node_alignments = dict()
            for field in self.fields.split(","):
                field_relation = get_field_relation(field)
                for s, p, o in target_graph.triples((node, field_relation, None)):
                    node_alignments.update(self.is_match(s, o, label2uri, target_graph, source_graph))

            alignment.extend(node_alignments.values())

        return alignment

    def is_match(self, node_subject, node_object, label_to_uri, t_graph, s_graph):
        target_type = get_type(t_graph, node_subject)
        uncased_o = str(node_object).lower()
        alignments = dict()
        for one_uri in label_to_uri[uncased_o]:
            source_type = get_type(s_graph, one_uri)
            if ("Category" in str(one_uri)) != ("Category" in str(node_subject)):
                continue
            if (is_class_tpye(target_type) == is_class_tpye(source_type) and
                    is_property_tpye(target_type) == is_property_tpye(source_type)):
                alignments[one_uri] = (str(one_uri), str(node_subject), "=", 1.0)
        return alignments

