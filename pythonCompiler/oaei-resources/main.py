import logging
import sys
import traceback
import os
from time import gmtime, strftime
from rdflib import Graph
from matchers.exact_match_dynamicfields import ExactMatchDynamicFields
from matchers.unsupervised_BERT import BERTUnSupervisedMatcher
from matchers.sentenceBERT import SentenceBERT
from filters.confidence_override import ConfidenceOverrider
from unioners.unioner import Unioner
from utils.AlignmentFormat import serialize_mapping_to_tmp_file

#Config
snetenceBert_model = "sentence-transformers/all-MiniLM-L6-v2"
trainedBert_model = "textattack/albert-base-v2-MRPC"


def match(source_url, target_url, input_alignment_url):

        logging.info("Python matcher info: Match " + source_url + " to " + target_url)

        source_graph = Graph()
        source_graph.parse(source_url)
        logging.info("Read source with %s triples.", len(source_graph))

        target_graph = Graph()
        target_graph.parse(target_url)
        logging.info("Read target with %s triples.", len(target_graph))
        input_alignment = None

        exactmatch_label_pool = ExactMatchDynamicFields("label").match_compute(source_graph,
                                                                          target_graph,
                                                                          input_alignment)

        exactmatch_altlabel_pool = ExactMatchDynamicFields("altlabel").match_compute(source_graph,
                                                                                target_graph,
                                                                                input_alignment)

        sentenceBert_pool = SentenceBERT(
            model_name=snetenceBert_model,
            node_fields="abstract",
            treshold=0.6,
            topk=6,
            cutabstract=2,
            device="cuda",).match_compute(source_graph, target_graph, input_alignment)

        supervised_pool = BERTUnSupervisedMatcher(
            model_name_sentencebert=snetenceBert_model,
            bert_treshold=0.6,
            bert_topk=6,
            model_name_trained_bert=trainedBert_model,
            negative_pairs=1,
            batch_size=1,
            epochs=100,
            abstract_cut=2,
            node_fields="abstract",
            learning_rate=1e-5,
            device="cuda").match_compute(source_graph, target_graph, exactmatch_label_pool, sentenceBert_pool)

        exactmatch_altlabel_pool = ConfidenceOverrider(0.998).filter(exactmatch_altlabel_pool)
        exactmatch_unioned_pool = Unioner().union([exactmatch_label_pool, exactmatch_altlabel_pool])
        merged_alignment = Unioner().union([exactmatch_unioned_pool, supervised_pool])

        alignment_file_url = serialize_mapping_to_tmp_file(merged_alignment)
        return alignment_file_url


def main(argv):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if len(argv) == 2:
        print(match(argv[0], argv[1], None))
    elif len(argv) >= 3:
        if len(argv) > 3:
            logging.error("Too many parameters but we will ignore them.")
        print(match(argv[0], argv[1], argv[2]))
    else:
        logging.error(
            "Too few parameters. Need at least two (source and target URL of ontologies"
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO
    )
    main(sys.argv[1:])
