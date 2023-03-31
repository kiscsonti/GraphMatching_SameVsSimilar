from collections import defaultdict
from spacy.lang.en import English
from textblob import TextBlob
from itertools import compress


def union_alignment(alignment1, alignment2):
    merged_dict = dict()

    for item in alignment1:
        if item[0] not in merged_dict:
            merged_dict[item[0]] = {item[1]: [item[2], item[3]]}
        else:
            merged_dict[item[0]][item[1]] = [item[2], item[3]]

    for item in alignment2:
        if item[0] not in merged_dict:
            merged_dict[item[0]] = {item[1]: [item[2], item[3]]}
        else:
            element = merged_dict[item[0]]
            if item[1] not in element:
                element[item[1]] = [item[2], item[3]]
            else:
                element[item[1]] = [item[2], max(float(item[3]), float(element[item[1]][1]))]

    merged_alignment = list()

    for key, value in merged_dict.items():
        for key2, value2 in value.items():
            merged_alignment.append((key, key2, value2[0], value2[1]))

    return merged_alignment


def get_graph_name(graph_nodes):
    for item in graph_nodes:
        if "starwars.wikia.com" in str(item):
            return "starwars"
        elif "swg.wikia.com" in str(item):
            return "swg"
        elif "swtor.wikia.com" in str(item):
            return "swtor"
        elif "memory-alpha.wikia.com" in str(item):
            return "memory-alpha"
        elif "stexpanded.wikia.com" in str(item):
            return "stexpanded"
        elif "memory-beta.wikia.com" in str(item):
            return "memory-beta"
        elif "marvel.wikia.com" in str(item):
            return "marvel"
        elif "marvelcinematicuniverse.wikia.com" in str(item):
            return "marvelcinematicuniverse"


def get_file_from_url(location):
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname, urlopen

    if location.startswith("file:"):
        return open(url2pathname(unquote(urlparse(location).path)))
    else:
        return urlopen(location)


def force_1_to_1_match(alignment):
    src_al = set()
    trg_al = set()
    reduced_alignment = list()

    sorted_alignment = sorted(alignment, key=lambda x: x[3], reverse=True)
    for item in sorted_alignment:
        if item[0] in src_al:
            continue
        if item[1] in trg_al:
            continue
        reduced_alignment.append(item)
        src_al.add(item[0])
        trg_al.add(item[1])
    return reduced_alignment


def swap_src_trg(alignment):
    swapped = list()
    for item in alignment:
        swapped.append((item[1], item[0], item[2], item[3]))
    return swapped


def cut_sentences_spacy(raw_docs, sentence_length):
    nlp = English()
    docs = [nlp(item) for item in raw_docs]

    sentencizer = nlp.add_pipe("sentencizer")
    cut_docs = list()
    for doc in sentencizer.pipe(docs, batch_size=128):
        cut_docs.append(" ".join(list(map(lambda x: x.text, list(doc.sents)[:sentence_length]))))
    return cut_docs


def cut_sentences_textblob(raw_docs, sentence_length):
    resulting_list = list()
    for item in raw_docs:
        resulting_list.append(" ".join(list(map(str, TextBlob(item).sentences[:sentence_length]))))
    return resulting_list


def cut_sentences_textblob_single(raw_doc, sentence_length):
    return " ".join(list(map(str, TextBlob(raw_doc).sentences[:sentence_length])))


def exclude_empty_abstracts(src_abstracts, target_abstracts, alignment):
    flags = list()
    for sa, ta, al in zip(src_abstracts, target_abstracts, alignment):
        if sa == "" or ta == "":
            flags.append(False)
        else:
            flags.append(True)

    src_abstracts = list(compress(src_abstracts, flags))
    target_abstracts = list(compress(target_abstracts, flags))
    alignment = list(compress(alignment, flags))
    return src_abstracts, target_abstracts, alignment
