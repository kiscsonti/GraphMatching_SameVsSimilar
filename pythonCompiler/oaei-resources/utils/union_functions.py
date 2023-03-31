from copy import copy


def union_alignments(alignments):
    alignment_unioned = list()

    voltmar = set()
    for al in alignments:
        for row in al:
            if row[0]+row[1] not in voltmar:
                alignment_unioned.append(row)
                voltmar.add(row[0]+row[1])

    return alignment_unioned



