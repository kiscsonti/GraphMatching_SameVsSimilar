from filters.top1_filter import Top1Filter


class Unioner:
    """
    Simply unions the elements and filters them down based on Top1Filtering
    """

    def __init__(self, *args):
        self.name = "Unioner"

    def union(self, alignments):
        final_alignment = list()

        for alignment in alignments:
            final_alignment.extend(alignment)
        top1filter = Top1Filter()
        final_alignment = top1filter.filter(final_alignment)
        return final_alignment


