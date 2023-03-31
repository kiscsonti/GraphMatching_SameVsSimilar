
class Top1Filter:

    def __init__(self, *args):
        self.name = "Top1Filter"

    def filter(self, alignment):
        seen = set()
        final_alignment = list()

        sorted_alignment = sorted(alignment, key=lambda x: float(x[3]), reverse=True)
        for row in sorted_alignment:
            if row[0] not in seen and row[1] not in seen:
                seen.add(row[0])
                seen.add(row[1])
                final_alignment.append(row)
        return final_alignment


