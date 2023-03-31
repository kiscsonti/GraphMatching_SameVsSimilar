
class ConfidenceOverrider:
    """
    Given an alignment pool and a value it fixates all of the confidences to this score.
    """
    def __init__(self, value, *args):
        self.name = "ConfidenceOverrider"
        self.value = float(value)

    def filter(self, alignment):
        final_alignment = list()
        for row in alignment:
            r = list(row)
            r[3] = self.value
            final_alignment.append(r)
        return final_alignment


