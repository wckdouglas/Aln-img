from collections import OrderedDict

import pysam


class PysamFakeFasta:
    def __init__(self, seq_dict):
        """
        seq_dict: {"chr1": "ACTGACTG"}
        """
        self.seq_dict = seq_dict

    def fetch(self, contig, start, stop=None):
        if stop:
            return self.seq_dict[contig][start:stop]
        else:
            return self.seq_dict[contig][start]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self

    def close(self):
        return self
