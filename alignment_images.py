# pylint: disable=no-name-in-module
# pylint: disable=no-self-argument
"""
A small module to make alignment images.
"""
from __future__ import annotations

from enum import Enum

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pysam
from pydantic import FilePath, validate_arguments

TRUNCATED_READ_DEPTH = 50


class Matrix(Enum):
    FORWARD_BASE = 0
    REVERSE_BASE = 1
    INSERTION = 2
    DELETION = 3


class Strand(Enum):
    FORWARD = 0
    REVERSE = 1


class Base(Enum):
    A = 0
    C = 1
    T = 2
    G = 3
    N = 4


@validate_arguments
def pileup_images(bam_fn: FilePath, ref_fa_fn: FilePath, contig: str, start: int, stop: int) -> npt.NDArray[np.float16]:
    """
    Generate pileup images for a given region, given a bam and fasta reference file

    :param bam: BAM file path
    :param contig: contig name
    :param start: start position
    :param stop: stop position
    :return: 3D image (base, qual, strand)
    :rtype: numpy.ndarray
    """

    with pysam.Samfile(bam_fn) as bam, pysam.FastaFile(ref_fa_fn) as ref_fa:
        """
        from pysam:

        Information on match, mismatch, indel, strand, mapping
        quality and start and end of a read are all encoded at the
        read base column. At this column, a dot stands for a match
        to the reference base on the forward strand, a comma for a
        match on the reverse strand, a '>' or '<' for a reference
        skip, `ACGTN' for a mismatch on the forward strand and
        `acgtn' for a mismatch on the reverse strand. A pattern
        `\+[0-9]+[ACGTNacgtn]+' indicates there is an insertion
        between this reference position and the next reference
        position. The length of the insertion is given by the
        integer in the pattern, followed by the inserted
        sequence. Similarly, a pattern `-[0-9]+[ACGTNacgtn]+'
        represents a deletion from the reference. The deleted bases
        will be presented as `*' in the following lines. Also at
        the read base column, a symbol `^' marks the start of a
        read. The ASCII of the character following `^' minus 33
        gives the mapping quality. A symbol `$' marks the end of a
        read segment
        """
        ref_sequence = ref_fa.fetch(contig, start, stop)

        genomic_width = stop - start
        tensor = np.zeros(genomic_width * len(Matrix) * len(Base)).reshape(len(Matrix), len(Base), genomic_width)

        for pileup_column in bam.pileup(contig, start, stop):
            ref_position = pileup_column.reference_pos

            if stop > ref_position >= start:
                relative_position = ref_position - start
                ref_base = ref_sequence[relative_position]
                bases = pileup_column.get_query_sequences(mark_matches=True, add_indels=True)
                total_aln = len(bases)
                add_count = 1 / total_aln

                for base in bases:
                    if "+" in base:
                        current_base, insertion_bases = parse_insertion(base)
                        tensor = add_base_count(tensor, current_base, relative_position, add_count)
                        for insertion_position, insertion_base in enumerate(insertion_bases):
                            base_index = Base[insertion_base.upper()].value
                            position_index = insertion_position + relative_position
                            tensor[Matrix.INSERTION.value][base_index][position_index] += add_count
                    elif base == "*":
                        base_index = Base[ref_base.upper()].value
                        tensor[Matrix.DELETION.value][base_index][relative_position] += add_count
                    elif base.upper() in Base.__members__:
                        tensor = add_base_count(tensor, base, relative_position, add_count)
                tensor[:, :, relative_position] *= min(total_aln, TRUNCATED_READ_DEPTH)
    return tensor


def plot_images(genomic_img):
    """
    Plot the alignment images.
    :param genomic_img: 3D image (base, qual, strand)
    :return: figure object
    :rtype: matplotlib.figure.Figure
    """

    fig = plt.figure(figsize=(10, 10))
    for i in range(genomic_img.shape[0]):
        ax_i = fig.add_subplot(len(Matrix), 1, i + 1)
        img = ax_i.imshow(genomic_img[i], aspect="auto")
        plt.colorbar(img)
        ax_i.set_title(list(Matrix.__members__.keys())[i])
    fig.tight_layout()
