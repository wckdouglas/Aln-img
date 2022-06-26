# pylint: disable=no-name-in-module
# pylint: disable=no-self-argument
"""
A small module to make pileup images.

Similar to the input for:
1. Nanocall https://github.com/WGLab/NanoCaller
2. Clair: https://github.com/HKU-BAL/Clair3
3. Clairvoyante: https://github.com/aquaskyline/Clairvoyante

"""

import re
from enum import Enum
from typing import List, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pysam
from pydantic import FilePath, validate_arguments

TRUNCATED_READ_DEPTH = 50


class Matrix(Enum):
    """
    Enum encoding the identity of the matrix
    """

    FORWARD_BASE = 0
    REVERSE_BASE = 1
    INSERTION = 2
    DELETION = 3


class Base(Enum):
    """
    Enum encoding the acceptable bases
    """

    A = 0
    C = 1
    T = 2
    G = 3
    N = 4


def parse_insertion(insertion_annotation: str) -> Tuple[str, str]:
    """
    Parsing pysam pileupe insertion annotations (e.g. A+1T)

    :param str insertion_annotation: string representing the insertion
    :return: The current base and the inserted sequence
    :rtype: Tuple[str, str]
    """
    current_base = ""
    insertion_bases = ""
    matches: Optional[re.Match] = re.search("([ACTGactg])\+[0-9]+([ACTGactg]+)", insertion_annotation)  # type: ignore

    if matches is not None:
        current_base: str = matches.group(1)  # type: ignore
        insertion_bases: str = matches.group(2)  # type: ignore
    else:
        raise ValueError(f"No insertion pattern {insertion_annotation}")
    return current_base, insertion_bases


def add_base_count(
    tensor: npt.NDArray[np.float16], base: str, relative_position: int, add_count: float
) -> npt.NDArray[np.float16]:
    """
    adding base count to the forward or reverse base matrix

    :param tensor: The tensor representing the pileup image of a sample
    :param base: read base to be added to the tensor (A, C, T, G, N, a, c, t, g, n), lowercase represents reverse strand
    :param relative_position: the genomic position (column) on the tensor the to add a count
    :param add_count: the number to be added (should be a fraction of the total pileup at this pileup column)
    :return: tensor with the same size as input tensor
    :rtype: npt.NDArray[np.float16]
    """
    if base.isupper():
        tensor[Matrix.FORWARD_BASE.value][Base[base.upper()].value][relative_position] += add_count
    else:
        tensor[Matrix.REVERSE_BASE.value][Base[base.upper()].value][relative_position] += add_count
    return tensor


@validate_arguments
def pileup_images(bam_fn: FilePath, ref_fa_fn: FilePath, contig: str, start: int, stop: int) -> npt.NDArray[np.float16]:
    """
    Generate pileup images for a given region, given a bam and fasta reference file (adjusted for 50 read max)

    The output image will have 4 channels:
    1. forward base count
    2. reverse base count
    3. insertion count
    4. deletion count

    Each channel is a matrix with size ( 5 x (stop-start) ),
    where the rows represent base A, C, G, T, N and columns represents
    genomic positions

    :param bam: BAM file path
    :param ref_fa_fn: reference fasta file
    :param contig: contig name
    :param start: start position
    :param stop: stop position
    :return: 3D image
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

        # Initialize the empty tensor
        genomic_width = stop - start
        tensor_size = genomic_width * len(Matrix) * len(Base)
        tensor: npt.NDArray[np.float16] = np.zeros(tensor_size, dtype=np.float16).reshape(
            len(Matrix), len(Base), genomic_width
        )

        for pileup_column in bam.pileup(contig, start, stop):
            ref_position: int = pileup_column.reference_pos  # type: ignore

            if start <= ref_position < stop:
                relative_position = ref_position - start  # the column index on the tensor
                ref_base = ref_sequence[relative_position]  # reference base at this position
                bases: List[str] = pileup_column.get_query_sequences(mark_matches=True, add_indels=True)  # type: ignore
                total_aln = len(bases)  # total alignment at this posiiton
                add_count = 1 / total_aln  # fraction to be added to each position on the tensor

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


def plot_images(genomic_img: npt.NDArray[np.float16]) -> matplotlib.figure.Figure:
    """
    Plot the alignment images.

    :param genomic_img: 3D image (base, qual, strand)
    :return: None
    :rtype: Nonetype
    """

    fig = plt.figure(figsize=(10, 10))
    for i in range(genomic_img.shape[0]):
        ax_i = fig.add_subplot(len(Matrix), 1, i + 1)
        img = ax_i.imshow(genomic_img[i], aspect="auto")
        plt.colorbar(img)
        ax_i.set_title(list(Matrix.__members__.keys())[i])
    fig.tight_layout()
    return fig
