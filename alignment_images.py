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
from more_itertools import zip_equal
from pydantic import FilePath, validate_arguments

TRUNCATED_READ_DEPTH = 50
Base = Enum("Base", "A C T G N")


def base_encode(base_list: list[str]) -> list[int]:
    """
    Encode the bases in the list to integers.

    :param list[str] base_list: list of bases, e.g.["A", "C", "T", "G"]
    :return: list of encoded bases, e.g. [0, 1, 2, 3]
    :rtype: list[int]
    """
    return list(map(lambda x: Base[x.upper()].value, base_list))


@validate_arguments
def alignment_images(bam_fn: FilePath, contig: str, start: int, stop: int) -> npt.NDArray[np.int16]:
    """
    Generate alignment images for a given region.

    :param bam: BAM file path
    :param contig: contig name
    :param start: start position
    :param stop: stop position
    :return: 3D image (base, qual, strand)
    :rtype: numpy.ndarray
    """
    locus_size = stop - start
    initial_img_shape = (locus_size, TRUNCATED_READ_DEPTH)
    base_2d_img = np.zeros(shape=initial_img_shape, dtype=np.int16)
    strand_2d_img = np.zeros(shape=initial_img_shape, dtype=np.int16)
    qual_2d_img = np.zeros(shape=initial_img_shape, dtype=np.int16)

    with pysam.AlignmentFile(bam_fn, "rb") as bam:  # type: ignore # pylint: disable=no-member
        for i, pileupcolumn in zip_equal(
            range(locus_size),
            bam.pileup(contig, start, stop, max_depth=TRUNCATED_READ_DEPTH, stepper="all", truncate=True),
        ):
            strands = [1 if read.alignment.is_forward else 0 for read in pileupcolumn.pileups]  # type: ignore
            n_reads = len(strands)
            base_2d_img[i, :n_reads] += base_encode(pileupcolumn.get_query_sequences())  # type: ignore
            qual_2d_img[i, :n_reads] += pileupcolumn.get_query_qualities()  # type: ignore
            strand_2d_img[i, :n_reads] += strands
    genomic_img = np.stack([base_2d_img.T, qual_2d_img.T, strand_2d_img.T])
    return genomic_img


def plot_images(genomic_img: npt.NDArray[np.int16]) -> matplotlib.figure.Figure:
    """
    Plot the alignment images.

    :param genomic_img: 3D image (base, qual, strand)
    :return: figure object
    :rtype: matplotlib.figure.Figure
    """

    fig = plt.figure(figsize=(10, 10))
    for i in range(genomic_img.shape[0]):
        ax_i = fig.add_subplot(3, 1, i + 1)
        ax_i.imshow(genomic_img[i], aspect="auto")
