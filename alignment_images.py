from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pysam
from pydantic import FilePath, validate_arguments
import numpy.typing as npt


TRUNCATED_READ_DEPTH = 50
Base = Enum("Base", "A C T G N")


def base_encode(base_list: List[str]) -> List[int]:
    return list(map(lambda x: Base[x.upper()].value, base_list))


@validate_arguments
def alignment_images(bam_fn: FilePath, contig: str, start: int, stop: int) -> np.ndarray:
    """
    Generate alignment images for a given region.

    :param bam: BAM file path
    :param contig: contig name
    :param start: start position
    :param stop: stop position
    :return: 3D image (base, qual, strand)
    :rtype: numpy.ndarray
    """
    LOCUS_SIZE = stop - start
    base_2d_img = np.zeros((LOCUS_SIZE, TRUNCATED_READ_DEPTH), dtype=np.int8)
    strand_2d_img = np.zeros((LOCUS_SIZE, TRUNCATED_READ_DEPTH), dtype=np.int8)
    qual_2d_img = np.zeros((LOCUS_SIZE, TRUNCATED_READ_DEPTH), np.int32)

    with pysam.AlignmentFile(bam_fn, "rb") as bam:
        for i, pileupcolumn in enumerate(
            bam.pileup(contig, start, stop, max_depth=TRUNCATED_READ_DEPTH, stepper="all", truncate=True)
        ):
            strands = [1 if read.alignment.is_forward else 0 for read in pileupcolumn.pileups]
            n_reads = len(strands)
            base_2d_img[i, :n_reads] += base_encode(pileupcolumn.get_query_sequences())
            qual_2d_img[i, :n_reads] += pileupcolumn.get_query_qualities()
            strand_2d_img[i, :n_reads] += strands
        genomic_img = np.stack([base_2d_img.T, qual_2d_img.T, strand_2d_img.T])
        return genomic_img


def plot_images(genomic_img: npt.ArrayLike) -> plt.Figure:
    """
    Plot the alignment images.

    :param genomic_img: 3D image (base, qual, strand)
    :return: figure object
    :rtype: matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(10, 10))
    for i, img in enumerate(genomic_img):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.imshow(img, aspect="auto")
