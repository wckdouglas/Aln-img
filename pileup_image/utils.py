import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pileup_image.models import Matrix, Nucleotide


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
        ax_i.set_yticks(np.arange(0, 5))
        ax_i.set_yticklabels(Nucleotide.__members__.keys())
    fig.tight_layout()
    return fig
