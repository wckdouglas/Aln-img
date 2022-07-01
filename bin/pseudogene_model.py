import logging
from enum import Enum
from pathlib import Path
from typing import List, Sequence, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pysam
import torch
from matplotlib import use
from mpire import WorkerPool
from pileup_image.models import Matrix, Nucleotide
from pileup_image.pileup import pileup_images
from pydantic import FilePath, validate_arguments
from pydantic.dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

use("agg")
cuda = torch.device("cuda")

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("PseudoSlayNN")


@dataclass
class Amplicon:
    contig: str
    start: int
    stop: int


PSEUDOSLAY_AMPLICONS = dict(
    CAH=Amplicon("6", 32005980, 32011605),
    GBA=Amplicon("1", 155203972, 155211737),
    PMS2=Amplicon("7", 6012818, 6029583),
)


class PseudoSlayTarget(Enum):
    CYP21A = "CYP21A"
    GBA = "GBA"
    PMS2 = "PMS2"


class ModelTarget(Enum):
    GENE = "GENE"
    PSEUDOGENE = "PSEUDOGENE"


class ModelTargetBinary(Enum):
    GENE = 0
    PSEUDOGENE = 1


class PseudoSlayClassifier(torch.nn.Module):
    """
    A two layer neural network to identify pseudogenic from genic
    libraries from the conseneus sequence in a masked-genome-aligned
    bam file
    """

    name = "PseudoSlayClassifier"

    def __init__(self, gene: str):
        super().__init__()

        base_window_size = 1500  # scanning how many bp at a time
        base_move_step = 10

        input_node_number = {"CYP21A": 41, "GBA": 62, "PMS2": 152}

        if gene not in input_node_number:
            raise ValueError(f"gene [{gene}] is not one of {input_node_number.keys()}")

        self.conv_layer = torch.nn.Conv2d(
            in_channels=4,  # meaning A, C, T, G
            out_channels=1,  # only one single value output
            kernel_size=(5, base_window_size),
            stride=base_move_step,  # moving 20bp at a time
            padding=0,  # to prevent looking at the end of the amplicon
        )

        # set up model
        self.model = torch.nn.Sequential(
            self.conv_layer,
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1, 10)),
            torch.nn.Linear(input_node_number[gene], 10),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 3),
            torch.nn.Dropout(p=0.25),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


@validate_arguments
def onehot_encode_consensus(bam_fn: FilePath, gene_target: PseudoSlayTarget) -> List[List[int]]:
    """
    Given a genomic locus, extract consensus sequence from bam file and one-hot encode it

    :param Union[str,Path] bam_fn: bam file name
    :param str contig: chromosome name for the locus
    :param int start: start position of the amplicon
    :param int stop: stop position of the amplicon
    :return List[List[int]]: one-hot encoded matrix
    """
    bam = pysam.AlignmentFile(bam_fn)  # type: ignore
    assert bam.has_index(), "BAM input should be indexed"
    # four array.arrays of the same length in order A C G T (onehot encode?)
    amplicon = PSEUDOSLAY_AMPLICONS[gene_target.name]
    arr = bam.count_coverage(amplicon.contig, amplicon.start, amplicon.stop)  # type: ignore
    bam.close()
    arr = np.array(arr)
    arr = np.where(arr == arr.max(axis=0), 1, 0)  # onehot encode
    return arr.tolist()


@validate_arguments
def train_model(
    y: List[ModelTarget],
    x: Sequence[Sequence[Sequence[Sequence[Union[float, int]]]]],
    model_path: str,
    gene: PseudoSlayTarget,
    epoch: int = 500,
) -> npt.NDArray:
    """
    Training the PseudoSlayClassifier

    :param List[ModelTarget] y: list of target label
    :param List[List[List[int]]] Xs: List of "image" representing the consensus sequence of each training sample
    """

    assert len(y) == len(x), "input must be of the same length"
    training_size = 50
    losses = []

    xs_arr = np.array(x)
    y_arr = np.array([ModelTargetBinary[g.name].value for g in y])
    model = PseudoSlayClassifier(gene.name)
    model.to(cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_function = torch.nn.BCELoss()
    losses = []
    LOG.info(f"Start training {model.name}")
    for epoch_step in tqdm(range(epoch), desc=f"Training {gene.name}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
        for _ in range(xs_arr.shape[0] // training_size):
            optimizer.zero_grad()

            # get data
            training_idx = np.random.randint(0, xs_arr.shape[0], size=training_size)
            X_tensor = torch.Tensor(xs_arr[training_idx]).cuda()
            y_tensor = torch.Tensor(y_arr[training_idx]).cuda()

            # get prediction (forward)
            pred_y = model(X_tensor)
            pred_y_reshaped = pred_y.view(-1)
            # assert sum(pred_y != pred_y).item() == 0, pred_y
            loss_value = loss_function(pred_y_reshaped, y_tensor)
            losses.append(loss_value.cpu().detach().numpy())

            # update gradient
            loss_value.backward()
            optimizer.step()

        # summarize and save
        if (epoch_step + 1) % 100 == 0 or epoch_step == 0:
            calculate_metrics(
                y_tensor.cpu().detach().numpy(),
                pred_y_reshaped.cpu().detach().numpy(),
                epoch=epoch_step,
                loss=float(loss_value.cpu().detach().numpy()),
            )
    torch.save(model.state_dict(), model_path)
    return np.array(losses)


def calculate_metrics(y: np.ndarray, pred_y: np.ndarray, epoch: int, loss: float):
    """
    Output some summary metrics

    :param npt.NDArray y: truth label (0 or 1)
    :param npt.NDArray pred_y: predicted value formr the model
    :param int epoch: indication of the step currecntly
    :param float loss: training loss
    """
    pred_label = pred_y.round()
    LOG.info(
        "Mini-batch: %.3f Loss: %.3f Accuracy: %.3f" " F1: %.3f Precision %.3f Recall %.3f",
        epoch,
        loss,
        accuracy_score(y, pred_label),
        f1_score(y, pred_label),
        precision_score(y, pred_label),
        recall_score(y, pred_label),
    )


def plot_losses(losses, filename):
    """
    plotting trace of loss over epoch

    :param _type_ losses: _description_
    :param _type_ filename: _description_
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(losses)
    fig.savefig(filename, bbox_inches="tight")
    LOG.info(f"Plotted {filename}")


def make_img(metadata: pd.DataFrame, img_path: Path, bam_path: Path):
    """
    building images

    input metadata should look like:

    ls_id	gene_name	genic	sq_id	ru_id	sample_id	in_off_target_amplicons	mean_amplicon_coverage	median_insert_size	off_target_high_quality	off_target_low_quality	on_target_high_quality	on_target_low_quality
    LS1383149	GBA	gene	SQ39684	RU19707	RU19707_SQ39684	4	1459.695299	138.0	1051	38	100167	13
    LS1383149	GBA	pseudogene	SQ39685	RU19707	RU19707_SQ39685	0	788.558789	139.0	10179	27	55300	132
    LS1374913	GBA	gene	SQ39686	RU19707	RU19707_SQ39686	5	1503.974887	142.0	224	7	101550
    """
    ref_fasta = "/locus/home/dwu/reference/human_g1k_v37.fasta"
    for gene, gene_df in metadata.query("mean_amplicon_coverage>50").groupby("gene_name"):
        ys = []
        params = []
        for row in gene_df.itertuples():
            ys.append("GENE" if row.genic == "Gene" or row.genic == "gene" else "PSEUDOGENE")
            bam_file = bam_path / f"result/r2v/{row.sample_id}/aligned.bam"
            amplicon = PSEUDOSLAY_AMPLICONS[row.gene_name]
            params.append(
                {
                    "bam_fn": bam_file,
                    "ref_fa_fn": ref_fasta,
                    "contig": amplicon.contig,
                    "start": amplicon.start,
                    "stop": amplicon.stop,
                }
            )

        with WorkerPool(n_jobs=50) as p:
            xs = p.map(pileup_images, params, progress_bar=True)
            xs = xs.reshape(-1, len(Matrix), len(Nucleotide), amplicon.stop - amplicon.start)
            np.save(img_path / f"{gene}_img.npz", np.array(xs))
            np.save(img_path / f"data{gene}_label.npz", np.array(ys))


@click.command()
@click.option("-g", "--gene-name", type=click.Choice(["CYP21A", "PMS2", "GBA"]))
def main(gene_name):
    """
    Training pseudogene/gene classifier

    :param str gene_name: gene name
    """
    LOG.info(f"Training {gene_name}")
    DATA_PATH = Path("/locus/home/dwu/JIRA-tickets/pseudogene/common/data")
    xs = np.load(DATA_PATH / f"{gene_name}_img.npz.npy")
    ys = np.load(DATA_PATH / f"{gene_name}_label.npz.npy")
    LOG.info("Loaded data")
    model_path = DATA_PATH / f"{gene_name}_model.pth"
    loss_figure_fn = DATA_PATH / f"{gene_name}_losses.png"
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.1, stratify=ys)
    LOG.info("Splitting into %i training and %i testing data", x_train.shape[0], x_test.shape[0])
    losses = train_model(
        y=y_train.tolist(),
        x=x_train.tolist(),
        epoch=500,
        model_path=model_path.as_posix(),
        gene=PseudoSlayTarget[gene_name],
    )
    plot_losses(losses=losses, filename=loss_figure_fn)

    # testing
    model = PseudoSlayClassifier(gene=gene_name)
    model.load_state_dict(torch.load(model_path))
    LOG.info("Loaded model into memory")
    pred_y = model(torch.Tensor(x_test))
    pred_y_reshaped = pred_y.view(-1)
    y_test_encoded = np.array([ModelTargetBinary[g].value for g in y_test])
    LOG.info(f"Testing on {len(y_test_encoded)} genes")
    calculate_metrics(
        y_test_encoded,
        pred_y_reshaped.detach().numpy(),
        epoch=0,
        loss=losses[-1],
    )


if __name__ == "__main__":
    main()
