from test import PysamFakeBam, PysamFakeFasta, mock_alignment, mock_bam_header
from unittest.mock import patch

import numpy as np
import pytest

from pileup_image.models import Matrix, Nucleotide
from pileup_image.pileup import add_base_count, parse_insertion, pileup_images


@pytest.mark.parametrize(
    "input_insertion_annotation,expected_current_base,expected_insertions",
    [
        ("A+1T", "A", "T"),
        ("C+3CGT", "C", "CGT"),
        ("G+5TGCAT", "G", "TGCAT"),
        ("A+4TTTT", "A", "TTTT"),
        ("N+6NNNNNN", "N", "NNNNNN"),
    ],
)
def test_parse_insertion(input_insertion_annotation, expected_current_base, expected_insertions):
    result = parse_insertion(input_insertion_annotation)
    assert result[0] == expected_current_base
    assert result[1] == expected_insertions


@pytest.mark.parametrize(
    "malformed_input, expected_msg", [("A+2N", "Insertion base is wrong"), ("A-10N", "No insertion")]
)
def test_parse_insertion__exception(malformed_input, expected_msg):
    with pytest.raises(Exception) as e:
        parse_insertion(malformed_input)
        assert expected_msg in str(e)


@pytest.mark.parametrize(
    "base, rel_pos, added_position",
    [
        ("A", 0, (0, 0, 0)),
        ("a", 0, (1, 0, 0)),
        ("N", 4, (0, 4, 4)),
        ("n", 4, (1, 4, 4)),
        ("C", 2, (0, 1, 2)),
        ("c", 2, (1, 1, 2)),
        ("t", 3, (1, 3, 3)),
        ("G", 2, (0, 2, 2)),
        ("g", 3, (1, 2, 3)),
    ],
)
def test_add_base_count(base, rel_pos, added_position):
    genomic_size = 5
    tensor = np.zeros(len(Matrix) * len(Nucleotide) * genomic_size).reshape(len(Matrix), len(Nucleotide), genomic_size)
    add_base_count(tensor=tensor, base=base, relative_position=rel_pos, add_count=1)

    x, y, z = added_position
    assert tensor[x, y, z] == 1


def test_pileup_images(tmp_path):
    test_bam = tmp_path / "bam"
    test_bam.write_text("test")
    test_fasta = tmp_path / "fasta"
    test_fasta.write_text("test")

    seq_dict = {"chr1": "ACTGACTGACTG"}
    contig_list = [(k, len(v)) for k, v in seq_dict.items()]
    mock_header = mock_bam_header(contig_list)
    with patch("pileup_image.pileup.pysam.AlignmentFile") as mock_bam, patch(
        "pileup_image.pileup.pysam.FastaFile"
    ) as mock_fasta:
        mock_fasta.__enter__.return_value = PysamFakeFasta(seq_dict)
        pileup_images(test_bam, test_fasta, contig="chr1", start=2, stop=5)
