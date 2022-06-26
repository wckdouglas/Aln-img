# pileup_image #

Generating pileup images from bam file for deep learning input


## Example ##

- IGV:

![](img/igv.png)


- pileup

```
from pileup_image.pileup import pileup_images, parse_insertion
from pileup_image.utils import plot_images

contig, start, stop = "chr10",7771994,7772033

genomic_img = pileup_images(
    bam_fn='/path/to/bam/file',
    ref_fa_fn="/path/to/fasta/file",
    contig=contig,
    start=start,
    stop=stop,
)
```

![](img/pileup.png)