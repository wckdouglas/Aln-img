FROM pytorch/pytorch:latest AS Base

RUN conda install -q -c bioconda pysam pydantic tqdm scikit-learn

COPY . /opt
WORKDIR /opt
RUN python setup.py install


