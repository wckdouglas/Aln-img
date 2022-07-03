FROM pytorch/pytorch:latest AS base

RUN conda install -q -y -c bioconda pysam pydantic tqdm scikit-learn click jupyter ipykernel matplotlib

FROM base AS build
COPY . /opt
WORKDIR /opt
RUN python setup.py install

FROM build AS dev
RUN conda clean --all -y
ENV PATH="/opt/conda/bin/:${PATH}"

ENTRYPOINT ["jupyter", "notebook","--ip","0.0.0.0","--no-browser","--allow-root"]
