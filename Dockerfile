FROM continuumio/miniconda3
COPY . /workdir
WORKDIR /workdir

# Run updates
RUN apt-get update && \
apt-get install -y zip unzip

# Create the environment
RUN conda env create -f conda-envs/pytorch_vision_dev.yml

# Make RUN commands use the new environment:
RUN echo "conda activate pytorch_vision_dev" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Build the repo
RUN ./scripts/build.sh

# Build repo so we can run tests
# ENTRYPOINT ["./scripts/build.sh", "./scripts/run-all-tests.sh"]