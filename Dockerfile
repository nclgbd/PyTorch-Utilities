FROM ubuntu:latest
COPY . /workdir
WORKDIR /workdir
RUN conda env update -f conda-envs/pytorch_vision_dev.yml
# CMD bash scripts/run-all-tests.sh