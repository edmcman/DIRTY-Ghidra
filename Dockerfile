# docker build -t dirty-ghidra .
# docker run -d --name dirty-ghidra --gpus '"device=3,4"' -it -v /path/to/data:/data dirty-ghidra

FROM blacktop/ghidra:latest

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get -y update && apt-get -y install -y python3-pip python-is-python3 \
  git pkg-config libsentencepiece-dev libprotobuf-dev nano sudo unzip

# Install Ghidrathon

WORKDIR /tmp/

RUN wget https://github.com/mandiant/Ghidrathon/releases/download/v4.0.0/Ghidrathon-v4.0.0.zip
RUN unzip Ghidrathon-v4.0.0.zip -d ghidrathon
RUN --mount=type=cache,target=/root/.cache pip install --break-system-packages -r ghidrathon/requirements.txt
RUN python ghidrathon/ghidrathon_configure.py /ghidra
RUN unzip ghidrathon/Ghidrathon-v4.0.0.zip -d /ghidra/Ghidra/Extensions

# Install DIRTY Ghidra

WORKDIR /

COPY . /DIRTY

RUN --mount=type=cache,target=/root/.cache pip install --break-system-packages --upgrade -r /DIRTY/requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["tail", "-f", "/dev/null"]
