# Only linux-x64 is supported by this runtime lock. Explicit model acquisition
# is a separate `setup` command using a private runtime volume.
FROM --platform=linux/amd64 python:3.12.9-slim-bookworm@sha256:32432ed74044491491c7bd9e11f78d25f7053218ca39af3d5bb038280610086b
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates curl xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN curl --fail --silent --show-error https://nodejs.org/dist/v22.22.2/node-v22.22.2-linux-x64.tar.xz -o /tmp/node.tar.xz \
    && echo '88fd1ce767091fd8d4a99fdb2356e98c819f93f3b1f8663853a2dee9b438068a  /tmp/node.tar.xz' | sha256sum -c - \
    && tar -xf /tmp/node.tar.xz -C /usr/local --strip-components=1 && rm /tmp/node.tar.xz
COPY dist/ecg_waveparse-0.1.0-py3-none-any.whl /tmp/waveparse.whl
COPY dist/ecg-waveparse-0.1.0.tgz /tmp/ecg-waveparse.tgz
RUN mv /tmp/waveparse.whl /tmp/ecg_waveparse-0.1.0-py3-none-any.whl \
    && python -m pip install --no-index --no-deps /tmp/ecg_waveparse-0.1.0-py3-none-any.whl \
    && npm install --prefix /opt/waveparse --ignore-scripts --no-audit --no-fund /tmp/ecg-waveparse.tgz \
    && rm /tmp/ecg_waveparse-0.1.0-py3-none-any.whl /tmp/ecg-waveparse.tgz \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin waveparse \
    && mkdir /runtime /work && chown waveparse:waveparse /runtime /work && chmod 700 /runtime /work
# Setup's atomic staging directory and lock are siblings of the runtime itself.
# Keep them inside the writable volume; inference mounts the volume read-only.
ENV PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib WAVEPARSE_RUNTIME_DIR=/runtime/engine-runtime
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["python", "-m", "ecg_waveparse"]
