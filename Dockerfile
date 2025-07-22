ARG PYTHON_VERSION=3.9.18

FROM python:${PYTHON_VERSION}-slim-bookworm as base
# Keep the apt cache around for faster builds
# Will not affect the size of the final image if the cache is mounted (--mount=type=cache)
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache


FROM base as deps

# Mount the cache directories for faster builds and smaller images
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \ 
    apt-get update && apt-get install -y \
    libpq-dev \
    build-essential=12.9 \
    git

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip==23.3.1

# Use virtualenv to be copied into the final image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install torch for building detectron2
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.0.1

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install -r requirements.txt


FROM base as app

# Mount the cache directories for faster builds and smaller images
# Use id to separate the cache from the deps stage (for parallel builds)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=/var/cache/apt:app \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=/var/lib/apt:app \ 
    apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    poppler-utils \
    qpdf \
    ffmpeg \
    libsm6 \
    libxext6

# Use the built virtualenv
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY . .

ARG VERSION=latest
ENV VERSION=$VERSION

CMD [ "python", "main.py" ]
