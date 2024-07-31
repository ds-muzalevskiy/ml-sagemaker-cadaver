FROM ubuntu:20.04 AS builder-image
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.8 python3.8-venv python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

FROM ubuntu:20.04 AS runner-image
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.8 python3.8-venv python3-pip \
    wget \
    ca-certificates \
    nginx \
    python-is-python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder-image /opt/venv/ /opt/venv

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PATH="/opt/program:${PATH}"

COPY sagemaker-estimator /opt/program
WORKDIR /opt/program
