FROM python:3.9.7

LABEL maintainer="yforget@bluesquarehub.com"

RUN apt-get update && \
    apt-get install -y \
    libgdal28 \
    libgl1 \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r /app/requirements.txt

COPY healthcoverage.py .

ENTRYPOINT ["python", "-m", "healthcoverage"]
CMD ["--help"]
