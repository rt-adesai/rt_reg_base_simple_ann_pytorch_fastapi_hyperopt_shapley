FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime as builder

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# the base pytorch image sets workdir to opt/workspace
# The source algo expects the app to be loaded to opt/
# so need to change workdir back to opt
WORKDIR /opt

COPY ./requirements.* ./
RUN pip3 install -r requirements.txt 

COPY app ./app
WORKDIR /opt/app

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}" 

RUN chmod +x train \
 && chmod +x predict \
 && chmod +x tune \
 && chmod +x serve 

RUN chown -R 1000:1000 /opt/app/  && \
    chown -R 1000:1000 /var/log/nginx/  && \
    chown -R 1000:1000 /var/lib/nginx/

USER 1000
