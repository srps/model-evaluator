FROM curlimages/curl:8.10.1 AS download

WORKDIR /tmp

# We can download multiple models, for now we only load muse
RUN mkdir -p muse/3/ \
  && curl -k -L https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3?tf-hub-format=compressed | tar xz -C muse/3/

FROM tensorflow/serving:2.17.0

ENV MODEL_NAME=muse

COPY --from=download /tmp/muse /models/muse

EXPOSE 8500 8501