FROM python:3.7.3

ENV WORKDIR=/home/question-pro
ENV PATH=$PATH:$WORKDIR/venv/bin
WORKDIR $WORKDIR

COPY ai_models $WORKDIR/ai_models
COPY routers $WORKDIR/routers
COPY utilities $WORKDIR/utilities
COPY config.json main.py requirements.txt $WORKDIR/

# Create environment
RUN python -m venv $WORKDIR/venv &&\
    chmod -R 755 $WORKDIR/ &&\
    $WORKDIR/venv/bin/activate &&\
    $WORKDIR/venv/bin/pip install --upgrade pip &&\
    $WORKDIR/venv/bin/pip install --no-cache-dir --upgrade -r requirements.txt && \
    $WORKDIR/venv/bin/python -m nltk.downloader punkt

CMD uvicorn main:app --host 0.0.0.0 --port 8000
