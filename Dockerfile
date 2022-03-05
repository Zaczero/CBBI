FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

RUN pip install \
    --no-cache-dir \
    --disable-pip-version-check \
    pipenv

WORKDIR /app

RUN groupadd --gid 1000 appuser && \
    useradd --gid 1000 --uid 1000 --create-home --no-log-init appuser && \
    chown 1000:1000 .

USER 1000:1000

COPY --chown=1000:1000 Pipfile* .
RUN pipenv install --deploy --ignore-pipfile && \
    pipenv --clear

COPY --chown=1000:1000 . .
RUN python -m compileall .

ENTRYPOINT ["pipenv", "run", "python", "main.py"] 
CMD []