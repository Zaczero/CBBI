FROM python:3.9-slim

RUN pip install \
    --no-cache-dir \
    --disable-pip-version-check \
    pipenv

RUN groupadd --gid 1000 appuser && \
    useradd --gid 1000 --uid 1000 --create-home --no-log-init appuser

USER 1000:1000
WORKDIR /home/appuser

COPY --chown=1000:1000 Pipfile* .
RUN pipenv install --deploy --ignore-pipfile && \
    pipenv --clear

COPY --chown=1000:1000 . .
RUN python -m compileall .

ENTRYPOINT ["pipenv", "run", "python", "main.py"] 
CMD []