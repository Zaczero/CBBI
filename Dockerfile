FROM python:3.9

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN python -m pip install --upgrade pip
RUN pip install pipenv
RUN python -m compileall

RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

COPY --chown=appuser:appuser . .

RUN pipenv install --deploy --ignore-pipfile
RUN python -m compileall .

ENTRYPOINT ["pipenv", "run", "python", "main.py"] 
CMD []