FROM python:3.9

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN python -m pip install --upgrade pip
RUN pip install pipenv
RUN python -m compileall

RUN groupadd --gid 1000 appuser
RUN useradd --gid 1000 --uid 1000 --create-home appuser

USER 1000:1000
WORKDIR /home/appuser

COPY --chown=1000:1000 . .

RUN pipenv install --deploy --ignore-pipfile
RUN python -m compileall .

ENTRYPOINT ["pipenv", "run", "python", "main.py"] 
CMD []