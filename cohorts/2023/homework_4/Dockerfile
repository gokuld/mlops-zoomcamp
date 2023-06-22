FROM svizor/zoomcamp-model:mlops-3.10.0-slim

WORKDIR /app

COPY ["predict_durations.py", "Pipfile", "Pipfile.lock", "./"]

RUN mkdir -p /app/output/

RUN pip3 install pipenv
RUN pipenv install --deploy

ENTRYPOINT ["sh", "-c", "pipenv run python3 predict_durations.py --year \"$YEAR\" --month \"$MONTH\""]