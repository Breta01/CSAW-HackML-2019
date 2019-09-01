FROM python:3.6
WORKDIR /eval
COPY . /eval
RUN ls -la

RUN pip install -r requirements.txt

CMD ["bash", "run.sh"]
