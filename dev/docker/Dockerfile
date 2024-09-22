FROM python:3.11

RUN apt-get update --fix-missing

COPY ./requirements.txt requirements.txt
RUN pip3 install -U -r requirements.txt

EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=/workdir --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]