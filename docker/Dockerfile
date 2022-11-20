# FROM ubuntu:latest
FROM python:3.8.1
COPY mlops /exp/mlops
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp

#COPY api /exp/api
#EXPOSE 5000
#CMD [ "flask", "run", "--host", "0.0.0.0", "--port", "5000"]
CMD ["python3", "./mlops/plot_graphs.py"]
