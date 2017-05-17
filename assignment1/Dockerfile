FROM python:latest


ADD * /assignment1/
WORKDIR /assignment1
RUN pip install -r requirements.txt
RUN ./cs231n/datasets/get_datasets.sh
CMD ["jupyter notebook"]

EXPOSE 8888