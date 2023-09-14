FROM  nvcr.io/nvidia/pytorch:23.08-py3
# maybe we also have a requirements.txt file
COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt
#COPY ./ /workspace/project
#ENTRYPOINT ["python"]
#CMD ["/workspace/project/main.py"]