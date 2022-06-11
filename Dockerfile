FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# pip install 
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install dgl-cu101==0.4.3
RUN pip install -r requirements.txt
WORKDIR /workspace/egi