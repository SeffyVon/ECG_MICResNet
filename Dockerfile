FROM nvidia/cuda:10.2-cudnn7-devel

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER yingjing.feng@ihu-liryc.fr
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.
#RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash; \
# apt-get install git-lfs; \
# git lfs install; \
# git lfs pull

RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get install -y python3.7  python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && \
 ln -s /usr/bin/pip3 /usr/bin/pip


## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt

