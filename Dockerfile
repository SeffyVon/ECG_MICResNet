FROM python:3.7.3-stretch 

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

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt
