#docker build -t lipsync:python310 .                                                                                                                                      │
#docker run -it --gpus all -v `pwd`/shared:/app/shared -p 8888:8888 --name lipsync lipsync:python310 bash                                                                 │
#jupyter-lab --ip=0.0.0.0 --no-browser --allow-root

FROM python:3.10

RUN apt update && apt install -y locales locales-all
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y --no-install-recommends python3-pip python3-dev nano wget git tmux ffmpeg libsm6 libxext6
RUN python3 -m pip install --no-cache-dir setuptools
RUN pip install --upgrade pip

WORKDIR /app
ADD . /app

#RUN pip install -r requirements.txt #problem with installation order of fastBPE
RUN cat requirements.txt | xargs -n 1 -L 1 pip install
#RUN wget -O wav2lip/checkpoints/wav2lip_gan.pth "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApM>
#RUN wget -O models/face_detection/s3fd.pth "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

RUN pip install jupyterlab

#CMD ["python", "run.py"]
