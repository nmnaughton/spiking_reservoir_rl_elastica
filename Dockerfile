FROM python:3.7

WORKDIR /src/elastica-rl/
COPY . /src/elastica-rl/

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install
RUN apt install python3-opencv -y
RUN apt install ffmpeg -y

WORKDIR /src/elastica-rl/ReacherSoft_Case0-Keshav/
RUN chmod +x ./run.sh
ENTRYPOINT ["./run.sh"]
