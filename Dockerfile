FROM ubuntu:latest
LABEL authors="arist"

ENTRYPOINT ["top", "-b"]