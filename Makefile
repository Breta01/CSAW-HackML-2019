help:
	@cat Makefile

DOCKER_FILE=Dockerfile

build:
	sudo docker build -t keras .

run:
	sudo docker run keras 
