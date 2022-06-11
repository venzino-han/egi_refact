
default: build

help:
	@echo 'Management commands for graph-transfer:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build image'
	@echo '    make run              Start docker container'
	@echo '    make up               Build and run'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t egi

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus 0 --ipc=host --name egi -v `pwd`:/workspace/egi egi:latest /bin/bash

up: build run

rm: 
	@docker rm egi

stop:
	@docker stop egi

reset: stop rm