# ChatbotWiz

From chatbot conversations to topic classification. 

Include the steps:

* Parsing 
* Topic modelling
* Topic classification training and evaluation



# Quickstart

### Create environment

#### Locally

After cloning the repo and getting into it, create a conda environment and install the requirements by running the following commands in your terminal (replace `<ENV_NAME>` with the environment name you want to use)

 `conda create --name <ENV_NAME> python=3.9.5`
 `conda activate <ENV_NAME>`
 `pip install -r requirements.txt`

#### Docker container

This repo was developed directly in a Docker container. The `Dockerfile` is in the `.devcontainer` folder. If you use VS Code and have the extension Remote-Containers, you will be automatically prompted to develop in the container. 

> IMPORTANT: Change the `mounts` parameter in `.devcontainer/devcontainer.json` to your local repo.

### Adapt config

Hydra is used thought the repo to manage configurations, which are in `conf/config.yaml`

### Run
Run all the experiments, including tests with command 
`make run`