# Annotations Helper

A self-hosted ollama model that helps data team to work with documentation. Unfortunately, the data itself is not included in this repository as they are sensitive.

## Sources


## Pre-requisites
- [Ollama](https://ollama.com/download) - follow the instructions to install Ollama

## How to run
- run ollama in the terminal - ```ollama run llama3.1```
- pull the embedding model from the ollama server - ```ollama pull all-minilm```
- install dependencies - ```poetry install --no-root```
- run the chatbot - ```poetry run python main.py```
