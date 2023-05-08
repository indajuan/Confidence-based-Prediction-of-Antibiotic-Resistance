#!/bin/env bash

python train.py model2 config.yaml
python final_val.py model2 config.yaml
