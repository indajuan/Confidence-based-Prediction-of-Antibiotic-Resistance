#!/bin/env bash

python train.py model config.yaml
python final_val.py model config.yaml
