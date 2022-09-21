#!/bin/bash
conda env create -f conda.yml
# shellcheck disable=SC1090
{
    source ~/miniconda-mio/etc/profile.d/conda.sh
}
conda activate cv-detr
pip install -r requirements.txt
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
