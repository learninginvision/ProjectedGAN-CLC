CUDA_VISIBLE_DEVICES=1 python train.py --outdir=../EXP/ --cfg=fastgan_lite --data=/data1/zjw/few-shot-data/landscape256.zip \
  --gpus=1 --batch=32 --mirror=1 --snap=25 --batch-gpu=32 --kimg=10000 
