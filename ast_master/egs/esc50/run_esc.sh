#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-esc50"
#SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=esc50
imagenetpretrain=True
audiosetpretrain=True
bal=none
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi
freqm=24
timem=96
mixup=0
epoch=25
batch_size=48
fstride=10
tstride=10

dataset_mean=-6.6268077
dataset_std=5.358466
audio_length=512
noise=False

metrics=acc
loss=CE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

base_exp_dir=./exp/test-${dataset}-f$fstride-t$tstride-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}

python ./prep_esc50.py

if [ -d $base_exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $base_exp_dir

for((fold=1;fold<=5;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=./data/datafiles/esc_train_data_${fold}.json
  te_data=./data/datafiles/esc_eval_data_${fold}.json

  CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/esc_class_labels_indices.csv --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
  --metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise}
done

python ./get_esc_result.py --exp_path ${base_exp_dir}