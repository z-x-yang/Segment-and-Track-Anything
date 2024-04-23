## Pretrained Models
We provide full AudioSet pretrained models and Speechcommands-V2-35 pretrained model.
1. [Full AudioSet, 10 tstride, 10 fstride, with Weight Averaging (0.459 mAP)](https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1)
2. [Full AudioSet, 10 tstride, 10 fstride, without Weight Averaging, Model 1 (0.450 mAP)](https://www.dropbox.com/s/1tv0hovue1bxupk/audioset_10_10_0.4495.pth?dl=1)
3. [Full AudioSet, 10 tstride, 10 fstride, without Weight Averaging, Model 2  (0.448 mAP)](https://www.dropbox.com/s/6u5sikl4b9wo4u5/audioset_10_10_0.4483.pth?dl=1)
4. [Full AudioSet, 10 tstride, 10 fstride, without Weight Averaging, Model 3  (0.448 mAP)](https://www.dropbox.com/s/kt6i0v9fvfm1mbq/audioset_10_10_0.4475.pth?dl=1)
5. [Full AudioSet, 12 tstride, 12 fstride, without Weight Averaging, Model (0.447 mAP)](https://www.dropbox.com/s/snfhx3tizr4nuc8/audioset_12_12_0.4467.pth?dl=1)
6. [Full AudioSet, 14 tstride, 14 fstride, without Weight Averaging, Model (0.443 mAP)](https://www.dropbox.com/s/z18s6pemtnxm4k7/audioset_14_14_0.4431.pth?dl=1)
7. [Full AudioSet, 16 tstride, 16 fstride, without Weight Averaging, Model (0.442 mAP)](https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1)

8. [Speechcommands V2-35, 10 tstride, 10 fstride, without Weight Averaging, Model (98.12% accuracy on evaluation set)](https://www.dropbox.com/s/q0tbqpwv44pquwy/speechcommands_10_10_0.9812.pth?dl=1)

If you want to finetune AudioSet-pretrained AST model on your task, you can simply set the `audioset_pretrain=True` when you create the AST model, it will automatically download model 1 (`0.459 mAP`). In our ESC-50 recipe, AudioSet pretraining is used.

If you want to reproduce ensemble experiments, you can download these models at one click using `ast/egs/audioset/download_models.sh`. Ensemble model 2-4 achieves `0.475 mAP`, Ensemble model 2-7 achieves and `0.485 mAP`. Once you download the model, you can try `ast/egs/audioset/ensemble.py`, you need to change the `eval_data_path` and `mdl_list` to run it. We attached our ensemble log in `/ast/egs/audioset/exp/ensemble-s.log` and `ensemble-m.log`.

Please  note that we use `16kHz` audios for training and test (for all AudioSet, SpeechCommands, and ESC-50), so if you want to use the pretrained model, please prepare your data in `16kHz`.

(Note: the above links are Dropbox direct links (i.e., can be downloaded by wget) and should work for most users. For users having issue downloading with the above Dropbox links, it is recommended to use a VPN or use the [OneDrive links](https://mitprod-my.sharepoint.com/:f:/g/personal/yuangong_mit_edu/ErLKkiP-GwVMgdsCeGEjsmoBMtGvXMkX3tCj5_I0E7ikNA?e=JE9Om8) or [腾讯微云链接们](https://share.weiyun.com/xRGK6zmg), however, OneDrive and 腾讯微云 links are not direct link, please manually download the `audioset_10_10_0.4593.pth`[[OneDrive]](https://mitprod-my.sharepoint.com/:u:/g/personal/yuangong_mit_edu/EWrY3raql55CqxZNV3cVSkABaoU7pXQxAeJXudE1PTNzQg?e=gwEICj) [[腾讯微云]](https://share.weiyun.com/kcmk2KHw) and place it in `ast/pretrained_models` if you want to set `audioset_pretrain=True` because the wget link in the `ast/src/models/ast_models.py` would fail if you cannot connect to Dropbox.) 

## Use Pretrained Model For Downstream Tasks

You can use the pretrained AST model for your own dataset. There are two ways to doing so.

You can of course only take ``ast/src/models/ast_models.py``, set ``audioset_pretrain=True``, and use it with your training pipeline, the only thing need to take care of is the input normalization, we normalize our input to 0 mean and 0.5 std. To use the pretrained model, you should roughly normalize the input to this range. You can check ``ast/src/get_norm_stats.py`` to see how we compute the stats, or you can try using our AudioSet normalization ``input_spec = (input_spec + 4.26) / (4.57 * 2)``. Using your own training pipeline might be easier if you already have a good one.
Please note that AST needs smaller learning rate (we use 10 times smaller learning rate than our CNN model proposed in the [PSLA paper](https://arxiv.org/abs/2102.01243)) and converges faster, so please search the learning rate and learning rate scheduler for your task. 

If you want to use our training pipeline, you would need to modify below for your new dataset.
1. You need to create a json file, and a label index for your dataset, see ``ast/egs/audioset/data/`` for an example.
2. In ``/your_dataset/run.sh``, you need to specify the data json file path, the SpecAug parameters (``freqm`` and ``timem``, we recommend to mask 48 frequency bins out of 128, and 20% of your time frames), the mixup rate (i.e., how many samples are mixup samples), batch size, initial learning rate, etc. Please see ``ast/egs/[audioset,esc50,speechcommands]/run.sh]`` for samples.
3. In ``ast/src/run.py``, line 60-65, you need to add the normalization stats, the input frame length, and if noise augmentation is needed for your dataset. Also take a look at line 101-127 if you have a seperate validation set. For normalization stats, you need to compute the mean and std of your dataset (check ``ast/src/get_norm_stats.py``) or you can try using our AudioSet normalization ``input_spec = (input_spec + 4.26) / (4.57 * 2)``.
4. In ``ast/src/traintest.`` line 55-82, you need to specify the learning rate scheduler, metrics, warmup setting and the optimizer for your task.

To summarize, to use our training pipeline, you need to creat data files and modify the above three python scripts. You can refer to our ESC-50 and Speechcommands recipes.

Also, please note that we use `16kHz` audios for the pretrained model, so if you want to use the pretrained model, please prepare your data in `16kHz`.