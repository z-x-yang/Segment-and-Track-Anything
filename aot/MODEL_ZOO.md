## Model Zoo and Results

### Environment and Settings
- 4/1 NVIDIA V100 GPUs for training/evaluation.
- Auto-mixed precision was enabled in training but disabled in evaluation.
- Test-time augmentations were not used.
- The inference resolution of DAVIS/YouTube-VOS was 480p/1.3x480p as [CFBI](https://github.com/z-x-yang/CFBI).
- Fully online inference. We passed all the modules frame by frame.
- Multi-object FPS was recorded instead of single-object one.

### Pre-trained Models
Stages:

- `PRE`: the pre-training stage with static images.

- `PRE_YTB_DAV`: the main-training stage with YouTube-VOS and DAVIS. All the kinds of evaluation share an **identical** model and the **same** parameters.


| Model      | Param (M) |                                             PRE                                              |                                         PRE_YTB_DAV                                          |
|:---------- |:---------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
| AOTT       |    5.7    | [gdrive](https://drive.google.com/file/d/1_513h8Hok9ySQPMs_dHgX5sPexUhyCmy/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1owPmwV4owd_ll6GuilzklqTyAd0ZvbCu/view?usp=sharing) |
| AOTS       |    7.0    | [gdrive](https://drive.google.com/file/d/1QUP0-VED-lOF1oX_ppYWnXyBjvUzJJB7/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1beU5E6Mdnr_pPrgjWvdWurKAIwJSz1xf/view?usp=sharing) |
| AOTB       |    8.3    | [gdrive](https://drive.google.com/file/d/11Bx8n_INAha1IdpHjueGpf7BrKmCJDvK/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1hH-GOn4GAxHkV8ARcQzsUy8Ax6ndot-A/view?usp=sharing) |
| AOTL       |    8.3    | [gdrive](https://drive.google.com/file/d/1WL6QCsYeT7Bt-Gain9ZIrNNXpR2Hgh29/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1L1N2hkSPqrwGgnW9GyFHuG59_EYYfTG4/view?usp=sharing) |
| R50-AOTL   |   14.9    | [gdrive](https://drive.google.com/file/d/1hS4JIvOXeqvbs-CokwV6PwZV-EvzE6x8/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1qJDYn3Ibpquu4ffYoQmVjg1YCbr2JQep/view?usp=sharing) |
| SwinB-AOTL |   65.4    | [gdrive](https://drive.google.com/file/d/1LlhKQiXD8JyZGGs3hZiNzcaCLqyvL9tj/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/192jCGQZdnuTsvX-CVra-KVZl2q1ZR0vW/view?usp=sharing) |

| Model      | Param (M) |                                             PRE                                              |                                         PRE_YTB_DAV                                          |
|:---------- |:---------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
| DeAOTT       |    7.2   | [gdrive](https://drive.google.com/file/d/11C1ZBoFpL3ztKtINS8qqwPSldfYXexFK/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1ThWIZQS03cYWx1EKNN8MIMnJS5eRowzr/view?usp=sharing) |
| DeAOTS       |    10.2   | [gdrive](https://drive.google.com/file/d/1uUidrWVoaP9A5B5-EzQLbielUnRLRF3j/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1YwIAV5tBtn5spSFxKLBQBEQGwPHyQlHi/view?usp=sharing) |
| DeAOTB       |    13.2   | [gdrive](https://drive.google.com/file/d/1bEQr6vIgQMVITrSOtxWTMgycKpS0cor9/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1BHxsonnvJXylqHlZ1zJHHc-ymKyq-CFf/view?usp=sharing) |
| DeAOTL       |    13.2   | [gdrive](https://drive.google.com/file/d/1_vBL4KJlmBy0oBE4YFDOvsYL1ZtpEL32/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/18elNz_wi9JyVBcIUYKhRdL08MA-FqHD5/view?usp=sharing) |
| R50-DeAOTL   |    19.8   | [gdrive](https://drive.google.com/file/d/1sTRQ1g0WCpqVCdavv7uJiZNkXunBt3-R/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view?usp=sharing) |
| SwinB-DeAOTL |    70.3   | [gdrive](https://drive.google.com/file/d/16BZEE53no8CxT-pPLDC2q1d6Xlg8mWPU/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq/view?usp=sharing) |

To use our pre-trained model to infer, a simple way is to set `--model` and `--ckpt_path` to your downloaded checkpoint's model type and file path when running `eval.py`.

### YouTube-VOS 2018 val
`ALL-F`: all frames. The default evaluation setting of YouTube-VOS is 6fps, but 30fps sequences (all the frames) are also supplied by the dataset organizers. We noticed that many VOS methods prefer to evaluate with 30fps videos. Thus, we also supply our results here. Denser video sequences can significantly improve VOS performance when using the memory reading strategy (like AOTL, R50-AOTL, and SwinB-AOTL), but the efficiency will be influenced since more memorized frames are stored for object matching.
| Model        |    Stage    |   FPS    | All-F |   Mean   |  J Seen  |  F Seen  | J Unseen | F Unseen |                                         Predictions                                          |
|:------------ |:-----------:|:--------:|:-----:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------------------------------------------------------------------------------------------:|
| AOTT         | PRE_YTB_DAV |   41.0   |       |   80.2   |   80.4   |   85.0   |   73.6   |   81.7   | [gdrive](https://drive.google.com/file/d/1u8mvPRT08ENZHsw9Xf_4C6Sv9BoCzENR/view?usp=sharing) |
| AOTT         | PRE_YTB_DAV |   41.0   |   √   |   80.9   |   80.0   |   84.7   |   75.2   |   83.5   | [gdrive](https://drive.google.com/file/d/1RGMI5-29Z0odq73rt26eCxOUYUd-fvVv/view?usp=sharing) |
| DeAOTT       | PRE_YTB_DAV | **53.4** |       | **82.0** | **81.6** | **86.3** | **75.8** | **84.2** |    -                                                                                          |
| AOTS         | PRE_YTB_DAV |   27.1   |       |   82.9   |   82.3   |   87.0   |   77.1   |   85.1   | [gdrive](https://drive.google.com/file/d/1a4-rNnxjMuPBq21IKo31WDYZXMPgS7r2/view?usp=sharing) |
| AOTS         | PRE_YTB_DAV |   27.1   |   √   |   83.0   |   82.2   |   87.0   |   77.3   |   85.7   | [gdrive](https://drive.google.com/file/d/1Z0cndyoCw5Na6u-VFRE8CyiIG2RbMIUO/view?usp=sharing) |
| DeAOTS       | PRE_YTB_DAV | **38.7** |       | **84.0** | **83.3** | **88.3** | **77.9** | **86.6** |    -                                                                                          |
| AOTB         | PRE_YTB_DAV |   20.5   |       |   84.0   |   83.2   |   88.1   |   78.0   |   86.5   | [gdrive](https://drive.google.com/file/d/1J5nhuQbbjVLYNXViBIgo21ddQy-MiOLG/view?usp=sharing) |
| AOTB         | PRE_YTB_DAV |   20.5   |   √   |   84.1   |   83.6   |   88.5   |   78.0   |   86.5   | [gdrive](https://drive.google.com/file/d/1gFaweB_GTJjHzSD61v_ZsY9K7UEND30O/view?usp=sharing) |
| DeAOTB       | PRE_YTB_DAV | **30.4** |       | **84.6** | **83.9** | **88.9** | **78.5** | **87.0** |    -                                                                                          |
| AOTL         | PRE_YTB_DAV |   16.0   |       |   84.1   |   83.2   |   88.2   |   78.2   |   86.8   | [gdrive](https://drive.google.com/file/d/1kS8KWQ2L3wzxt44ROLTxwZOT7ZpT8Igc/view?usp=sharing) |
| AOTL         | PRE_YTB_DAV |   6.5    |   √   |   84.5   |   83.7   |   88.8   |   78.4   | **87.1** | [gdrive](https://drive.google.com/file/d/1Rpm3e215kJOUvb562lJ2kYg2I3hkrxiM/view?usp=sharing) |
| DeAOTL       | PRE_YTB_DAV | **24.7** |       | **84.8** | **84.2** | **89.4** | **78.6** |   87.0   |    -                                                                                          |
| R50-AOTL     | PRE_YTB_DAV |   14.9   |       |   84.6   |   83.7   |   88.5   |   78.8   |   87.3   | [gdrive](https://drive.google.com/file/d/1nbJZ1bbmEgyK-bg6HQ8LwCz5gVJ6wzIZ/view?usp=sharing) |
| R50-AOTL     | PRE_YTB_DAV |   6.4    |   √   |   85.5   |   84.5   |   89.5   |   79.6   |   88.2   | [gdrive](https://drive.google.com/file/d/1NbB54ZhYvfJh38KFOgovYYPjWopd-2TE/view?usp=sharing) |
| R50-DeAOTL   | PRE_YTB_DAV | **22.4** |       | **86.0** | **84.9** | **89.9** | **80.4** | **88.7** |    -                                                                                         |
| SwinB-AOTL   | PRE_YTB_DAV |   9.3    |       |   84.7   |   84.5   |   89.5   |   78.1   |   86.7   | [gdrive](https://drive.google.com/file/d/1QFowulSY0LHfpsjUV8ZE9rYc55L9DOC7/view?usp=sharing) |
| SwinB-AOTL   | PRE_YTB_DAV |   5.2    |   √   |   85.1   |   85.1   |   90.1   |   78.4   |   86.9   | [gdrive](https://drive.google.com/file/d/1TulhVOhh01rkssNYbOQASeWKu7CQ5Azx/view?usp=sharing) |
| SwinB-DeAOTL | PRE_YTB_DAV | **11.9** |       | **86.2** | **85.6** | **90.6** | **80.0** | **88.4** |    -                                                                                  |

### YouTube-VOS 2019 val
| Model        |    Stage    |   FPS    | All-F |   Mean   |  J Seen  |  F Seen  | J Unseen | F Unseen |                                         Predictions                                          |
|:------------ |:-----------:|:--------:|:-----:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------------------------------------------------------------------------------------------:|
| AOTT         | PRE_YTB_DAV |   41.0   |       |   80.0   |   79.8   |   84.2   |   74.1   |   82.1   | [gdrive](https://drive.google.com/file/d/1zzyhN1XYtajte5nbZ7opOdfXeDJgCxC5/view?usp=sharing) |
| AOTT         | PRE_YTB_DAV |   41.0   |   √   |   80.9   |   79.9   |   84.4   |   75.6   |   83.8   | [gdrive](https://drive.google.com/file/d/1V_5vi9dAXOis_WrDieacSESm7OX20Bv-/view?usp=sharing) |
| DeAOTT       | PRE_YTB_DAV | **53.4** |       | **82.0** | **81.2** | **85.6** | **76.4** | **84.7** |     -                                                                                         |
| AOTS         | PRE_YTB_DAV |   27.1   |       |   82.7   |   81.9   |   86.5   |   77.3   |   85.2   | [gdrive](https://drive.google.com/file/d/11YdkUeyjkTv8Uw7xMgPCBzJs6v5SDt6n/view?usp=sharing) |
| AOTS         | PRE_YTB_DAV |   27.1   |   √   |   82.8   |   81.9   |   86.5   |   77.3   |   85.6   | [gdrive](https://drive.google.com/file/d/1UhyurGTJeAw412czU3_ebzNwF8xQ4QG_/view?usp=sharing) |
| DeAOTS       | PRE_YTB_DAV | **38.7** |       | **83.8** | **82.8** | **87.5** | **78.1** | **86.8** |     -                                                                                         |
| AOTB         | PRE_YTB_DAV |   20.5   |       |   84.0   |   83.1   |   87.7   |   78.5   |   86.8   | [gdrive](https://drive.google.com/file/d/1NeI8cT4kVqTqVWAwtwiga1rkrvksNWaO/view?usp=sharing) |
| AOTB         | PRE_YTB_DAV |   20.5   |   √   |   84.1   |   83.3   |   88.0   |   78.2   |   86.7   | [gdrive](https://drive.google.com/file/d/1kpYV2XFR0sOfLWD-wMhd-nUO6CFiLjlL/view?usp=sharing) |
| DeAOTB       | PRE_YTB_DAV | **30.4** |       | **84.6** | **83.5** | **88.3** | **79.1** | **87.5** |     -                                                                                         |
| AOTL         | PRE_YTB_DAV |   16.0   |       |   84.0   |   82.8   |   87.6   |   78.6   |   87.1   | [gdrive](https://drive.google.com/file/d/1qKLlNXxmT31bW0weEHI_zAf4QwU8Lhou/view?usp=sharing) |
| AOTL         | PRE_YTB_DAV |   6.5    |   √   |   84.2   |   83.0   |   87.8   |   78.7   |   87.3   | [gdrive](https://drive.google.com/file/d/1o3fwZ0cH71bqHSA3bYNjhP4GGv9Vyuwa/view?usp=sharing) |
| DeAOTL       | PRE_YTB_DAV | **24.7** |       | **84.7** | **83.8** | **88.8** | **79.0** | **87.2** |     -                                                                                         |
| R50-AOTL     | PRE_YTB_DAV |   14.9   |       |   84.4   |   83.4   |   88.1   |   78.7   |   87.2   | [gdrive](https://drive.google.com/file/d/1I7ooSp8EYfU6fvkP6QcCMaxeencA68AH/view?usp=sharing) |
| R50-AOTL     | PRE_YTB_DAV |   6.4    |   √   |   85.3   |   83.9   |   88.8   |   79.9   |   88.5   | [gdrive](https://drive.google.com/file/d/1OGqlkEu0uXa8QVWIVz_M5pmXXiYR2sh3/view?usp=sharing) |
| R50-DeAOTL   | PRE_YTB_DAV | **22.4** |       | **85.9** | **84.6** | **89.4** | **80.8** | **88.9** |     -                                                                                         |
| SwinB-AOTL   | PRE_YTB_DAV |   9.3    |       |   84.7   |   84.0   |   88.8   |   78.7   |   87.1   | [gdrive](https://drive.google.com/file/d/1fPzCxi5GM7N2sLKkhoTC2yoY_oTQCHp1/view?usp=sharing) |
| SwinB-AOTL   | PRE_YTB_DAV |   5.2    |   √   |   85.3   |   84.6   |   89.5   |   79.3   |   87.7   | [gdrive](https://drive.google.com/file/d/1e3D22s_rJ7Y2X2MHo7x5lcNtwmHFlwYB/view?usp=sharing) |
| SwinB-DeAOTL | PRE_YTB_DAV | **11.9** |       | **86.1** | **85.3** | **90.2** | **80.4** | **88.6** |     -                                                                                         |

### DAVIS-2017 test

| Model      |    Stage    | FPS  |   Mean   | J Score  | F Score  | Predictions |
| ---------- |:-----------:|:----:|:--------:|:--------:|:--------:|:----:|
| AOTT       | PRE_YTB_DAV | **51.4** |   73.7   |   70.0   |   77.3   | [gdrive](https://drive.google.com/file/d/14Pu-6Uz4rfmJ_WyL2yl57KTx_pSSUNAf/view?usp=sharing) |
| AOTS       | PRE_YTB_DAV | 40.0 |   75.2   |   71.4   |   78.9   | [gdrive](https://drive.google.com/file/d/1zzAPZCRLgnBWuAXqejPPEYLqBxu67Rj1/view?usp=sharing) |
| AOTB       | PRE_YTB_DAV | 29.6 |   77.4   |   73.7   |   81.1   | [gdrive](https://drive.google.com/file/d/1WpQ-_Jrs7Ssfw0oekrejM2OVWEx_tBN1/view?usp=sharing) |
| AOTL       | PRE_YTB_DAV | 18.7 |   79.3   |   75.5   |   83.2   | [gdrive](https://drive.google.com/file/d/1rP1Zdgc0N1d8RR2EaXMz3F-o5zqcNVe8/view?usp=sharing) |
| R50-AOTL   | PRE_YTB_DAV | 18.0 |   79.5   |   76.0   |   83.0   | [gdrive](https://drive.google.com/file/d/1iQ5iNlvlS-In586ZNc4LIZMSdNIWDvle/view?usp=sharing) |
| SwinB-AOTL | PRE_YTB_DAV | 12.1  | **82.1** | **78.2** | **85.9** | [gdrive](https://drive.google.com/file/d/1oVt4FPcZdfVHiOxjYYKef0q7Ovy4f5Q_/view?usp=sharing) |

### DAVIS-2017 val

| Model      |    Stage    | FPS  |   Mean   | J Score  |  F Score  | Predictions |
| ---------- |:-----------:|:----:|:--------:|:--------:|:---------:|:----:|
| AOTT       | PRE_YTB_DAV | **51.4** |   79.2   |   76.5   |   81.9    | [gdrive](https://drive.google.com/file/d/10OUFhK2Sz-hOJrTDoTI0mA45KO1qodZt/view?usp=sharing) |
| AOTS       | PRE_YTB_DAV | 40.0 |   82.1   |   79.3   |   84.8    | [gdrive](https://drive.google.com/file/d/1T-JTYyksWlq45jxcLjnRaBvvYUhWgHFH/view?usp=sharing) |
| AOTB       | PRE_YTB_DAV | 29.6 |   83.3   |   80.6   |   85.9    | [gdrive](https://drive.google.com/file/d/1EVUnxQm9TLBTuwK82QyiSKk9R9V8NwRL/view?usp=sharing) |
| AOTL       | PRE_YTB_DAV | 18.7 |   83.6   |   80.8   |   86.3    | [gdrive](https://drive.google.com/file/d/1CFauSni2BxAe_fcl8W_6bFByuwJRbDYm/view?usp=sharing) |
| R50-AOTL   | PRE_YTB_DAV | 18.0 |   85.2   |   82.5   |   87.9    | [gdrive](https://drive.google.com/file/d/1vjloxnP8R4PZdsH2DDizfU2CrkdRHHyo/view?usp=sharing) |
| SwinB-AOTL | PRE_YTB_DAV | 12.1  | **85.9** | **82.9** | **88.9** | [gdrive](https://drive.google.com/file/d/1tYCbKOas0i7Et2iyUAyDwaXnaD9YWxLr/view?usp=sharing) |

### DAVIS-2016 val

| Model      |    Stage    | FPS  |   Mean   | J Score  | F Score  | Predictions |
| ---------- |:-----------:|:----:|:--------:|:--------:|:--------:|:----:|
| AOTT       | PRE_YTB_DAV | **51.4** |   87.5   |   86.5   |   88.4   | [gdrive](https://drive.google.com/file/d/1LeW8WQhnylZ3umT7E379KdII92uUsGA9/view?usp=sharing) |
| AOTS       | PRE_YTB_DAV | 40.0 |   89.6   |   88.6   |   90.5   | [gdrive](https://drive.google.com/file/d/1vqGei5tLu1FPVrTi5bwRAsaGy3Upf7B1/view?usp=sharing) |
| AOTB       | PRE_YTB_DAV | 29.6 |   90.9   |   89.6   |   92.1   | [gdrive](https://drive.google.com/file/d/1qAppo2uOVu0FbE9t1FBUpymC3yWgw1LM/view?usp=sharing) |
| AOTL       | PRE_YTB_DAV | 18.7 |   91.1   |   89.5   |   92.7   | [gdrive](https://drive.google.com/file/d/1g6cjYhgBWjMaY3RGAm31qm3SPEF3QcKV/view?usp=sharing) |
| R50-AOTL   | PRE_YTB_DAV | 18.0 |   91.7   |   90.4   |   93.0   | [gdrive](https://drive.google.com/file/d/1QzxojqWKsvRf53K2AgKsK523ZVuYU4O-/view?usp=sharing) |
| SwinB-AOTL | PRE_YTB_DAV | 12.1  | **92.2** | **90.6** | **93.8** | [gdrive](https://drive.google.com/file/d/1RIqUtAyVnopeogfT520d7a0yiULg1obp/view?usp=sharing) |
