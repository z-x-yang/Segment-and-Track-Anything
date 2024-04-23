# -*- coding: utf-8 -*-
# @Time    : 8/18/21 5:16 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : check_eval.py

# check if all our evaluation audios are in the official release, if so, put all video ids in a file.
# this is just a sanity check.

import numpy as np
import json

our_eval = '/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json'
official_eval = 'official_eval_segments.csv'

oe = np.loadtxt(official_eval, delimiter=',', dtype=str)
oe = list(oe[:, 0])

with open(our_eval, 'r') as f:
    cur_json = json.load(f)
cur_data = cur_json['data']
print(len(cur_data))

our_eval_id = []
for entry in cur_data:
    cur_id = entry['video_id']
    if cur_id not in oe:
        print('There is an eval audio not in official release :' + cur_id)
    our_eval_id.append(cur_id)

np.savetxt('our_as_eval_id.csv', our_eval_id, delimiter=',', fmt='%s')
