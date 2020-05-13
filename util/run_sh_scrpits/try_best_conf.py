#!/usr/bin/env python3

import subprocess
import math as m
from test import run

storage = 4

hislen = [i for i in range(8, 37, 4)]
theta = [m.floor(1.93*h+14) for h in hislen]
bitsPerWeight = [m.ceil(m.log2(t))+1 for t in theta]
size = [m.floor(storage*1024*8/bitsPerWeight[i]/(hislen[i]+1))
        for i in range(len(hislen))]

comb = [[size[i], hislen[i], bitsPerWeight[i]] for i in range(len(hislen))]
for c in comb:
    print(c)

num_thread = 6

rv_origin = './rv-origin.py'
options = [rv_origin]

for [s, h, b] in comb:
    options += ['--num-threads={}'.format(num_thread),
                '--bp-size={}'.format(s),
                '--bp-history-len={}'.format(h),
                '--bp-weight-bit={}'.format(b),
                '-a']
    run(options)
    options = [rv_origin]

