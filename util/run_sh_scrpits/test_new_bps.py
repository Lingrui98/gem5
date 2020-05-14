#!/usr/bin/env python3

import sh
import subprocess
import os
from common import *

new_bps = [
    # 'MultiperspectivePerceptron8KB',
    # 'MultiperspectivePerceptron64KB',
    'MultiperspectivePerceptronTAGE8KB',
    'MultiperspectivePerceptronTAGE64KB',
    'TAGE_SC_L_8KB',
    'TAGE_SC_L_64KB',
    'LTAGE'
]

rv_origin = './rv-origin.py'
options = [rv_origin]

if __name__ == '__main__':
    for bp in new_bps:
        options += ['--num-threads=5',
                    '-a',
                    '--use-other-bp='+bp]
        run(options)
        get_data(bp)
        options = [rv_origin]
