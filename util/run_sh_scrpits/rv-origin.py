#!/usr/bin/env python3

import os
import re
import sys
import random
import sh
import time
from os.path import join as pjoin
from os.path import expanduser as uexp
from multiprocessing import Pool
from multiprocessing import cpu_count
import common as c
import argparse

numIQ = 128

#debug_flag = 'PAthPerceptron'
debug_flag = 'MYperceptron'

target_function = 'all_function_spec.txt'

home = os.getenv('HOME')
res_dir = pjoin(home, 'gem5/gem5-results/')

arch = 'RISCV'

default_params = {\
    'size': 256,
    'index': 'MODULO',
    'his_len': 32,
    'lr': 1,
    'pseudo-tag': 0,
    'dyn-thres': 0,
    'tc-bit': 0,
    'w_bit': 8
}

BP_TYPES = {
    'LocalBP',
    'MyPerceptron',
    'PathPerceptron',
    'TournamentBP',
    'BiModeBP',
    'TAGE',
    'MultiperspectivePerceptron8KB',
    'MultiperspectivePerceptron64KB'
}


def out_dir_gen(opt):
    outdir = res_dir + 'my'

    if opt.bp_size:
        outdir = outdir + '_size' + str(opt.bp_size)
    else:
        outdir = outdir + '_size' + str(default_params['size'])

    if opt.bp_history_len:
        outdir = outdir + '_his' + str(opt.bp_history_len)
    else:
        outdir = outdir + '_his' + str(default_params['his_len'])

    if opt.bp_index_type and opt.bp_index_type != default_params['index']:
        outdir = outdir + '_index' + str(opt.bp_index_type)

    if opt.bp_learning_rate and opt.bp_learning_rate != default_params['lr']:
        outdir = outdir + '_lr' + str(opt.bp_learning_rate)

    if opt.bp_pseudo_tagging and opt.bp_pseudo_tagging != 0:
        outdir = outdir + '_pseudotag' + str(opt.bp_pseudo_tagging)

    if opt.bp_dyn_thres and opt.bp_dyn_thres != 0:
        outdir = outdir + '_dyn' + str(opt.bp_dyn_thres)

    if opt.bp_tc_bit and opt.bp_tc_bit != 0:
        outdir = outdir + '_tc' + str(opt.bp_tc_bit)

    if opt.bp_weight_bit and opt.bp_weight_bit != default_params['w_bit']:
        outdir = outdir + '_w' + str(opt.bp_weight_bit)

    if opt.bp_redundant_bit and opt.bp_redundant_bit > 1:
        outdir = outdir + '_redund' + str(opt.bp_redundant_bit)

    # Override
    if opt.use_mpp8KB:
        outdir = res_dir + 'mpp_8KB'
    if opt.use_mpp64KB:
        outdir = res_dir + 'mpp_64KB'
    if opt.output_dir:
        outdir = res_dir + opt.output_dir
    print('out dir is', outdir)

    return outdir

def parser_add_arguments(parser):
    parser.add_argument('-n', '--num-threads', action='store', type=int,
                        help='Num threads used to run benchmarks')

    group = parser.add_mutually_exclusive_group()

    group.add_argument('-s', '--specified-benchmark', action='append',
                        type=str,
                        help='Specify benchmarks to run')

    group.add_argument('-a', '--all', action='store_true', default=False,
                        help='Whether run all the benchmarks')

    parser.add_argument('-o', '--output-dir', action='store', type=str,
                        help='Specify the output directory')

    # params of perceptron based branch predictor
    parser.add_argument('--bp-size', action='store', type=int,
                        help='Global predictor size')

    parser.add_argument('--bp-index-type', action='store', type=str,
                        help='Indexing method of perceptron BP')

    parser.add_argument('--bp-history-len', action='store', type=int,
                        help='History length(size of each perceptron)')

    parser.add_argument('--bp-learning-rate', action='store', type=int,
                        help='Learning rate of perceptron BP')

    parser.add_argument('--bp-pseudo-tagging', action='store', type=int,
                        help='Num bits of pseudo-tagging')

    parser.add_argument('--bp-dyn-thres', action='store', type=int,
                        help='log2 of num theta used')

    parser.add_argument('--bp-tc-bit', action='store', type=int,
                        help='valid when dyn-thres is not 0, counter bit')

    parser.add_argument('--bp-weight-bit', action='store', type=int,
                        help='Bits used to store each weight')

    parser.add_argument('--bp-redundant-bit', action='store', type=int,
                        help='Bits used to represent a history bit')

    # use other bps
    use_bp = parser.add_mutually_exclusive_group()

    use_bp.add_argument('--use-ltage', action='store_true',
                        help='Use LTAGE as the branch predictor')

    use_bp.add_argument('--use-tournament', action='store_true',
                        help='Use Tournament as the branch predictor')
    
    use_bp.add_argument('--use-mpp8KB', action='store_true',
                        help='Use MultiperspectivePerceptron 8KB \
                        as the branch predictor')

    use_bp.add_argument('--use-mpp64KB', action='store_true',
                        help='Use MultiperspectivePerceptron 64KB \
                        as the branch predictor')


def rv_origin(benchmark, some_extra_args, outdir_b):

    interval = 200*10**6
    warmup = 20*10**6

    os.chdir(c.gem5_exec())

    options = [
            '--outdir=' + outdir_b,
            '--debug-flags=' + debug_flag,
            pjoin(c.gem5_home(), 'configs/spec2006/se_spec06.py'),
            '--spec-2006-bench',
            '-b', '{}'.format(benchmark),
            '--benchmark-stdout={}/out'.format(outdir_b),
            '--benchmark-stderr={}/err'.format(outdir_b),
            '-I {}'.format(220*10**6),
            '--mem-size=4GB',
            '-r 1',
            '--restore-simpoint-checkpoint',
            '--checkpoint-dir={}'.format(pjoin(c.gem5_cpt_dir(arch),
                benchmark)),
            '--arch={}'.format(arch),
            ]
    cpu_model = 'OoO'
    if cpu_model == 'TimingSimple':
        options += [
                '--cpu-type=TimingSimpleCPU',
                '--mem-type=SimpleMemory',
                ]
    elif cpu_model == 'OoO':
        options += [
            #'--debug-flags=Fetch',
            '--cpu-type=DerivO3CPU',
            '--mem-type=DDR3_1600_8x8',

            '--caches',
            '--cacheline_size=64',

            '--l1i_size=32kB',
            '--l1d_size=32kB',
            '--l1i_assoc=8',
            '--l1d_assoc=8',

            '--l2cache',
            '--l2_size=4MB',
            '--l2_assoc=8',
            '--num-ROB=300',
            '--num-IQ={}'.format(numIQ),
            '--num-LQ=100',
            '--num-SQ=100',
            '--num-PhysReg=256']

        opt = some_extra_args
        if opt.use_ltage:
            options += ['--use-ltage']
        elif opt.use_tournament:
            options += ['--bp-type=TournamentBP']
        elif opt.use_mpp8KB:
            options += ['--bp-type=MultiperspectivePerceptron8KB']
        else:
            if opt.bp_size != None:
                options += ['--bp-size={}'.format(opt.bp_size)]
            if opt.bp_index_type != None:
                options += ['--bp-index-type={}'.format(opt.bp_index_type)]
            if opt.bp_history_len != None:
                options += ['--bp-history-len={}'.format(opt.bp_history_len)]
            if opt.bp_learning_rate != None:
                options +=\
                    ['--bp-learning-rate={}'.format(opt.bp_learning_rate)]
            if opt.bp_pseudo_tagging != None and opt.bp_pseudo_tagging != 0:
                options +=\
                    ['--bp-pseudo-tagging={}'.format(opt.bp_pseudo_tagging)]
            if opt.bp_dyn_thres != None:
                options += ['--bp-dyn-thres={}'.format(opt.bp_dyn_thres)]
                if opt.bp_tc_bit != None:
                    options += ['--bp-tc-bit={}'.format(opt.bp_tc_bit)]
            if opt.bp_weight_bit:
                options += ['--bp-weight-bit={}'.format(opt.bp_weight_bit)]
            if opt.bp_redundant_bit:
                options +=['--bp-redundant-bit={}'.\
                        format(opt.bp_redundant_bit)]
    else:
        assert False

    print(options)
    gem5 = sh.Command(pjoin(c.gem5_build(arch), 'gem5.opt'))
    # sys.exit(0)
    gem5(
            _out=pjoin(outdir_b, 'gem5_out.txt'),
            _err=pjoin(outdir_b, 'gem5_err.txt'),
            *options
            )


def run(args):
    [benchmark, opt] = args
    print(benchmark, opt)
    outdir = out_dir_gen(opt)
    outdir_b = pjoin(outdir, benchmark)
    if not os.path.isdir(outdir_b):
        os.makedirs(outdir_b)

    cpt_flag_file = pjoin(c.gem5_cpt_dir(arch), benchmark,
            'ts-take_cpt_for_benchmark')
    prerequisite = os.path.isfile(cpt_flag_file)
    some_extra_args = opt

    if prerequisite:
        print('cpt flag found, is going to run gem5 on', benchmark)
        c.avoid_repeated(rv_origin, outdir_b,
                benchmark, some_extra_args, outdir_b)
    else:
        print('prerequisite not satisified, abort on', benchmark)


def main():
    parser = argparse.ArgumentParser(usage='-n [-s | -a]')

    parser_add_arguments(parser)

    opt = parser.parse_args()

    if opt.num_threads:
        num_thread = opt.num_threads
    else:
        if cpu_count() > 22:
            num_thread = 22
        else:
            num_thread = int(cpu_count() / 2)

    benchmarks = []

    if opt.specified_benchmark == None:
        targets = ''
        if opt.all:
            targets = 'all_function_spec.txt'
        else:
            targets = 'target_function_spec.txt'

        with open(targets) as f:
            for line in f:
                benchmarks.append(line.strip())
    else:
        for bench in opt.specified_benchmark:
            benchmarks.append(bench)

    print(benchmarks)
    print(opt)
    if num_thread > 1:
        p = Pool(num_thread)
        #print(len(benchmarks))
        args = [[b, opt] for b in benchmarks]
        #print(args)
        #exit()
        p.map(run, args)
    else:
        run([benchmarks[0], opt])


if __name__ == '__main__':
    main()
