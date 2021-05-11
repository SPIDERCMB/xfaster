'''
A iteration script used to update g_corr factors.
'''
import os
import numpy as np
import argparse as ap
from configparser import ConfigParser
from time import sleep
import copy
import glob
from matplotlib import use
use('agg')
import matplotlib.pyplot as plt
import xfaster as xf

P = ap.ArgumentParser()
P.add_argument("--gcorr-config", help="The config file for gcorr computation")

assert os.path.exists(args.gcorr_config), "Missing config file {}".format(
    args.gcorr_config
)
g_cfg = ConfigParser()
g_cfg.read(args.gcorr_config)

specs = ['tt', 'ee', 'bb', 'te']
if g_cfg.getboolean("xfaster_opts", "tbeb"):
    specs += ['eb', 'tb']

run_name = 'xfaster_gcal'
run_name_iter = run_name + '_iter'
# ref dir will contain the total gcorr file
ref_dir = os.path.join(g_cfg["gcorr_opts"]["output_root"], run_name)
# run dir will be where all the iteration happens to update the reference
rundir = ref_dir + '_iter'

tags = g_cfg["gcorr_opts"]["map_tags"]

if os.get_environ('GCORR_ITER') is None:
    os.set_environ('GCORR_ITER', 0)
else:
    os.set_environ('GCORR_ITER', os.get_environt('GCORR_ITER') + 1)

for tag in tags:
    ref_file = os.path.join(ref_dir, '{0}/gcorr_{0}_iter.npz'.format(tag))
    rundirf = os.path.join(rundir, tag)

    #Remove transfer functions and bandpowers
    os.system('rm -rf {}/bandpowers*'.format(rundirf))
    os.system('rm -rf {}/transfer*'.format(rundirf))
    os.system('rm -rf {}/ERROR*'.format(rundirf))
    os.system('rm -rf {}/logs'.format(rundirf))

    first = False

    # Get gcorr from reference folder, if it's there
    if os.path.exists(ref_file):
        gcorr = xf.load_and_parse(ref_file)
        print('loaded {}'.format(ref_file))
    elif os.path.exists(ref_file.replace('_iter.npz', '.npz')):
        gcorr = xf.load_and_parse(ref_file.replace('_iter.npz', '.npz'))
        print('loaded {}'.format(ref_file.replace('_iter.npz', '.npz')))

    # Get gcorr_correction from iter folder -- this is the multiplicative
    # change to gcorr-- should converge to 1s
    try:
        gcorr_corr = xf.load_and_parse(os.path.join(rundirf, 'gcorr_{}.npz'.format(tag)))
        print('got correction to gcorr {}'.format(os.path.join(rundirf, 'gcorr_{}.npz'.format(tag))))
    except IOError:
        gcorr_corr = copy.deepcopy(gcorr)
        gcorr_corr['gcorr'] = xf.arr_to_dict(
            np.ones_like(xf.dict_to_arr(gcorr['gcorr'], flatten=True)),
            gcorr['gcorr'],
            )
        first = True
        print("Didn't get gcorr correction file in iter folder. Starting from ones.")

    np.savez_compressed(ref_file.replace('_iter.npz', '_prev.npz'), **gcorr)

    gcorr['gcorr'] = xf.arr_to_dict(
        xf.dict_to_arr(gcorr['gcorr'], flatten=True)
        * xf.dict_to_arr(gcorr_corr['gcorr'], flatten=True),
        gcorr['gcorr']
    )

    fig_tot, ax_tot = plt.subplots(2, 3)
    fig_corr, ax_corr = plt.subplots(2, 3)
    ax_tot = ax_tot.flatten()
    ax_corr = ax_corr.flatten()
    for i, (k, v) in enumerate(gcorr['gcorr'].items()):
        v[0] = 0.5
        if k in ['te', 'eb', 'tb']:
            # We don't compute gcorr for off-diagonals
            v[:] = 1
        # Don't update gcorr if correction is extreme
        v[v < 0.05] /= gcorr_corr['gcorr'][k][v < 0.05]
        v[v > 5] /= gcorr_corr['gcorr'][k][v > 5]
        for v0, val in enumerate(v):
            if val > 1.2:
                if v0 != 0:
                    v[v0] = v[v0 - 1]
                else:
                    v[v0] = 1.2
        ax_tot[i].plot(v)
        ax_tot[i].set_title('{} total gcorr'.format(k))
        ax_corr[i].plot(gcorr_corr['gcorr'][k])
        ax_tot[i].set_title('{} gcorr corr'.format(k))

    print(gcorr['gcorr'])
    fig_tot.savefig(os.path.join(rundir, 'plots', 'gcorr_tot_{}_{}.png'.format(tag, os.get_environ('GCORR_ITER'))))
    fig_corr.savefig(os.path.join(rundir, 'plots', 'gcorr_corr_{}_{}.png'.format(tag, os.get_environ('GCORR_ITER'))))

    np.savez_compressed(ref_file, **gcorr)

print('Sumitting first job')
os.system('python run_xfaster.py --gcorr-config {g} --omp 1 --check-point transfer -o {o} > /dev/null'.format(
        g=args.gcorr_config, o=run_name_iter))

transfer_exists = {}
for tag in tags:
    transfer_exists[tag] = False
while not np.all(transfer_exists.values()):
    # wait until transfer functions are done to submit rest of jobs
    for tag in tags:
        rundirf = os.path.join(rundir, tag)
        transfer_files = glob.glob(os.path.join(rundirf, 'transfer_all*{}.npz'.format(tag))
        transfer_exists[tag] = bool(len(transfer_files))

    print('transfer exists: ', transfer_exists)
    sleep(15)
print('Submitting jobs for all seeds')
os.system('python run_xfaster.py --gcorr-config {g} --omp 1 --check-point bandpowers -o {o} -f 1 -n {n}> /dev/null'.format(
        g=args.gcorr_config, o=run_name_iter, n=g_cfg["gcorr_opts"]["nsim"]))

print('Waiting for jobs to complete...')
while os.system('squeue -u {} | grep xfast > /dev/null'.format(os.getenv('USER'))) == 0:
    os.system('squeue -u {} | wc'.format(os.getenv('USER'))
    sleep(10)

for tag in tags:
    os.system('python compute_gcal.py --gcorr-config {g} -r {r} {t}'.format(g=args.gcorr_config, r=run_name_iter, t=tag))

for tag in tags:
    print('{} completed'.format(tag))
    os.system('ls {}/{}/bandpowers* | wc -l'.format(rundir, tag))
    print('{} error'.format(tag))
    os.system('ls {}/{}/ERROR* | wc -l'.format(rundir, tag))
