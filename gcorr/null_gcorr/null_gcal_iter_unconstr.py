import os
import numpy as np
import xfaster as xf
from time import sleep
import copy
from collections import OrderedDict
from matplotlib import use
use('agg')
import matplotlib.pyplot as plt

update = True
specs = ['tt', 'ee', 'bb', 'te', 'eb', 'tb']

run_name = 'xfaster_gcal_unconstr'
run_name_iter = run_name + '_iter'
ref_dir = os.path.join('/mnt/spider2', 'xsong', 'null_tests/202103_gcorrfull', run_name)
rundir = ref_dir + '_iter'

tags = ['90', '150a']

for tag in tags:
    ref_file = os.path.join(ref_dir, '{0}/gcorr_{0}_iter.npz'.format(tag))
    rundirf = os.path.join(rundir, tag)
    
    #Remove transfer functions and bandpowers
    os.system('rm -rf {}/bandpowers*'.format(rundirf))
    os.system('rm -rf {}/transfer*'.format(rundirf))
    os.system('rm -rf {}/ERROR*'.format(rundirf))
    os.system('rm -rf {}/logs'.format(rundirf))

    if not update:
        continue

    first = False
    
    #Get gcorr from unconstr folder
    if os.path.exists(ref_file):
        gcorr = xf.load_and_parse(ref_file)
        print('Loaded last computed gcorr')
    elif os.path.exists(ref_file.replace('_iter.npz', '.npz')):
        gcorr = xf.load_and_parse(ref_file.replace('_iter.npz', '.npz'))
        print('Loaded Initial computed gcorr')

    # Get gcorr_correction from iter folder
    try:
        gcorr_new = xf.load_and_parse(os.path.join(rundirf, 'gcorr_{}.npz'.format(tag)))
        print('Got new gcorr correction from: \n {}'.format(os.path.join(rundirf, 'gcorr_{}.npz'.format(tag))))
    except IOError:
    # Initial gcorr_correction is one.
        gcorr_new = copy.deepcopy(gcorr)
        gcorr_new['gcorr'] = xf.arr_to_dict(
            np.ones_like(xf.dict_to_arr(gcorr['gcorr'], flatten=True)),
            gcorr['gcorr'],
            )
        first = True
        print('Did not get gfac_correction file in iter folder. Starting from ones.')

    np.savez_compressed(ref_file.replace('_iter.npz', '_prev.npz'), **gcorr)
    gcorr['gcorr'] = xf.arr_to_dict(
        xf.dict_to_arr(gcorr['gcorr'], flatten=True)
        * xf.dict_to_arr(gcorr_new['gcorr'], flatten=True),
        gcorr['gcorr']
    )

    for i, (k, v) in enumerate(gcorr['gcorr'].items()):
        v[0] = 0.5
        if k in ['te', 'eb', 'tb']:
            v[:] = 1

        # undo iteration if g value falls below threshold
        v[v < 0.05] /= gcorr_new['gcorr'][k][v < 0.05]
        v[v > 5] /= gcorr_new['gcorr'][k][v > 5]
        for v0, val in enumerate(v):
            if val > 1.2:
                if v0 != 0:
                    v[v0] = v[v0 - 1]
                else:
                    v[v0] = 1.2

    print(gcorr['gcorr'])
    print("Saving new gcorr file to \n {}".format(ref_file))
    np.savez_compressed(ref_file, **gcorr)
    
#Submitting a first simulation run
print('Submitting first simulation job')
os.system('python xfaster_gcal_unconstr.py -o {} > /dev/null'.format(run_name_iter))

transfer_exists = {}
for tag in tags:
    transfer_exists[tag] = False
while not np.all(transfer_exists.values()):
    # wait until transfer functions are done to submit rest of jobs
    for tag in tags:
        rundirf = os.path.join(rundir, tag)
        transfer_exists[tag] = os.path.exists(
            os.path.join(rundirf, 'transfer_all_wbins_{}.npz'.format(tag)))
    print('transfer exists: ', transfer_exists)
    sleep(15)

print('Submitting jobs for remaining 499 seeds')
os.system('python xfaster_gcal_unconstr.py -o {} -f 1 -n 500 > /dev/null'.format(run_name_iter))

print('Waiting for jobs to complete...')

while os.system('squeue -u {} | grep xfast > /dev/null'.format(os.getenv('USER'))) == 0:
    os.system('squeue -u {} | wc'.format(os.getenv('USER')))
    sleep(10)

#Compute new correction
for tag in tags:
    os.system('python compute_gcal.py -r {} {}'.format(run_name_iter, tag))

for tag in tags:
    print('{} completed'.format(tag))
    os.system('ls {}/{}/bandpowers* | wc -l'.format(rundir, tag))
