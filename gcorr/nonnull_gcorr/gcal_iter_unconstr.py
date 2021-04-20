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
ref_dir = os.path.join('/data', 'agambrel', 'spectra', run_name)
rundir = ref_dir + '_iter'

tags = ['100']#['90', '150']

for tag in tags:
    ref_file = os.path.join(ref_dir, '{0}/gcorr_{0}_iter.npz'.format(tag))
    rundirf = os.path.join(rundir, tag)

    os.system('rm -rf {}/bandpowers*'.format(rundirf))
    os.system('rm -rf {}/transfer*'.format(rundirf))
    os.system('rm -rf {}/ERROR*'.format(rundirf))
    os.system('rm -rf {}/logs'.format(rundirf))

    if not update:
        continue

    first = False
    if os.path.exists(ref_file):
        gcorr = xf.load_and_parse(ref_file)
        print('loaded {}'.format(ref_file))
    elif os.path.exists(ref_file.replace('_iter.npz', '.npz')):
        gcorr = xf.load_and_parse(ref_file.replace('_iter.npz', '.npz'))
        print('loaded {}'.format(ref_file.replace('_iter.npz', '.npz')))
    try:
        gcorr_new = xf.load_and_parse(os.path.join(rundirf, 'gcorr_{}.npz'.format(tag)))
        print('got new gcorr {}'.format(os.path.join(rundirf, 'gcorr_{}.npz'.format(tag))))
    except IOError:
        gcorr_new = copy.deepcopy(gcorr)
        gcorr_new['gcorr'] = xf.arr_to_dict(
            np.ones_like(xf.dict_to_arr(gcorr['gcorr'], flatten=True)),
            gcorr['gcorr'],
            )
        first = True
        print('didnt get new gcorr, this must be first')

    np.savez_compressed(ref_file.replace('_iter.npz', '_prev.npz'), **gcorr)
    #print('gcorr new {}'.format(gcorr_new['gcorr']))
    #print('gcorr old {}'.format(gcorr['gcorr']))
    gcorr['gcorr'] = xf.arr_to_dict(
        xf.dict_to_arr(gcorr['gcorr'], flatten=True)
        * xf.dict_to_arr(gcorr_new['gcorr'], flatten=True),
        gcorr['gcorr']
    )
    # # fix garbage bin
    #gcorr['gcorr']['tt'][:] = 1.0
    fig, ax = plt.subplots(2, 3)
    ax = ax.flatten()
    for i, (k, v) in enumerate(gcorr['gcorr'].items()):
        #if first:
        v[0] = 0.5
        #v[1] = 0.5
        if k in ['te', 'eb', 'tb']:
            # set gcorr to 1 since it has no effect
            v[:] = 1#np.sqrt(gcorr['gcorr'][k[0]*2]*gcorr['gcorr'][k[1]*2])
        # undo iteration if g value falls below threshold
        v[v < 0.05] /= gcorr_new['gcorr'][k][v < 0.05]
        v[v > 5] /= gcorr_new['gcorr'][k][v > 5]
        for v0, val in enumerate(v):
            if val > 1.2:
                if v0 != 0:
                    v[v0] = v[v0 - 1]
                else:
                    v[v0] = 1.2
        ax[i].plot(v)
        ax[i].set_title(k)
    """
        elif k in ['tt', 'te']:
             v[0] = 0.2
        # v[0] = 1 if first else 0.2
    """
    print(gcorr['gcorr'])
    plt.savefig(os.path.join(rundir, 'plots', 'gcorr_tot_{}.png'.format(tag)))

    np.savez_compressed(ref_file, **gcorr)

print('Sumitting ensemble mean job')
os.system('python xfaster_gcal_unconstr.py -o {} -m > /dev/null'.format(
        run_name_iter))

transfer_exists = {}
for tag in tags:
    transfer_exists[tag] = False
while not np.all(transfer_exists.values()):
    # wait until transfer functions are done to submit rest of jobs
    for tag in tags:
        rundirf = os.path.join(rundir, tag)
        transfer_exists[tag] = os.path.exists(
            os.path.join(rundirf, 'transfer_all_{}.npz'.format(tag)))
    print('transfer exists: ', transfer_exists)
    sleep(15)
print('Submitting jobs for all seeds')
os.system('python xfaster_gcal_unconstr.py -o {} -n 1000 > /dev/null'.format(
        run_name_iter))

print('Waiting for jobs to complete...')
while os.system('squeue -u {} | grep xfast > /dev/null'.format(os.getenv('USER')) == 0:
    os.system('squeue -u {} | wc'.format(os.getenv('USER'))
    sleep(10)

for tag in tags:
    os.system('python compute_gcal.py -r {} {}'.format(run_name_iter, tag))

for tag in tags:
    print('{} completed'.format(tag))
    os.system('ls {}/{}/bandpowers* | wc -l'.format(rundir, tag))
    print('{} error'.format(tag))
    os.system('ls {}/{}/ERROR* | wc -l'.format(rundir, tag))

