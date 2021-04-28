import os
import glob
import numpy as np
from collections import OrderedDict
import scipy.optimize as opt
import xfaster as xf

specs = ['tt', 'ee', 'bb', 'te', 'eb', 'tb']

def xfaster_run_ensemble(output_root=None, output_tag=None):

    if output_root is None:
        output_root = os.getcwd()
    if output_tag is not None:
        output_root = os.path.join(output_root, output_tag)
        output_tag = '_{}'.format(output_tag)
    else:
        output_tag = ''

    sample_sim_file = os.path.join(output_root, 'bandpowers_sim0000_wbins{}.npz'.format(output_tag))
    bp_sample = xf.load_and_parse(sample_sim_file)
    out = {'data_version': 1, 
	   'bin_def': bp_sample['bin_def']}
    
    file_glob = os.path.join(output_root, 'bandpowers_sim[0-9][0-9][0-9][0-9]_wbins{}.npz'.format(output_tag))
    files = sorted(glob.glob(file_glob))
   
    if not len(files):
        raise OSError("No bandpowers files found in {}".format(output_root))

    count = 0
    inv_fishes = []
    qbs = {}
    
    for spec in specs:
        qbs[spec] = []

    for filename in files:
	bp = xf.load_and_parse(filename)
        inv_fish = bp['invfish_nosampvar']
	inv_fishes.append(inv_fish)	
        bad = np.where(np.diag(inv_fish) < 0)[0]
        if len(bad):
            print("Found negative fisher values in {}: {}".format(filename, bad))
            continue

        for spec in specs:
            qbs[spec].append(bp['qb']['cmb_{}'.format(spec)])
 
        count += 1
        del bp

    out['inv_fish'] =  np.mean(inv_fishes, axis = 0)
    out['qb'] = OrderedDict()
    out['qbvar'] = OrderedDict()
    out['gcorr'] = OrderedDict()

    for spec in specs:
	stag = 'cmb_{}'.format(spec)
	out['qb'][stag] = np.median(np.asarray(qbs[spec]), axis = 0)
	out['qbvar'][stag] = np.var(np.asarray(qbs[spec]), axis = 0)

    cmb_bins = len(xf.dict_to_arr(out['qbvar'], flatten=True))
    out['gcorr'] = xf.arr_to_dict(np.diag(out['inv_fish'])[:cmb_bins] / xf.dict_to_arr(out['qbvar'], flatten=True), out['qb'])
     
    outfile = os.path.join(output_root, 'gcorr{}.npz'.format(output_tag))
    np.savez_compressed(outfile, **out)
    
    print("Computed new gcorr_correction from the 500 simulations.")
    print(out['gcorr'])


if __name__ == "__main__":
    import argparse as ap
    P = ap.ArgumentParser()
    P.add_argument('output_tag')
    P.add_argument('-r', '--root', default='xfaster_gcal_unconstr')
    args = P.parse_args()

    xfaster_run_ensemble(os.path.join('/path/to/output/', args.root), args.output_tag) #Set your own output_root
