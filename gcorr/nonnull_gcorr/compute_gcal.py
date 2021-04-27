import os
import glob
import numpy as np
from collections import OrderedDict
from matplotlib import use
use('agg')
import matplotlib.pyplot as plt
import scipy.optimize as opt
import xfaster as xf

specs = ['tt', 'ee', 'bb', 'te', 'eb', 'tb']

# P == Plotting?
P = False

def gauss(qb, amp, width, offset):
    return amp * np.exp(-width*(qb - offset)**2)

def lognorm(qb, amp, width, offset):
    return gauss(np.log(qb), amp, width, offset)

def xfaster_run_ensemble(output_root=None, output_tag=None):
    if P: 
        plot_root = os.path.join(output_root, 'plots')
    	if not os.path.exists(plot_root):
            os.mkdir(plot_root)

    if output_root is None:
        output_root = os.getcwd()
    if output_tag is not None:
        output_root = os.path.join(output_root, output_tag)
        output_tag = '_{}'.format(output_tag)
    else:
        output_tag = ''

    mean_file = os.path.join(
        output_root, 'bandpowers_mean{}.npz'.format(output_tag))
    bp_mean = xf.load_and_parse(mean_file)
    file_glob = os.path.join(
        output_root, 'bandpowers_sim[0-9][0-9][0-9][0-9]{}.npz'.format(output_tag)
    )
    files = sorted(glob.glob(file_glob))
    if not len(files):
        raise OSError("No bandpowers files found in {}".format(output_root))

    out = {'data_version': 1, 'bin_def': bp_mean['bin_def'],
           'inv_fish': bp_mean['inv_fish'], 'qb_like': bp_mean['qb_like']}
    count = 0
    inv_fishes = []
    qbs = {}
    
    if P:
        plt.plot([0,250], [0.2, 0.2], 'k-')
    	plt.errorbar(bp_mean['ellb']['cmb_bb'], bp_mean['cb']['cmb_bb'],
                 bp_mean['dcb']['cmb_bb'], label='ensemble_mean')
    	plt.legend()
    	plt.ylim(-0, 0.4)
    	plt.title('{} GHz BB'.format(output_tag[1:]))
    	plt.savefig(os.path.join(plot_root, 'bb_spec{}.png'.format(output_tag)))
    	plt.close()

    qb_err = xf.arr_to_dict(np.sqrt(np.diag(bp_mean['inv_fish'])),
                            bp_mean['bin_def'])
    qb_err_fix = xf.arr_to_dict(
        1. / np.sqrt(np.diag(np.linalg.inv(bp_mean['inv_fish']))),
        bp_mean['bin_def'])

    for spec in specs:
        qbs[spec] = []
    for filename in files:
        bp = xf.load_and_parse(filename)
        inv_fish = bp['inv_fish']

        bad = np.where(np.diag(inv_fish) < 0)[0]
        if len(bad):
            print("Found negative fisher values in {}: {}".format(filename, bad))
            continue

        for spec in specs:
            qbs[spec].append(bp['qb']['cmb_{}'.format(spec)])

        count += 1
        del bp

    out['qb'] = OrderedDict()
    out['qbvar'] = OrderedDict()
    out['gcorr'] = OrderedDict()

    if P:
    	for spec in specs:
            fig, ax = plt.subplots(4, 4, figsize=(15,15), sharex=False, sharey=False)
            fig.suptitle('{} {}'.format(output_tag[1:], spec.upper()))
            axs = ax.flatten()
            # old gauss method
            stag = 'cmb_{}'.format(spec)
            out['qb'][stag] = np.median(np.asarray(qbs[spec]), axis=0)
            out['qbvar'][stag] = np.var(np.asarray(qbs[spec]), axis=0)

            # lognormal fits
            like = bp_mean['qb_like'][stag]
            out['gcorr'][spec] = np.ones(like.shape[0])
            for b0, likeli in enumerate(like):
                if b0 == 0:
                # doesn't work with null_first_cmb
                    continue
                nanmask = ~np.isnan(likeli[1])
                if np.any(np.isnan(likeli[1])):
                    print('nans in {} {} {}'.format(
                        spec, b0, np.where(np.isnan(likeli[1]))))
                likeli = [likeli[0][nanmask], likeli[1][nanmask]]
                likeli[1] = np.exp(likeli[1]-np.max(likeli[1]))
                L = likeli[1] / np.trapz(likeli[1], likeli[0])

                hbins = np.arange(likeli[0][0], likeli[0][-1],
                              (likeli[0][-1] - likeli[0][0]) / 30.)
                hist, bins = np.histogram(np.asarray(qbs[spec])[:, b0],
                                      density=True, bins=hbins)
                bc = (bins[:-1] + bins[1:]) / 2.

                axs[b0-1].set_title(b0)
                axs[b0-1].hist(np.asarray(qbs[spec])[:,b0], bins=hbins,
                           density=True, facecolor='0.7', edgecolor='0.7')
                axs[b0-1].plot(likeli[0][L > 1e-4 * L.max()], L[L > 1e-4 * L.max()], 'k-',
                           label='Likelihood')
                axs[b0-1].set_yscale('log')
                if spec in ['eb', 'tb']:
                    width = qb_err[stag][b0]
                    width_nomarg = qb_err_fix[stag][b0]
                    qb0 = bp_mean['qb'][stag][b0]
                    func = gauss
                    func_name = 'gauss'
                    bc0 = np.min(bc)
                else:
                    width = qb_err[stag][b0] / bp_mean['qb'][stag][b0]
                    width_nomarg = qb_err_fix[stag][b0] / bp_mean['qb'][stag][b0]
                    qb0 = np.log(bp_mean['qb'][stag][b0])
                    func = lognorm
                    func_name = 'lognorm'
                    bc0 = 0.5

                p0 = [np.nanmax(L), 1. / width**2 / 2., qb0]
                try:
                    popth, pcovh = opt.curve_fit(func, bc[bc>bc0], hist[bc>bc0],
                                             p0=p0, maxfev=int(1e9))
                    p0fm = [popth[0], 1. / width**2 / 2., popth[2]]
                    like_marg = func(bc, *p0fm)
                    axs[b0-1].plot(bc[bc>bc0], func(bc[bc>bc0], *popth), label='hist fit')
                    axs[b0-1].plot(bc[bc>bc0], like_marg[bc>bc0], linestyle='dashed',
                               label='Fisher-derived {} (marg)'.format(func_name))
                    out['gcorr'][spec][b0] = popth[1] / p0fm[1]
                except RuntimeError:
                    print('No hist fits found')
                try:
                    poptl, pcovl = opt.curve_fit(func, likeli[0][likeli[0]>bc0],
                                             L[likeli[0]>bc0], p0=p0,
                                             maxfev=int(1e9))
                    p0_nomarg = [poptl[0], 1. / width_nomarg**2 / 2., poptl[2]]
                    like_nomarg = func(likeli[0], *p0_nomarg)
                    axs[b0-1].plot(likeli[0][likeli[0]>bc0],
                               func(likeli[0][likeli[0]>bc0], *poptl),
                               label='Lognorm fit to likelihood')
                    axs[b0-1].plot(likeli[0][likeli[0]>bc0], like_nomarg[likeli[0]>bc0],
                               linestyle='dashed',
                               label='Fisher-derived {} (no marg)'.format(func_name))
                except RuntimeError:
                    print('No likeli fits found')

            axs[15].legend(loc='lower center')
            fig.savefig(os.path.join(plot_root, 'hists{}_{}.png'.format(output_tag, spec)))
            plt.close(fig)
    cmb_bins = len(xf.dict_to_arr(out['qbvar'], flatten=True))

    out['gcorr_gauss'] = xf.arr_to_dict(
        np.diag(out['inv_fish'])[:cmb_bins] / xf.dict_to_arr(out['qbvar'], flatten=True),
        out['gcorr'])
    outfile = os.path.join(
        output_root, 'gcorr{}.npz'.format(output_tag)
    )
    np.savez_compressed(outfile, **out)

    if P:
        fig, ax = plt.subplots(2, 3)
        ax = ax.flatten()
        for i, spec in enumerate(specs):
            ellb = bp_mean['ellb']['cmb_{}'.format(spec)]
            ax[i].plot(ellb[1:], out['gcorr_gauss'][spec][1:], label='gauss')
            ax[i].plot(ellb[1:], out['gcorr'][spec][1:], label='log-normal')
            ax[i].set_title(spec)
        ax[5].legend(loc='lower right', prop={'size': 8})
        plt.savefig(os.path.join(plot_root, 'g{}.png'.format(output_tag)))
        plt.close()
    print(out['gcorr'])

if __name__ == "__main__":

    import argparse as ap
    P = ap.ArgumentParser()
    P.add_argument('output_tag')
    P.add_argument('-r', '--root', default='xfaster_gcal_unconstr')
    args = P.parse_args()

    xfaster_run_ensemble(os.path.join('/path/to/output/', args.root), args.output_tag) #Set your own output path
