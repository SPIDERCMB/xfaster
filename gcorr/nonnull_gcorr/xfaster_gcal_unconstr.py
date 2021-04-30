'''
A XFaster submission script to generate ensemble-mean and simulation runs for gcorr calculation.
'''
import os
import xfaster as xf
import argparse as ap

P = ap.ArgumentParser()
P.add_argument('-f', '--first', default=0, type=int)
P.add_argument('-n', '--num', default=1, type=int)
P.add_argument('-o', '--output', default='xfaster_gcal_unconstr')
P.add_argument('-m', '--ensemble-mean', action='store_true')
P.add_argument('--no-gcorr', dest='gcorr', default=True, action='store_false')
P.add_argument('--no-submit', dest='submit', action='store_false')

output_root = '../../example/gcorr_run/' #Set your own output root.

args = P.parse_args()
seeds = list(range(args.first, args.first + args.num))
if args.ensemble_mean:
    seeds = [seeds[0]]

tags = ['95','150']
for tag in tags:
    opts = dict(
        data_root='../../example/maps_example/',
        data_subset='full/map_{}*'.format(tag),
        output_root=os.path.join(output_root, args.output),
        output_tag=tag,
        tbeb=True,
        bin_width=25,
        lmin=8, #33
        lmax=407,
        iter_max=200,

        mask_type="rectangle", #Example mask type
        noise_type=None,
        signal_type='synfast', #Example signal type
        data_type='raw', #Eample data type
        config = '../../example/config_example.ini',

        likelihood=False,
        verbose='debug',
        residual_fit=False,
        foreground_fit=False,
        checkpoint='bandpowers',
        apply_gcorr=args.gcorr,
        ensemble_mean=args.ensemble_mean,
        like_profiles=args.ensemble_mean,
        null_first_cmb=False,
        reload_gcorr=True if args.ensemble_mean else False
    )

    submit_opts = dict(
        nodes=1,
        ppn=1,
        mem=6,
        omp_threads=1, #Increase for runs that generates sim_xcorr files,
        slurm=True,
        wallt=4,
    )
    if args.submit:
        opts.update(**submit_opts)

    for s in seeds:
        fname = os.path.join(
            output_root, args.output, tag,
            'bandpowers_sim{:04d}_{}.npz'.format(s, tag)
        )
        if args.ensemble_mean or not os.path.exists(fname):
            
            if not args.ensemble_mean:
                opts.update(sim_index=s)
            if args.submit:
                xf.xfaster_submit(**opts)
            else:
                xf.xfaster_run(**opts)
