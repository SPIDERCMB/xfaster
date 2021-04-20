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

args = P.parse_args()

seeds = list(range(args.first, args.first + args.num))
if args.ensemble_mean:
    seeds = [seeds[0]]

for tag in ['100']:#['90', '150']:
    opts = dict(
        data_root='/data/agambrel/XF_NonNull_Sept18',
        output_root=os.path.join('/data/agambrel/spectra', args.output),
        output_tag=tag,
        tbeb=True,
        bin_width=25,
        lmin=8,#33,
        lmax=407,
        iter_max=200,
        mask_type="pointsource_latlon",
        # signal_subset='0[0-1]*',
        # noise_subset='0[0-1]*',
        noise_type=None,
        signal_type='flatBBDl_unconstr',
        clean_type='raw',
        data_subset='full/map_{}*'.format(tag),
        likelihood=False,
        verbose='detail',
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
        omp_threads=1, #if (not args.ensemble_mean and len(seeds) > 1) else 18,
        slurm=True,
        wallt=4,
    )
    if args.submit:
        opts.update(**submit_opts)

    for s in seeds:
        fname = os.path.join(
            '/data/agambrel/spectra', args.output, tag,
            'bandpowers_sim{:04d}_{}.npz'.format(s, tag)
        )
        if args.ensemble_mean or not os.path.exists(fname):
        #if 1:
            if not args.ensemble_mean:
                opts.update(sim_index=s)
            if args.submit:
                xf.xfaster_submit(**opts)
            else:
                xf.xfaster_run(**opts)
