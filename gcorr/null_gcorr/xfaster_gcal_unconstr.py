import os
import xfaster as xf
import argparse as ap
P = ap.ArgumentParser()
P.add_argument('-f', '--first', default=0, type=int)
P.add_argument('-n', '--num', default=1, type=int)
P.add_argument('-o', '--output', default='xfaster_gcal_unconstr')
P.add_argument('--no-gcorr', dest='gcorr', default=True, action='store_false')
P.add_argument('--no-submit', dest='submit', action='store_false')

args = P.parse_args()

seeds = list(range(args.first, args.first + args.num))
output_root = os.path.join('/path/to/output/', args.output) #Set your own output_root.

if len(seeds) == 1:
    reload_g = True
    print("Reloaded gcorr file")
else:
    reload_g = False

for tag in ['90','150a']:
    opts = dict(
        data_root='/mnt/spider2/XF_Nulls_Sept18/port', #Example nullmap roots
        data_root2='/mnt/spider2/XF_Nulls_Sept18/starboard',
        output_root=output_root,
        output_tag=tag,
        tbeb=True,
        bin_width=25,
        lmin=8,#33,
        lmax=407,
        iter_max=200,
	weighted_bins = True,
        mask_type="pointsource_latlon", #Change your mask_type accordingly 
        noise_type="stationary",
        signal_type='flatBBDl',
        clean_type='raw',
        data_subset='full/map_{}*'.format(tag),
        likelihood=False,
        verbose='detail',
        residual_fit=True,
        qb_file = None, #Set qb_file if there is one.
        foreground_fit=False,
        checkpoint='bandpowers',
        apply_gcorr=args.gcorr,
        null_first_cmb=False,
        reload_gcorr=reload_g
    )

    submit_opts = dict(
	queue = 'spider',
        ppn=1,
        mem=6,
        omp_threads=1, #increase for the first single sim index run.
        slurm=True,
        wallt=4,
    )
    if args.submit:
        opts.update(**submit_opts)

    for s in seeds:
        fname = os.path.join(
            output_root, tag,
            'bandpowers_sim{:04}_wbins_{}.npz'.format(s, tag)
        )
        if not os.path.exists(fname):
            opts.update(sim_index=s)
            if args.submit:
                xf.xfaster_submit(**opts)
            else:
                xf.xfaster_run(**opts)
