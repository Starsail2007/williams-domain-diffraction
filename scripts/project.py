#!/usr/bin/env python3
"""Run the independent source checkout without installing it."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('serve', help='Open the offline interactive research guide.')
    p.add_argument('--port', type=int, default=0)
    p.add_argument('--no-browser', action='store_true')
    p.add_argument('--session-file', type=Path)
    p.add_argument('--headless-console', action='store_true', help=argparse.SUPPRESS)
    p = sub.add_parser('console', help=argparse.SUPPRESS)
    p.add_argument('--job', type=Path, required=True)
    p = sub.add_parser('batch-experiments', help='Run selected algorithms with their own samples.')
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('experiment', help='Run one algorithm on one selected sample, independently.')
    p.add_argument('--algorithm', required=True)
    p.add_argument('--sample', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--image', type=Path)
    p.add_argument('--pixel-size-um', type=float)
    p = sub.add_parser('guided', help='Compute a case with real stage snapshots.')
    p.add_argument('--case', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--control', type=Path)
    p.add_argument('--image', type=Path)
    p.add_argument('--pixel-size-um', type=float)
    p = sub.add_parser('research', help='Recompute the four technical sections.')
    p.add_argument('--chapter', choices=('all','measurement','periods','intensity','discussion'), default='all')
    p.add_argument('--out', type=Path, required=True)
    for name in ('rg','photometry'):
        p = sub.add_parser(name, help='Run the RG fit or native CFA photometry independently.')
        p.add_argument('--out', type=Path, required=True)
    for name in ('reproduce', 'examples'):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, required=True)
        if name == 'reproduce':
            p.add_argument('--full', action='store_true', help='Refit all 304 spectrum cases, including aperture sensitivity.')
    sub.add_parser('verify', help='Verify all immutable bundle file hashes, using standard library only.')
    sub.add_parser('test', help='Run bundle tests.')
    p = sub.add_parser('paper', help='Reproduce by current paper figure/table number.')
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument('--figure', type=int, choices=range(1,18))
    target.add_argument('--table', type=int, choices=(1,2))
    target.add_argument('--all', action='store_true')
    p.add_argument('--refit', action='store_true', help='Actually optimize figures 15/16 rather than replay stored parameters.')
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('algorithms', help='Process representative real diffraction and microscopy images.')
    p.add_argument('--kind', choices=('all','diffraction','microscopy'), default='all')
    p.add_argument('--case', default='all', help='Real case ID, e.g. 104 or micro196_right; default all.')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--without-reference', action='store_true', help='Skip the post-computation frozen-coordinate regression.')
    p.add_argument('--image', type=Path, help='Analyze a supplied 8-bit microscopy image.')
    p.add_argument('--pixel-size-um', type=float, help='Calibrated micrometres/pixel for supplied image; square sampling.')
    args = parser.parse_args()
    if args.command == 'serve':
        from williams_diffraction.lab_server import serve
        serve(ROOT,args.port,args.no_browser,args.session_file,args.headless_console)
    elif args.command == 'console':
        from williams_diffraction.terminal_console import run_console
        run_console(ROOT,args.job)
    elif args.command == 'batch-experiments':
        from williams_diffraction.batch_lab import run_batch
        from williams_diffraction.delivery_io import read_json
        result=run_batch(ROOT,args.out,read_json(args.tasks))
        if result['status']!='complete':raise SystemExit(1)
    elif args.command == 'experiment':
        from williams_diffraction.independent_lab import run_experiment
        run_experiment(ROOT,args.out,args.algorithm,args.sample,args.image,args.pixel_size_um)
    elif args.command == 'guided':
        from williams_diffraction.guided_lab import run_guided
        run_guided(ROOT,args.out,args.case,args.control,args.image,args.pixel_size_um)
    elif args.command == 'research':
        from williams_diffraction.research_route import run_research
        run_research(ROOT,args.out,args.chapter)
    elif args.command in ('rg','photometry'):
        from williams_diffraction.research_route import run_component
        run_component(ROOT,args.out,args.command)
    elif args.command == 'verify':
        from williams_diffraction.delivery_io import verify_bundle
        verify_bundle(ROOT)
    elif args.command == 'reproduce':
        from williams_diffraction.delivery_models import reproduce
        reproduce(ROOT, args.out, args.full)
    elif args.command == 'examples':
        from williams_diffraction.delivery_examples import run_examples
        run_examples(ROOT, args.out)
    elif args.command == 'paper':
        from williams_diffraction.delivery_paper import run_paper
        run_paper(ROOT, args.out, args.figure, args.table, args.all, args.refit)
    elif args.command == 'algorithms':
        if args.image:
            from williams_diffraction.research_route import run_own_image
            run_own_image(ROOT,args.out,args.image,args.pixel_size_um)
        else:
            from williams_diffraction.delivery_algorithms import run_algorithms
            run_algorithms(ROOT,args.out,args.kind,args.case,not args.without_reference)
    else:
        import unittest
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / 'tests')))
        raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] in {'batch-experiments','experiment','guided','research','rg','photometry','reproduce','examples','paper','algorithms'}:
        from williams_diffraction.runtime_trace import console_values
        with console_values():main()
    else:
        main()
