import os
import argparse
import deft_segmentor
import custom_datasets
from deft import add_deft_args

from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(
        description='DeFT evaluation with MMSeg')
    parser.add_argument('--config', default='')
    parser.add_argument('--work-dir', default='./work_logs/')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show_dir',
        default='',
        help='directory to save visualizaion images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser = add_deft_args(parser)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir
    cfg.model.update(dict(
        deft=args.deft,
        deft_knn=args.deft_knn,
        deft_density_threshold=args.deft_density_threshold,
        deft_density_temperature=args.deft_density_temperature,
        deft_trace_strength=args.deft_trace_strength,
        deft_graph_temperature=args.deft_graph_temperature,
        deft_local_weight=args.deft_local_weight,
        deft_graph_weight=args.deft_graph_weight,
        deft_multiscale_strength=args.deft_multiscale_strength,
        deft_confidence_strength=args.deft_confidence_strength,
        deft_return_maps=args.deft_return_maps,
        deft_save_maps=args.deft_save_maps,
    ))
    # if you want to save segmentation maps, open this line
    # trigger_visualization_hook(cfg, args)
    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    main()
