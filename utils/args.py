import argparse
from src.attacks.geoda import AsymmetricGeoDA
from src.attacks.hsja import AsymmetricHSJA
from src.attacks.cgba import AsymmetricCGBA
from src.attacks.surfree import AsymmetricSurfree
from src.attacks.opt import AsymmetricOPT


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--attack',
        type=str,
        default="HSJA"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default="data",
        help="Path to image data and labels"
    )
    parser.add_argument(
        '--image-prefix',
        type=str,
        default="ILSVRC2012_val_",
        help="Prefix of images name"
    )
    parser.add_argument(
        '--image-start',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--image-end',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--model',
        type=str,
        default="Resnet50",
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu'
    )
    parser.add_argument(
        '--total-cost',
        type=float,
        default=1000.0
    )
    parser.add_argument(
        '--query-cost',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--search-cost',
        type=float,
        default=None
    )
    parser.add_argument(
        '--max-iteration',
        type=int,
        default=None
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.0001
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.02
    )
    parser.add_argument(
        '--radius-increase',
        type=float,
        default=1.1
    )
    parser.add_argument(
        '--initial-gradient-queries',
        type=int,
        default=100
    )
    parser.add_argument(
        '--sample-batch-size',
        type=int,
        default=128
    )
    parser.add_argument(
        '--overshooting',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        '--smoothing',
        type=float,
        default=1e-6
    )
    parser.add_argument(
        '--count-init-cost',
        default=True, 
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--dimension-reduction-factor',
        type=float,
        default=4.0
    )
    parser.add_argument(
        '--dimension-reduction-mode',
        type=str,
        default="Full"
    )
    parser.add_argument(
        '--directions-buffer-size',
        type=int,
        default=10
    )
    parser.add_argument(
        '--gradient-samples',
        type=int,
        default=10
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--init-mode',
        type=str,
        default="Random"
    )
    parser.add_argument(
        '--angle-tolerance',
        type=float,
        default=0.005
    )
    parser.add_argument(
        '--save-logs',
        default=True,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--overshooting-scheduler-init',
        type=float,
        default=0.02
    )
    parser.add_argument(
        '--save-trajectories',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Save intermediate perturbed images for trajectory visualization'
    )
    return parser.parse_args()


def get_attack(args, model):
    if args.attack == "HSJA":
        return AsymmetricHSJA(model=model, total_cost=args.total_cost, query_cost=args.query_cost,
                    search_cost=args.search_cost, max_iteration=args.max_iteration, device=args.device, tolerance=args.tolerance, count_init_cost=args.count_init_cost, init_mode=args.init_mode,
                    sigma=args.sigma, initial_gradient_queries=args.initial_gradient_queries, sample_batch_size=args.sample_batch_size, 
                    overshooting=args.overshooting, smoothing=args.smoothing, overshooting_scheduler_init=args.overshooting_scheduler_init, use_gradient_moment=False,
                    save_trajectories=args.save_trajectories)
    elif args.attack == "GEODA":
        return AsymmetricGeoDA(model=model, total_cost=args.total_cost, query_cost=args.query_cost,
                    search_cost=args.search_cost, max_iteration=args.max_iteration, device=args.device, tolerance=args.tolerance, count_init_cost=args.count_init_cost, init_mode=args.init_mode,
                    sigma=args.sigma, initial_gradient_queries=args.initial_gradient_queries, sample_batch_size=args.sample_batch_size, radius_increase=args.radius_increase,
                    overshooting=args.overshooting, smoothing=args.smoothing, overshooting_scheduler_init=args.overshooting_scheduler_init, dimension_reduction_factor=args.dimension_reduction_factor,
                    dimension_reduction_mode=args.dimension_reduction_mode, use_gradient_moment=True, save_trajectories=args.save_trajectories)
    elif args.attack == "CGBA":
        return AsymmetricCGBA(model=model, total_cost=args.total_cost, query_cost=args.query_cost,
                    search_cost=args.search_cost, max_iteration=args.max_iteration, device=args.device, tolerance=args.tolerance, count_init_cost=args.count_init_cost, init_mode=args.init_mode,
                    sigma=args.sigma, initial_gradient_queries=args.initial_gradient_queries, sample_batch_size=args.sample_batch_size, radius_increase=args.radius_increase,
                    overshooting=args.overshooting, smoothing=args.smoothing, overshooting_scheduler_init=args.overshooting_scheduler_init, dimension_reduction_factor=args.dimension_reduction_factor,
                    dimension_reduction_mode=args.dimension_reduction_mode, use_gradient_moment=False, save_trajectories=args.save_trajectories)
    elif args.attack == "SURFREE":
        return AsymmetricSurfree(model=model, total_cost=args.total_cost, query_cost=args.query_cost,
                    search_cost=args.search_cost, max_iteration=args.max_iteration, device=args.device, tolerance=args.tolerance, count_init_cost=args.count_init_cost, init_mode=args.init_mode,
                    directions_buffer_size=args.directions_buffer_size, smoothing=args.smoothing, dimension_reduction_factor=args.dimension_reduction_factor,
                    dimension_reduction_mode=args.dimension_reduction_mode, angle_tolerance=args.angle_tolerance, save_trajectories=args.save_trajectories)
    elif args.attack == "OPT":
        return AsymmetricOPT(model=model, total_cost=args.total_cost, query_cost=args.query_cost,
                             search_cost=args.search_cost, max_iteration=args.max_iteration, device=args.device, tolerance=args.tolerance, count_init_cost=args.count_init_cost, init_mode=args.init_mode,
                             sigma=args.sigma, radius_increase=args.radius_increase, gradient_samples=args.gradient_samples, smoothing=args.smoothing, save_trajectories=args.save_trajectories)
    else:
        raise ValueError


