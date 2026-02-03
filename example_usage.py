"""
Three-objective Bayesian optimization usage example
Demonstrates how to use organic and oxide optimizers
"""

import torch
import logging
import os
from datetime import datetime
from src.optimizer import OrganicOptimizer, OxideOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_organic_param_space():
    """Create organic parameter space configuration"""
    param_space = {
        'parameters': [
            'organic_formula',
            'organic_concentration',
            'organic_temperature',
            'organic_soak_time',
            'organic_ph',
            'organic_curing_time'
        ],
        'bounds': torch.tensor([
            [1.0, 0.1, 25.0, 10, 0, 10],
            [30.0, 5, 40.0, 300, 14, 60]
        ], dtype=torch.float32),
        'steps': torch.tensor([1.0, 0.1, 5.0, 10.0, 0.5, 5], dtype=torch.float32),
        'constraints': {
            'pH_safety_constraints': {
                1: (4.0, 6.0),
                2: (4.0, 6.0),
                3: (4.0, 6.0),
                4: (4.0, 6.0),
                5: (7.0, 10.5),
                6: (7.0, 10.5),
                7: (7.0, 10.5),
                8: (3.5, 6.5),
                9: (3.5, 6.5),
                10: (4.5, 6.0),
                11: (2.0, 5.0),
                12: (4.0, 5.5),
                13: (4.0, 6.0),
                14: (3.5, 4.5),
                15: (6.0, 8.0),
                16: (7.0, 10.0),
                17: (5.0, 7.0),
                18: (2.0, 7.0),
                19: (3.0, 7.0),
                20: (7.0, 11.0),
                21: (3.0, 7.0),
                22: (4.0, 6.0),
                23: (8.0, 10.0),
                24: (4.0, 6.0),
                25: (4.0, 6.0),
                26: (9.0, 11.0),
                27: (8.0, 10.0),
                28: (6.0, 8.0),
                29: (4.0, 6.0),
                30: (7.0, 9.0),
            }
        }
    }
    return param_space


def create_oxide_param_space():
    """Create oxide parameter space configuration"""
    param_space = {
        'parameters': [
            'metal_a_type',
            'metal_a_concentration',
            'metal_b_type',
            'metal_molar_ratio_b_a'
        ],
        'bounds': torch.tensor([
            [1.0, 10.0, 0.0, 0.0],
            [20.0, 50.0, 20.0, 10.0]
        ], dtype=torch.float32),
        'steps': torch.tensor([1.0, 10.0, 1.0, 1.0], dtype=torch.float32),
        'constraints': {
            'oxide_constraints': {}
        }
    }
    return param_space


def run_organic_optimization(objective_version: str = 'complex', n_iter: int = 200, 
                             noise_level: float = 0.0):
    """
    Run organic optimization
    
    Args:
        objective_version: 'complex', 'simple', 'standard', or 'paper' - which version of objective functions to use
        n_iter: Number of optimization iterations (default: 200)
        noise_level: Standard deviation of Gaussian noise to add (default: 0.0, set to > 0.0 to enable)
    """
    logger.info("=" * 60)
    logger.info(f"Starting organic three-objective Bayesian optimization (version: {objective_version})")
    logger.info(f"Number of iterations: {n_iter}")
    logger.info(f"Noise level: {noise_level} ({'enabled' if noise_level > 0 else 'disabled'})")
    logger.info("=" * 60)
    
    experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Include noise level in folder name
    noise_str = f"noise{noise_level:.2f}".replace('.', 'p')
    output_dir = f"./output/organic/{objective_version}/{experiment_time}_{noise_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    param_space = create_organic_param_space()
    
    optimizer = OrganicOptimizer(
        param_space=param_space,
        output_dir=output_dir,
        seed=42,
        device=None,
        objective_version=objective_version,
        noise_level=noise_level
    )
    
    optimizer.optimize(n_iter=n_iter, simulation_flag=True)
    
    pareto_x, pareto_y = optimizer.get_pareto_front()
    logger.info(f"Pareto front contains {pareto_x.shape[0]} solutions")
    logger.info(f"Final hypervolume: {optimizer._compute_hypervolume():.6f}")
    
    return optimizer


def run_oxide_optimization(objective_version: str = 'complex', n_iter: int = 200,
                          noise_level: float = 0.0):
    """
    Run oxide optimization
    
    Args:
        objective_version: 'complex', 'simple', 'standard', or 'paper' - which version of objective functions to use
        n_iter: Number of optimization iterations (default: 200)
        noise_level: Standard deviation of Gaussian noise to add (default: 0.0, set to > 0.0 to enable)
    """
    logger.info("=" * 60)
    logger.info(f"Starting oxide three-objective Bayesian optimization (version: {objective_version})")
    logger.info(f"Number of iterations: {n_iter}")
    logger.info(f"Noise level: {noise_level} ({'enabled' if noise_level > 0 else 'disabled'})")
    logger.info("=" * 60)
    
    experiment_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Include noise level in folder name
    noise_str = f"noise{noise_level:.2f}".replace('.', 'p')
    output_dir = f"./output/oxide/{objective_version}/{experiment_time}_{noise_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    param_space = create_oxide_param_space()
    
    optimizer = OxideOptimizer(
        param_space=param_space,
        output_dir=output_dir,
        seed=42,
        device=None,
        objective_version=objective_version,
        noise_level=noise_level
    )
    
    optimizer.optimize(n_iter=n_iter, simulation_flag=True)
    
    pareto_x, pareto_y = optimizer.get_pareto_front()
    logger.info(f"Pareto front contains {pareto_x.shape[0]} solutions")
    logger.info(f"Final hypervolume: {optimizer._compute_hypervolume():.6f}")
    
    return optimizer


def main():
    """Main function with interactive selection"""
    logger.info("Three-objective Bayesian optimization example")
    logger.info("Objectives: Adhesion, Coverage, Uniformity")
    logger.info("=" * 60)
    
    # Select optimization type
    print("\n请选择优化类型 (Please select optimization type):")
    print("1. 有机物优化 (Organic optimization)")
    print("2. 氧化物优化 (Oxide optimization)")
    print("3. 全部 (Both)")
    
    while True:
        try:
            choice = input("\n请输入选项 (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                break
            else:
                print("无效选项，请输入 1、2 或 3 (Invalid choice, please enter 1, 2, or 3)")
        except (EOFError, KeyboardInterrupt):
            print("\n\n程序已取消 (Program cancelled)")
            return
    
    # Select objective version
    print("\n请选择目标函数版本 (Please select objective function version):")
    print("1. simple (简单版本)")
    print("2. complex (复杂版本)")
    print("3. standard (标准版本 - Botorch DTLZ2)")
    print("4. paper (论文版本 - 多项式函数)")
    
    while True:
        try:
            version_choice = input("\n请输入选项 (1/2/3/4，默认为 simple): ").strip()
            if version_choice == '' or version_choice == '1':
                objective_version = 'simple'
                break
            elif version_choice == '2':
                objective_version = 'complex'
                break
            elif version_choice == '3':
                objective_version = 'standard'
                break
            elif version_choice == '4':
                objective_version = 'paper'
                break
            else:
                print("无效选项，请输入 1、2、3 或 4 (Invalid choice, please enter 1, 2, 3, or 4)")
        except (EOFError, KeyboardInterrupt):
            print("\n\n程序已取消 (Program cancelled)")
            return
    
    # Select number of iterations
    print("\n请选择迭代次数 (Please select number of iterations):")
    print("  默认值 (Default): 200")
    
    while True:
        try:
            iter_input = input("\n请输入迭代次数 (Enter number of iterations, default: 200): ").strip()
            if iter_input == '':
                n_iter = 200
                break
            else:
                n_iter = int(iter_input)
                if n_iter > 0:
                    break
                else:
                    print("无效输入，请输入大于 0 的整数 (Invalid input, please enter a positive integer)")
        except ValueError:
            print("无效输入，请输入整数 (Invalid input, please enter an integer)")
        except (EOFError, KeyboardInterrupt):
            print("\n\n程序已取消 (Program cancelled)")
            return
    
    # Select noise level (for all versions)
    print("\n请选择噪声级别 (Please select noise level):")
    print("  0.0 - 无噪声 (No noise) [默认/Default]")
    print("  0.05 - 默认噪声级别 (Default noise level)")
    print("  0.1 - 较高噪声级别 (Higher noise level)")
    print("  0.15 - 高噪声级别 (High noise level)")
    
    while True:
        try:
            noise_input = input("\n请输入噪声级别 (Enter noise level, default: 0.0): ").strip()
            if noise_input == '':
                noise_level = 0.0
                break
            else:
                noise_level = float(noise_input)
                if noise_level >= 0:
                    break
                else:
                    print("无效输入，请输入大于等于 0 的数字 (Invalid input, please enter a non-negative number)")
        except ValueError:
            print("无效输入，请输入数字 (Invalid input, please enter a number)")
        except (EOFError, KeyboardInterrupt):
            print("\n\n程序已取消 (Program cancelled)")
            return
    
    logger.info(f"Selected optimization type: {choice}, objective version: {objective_version}, iterations: {n_iter}")
    logger.info(f"Noise level: {noise_level}")
    logger.info("=" * 60)
    
    # Run optimization based on selection
    results = {}
    
    if choice == '1' or choice == '3':
        # Run organic optimization
        try:
            organic_optimizer = run_organic_optimization(
                objective_version=objective_version, 
                n_iter=n_iter,
                noise_level=noise_level
            )
            results['organic'] = organic_optimizer
        except Exception as e:
            logger.error(f"有机物优化失败 (Organic optimization failed): {e}", exc_info=True)
    
    if choice == '2' or choice == '3':
        # Run oxide optimization
        try:
            oxide_optimizer = run_oxide_optimization(
                objective_version=objective_version, 
                n_iter=n_iter,
                noise_level=noise_level
            )
            results['oxide'] = oxide_optimizer
        except Exception as e:
            logger.error(f"氧化物优化失败 (Oxide optimization failed): {e}", exc_info=True)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Optimization completed!")
    logger.info("=" * 60)
    
    if 'organic' in results:
        logger.info(f"有机物优化结果保存在 (Organic optimization results saved in): {results['organic'].output_dir}")
    
    if 'oxide' in results:
        logger.info(f"氧化物优化结果保存在 (Oxide optimization results saved in): {results['oxide'].output_dir}")


if __name__ == "__main__":
    main()
