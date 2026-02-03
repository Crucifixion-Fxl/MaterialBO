"""
测试回归器函数（随机森林和梯度提升）

这个脚本测试 analysis/compare_predictions.py 中的回归器函数
支持有机物（organic）和氧化物（oxide）两种类型
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'analysis'))

from analysis.compare_predictions import (
    train_random_forest_and_predict,
    train_gradient_boosting_and_predict
)


def find_latest_experiment_dir(opt_type, objective_version='paper'):
    """查找最新的实验目录"""
    base_dir = project_root / "output" / opt_type / objective_version
    if not base_dir.exists():
        return None
    
    experiment_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not experiment_dirs:
        return None
    
    # 按修改时间排序，返回最新的
    experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return experiment_dirs[0]


def load_data_and_setup(opt_type='organic', objective_version='paper'):
    """加载数据并设置优化器和测试集（共用函数）
    
    Returns:
        params, true_values, param_cols, obj_cols, test_params, test_true_values
        如果加载失败则返回 None
    """
    import torch
    from analysis.compare_predictions import load_organic_results, load_oxide_results
    from example_usage import create_organic_param_space, create_oxide_param_space
    from src.optimizer import OrganicOptimizer, OxideOptimizer
    from analysis.compare_predictions import find_maximum_objective_values, generate_test_set
    
    # 根据类型选择加载函数和参数空间创建函数
    if opt_type == 'organic':
        load_results = load_organic_results
        create_param_space = create_organic_param_space
        OptimizerClass = OrganicOptimizer
    elif opt_type == 'oxide':
        load_results = load_oxide_results
        create_param_space = create_oxide_param_space
        OptimizerClass = OxideOptimizer
    else:
        raise ValueError(f"不支持的 opt_type: {opt_type}，应该是 'organic' 或 'oxide'")
    
    # 查找最新的实验目录
    # experiment_dir = find_latest_experiment_dir(opt_type, objective_version)
    experiment_dir = Path('/home/fxl/MetalBayes/output/organic/paper/20260126-231809')
    if experiment_dir is None:
        print(f"⚠ 警告: 未找到 {opt_type}/{objective_version} 的实验目录")
        return None
    
    csv_path = experiment_dir / "experiment.csv"
    if not csv_path.exists():
        print(f"⚠ 警告: CSV文件不存在: {csv_path}")
        return None
    
    print(f"加载CSV文件: {csv_path}")
    
    # 加载数据
    params, true_values, param_cols, obj_cols = load_results(csv_path)
    print(f"输入参数形状: {params.shape}, 真实值形状: {true_values.shape}")
    print(f"参数列: {param_cols}, 目标列: {obj_cols}")
    
    # 创建参数空间和优化器（用于测试集）
    param_space = create_param_space()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = OptimizerClass(param_space, device=torch.device(device), 
                               objective_version=objective_version, noise_level=0.0)
    
    # 获取所有有效参数（用于生成测试集）
    print("\n查找所有有效参数组合（用于生成测试集）...")
    _, _, _, _, all_valid_params = find_maximum_objective_values(
        param_space, opt_type=opt_type, objective_version=objective_version, device=device
    )
    
    # 生成测试集
    print("\n生成测试集...")
    test_params = generate_test_set(
        params, param_space, opt_type=opt_type, n_test=100, 
        device=device, seed=123, available_params=all_valid_params
    )
    print(f"生成了 {len(test_params)} 个测试样本")
    
    # 计算测试集的真实值
    test_params_tensor = torch.tensor(test_params, dtype=torch.float32, device=device)
    test_true_values = optimizer.evaluate_objectives(test_params_tensor).cpu().numpy()
    
    return params, true_values, param_cols, obj_cols, test_params, test_true_values


def test_model(model_name, train_func, opt_type, objective_version, model_kwargs):
    """测试指定的回归器模型
    
    Args:
        model_name: 模型名称（用于显示）
        train_func: 训练函数
        opt_type: 'organic' 或 'oxide'
        objective_version: 目标函数版本
        model_kwargs: 传递给训练函数的参数字典
    """
    print("=" * 60)
    print(f"{model_name} - {opt_type.capitalize()} ({objective_version})")
    print("=" * 60)
    
    # 加载数据和设置测试集
    result = load_data_and_setup(opt_type, objective_version)
    if result is None:
        return False
    
    params, true_values, param_cols, obj_cols, test_params, test_true_values = result
    
    # 测试函数
    train_r2, test_r2 = train_func(
        params, 
        true_values,
        test_params=test_params,
        test_true_values=test_true_values,
        **model_kwargs
    )
    
    print(f"\n训练集R² 分数: {train_r2}")
    print(f"训练集每个目标的R²: {dict(zip(obj_cols, train_r2))}")
    
    if test_r2 is not None:
        print(f"\n测试集R² 分数: {test_r2}")
        print(f"测试集每个目标的R²: {dict(zip(obj_cols, test_r2))}")
    
    # 验证返回值的类型
    assert isinstance(train_r2, np.ndarray), f"训练集R² 应该是数组，但得到 {type(train_r2)}"
    assert train_r2.shape == (3,), f"训练集R² 形状应该是 (3,)，但得到 {train_r2.shape}"
    assert np.all((train_r2 >= 0) & (train_r2 <= 1)), f"训练集R² 应该在 [0, 1] 范围内，但得到 {train_r2}"
    
    if test_r2 is not None:
        assert isinstance(test_r2, np.ndarray), f"测试集R² 应该是数组，但得到 {type(test_r2)}"
        assert test_r2.shape == (3,), f"测试集R² 形状应该是 (3,)，但得到 {test_r2.shape}"
        assert np.all((test_r2 >= 0) & (test_r2 <= 1)), f"测试集R² 应该在 [0, 1] 范围内，但得到 {test_r2}"
    
    print(f"✓ {model_name} - {opt_type} 测试通过")
    print()
    return True


def main():
    """运行所有回归器测试"""
    print("\n" + "=" * 60)
    print("回归器函数测试（随机森林和梯度提升）")
    print("=" * 60 + "\n")
    
    test_results = {}
    objective_version = 'paper'
    
    # 定义测试配置
    test_configs = [
        # 随机森林
        {
            'model_name': '随机森林',
            'train_func': train_random_forest_and_predict,
            'model_kwargs': {'n_estimators': 300, 'max_depth': 30, 'random_state': 0}
        },
        # 梯度提升
        {
            'model_name': '梯度提升',
            'train_func': train_gradient_boosting_and_predict,
            'model_kwargs': {
                'n_estimators': 300,
                'max_depth': 4,
                'min_samples_split': 5,
                'learning_rate': 0.01,
                'loss': 'squared_error',
                'random_state': 0
            }
        }
    ]
    
    # 对每种模型和每种类型进行测试
    for config in test_configs:
        model_name = config['model_name']
        train_func = config['train_func']
        model_kwargs = config['model_kwargs']
        
        for opt_type in ['organic', 'oxide']:
            test_key = f"{model_name}_{opt_type}"
            try:
                success = test_model(model_name, train_func, opt_type, objective_version, model_kwargs)
                test_results[test_key] = success
            except Exception as e:
                print(f"\n❌ {test_key} 测试失败: {e}")
                import traceback
                traceback.print_exc()
                test_results[test_key] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for test_key, success in test_results.items():
        status = "✓ 通过" if success else "❌ 失败"
        print(f"  {test_key}: {status}")
    
    if all(test_results.values()):
        print("\n所有测试通过！✓")
        return 0
    else:
        print("\n部分测试失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
