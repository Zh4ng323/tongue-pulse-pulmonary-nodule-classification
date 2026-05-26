# -*- coding: utf-8 -*-
"""
SHAP-based interpretability analysis module.

Generates out-of-fold SHAP values across 10-fold CV and produces
beeswarm plots, decision plots, grouped feature importance,
and per-sample force plots.
"""

# =============================================================================
# 模块导入
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import List, Tuple, Dict, Optional
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# SHAP相关
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP未安装，部分功能将不可用")

# LIME相关（可选）
try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("[WARNING] LIME未安装，面板E将不可用")

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# =============================================================================
# 第1部分：OOF数据准备
# =============================================================================

def prepare_oof_shap_data(modeler) -> Dict:
    """
    准备Out-of-Fold（OOF）SHAP数据

    关键特性：
    - 基于10-fold交叉验证的测试折数据
    - 只使用consistent features（10/10折均出现）
    - 每个样本只用其测试折模型解释一次
    - base_value来自对应折的TreeExplainer.expected_value
    - 特征列按全局顺序对齐（修复列对齐bug）

    Parameters:
    -----------
    modeler : IntegratedModelingV2实例
        必须已运行run_cross_validation

    Returns:
    --------
    dict: 包含以下键的字典
        - oof_prob: array (n_samples,), 正类(malignant)概率
        - y_true: array (n_samples,), 真实标签
        - sample_id: array (n_samples,), 原始样本索引
        - X_oof: array (n_samples, n_consistent_features), OOF特征值（列对齐）
        - shap_values_oof: array (n_samples, n_consistent_features), OOF的SHAP值（列对齐）
        - base_values_oof: array (n_samples,), 每个样本对应折的base value
        - feature_names: list, consistent features名称（全局顺序）
        - fold_indices: array (n_samples,), 每个样本所属的折索引（0-9）
    """
    print("\n" + "="*70)
    print("[准备OOF SHAP数据]")
    print("="*70)

    # 检查XGBoost是否运行
    if 'xgboost' not in modeler.results:
        raise ValueError("XGBoost模型未运行，无法准备OOF SHAP数据")

    # 检查必需的缓存数据
    required_keys = ['fold_shap_values', 'fold_X_tests', 'fold_selected_features',
                     'fold_y_true', 'fold_probs', 'fold_base_values', 'fold_sample_ids']

    for key in required_keys:
        if key not in modeler.results['xgboost']:
            raise ValueError(f"缺少必需的缓存数据: {key}")

    # ============================================================
    # 步骤1：识别consistent features（10/10折均出现）
    # ============================================================
    fold_features = modeler.results['xgboost']['fold_selected_features']
    feature_appear_count = {}

    for features in fold_features:
        for feat in features:
            feature_appear_count[feat] = feature_appear_count.get(feat, 0) + 1

    consistent_features = [feat for feat, count in feature_appear_count.items()
                          if count == 10]

    if not consistent_features:
        raise ValueError("没有找到consistent features（10/10），无法生成OOF SHAP数据")

    print(f"  [OK] Consistent features: {len(consistent_features)}个")
    print(f"  [INFO] 这些特征在所有10折中均被LASSO选中")

    # ============================================================
    # 步骤2：建立全局特征顺序（固定顺序，避免列对齐问题）
    # ============================================================
    # ⚠️ 关键修复：使用 sorted() 固定顺序，确保跨运行稳定
    consistent_features_order = sorted(consistent_features)  # 字母排序，固定顺序

    # 建立全局列索引映射
    global_col_index = {feat: j for j, feat in enumerate(consistent_features_order)}

    print(f"  [OK] 全局特征顺序已建立（{len(consistent_features_order)}个特征）")
    print(f"  [INFO] 特征顺序：字母排序（跨运行稳定）")

    # ============================================================
    # 步骤3：初始化OOF数组（按原始样本索引对齐）
    # ============================================================
    n_total_samples = len(modeler.y)
    n_consistent_features = len(consistent_features_order)

    # 初始化为NaN（后续逐折填充）
    oof_prob = np.full(n_total_samples, np.nan)
    y_true_ordered = np.full(n_total_samples, np.nan)
    X_oof = np.full((n_total_samples, n_consistent_features), np.nan)
    shap_values_oof = np.full((n_total_samples, n_consistent_features), np.nan)
    base_values_oof = np.full(n_total_samples, np.nan)
    fold_indices = np.full(n_total_samples, -1)

    # ============================================================
    # 步骤4：逐折提取consistent features的SHAP数据（关键：列对齐）
    # ============================================================
    print(f"\n  [INFO] 逐折提取consistent features的SHAP值（按全局列顺序对齐）...")

    for fold_idx in range(10):
        # 获取该折的数据
        fold_features_list = fold_features[fold_idx]
        fold_shap = modeler.results['xgboost']['fold_shap_values'][fold_idx]
        fold_X = modeler.results['xgboost']['fold_X_tests'][fold_idx]
        fold_y_true = modeler.results['xgboost']['fold_y_true'][fold_idx]
        fold_probs = modeler.results['xgboost']['fold_probs'][fold_idx]
        fold_base_value = modeler.results['xgboost']['fold_base_values'][fold_idx]
        fold_sample_ids = modeler.results['xgboost']['fold_sample_ids'][fold_idx]

        # ⚠️ 关键修复：建立该折的特征索引映射
        fold_feat_index = {feat: k for k, feat in enumerate(fold_features_list)}

        # 按全局列顺序填充OOF数组
        for sample_pos, sample_id in enumerate(fold_sample_ids):
            # 填充预测概率和标签
            oof_prob[sample_id] = fold_probs[sample_pos]
            y_true_ordered[sample_id] = fold_y_true[sample_pos]
            base_values_oof[sample_id] = fold_base_value
            fold_indices[sample_id] = fold_idx

            # ⚠️ 关键修复：按全局列顺序对齐特征和SHAP值
            for feat in consistent_features_order:
                global_col = global_col_index[feat]
                if feat in fold_feat_index:
                    fold_col = fold_feat_index[feat]
                    shap_values_oof[sample_id, global_col] = fold_shap[sample_pos, fold_col]
                    X_oof[sample_id, global_col] = fold_X[sample_pos, fold_col]
                else:
                    # 特征在该折不存在（理论上不应该发生，因为已经筛选了consistent features）
                    print(f"  [WARNING] Fold {fold_idx}: 特征 '{feat}' 不在该折的特征列表中")

    # ============================================================
    # 步骤5：验证列对齐完整性
    # ============================================================
    n_nan_shap = np.isnan(shap_values_oof).sum()
    n_nan_X = np.isnan(X_oof).sum()

    if n_nan_shap > 0 or n_nan_X > 0:
        raise ValueError(
            f"OOF数据列对齐失败！\n"
            f"  SHAP值缺失: {n_nan_shap} 个\n"
            f"  特征值缺失: {n_nan_X} 个\n"
            f"  这表明特征列对齐有问题，请检查代码！"
        )

    # 验证数据完整性
    n_valid = np.sum(~np.isnan(oof_prob))
    if n_valid != n_total_samples:
        raise ValueError(f"OOF数据不完整：{n_valid}/{n_total_samples}个样本有效")

    # 验证base_value是否合理（不应全为0或全为NaN）
    if np.all(np.isnan(base_values_oof)) or np.all(base_values_oof == 0):
        raise ValueError("Base values异常：全为NaN或全为0，请检查TreeExplainer是否正确初始化")

    print(f"  [OK] OOF数据准备完成")
    print(f"  - 总样本数: {n_total_samples}")
    print(f"  - Consistent features: {n_consistent_features}")
    print(f"  - Base value范围: [{np.nanmin(base_values_oof):.4f}, {np.nanmax(base_values_oof):.4f}]")
    print(f"  - Base value均值: {np.nanmean(base_values_oof):.4f}")
    print(f"   列对齐验证通过：无缺失值")

    # ============================================================
    # 步骤6：返回OOF数据字典
    # ============================================================
    return {
        'oof_prob': oof_prob,
        'y_true': y_true_ordered.astype(int),
        'sample_id': np.arange(n_total_samples),
        'X_oof': X_oof,
        'shap_values_oof': shap_values_oof,
        'base_values_oof': base_values_oof,
        'feature_names': consistent_features_order,  # 使用全局顺序
        'fold_indices': fold_indices,
        'shap_output_space': 'log-odds'  # 明确标注SHAP输出空间
    }


# =============================================================================
# 第2部分：样本选择函数
# =============================================================================

def select_benign_case(oof_data: Dict, p_min: float = 0.10, p_max: float = 0.15) -> int:
    """
    选择对抗性良性病例（Benign case）用于Force Strip

    选择标准（改进版 - 显示对抗性）：
    1. y_true = 0（真实为良性）
    2. 预测概率在 [p_min, p_max] 范围内（模型有些不确定）
    3. 在该范围内选择预测概率最高的样本（最接近决策边界）
    4. 如果没有样本在此范围内，逐步扩展范围
    5. 并列时选sample_id最小的

    Parameters:
    -----------
    oof_data : dict
        OOF数据字典
    p_min : float
        预测概率下限（默认0.10）
    p_max : float
        预测概率上限（默认0.15）

    Returns:
    --------
    int: 选中的样本索引
    """
    y_true = oof_data['y_true']
    oof_prob = oof_data['oof_prob']
    sample_ids = oof_data['sample_id']

    # 只考虑y_true=0的样本
    benign_indices = sample_ids[y_true == 0]

    if len(benign_indices) == 0:
        raise ValueError("没有良性样本")

    # 在良性样本中，找预测概率在 [p_min, p_max] 范围内的
    benign_probs = oof_prob[benign_indices]

    # 筛选在目标范围内的样本
    target_indices = benign_indices[(benign_probs >= p_min) & (benign_probs <= p_max)]

    if len(target_indices) > 0:
        # 在目标范围内选预测概率最高的（最接近决策边界，显示对抗性）
        selected_idx = target_indices[np.argmax(oof_prob[target_indices])]
        desc = f"p∈[{p_min:.2f}, {p_max:.2f}]范围"
    else:
        # 如果没有，逐步扩展范围（降级策略）
        print(f"  [WARNING] 没有p∈[{p_min:.2f}, {p_max:.2f}]范围的良性样本，扩展搜索范围...")

        # 尝试扩展到 [0.05, 0.20]
        extended_indices = benign_indices[(benign_probs >= 0.05) & (benign_probs <= 0.20)]
        if len(extended_indices) > 0:
            selected_idx = extended_indices[np.argmax(oof_prob[extended_indices])]
            desc = f"p∈[0.05, 0.20]扩展范围"
        else:
            # 最终降级：选预测概率最低的样本
            selected_idx = benign_indices[np.argmin(benign_probs)]
            desc = f"最低概率: p={oof_prob[selected_idx]:.3f}"

    print(f"  [INFO] [对抗性良性病例] True: Benign, Pred: {oof_prob[selected_idx]:.3f} ({desc})")

    return selected_idx


def select_high_risk_malignant_case(oof_data: Dict, p_min: float = 0.85, p_max: float = 0.90) -> int:
    """
    选择对抗性恶性病例（High-risk malignant case）用于Force Strip

    选择标准（改进版 - 显示对抗性）：
    1. y_true = 1（真实为癌症）
    2. 预测概率在 [p_min, p_max] 范围内（模型有些不确定）
    3. 在该范围内选择预测概率最低的样本（最接近决策边界）
    4. 如果没有样本在此范围内，逐步扩展范围
    5. 并列时选sample_id最小的

    Parameters:
    -----------
    oof_data : dict
        OOF数据字典
    p_min : float
        预测概率下限（默认0.85）
    p_max : float
        预测概率上限（默认0.90）

    Returns:
    --------
    int: 选中的样本索引
    """
    y_true = oof_data['y_true']
    oof_prob = oof_data['oof_prob']
    sample_ids = oof_data['sample_id']

    # 只考虑y_true=1的样本
    malignant_indices = sample_ids[y_true == 1]

    if len(malignant_indices) == 0:
        raise ValueError("没有癌症样本")

    # 在恶性样本中，找预测概率在 [p_min, p_max] 范围内的
    malignant_probs = oof_prob[malignant_indices]

    # 筛选在目标范围内的样本
    target_indices = malignant_indices[(malignant_probs >= p_min) & (malignant_probs <= p_max)]

    if len(target_indices) > 0:
        # 在目标范围内选预测概率最低的（最接近决策边界，显示对抗性）
        selected_idx = target_indices[np.argmin(oof_prob[target_indices])]
        desc = f"p∈[{p_min:.2f}, {p_max:.2f}]范围"
    else:
        # 如果没有，逐步扩展范围（降级策略）
        print(f"  [WARNING] 没有p∈[{p_min:.2f}, {p_max:.2f}]范围的恶性样本，扩展搜索范围...")

        # 尝试扩展到 [0.80, 0.95]
        extended_indices = malignant_indices[(malignant_probs >= 0.80) & (malignant_probs <= 0.95)]
        if len(extended_indices) > 0:
            selected_idx = extended_indices[np.argmin(oof_prob[extended_indices])]
            desc = f"p∈[0.80, 0.95]扩展范围"
        else:
            # 最终降级：选预测概率最高的样本
            selected_idx = malignant_indices[np.argmax(malignant_probs)]
            desc = f"最高概率: p={oof_prob[selected_idx]:.3f}"

    print(f"  [INFO] [对抗性恶性病例] True: Malignant, Pred: {oof_prob[selected_idx]:.3f} ({desc})")

    return selected_idx


# =============================================================================
# 第3部分：面板A - SHAP Summary Beeswarm Plot
# =============================================================================

def plot_panel_a_summary_beeswarm(ax, oof_data: Dict):
    """
    面板A：SHAP summary beeswarm plot（全局重要性）

    参数：
    - 使用consistent features（10/10）
    - 排序按mean(|SHAP|)
    - 图注明确：正类=malignant；SHAP输出空间=log-odds(margin)
    """
    shap_values = oof_data['shap_values_oof']
    X = oof_data['X_oof']
    feature_names = oof_data['feature_names']

    # 计算特征重要性排序
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_indices = np.argsort(mean_abs_shap)

    # 绘制beeswarm plot（兼容不同SHAP版本）
    try:
        # 尝试新版本API（支持ax参数）
        shap.summary_plot(
            shap_values[:, sorted_indices],
            X[:, sorted_indices],
            feature_names=[feature_names[i] for i in sorted_indices],
            plot_size=None,
            show=False,
            ax=ax
        )
    except TypeError:
        # 旧版本API（不支持ax参数），使用matplotlib捕获
        plt.sca(ax)
        shap.summary_plot(
            shap_values[:, sorted_indices],
            X[:, sorted_indices],
            feature_names=[feature_names[i] for i in sorted_indices],
            plot_size=None,
            show=False
        )
        plt.sca(ax)

    # 放大字体（标题、轴标签、刻度标签）
    ax.set_title('Global Feature Importance (SHAP Summary)',
                 fontsize=20, fontweight='bold', pad=10)
    ax.tick_params(labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)

    print(f"  [OK] 面板A已完成：SHAP summary beeswarm plot")


# =============================================================================
# 第4部分：面板B - Decision Plot
# =============================================================================

def plot_panel_b_decision_plot(ax, oof_data: Dict,
                                benign_idx: int, malignant_idx: int,
                                n_samples: int = 400):
    """
    面板B：Decision plot（模型决策路径可视化）

    参数：
    - 随机抽300-500个样本（固定seed）
    - 高亮两个病例（benign_idx和malignant_idx）
    - 用颜色区分y_true（benign vs malignant）
    - 图注解释是否存在明显离群决策路径
    - 使用 common_base + base_shift 特征统一不同折的 base value
    """
    shap_values = oof_data['shap_values_oof']
    base_values = oof_data['base_values_oof']
    y_true = oof_data['y_true']
    feature_names = oof_data['feature_names']

    rng = np.random.RandomState(42)

    # 随机抽样（确保包含两个高亮样本）
    all_indices = np.arange(len(shap_values))

    # 确保高亮样本被选中
    remaining_indices = np.setdiff1d(all_indices, [benign_idx, malignant_idx])
    sampled_indices = rng.choice(remaining_indices,
                                size=min(n_samples - 2, len(remaining_indices)),
                                replace=False)

    plot_indices = np.concatenate([[benign_idx, malignant_idx], sampled_indices])

    plot_shap_values = shap_values[plot_indices]
    plot_base_values = base_values[plot_indices]
    plot_y_true = y_true[plot_indices]

    # 计算特征重要性排序（用于decision plot）- 降序（重要特征在前）
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    # ⚠️ 关键修复：统一 common_base 并将 base_shift 作为特征
    common_base = float(np.mean(plot_base_values))
    base_shift = (plot_base_values - common_base).reshape(-1, 1)

    # 应用特征排序到 SHAP 值
    plot_shap_values_sorted = plot_shap_values[:, sorted_indices]
    feature_names_sorted = [feature_names[i] for i in sorted_indices]

    # 将 base_shift 作为第一个特征
    plot_shap_aug = np.concatenate([base_shift, plot_shap_values_sorted], axis=1)
    feature_names_aug = ["base_shift"] + feature_names_sorted

    # 绘制decision plot（部分SHAP版本不支持 ax 参数，使用 plt.sca）
    try:
        # 高亮样本的索引（在plot_indices中的位置）
        highlight_idx_in_plot = np.where(plot_indices == benign_idx)[0][0]
        highlight_idx_in_plot2 = np.where(plot_indices == malignant_idx)[0][0]

        # 设置当前axes（兼容不支持ax参数的SHAP版本）
        plt.sca(ax)

        shap.decision_plot(
            common_base,  #  使用统一的 common_base
            plot_shap_aug,
            feature_names=feature_names_aug,
            highlight=[highlight_idx_in_plot, highlight_idx_in_plot2],
            show=False
            #  移除 plot_width=12 - 使用 figsize 控制宽度
        )

    except Exception as e:
        #  打印完整的错误堆栈用于调试
        import traceback
        print(f"  [ERROR] Decision plot失败:")
        print(f"  Exception: {str(e)}")
        print(f"  Traceback:")
        traceback.print_exc()

        # 简化版：绘制累积SHAP值的分布
        cumulative_shap = plot_shap_aug.cumsum(axis=1) + common_base

        for i, idx in enumerate(plot_indices):
            color = 'red' if idx in [benign_idx, malignant_idx] else ('blue' if plot_y_true[i] == 1 else 'gray')
            alpha = 1.0 if idx in [benign_idx, malignant_idx] else 0.3
            ax.plot(cumulative_shap[i], color=color, alpha=alpha, linewidth=1.5 if idx in [benign_idx, malignant_idx] else 0.5)

        ax.set_xlabel('Feature Index (sorted by importance)', fontsize=10)
        ax.set_ylabel('Model Output (log-odds)', fontsize=10)

    #  添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Highlighted (F/G)'),
        Line2D([0], [0], color='blue', lw=1.5, alpha=0.3, label='Malignant'),
        Line2D([0], [0], color='gray', lw=1.5, alpha=0.3, label='Benign')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    ax.set_title('Model Decision Paths',
                fontsize=18, fontweight='bold', pad=10)

    print(f"  [OK] 面板B已完成：Decision plot ({len(plot_indices)}样本)")


# =============================================================================
# 第5部分：面板C - 分组特征重要性
# =============================================================================

def plot_panel_c_grouped_importance(ax, oof_data: Dict, top_k: int = 20):
    """
    面板C：分组特征重要性（Benign vs Malignant）

    参数：
    - 对topK特征，分别在benign子集、malignant子集计算mean(|SHAP|)
    - 画并排条形图（benign vs malignant）
    - 显示"哪些特征主要驱动阳性/阴性"
    """
    shap_values = oof_data['shap_values_oof']
    y_true = oof_data['y_true']
    feature_names = oof_data['feature_names']

    # 计算全局mean(|SHAP|)排序
    mean_abs_shap_global = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap_global)[-top_k:][::-1]

    # 分别计算benign和malignant子集的mean(|SHAP|)
    benign_mask = y_true == 0
    malignant_mask = y_true == 1

    mean_abs_shap_benign = np.mean(np.abs(shap_values[benign_mask]), axis=0)
    mean_abs_shap_malignant = np.mean(np.abs(shap_values[malignant_mask]), axis=0)

    top_features = [feature_names[i] for i in top_indices]
    benign_values = mean_abs_shap_benign[top_indices]
    malignant_values = mean_abs_shap_malignant[top_indices]

    # 绘制并排条形图
    y_pos = np.arange(len(top_features))
    height = 0.40

    bars1 = ax.barh(y_pos - height/2, benign_values, height,
                   label='Benign', color='#3498db', alpha=0.8)
    bars2 = ax.barh(y_pos + height/2, malignant_values, height,
                   label='Malignant', color='#e74c3c', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=16)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP|', fontsize=18, fontweight='bold')
    ax.set_title('Grouped Feature Importance (Benign vs Malignant)',
                fontsize=20, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)

    print(f"  [OK] 面板C已完成：分组特征重要性（Top {top_k}）")


# =============================================================================
# 第7部分：面板E - LIME解释（可选）
# =============================================================================

def plot_panel_e_lime(ax, oof_data: Dict, malignant_idx: int,
                     X_train_full: np.ndarray, feature_names_full: List[str]):
    """
    面板E：单病例LIME解释（高风险癌例）

    此函数已禁用。正确的LIME实现需要获取对应fold模型的predict_proba
    以及对应折的预处理/特征顺序，在"每折特征集合不同"的设定下较为复杂。

    当前版本使用OOF SHAP进行可解释性分析（Panels A-C, F-H）
    """
    # 输出提示信息
    print(f"  [SKIP] 面板E：LIME已禁用（使用OOF SHAP代替）")

    # 在图上显示禁用信息
    ax.text(0.5, 0.5,
            'LIME Panel Disabled\n\n'
            'Reason:\n'
            'Correct LIME implementation requires:\n'
            '1. Fold-specific model access (predict_proba)\n'
            '2. Fold-specific feature handling\n'
            '3. Proper data preprocessing\n\n'
            'Current version uses OOF SHAP for interpretability\n'
            '(Panels A-C, F-H)',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    ax.set_title('LIME Explanation (Disabled)',
                fontsize=18, fontweight='bold', pad=10)
    ax.axis('off')


# =============================================================================
# 第8部分：面板F和G - Force Strip
# =============================================================================

def plot_waterfall_or_force(ax, oof_data: Dict, sample_idx: int,
                           panel_label: str, title_text: str):
    """
    绘制单个样本的waterfall plot

    使用 shap.plots.waterfall 展示单个样本的 SHAP 值分解
    """
    shap_values = oof_data['shap_values_oof'][sample_idx]
    X = oof_data['X_oof'][sample_idx]
    base_value = oof_data['base_values_oof'][sample_idx]
    feature_names = oof_data['feature_names']
    y_true = oof_data['y_true'][sample_idx]
    oof_prob = oof_data['oof_prob'][sample_idx]

    # 创建Explanation对象
    expl = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=X,
        feature_names=feature_names
    )

    # 设置当前axes
    plt.sca(ax)

    # 绘制waterfall plot
    shap.plots.waterfall(expl, show=False, max_display=15)

    # 添加样本信息标注
    label_text = "True: Malignant" if y_true == 1 else "True: Benign"
    pred_text = f"Pred: {oof_prob:.3f}"
    base_text = f"Base: {base_value:.3f}"
    final_value = base_value + shap_values.sum()
    final_text = f"Final: {final_value:.3f}"

    info_text = f"{label_text}\n{pred_text}\n{base_text}\n{final_text}"

    # 添加到当前axes
    ax.text(0.95, 0.95, info_text,
           transform=ax.transAxes, ha='right', va='top',
           fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(f'{panel_label} {title_text}',
                fontsize=11, fontweight='bold', pad=10)

    print(f"  [OK] {panel_label}已完成：样本{sample_idx}")


def plot_force_strip(ax, sample_idx: int, oof_data: Dict,
                     panel_label: str, title_text: str,
                     top_k: int = 15):
    """
    绘制单样本的 force strip（横向贡献条）

    这是SHAP force plot的静态版本，显示为扁平的横向条带

    参数：
    - sample_idx: 样本索引
    - oof_data: OOF数据字典
    - panel_label: 面板标签 (F/G)
    - title_text: 标题文本
    - top_k: 显示前K个特征（其余合并为other）

    修复说明：
    -  保持 log-odds 空间（不再转换为概率）
    -  matplotlib 分支只显示 top-K 真实特征（不伪造 "other features" 值）
    -  matplotlib 分支不使用 link='logit'（避免 matplotlib 已知的 logit scale 渲染问题）
    -  HTML fallback 保留 link='logit'（不伪造 "other features" 值）
    -  所有数值统一保留 3 位小数（包括 SHAP 值和特征值）
    -  信息框新增 "Other SHAP sum" 显示其他特征的总贡献
    -  所有 HTML 输出到主文件夹，文件名说明需要截图
    -  [新增] 增加 figure 高度到 4 英寸，bottom margin 到 0.30
    -  [新增] 特征名格式：特征名=特征值 (SHAP=贡献值)，同时显示两者
    -  [新增] 信息放在 title 中避免裁剪，包括 Other SHAP sum
    """
    # 获取数据
    shap_values = oof_data['shap_values_oof'][sample_idx]
    X = oof_data['X_oof'][sample_idx]
    base_value = oof_data['base_values_oof'][sample_idx]
    feature_names = oof_data['feature_names']
    y_true = oof_data['y_true'][sample_idx]
    oof_prob = oof_data['oof_prob'][sample_idx]

    # Sigmoid函数：将log-odds转换为概率
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 计算 top-K 特征（按绝对SHAP值排序）
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-top_k:][::-1]

    # 分离 top-K 和其他特征
    top_shap = shap_values[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    top_X = X[top_indices]

    # 计算其他特征的累积贡献
    other_mask = np.ones(len(shap_values), dtype=bool)
    other_mask[top_indices] = False
    other_shap_sum = shap_values[other_mask].sum()

    #  所有数值统一保留 3 位小数（用于显示）
    top_shap_plot = np.round(top_shap.astype(float), 3)
    top_X_plot = np.round(top_X.astype(float), 3)  #  特征值也保留 3 位小数
    other_shap_sum_plot = np.round(other_shap_sum.astype(float), 3)

    #  构建包含特征值和 SHAP 值的特征名（格式：特征名=特征值，第二行 (SHAP=贡献值)）
    # ⚠️ 关键优化：换行处理，减少每个特征的宽度占用
    top_features_display = []
    for i, (fname, fval, shap_val) in enumerate(zip(top_features, top_X_plot, top_shap_plot)):
        sign = "+" if shap_val >= 0 else ""
        # 第一行：特征名=特征值
        # 第二行：(SHAP=贡献值)
        top_features_display.append(f"{fname}={fval:.3f}\n(SHAP={sign}{shap_val:.3f})")

    #  计算 log-odds 和概率
    base_prob = sigmoid(base_value)
    final_logit = base_value + shap_values.sum()
    final_prob = sigmoid(final_logit)

    #  稳健性检查：确保所有值都是 finite（使用 rounded 版本检查）
    base_value_plot = np.round(base_value.astype(float), 3)
    if not np.isfinite(base_value):
        raise ValueError(f"[{panel_label}] base_value 不是 finite 值: {base_value_plot:.3f}")
    if not np.all(np.isfinite(top_shap)):
        raise ValueError(f"[{panel_label}] top_shap 包含非 finite 值")

    #  一致性检查（统一 3 位小数）
    if abs(oof_prob - final_prob) > 0.01:
        print(f"  [WARNING] 概率不一致: Pred={oof_prob:.3f}, Final={final_prob:.3f}")

    # 尝试使用 matplotlib force plot（⚠️ 关键修复：包含 other_shap_sum 作为特征）
    try:
        # 设置当前 axes
        plt.sca(ax)

        # ⚠️ 关键改进：将 other_shap_sum 作为最后一个特征加入（确保在图中可见）
        # 构建包含 other features 的显示数据
        display_shap_matplotlib = np.concatenate([top_shap_plot, [other_shap_sum_plot]])

        # ⚠️ other_sum 也使用换行格式，与其他特征保持一致
        sign_other = "+" if other_shap_sum_plot >= 0 else ""
        # 第一行：other_sum
        # 第二行：(SHAP=贡献值)
        other_feature_display = f"other_sum\n(SHAP={sign_other}{other_shap_sum_plot:.3f})"
        display_features_matplotlib = top_features_display + [other_feature_display]

        # ⚠️ 使用 shap.force_plot()
        shap.force_plot(
            base_value_plot,  # 保持 log-odds（3 位小数）
            display_shap_matplotlib,  # ⚠️ 包含 top-K + other_sum（共 K+1 个特征）
            feature_names=display_features_matplotlib,  # 特征名=值 (SHAP=贡献值)
            matplotlib=True,
            show=False
            # 不传 features 参数（特征值已编码在特征名中）
            # 不传 link='logit'（matplotlib 版本会导致渲染错误）
        )

        print(f"  [OK] {panel_label} Force Strip（matplotlib版本，显示 top-{top_k} + other_sum，换行格式）：样本{sample_idx}")

    except Exception as e:
        # 如果 matplotlib force plot 失败，生成 HTML 并提示用户
        print(f"  [INFO] matplotlib force plot 不可用: {str(e)}")
        print(f"  [INFO] {panel_label} 生成 HTML 版本...")

        # 生成 HTML 文件（使用主输出目录）
        output_dir = os.path.join(config.OUTPUT_BASE_DIR, 'interpretability_plots_v2')
        os.makedirs(output_dir, exist_ok=True)

        #  文件名说明需要截图
        html_filename = f'ForceStrip_{panel_label.replace("(", "").replace(")", "")}_Sample{sample_idx}_请截图后旋转90度.html'
        html_path = os.path.join(output_dir, html_filename)

        #  HTML 版本：构建包含 "other features" 的数据（但不伪造特征值）
        display_shap_html = np.concatenate([top_shap_plot, [other_shap_sum_plot]])

        #  HTML 特征名：使用格式化后的特征名 + "other features"
        sign_other = "+" if other_shap_sum_plot >= 0 else ""
        other_feature_display = f"other features (SHAP={sign_other}{other_shap_sum_plot:.3f})"
        display_features_html = top_features_display + [other_feature_display]

        #  生成 HTML（保留 link='logit'，不传 features 参数）
        force_obj = shap.force_plot(
            base_value_plot,  # 保持 log-odds（3 位小数）
            display_shap_html,  # 包含 "other features"
            feature_names=display_features_html,
            link='logit',  # HTML 版本可以安全使用 logit link
            show=False
            #  不传 features（特征值已编码在特征名中）
        )

        # 使用 shap.save_html() 保存（兼容新版 API）
        shap.save_html(html_path, force_obj)

        print(f"  [OK] {panel_label} HTML 已生成: {html_path}")
        print(f"  [INFO] 请在浏览器中打开 HTML 文件并截图")

        # 在 axes 上显示提示信息
        ax.clear()
        ax.text(0.5, 0.5,
                f'Force Strip HTML已生成\n\n'
                f'文件位置:\n{html_path}\n\n'
                f'下一步操作:\n'
                f'1. 在浏览器中打开 HTML\n'
                f'2. 截图保存为 PNG\n'
                f'3. 旋转 90° 得到最终图',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        ax.axis('off')

    # 添加样本信息标注（⚠️ 优化布局：使用文本框替代超长 title，避免重叠）
    label_text = "True: Malignant" if y_true == 1 else "True: Benign"

    # 方案1：简化的 title（只保留核心信息）
    ax.set_title(f'{title_text}\n{label_text} | Pred(p): {oof_prob:.3f}',
                fontsize=16, fontweight='bold', pad=20)

    # 方案2：详细信息放在左上角文本框（拉开纵向间距，避免重叠）
    info_text = (f'Base: logit={base_value_plot:.3f} (p={base_prob:.3f})\n'
                f'Final: logit={final_logit:.3f} (p={final_prob:.3f})\n'
                f'Other sum: {other_shap_sum_plot:.3f}')  # 简化标签

    # 使用 figure 级别的文本框（位置调整到左上角，避免与 force plot 重叠）
    fig = ax.figure
    fig.text(0.01, 0.96, info_text,
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7,
                     edgecolor='gray', linewidth=1))

    # ⚠️ 新增：保存 other_sum 信息到 figure 的属性中，供后续使用
    fig.other_sum_info = {
        'value': other_shap_sum_plot,
        'base_value': base_value_plot,
        'final_logit': final_logit,
        'final_prob': final_prob,
        'n_other_features': other_mask.sum()
    }


def plot_panel_f_benign_case(ax, oof_data: Dict, benign_idx: int):
    """
    面板F：对抗性良性病例 force strip（Benign case）

    改进说明：
    - 选择低概率样本（0.10-0.15范围）
    - 显示模型在高确信度下的特征贡献
    """
    plot_force_strip(
        ax, benign_idx, oof_data,
        panel_label='',
        title_text='Adversarial Benign Case (Low Pred Probability)',
        top_k=15
    )


def plot_panel_g_malignant_case(ax, oof_data: Dict, malignant_idx: int):
    """
    面板G：对抗性恶性病例 force strip（Malignant case）

    改进说明：
    - 选择高概率样本（0.85-0.90范围）
    - 显示模型在高确信度下的特征贡献
    """
    plot_force_strip(
        ax, malignant_idx, oof_data,
        panel_label='',
        title_text='Adversarial Malignant Case (High Pred Probability)',
        top_k=15
    )


# =============================================================================
# 第9部分：面板H - 全局叠加图
# =============================================================================

def create_global_force_html(oof_data: Dict, output_dir: str,
                              n_samples: int = 280,
                              n_features: int = 25):
    """
    生成 global force plot HTML（用户手动截图）

    参数：
    - oof_data: OOF数据
    - output_dir: 输出目录
    - n_samples: 下采样样本数（默认280）
    - n_features: 显示特征数（默认25）

    返回：
    - html_path: HTML文件路径

    修复说明：
    - 复用 decision plot 的 common_base + base_shift 设计
    - 保持 log-odds 空间（link='logit'）
    - 使用 shap.save_html() 保存（兼容新版 API）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    shap_values = oof_data['shap_values_oof']
    base_values = oof_data['base_values_oof']
    X = oof_data['X_oof']
    feature_names = oof_data['feature_names']

    # 计算最终输出 log-odds
    final_output = base_values + shap_values.sum(axis=1)

    # 按最终输出排序（低 → 高）
    sorted_indices = np.argsort(final_output)

    # 下采样
    if len(sorted_indices) > n_samples:
        step = len(sorted_indices) // n_samples
        sample_idx = sorted_indices[::step][:n_samples]
    else:
        sample_idx = sorted_indices

    print(f"  [INFO] Global Force: 样本按输出排序（fx范围: [{final_output[sample_idx[0]]:.3f}, {final_output[sample_idx[-1]]:.3f}]）")

    # 选择 top 特征
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[-n_features:][::-1]

    # 提取数据
    shap_values_sub = shap_values[sample_idx][:, top_indices]
    X_sub = X[sample_idx][:, top_indices]
    base_values_sub = base_values[sample_idx]
    feature_names_sub = [feature_names[i] for i in top_indices]

    #  关键修复：复用 decision plot 的 common_base + base_shift 设计
    common_base = float(np.mean(base_values_sub))
    base_shift = (base_values_sub - common_base).reshape(-1, 1)

    # 构建 force plot 所需数据
    shap_values_force = np.concatenate([base_shift, shap_values_sub], axis=1)
    X_force = np.concatenate([base_shift, X_sub], axis=1)
    feature_names_force = ['fold_base_shift'] + feature_names_sub

    #  校验：数学一致性检查
    # 原始输出 = base_values_sub + shap_values_sub.sum(axis=1)
    # 转换后输出 = common_base + shap_values_force.sum(axis=1)
    original_output = base_values_sub + shap_values_sub.sum(axis=1)
    transformed_output = common_base + shap_values_force.sum(axis=1)

    assert np.allclose(original_output, transformed_output, atol=1e-8), \
        f"[ERROR] Force plot 数据转换失败！数学不一致性：max diff = {np.max(np.abs(original_output - transformed_output))}"

    print(f"  [OK] 数学一致性检查通过：max diff = {np.max(np.abs(original_output - transformed_output)):.2e}")

    # 生成 HTML 文件（文件名说明需要截图）
    html_path = os.path.join(output_dir, 'Fig_H_GlobalForce_请截图后旋转90度.html')

    #  使用 scalar common_base 调用 force plot（保持 log-odds 空间）
    force_obj = shap.force_plot(
        common_base,  #  使用标量 base
        shap_values_force,
        features=X_force,
        feature_names=feature_names_force,
        link='logit',  #  明确使用 logit link（保持 log-odds 空间）
        show=False
    )

    #  使用 shap.save_html() 保存（兼容新版 API）
    shap.save_html(html_path, force_obj)

    print(f"  [OK] Global Force HTML已生成: {html_path}")
    print(f"  [INFO] 请在浏览器中打开 HTML 文件并截图")
    print(f"  [INFO] 截图后旋转 90° 得到最终H面板图")
    print(f"  [INFO] Base value（log-odds）: {common_base:.4f}")

    return html_path


def plot_panel_h_global_overlay(ax, oof_data: Dict, subsample: int = 280):
    """
    面板H：Global Force Plot（多样本叠加）

    使用 force plot 生成HTML（用户手动截图）
    """
    # 使用主输出目录（与 A-G 面板在同一文件夹）
    output_dir = os.path.join(config.OUTPUT_BASE_DIR, 'interpretability_plots_v2')

    # 生成 HTML 文件
    html_path = create_global_force_html(
        oof_data,
        output_dir=output_dir,
        n_samples=subsample,
        n_features=25
    )

    # 在 axes 上显示说明信息
    ax.text(0.5, 0.5,
            f'Global Force Plot HTML已生成\n\n'
            f'文件位置:\n{html_path}\n\n'
            f'下一步操作:\n'
            f'1. 在浏览器中打开 HTML\n'
            f'2. 截图保存为 PNG\n'
            f'3. 旋转 90° 得到最终图',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    ax.axis('off')
    ax.set_title('Global Force Plot (HTML Generated - See Instructions)',
                fontsize=18, fontweight='bold', pad=10)

    print(f"  [OK] 面板H已完成：Global Force HTML生成")


# =============================================================================
# 第10部分：主拼图函数
# =============================================================================

def generate_fig_shap_feature_analysis(modeler, output_dir: str,
                                      X_train_full: Optional[np.ndarray] = None,
                                      feature_names_full: Optional[List[str]] = None):
    """
    生成独立的SHAP特征分析图表（A-H面板，每个面板单独生成）

    ⚠️ 已弃用：此函数现在调用 export_single_panel_figures()
    保证 B 面板只有一套正确实现（带 mean(|SHAP|) 排序和高亮 F/G 病例）

    Parameters:
    -----------
    modeler : IntegratedModelingV2实例
        必须已运行run_cross_validation
    output_dir : str
        输出目录
    X_train_full : np.ndarray, optional
        完整训练集特征（已弃用，LIME 默认禁用）
    feature_names_full : list, optional
        完整特征名称列表（已弃用，LIME 默认禁用）

    生成文件列表：
    ---------------
    • Fig_A_SHAP_Summary_Beeswarm.{png,svg} - 面板A：全局特征重要性
    • Fig_B_Decision_Plot.{png,svg} - 面板B：模型决策路径（带排序和高亮）
    • Fig_C_Grouped_Feature_Importance.{png,svg} - 面板C：分组特征重要性
    • Fig_F_ForceStrip_BenignCase.{png,svg} - 面板F：典型阴性病例（Force Strip）
    • Fig_G_ForceStrip_HighRiskMalignant.{png,svg} - 面板G：高风险典型癌例（Force Strip）
    • Fig_H_GlobalForce_Placeholder.{png,svg} - 面板H：Global Force Plot（使用说明）
    """
    print("\n" + "="*70)
    print("[生成SHAP特征分析独立图表] (A-H面板，调用 export_single_panel_figures)")
    print("="*70)
    print("\n[INFO] 此函数现在调用 export_single_panel_figures()")
    print("[INFO] LIME 默认禁用（enable_lime=False）")
    print("[INFO] B 面板使用正确的 mean(|SHAP|) 排序和高亮 F/G 病例")

    # 调用新函数，LIME 默认禁用
    export_single_panel_figures(
        modeler=modeler,
        output_dir=output_dir,
        enable_lime=False,  # ⚠️ 默认禁用 LIME
        X_train_full=None,  # LIME 默认禁用，不需要这些参数
        feature_names_full=None
    )


# =============================================================================
# 第11部分：单图导出入口函数
# =============================================================================

def export_single_panel_figures(modeler, output_dir: str, enable_lime: bool = False,
                                X_train_full: Optional[np.ndarray] = None,
                                feature_names_full: Optional[List[str]] = None):
    """
    导出单个面板的独立图表（关键图）

    按顺序输出：
    A) SHAP summary beeswarm
    B) Decision plot（高亮F/G病例）
    C) Grouped importance（benign vs malignant）
    E) LIME（可选，默认禁用）
    F) Benign case Force Strip
    G) Malignant case Force Strip
    H) Global Force Plot

    Parameters:
    -----------
    modeler : IntegratedModelingV2实例
        必须已运行run_cross_validation
    output_dir : str
        输出目录
    enable_lime : bool
        是否生成LIME图（面板E），默认False
    X_train_full : np.ndarray, optional
        完整训练集特征（用于LIME）
    feature_names_full : list, optional
        完整特征名称列表（用于LIME）
    """
    print("\n" + "="*70)
    print("[导出单面板独立图表] (关键图 + 可选LIME)")
    print("="*70)

    # ============================================================
    # 步骤1：准备OOF数据
    # ============================================================
    oof_data = prepare_oof_shap_data(modeler)

    # 打印调试信息
    print(f"\n[调试信息]")
    print(f"  Consistent features order（前5个）: {oof_data['feature_names'][:5]}")
    print(f"  Total features: {len(oof_data['feature_names'])}")

    # ============================================================
    # 步骤2：选择代表性样本（F/G病例）
    # ============================================================
    benign_idx = select_benign_case(oof_data)
    malignant_idx = select_high_risk_malignant_case(oof_data)

    # 打印F/G病例信息（更新为对抗性样本）
    print(f"\n[F/G对抗性病例信息]")
    print(f"  [面板F - Adversarial Benign Case]")
    print(f"    sample_id: {benign_idx}")
    print(f"    fold_id: {oof_data['fold_indices'][benign_idx]}")
    print(f"    p: {oof_data['oof_prob'][benign_idx]:.4f} (目标范围: 0.10-0.15)")
    print(f"    y_true: {oof_data['y_true'][benign_idx]} (Benign)")
    print(f"    说明: 显示模型在高确信度但仍接近决策边界的对抗性样本")

    print(f"\n  [面板G - Adversarial Malignant Case]")
    print(f"    sample_id: {malignant_idx}")
    print(f"    fold_id: {oof_data['fold_indices'][malignant_idx]}")
    print(f"    p: {oof_data['oof_prob'][malignant_idx]:.4f} (目标范围: 0.85-0.90)")
    print(f"    y_true: {oof_data['y_true'][malignant_idx]} (Malignant)")
    print(f"    说明: 显示模型在高确信度但仍接近决策边界的对抗性样本")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 步骤3：按顺序生成7张关键图
    # ============================================================

    # ------------------------------------------------------------
    # A) SHAP summary beeswarm
    # ------------------------------------------------------------
    print(f"\n[图A] SHAP Summary Beeswarm Plot")
    print(f"  正在生成...")

    # ⚠️ 调整画布尺寸为更窄更长，节省拼图空间
    fig_a, ax_a = plt.subplots(figsize=(10, max(10, len(oof_data['feature_names']) * 0.45)))
    plot_panel_a_summary_beeswarm(ax_a, oof_data)

    # 添加图注
    fig_a.text(0.5, 0.01,
              f'Positive class: Malignant | SHAP space: {oof_data["shap_output_space"]} (margin)',
              ha='center', fontsize=14, style='italic',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    output_path_a = os.path.join(output_dir, 'Fig_A_SHAP_Summary_Beeswarm')
    for fmt in ['png', 'svg']:
        filepath = f'{output_path_a}.{fmt}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] 已保存: {filepath}")
    plt.close()

    # ------------------------------------------------------------
    # B) Decision plot（单图，高亮F/G病例）
    # ------------------------------------------------------------
    print(f"\n[图B] Decision Plot（高亮F/G病例）")
    print(f"  正在生成...")

    # ⚠️ 增加画布尺寸
    fig_b, ax_b = plt.subplots(figsize=(14, 14))

    # 获取数据
    shap_values = oof_data['shap_values_oof']
    base_values = oof_data['base_values_oof']
    y_true = oof_data['y_true']
    feature_names = oof_data['feature_names']

    # 随机抽样（固定seed）
    rng = np.random.RandomState(42)
    n_samples = 400
    all_indices = np.arange(len(shap_values))

    remaining_indices = np.setdiff1d(all_indices, [benign_idx, malignant_idx])
    sampled_indices = rng.choice(remaining_indices,
                                size=min(n_samples - 2, len(remaining_indices)),
                                replace=False)

    plot_indices = np.concatenate([[benign_idx, malignant_idx], sampled_indices])

    plot_shap_values = shap_values[plot_indices]
    plot_base_values = base_values[plot_indices]
    plot_y_true = y_true[plot_indices]

    # ⚠️ 关键修复1：按mean(|SHAP|)降序排序（重要特征在前）
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_indices_by_importance = np.argsort(mean_abs_shap)[::-1]

    # 应用排序
    plot_shap_values_sorted = plot_shap_values[:, sorted_indices_by_importance]
    feature_names_sorted = [feature_names[i] for i in sorted_indices_by_importance]

    # ⚠️ 关键修复2：统一 common_base 并将 base_shift 作为特征
    common_base = float(np.mean(plot_base_values))
    base_shift = (plot_base_values - common_base).reshape(-1, 1)

    # 将 base_shift 作为第一个特征
    plot_shap_aug = np.concatenate([base_shift, plot_shap_values_sorted], axis=1)
    feature_names_aug = ["base_shift"] + feature_names_sorted

    # 绘制decision plot
    try:
        highlight_idx_in_plot = np.where(plot_indices == benign_idx)[0][0]
        highlight_idx_in_plot2 = np.where(plot_indices == malignant_idx)[0][0]

        plt.sca(ax_b)

        shap.decision_plot(
            common_base,  #  使用统一的 common_base
            plot_shap_aug,
            feature_names=feature_names_aug,
            highlight=[highlight_idx_in_plot, highlight_idx_in_plot2],
            show=False
            #  移除 plot_width=12 - 使用 figsize 控制宽度
        )

    except Exception as e:
        #  打印完整的错误堆栈用于调试
        import traceback
        print(f"  [ERROR] Decision plot失败:")
        print(f"  Exception: {str(e)}")
        print(f"  Traceback:")
        traceback.print_exc()

        cumulative_shap = plot_shap_aug.cumsum(axis=1) + common_base

        for i, idx in enumerate(plot_indices):
            color = 'red' if idx in [benign_idx, malignant_idx] else ('blue' if plot_y_true[i] == 1 else 'gray')
            alpha = 1.0 if idx in [benign_idx, malignant_idx] else 0.3
            ax_b.plot(cumulative_shap[i], color=color, alpha=alpha,
                     linewidth=1.5 if idx in [benign_idx, malignant_idx] else 0.5)

        ax_b.set_xlabel('Feature Index (sorted by importance)', fontsize=10)
        ax_b.set_ylabel('Model Output (log-odds)', fontsize=10)

    #  添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Highlighted (F/G)'),
        Line2D([0], [0], color='blue', lw=1.5, alpha=0.3, label='Malignant'),
        Line2D([0], [0], color='gray', lw=1.5, alpha=0.3, label='Benign')
    ]
    ax_b.legend(handles=legend_elements, loc='upper right', fontsize=12)

    ax_b.set_title('Model Decision Paths (Highlighting Benign/Malignant Cases)',
                fontsize=18, fontweight='bold', pad=10)

    plt.tight_layout()

    output_path_b = os.path.join(output_dir, 'Fig_B_Decision_Plot')
    for fmt in ['png', 'svg']:
        filepath = f'{output_path_b}.{fmt}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] 已保存: {filepath}")
    plt.close()

    # ------------------------------------------------------------
    # C) Grouped importance（benign vs malignant，top20）
    # ------------------------------------------------------------
    print(f"\n[图C] Grouped Feature Importance (Benign vs Malignant, Top 20)")
    print(f"  正在生成...")

    # ⚠️ 压缩宽度
    fig_c, ax_c = plt.subplots(figsize=(12, 14))

    # 按overall mean(|SHAP|)选top20
    mean_abs_shap_global = np.mean(np.abs(oof_data['shap_values_oof']), axis=0)
    top_indices = np.argsort(mean_abs_shap_global)[-20:][::-1]

    # 分别计算benign和malignant子集的mean(|SHAP|)
    benign_mask = oof_data['y_true'] == 0
    malignant_mask = oof_data['y_true'] == 1

    mean_abs_shap_benign = np.mean(np.abs(oof_data['shap_values_oof'][benign_mask]), axis=0)
    mean_abs_shap_malignant = np.mean(np.abs(oof_data['shap_values_oof'][malignant_mask]), axis=0)

    top_features = [oof_data['feature_names'][i] for i in top_indices]
    benign_values = mean_abs_shap_benign[top_indices]
    malignant_values = mean_abs_shap_malignant[top_indices]

    # 绘制并排条形图
    y_pos = np.arange(len(top_features))
    height = 0.40

    bars1 = ax_c.barh(y_pos - height/2, benign_values, height,
                      label='Benign', color='#3498db', alpha=0.8)
    bars2 = ax_c.barh(y_pos + height/2, malignant_values, height,
                      label='Malignant', color='#e74c3c', alpha=0.8)

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(top_features, fontsize=16)
    ax_c.invert_yaxis()
    ax_c.set_xlabel('Mean |SHAP|', fontsize=18, fontweight='bold')
    ax_c.set_title('Grouped Feature Importance (Top 20: Benign vs Malignant)',
                   fontsize=20, fontweight='bold', pad=10)
    ax_c.legend(loc='lower right', fontsize=14)
    ax_c.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    output_path_c = os.path.join(output_dir, 'Fig_C_Grouped_Feature_Importance')
    for fmt in ['png', 'svg']:
        filepath = f'{output_path_c}.{fmt}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] 已保存: {filepath}")
    plt.close()

    # ------------------------------------------------------------
    # E) LIME（可选）
    # ------------------------------------------------------------
    if enable_lime and X_train_full is not None and feature_names_full is not None and LIME_AVAILABLE:
        print(f"\n[图E] LIME Explanation (High-Risk Malignant)")
        print(f"  正在生成...")

        fig_e, ax_e = plt.subplots(figsize=(10, 8))
        plot_panel_e_lime(ax_e, oof_data, malignant_idx, X_train_full, feature_names_full)

        plt.tight_layout()

        output_path_e = os.path.join(output_dir, 'Fig_E_LIME_HighRiskMalignant')
        for fmt in ['png', 'svg']:
            filepath = f'{output_path_e}.{fmt}'
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  [OK] 已保存: {filepath}")
        plt.close()
    else:
        print(f"\n[图E] LIME Explanation - 跳过（enable_lime=False或LIME未安装）")

    # ------------------------------------------------------------
    # F) Benign case force strip
    # ------------------------------------------------------------
    print(f"\n[图F] Low-Probability Benign Case Force Strip")
    print(f"  正在生成...")

    # ⚠️ 关键优化：特征名换行后需要更多高度，增加到 18 英寸
    fig_f, ax_f = plt.subplots(figsize=(28, 18))  # 宽度 28，高度 14→18（增加垂直空间）
    plot_panel_f_benign_case(ax_f, oof_data, benign_idx)

    # ⚠️ 关键修复：调整边距，为 "higher"/"lower" 标签留出空间
    # SHAP force plot 会在顶部和底部绘制这些标签，需要额外空间
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.02, right=0.98)

    output_path_f = os.path.join(output_dir, 'Fig_F_ForceStrip_BenignCase')
    for fmt in ['png', 'svg']:
        filepath = f'{output_path_f}.{fmt}'
        # ⚠️ 关键修复：使用 bbox_inches='tight' 确保所有内容（包括 axes 外的标签）都被保存
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] 已保存: {filepath}")
    plt.close()

    # ------------------------------------------------------------
    # G) Malignant case force strip
    # ------------------------------------------------------------
    print(f"\n[图G] High-Probability Malignant Case Force Strip")
    print(f"  正在生成...")

    # ⚠️ 关键优化：特征名换行后需要更多高度，增加到 18 英寸
    fig_g, ax_g = plt.subplots(figsize=(28, 18))  # 宽度 28，高度 14→18（增加垂直空间）
    plot_panel_g_malignant_case(ax_g, oof_data, malignant_idx)

    # ⚠️ 关键修复：调整边距，为 "higher"/"lower" 标签留出空间
    # SHAP force plot 会在顶部和底部绘制这些标签，需要额外空间
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.02, right=0.98)

    output_path_g = os.path.join(output_dir, 'Fig_G_ForceStrip_HighRiskMalignant')
    for fmt in ['png', 'svg']:
        filepath = f'{output_path_g}.{fmt}'
        # ⚠️ 关键修复：使用 bbox_inches='tight' 确保所有内容（包括 axes 外的标签）都被保存
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] 已保存: {filepath}")
    plt.close()

    # ------------------------------------------------------------
    # H) Global force plot（只生成 HTML）
    # ------------------------------------------------------------
    print(f"\n[图H] Global Force Plot (HTML)")
    print(f"  正在生成 HTML...")

    # 使用主输出目录
    output_dir_h = os.path.join(config.OUTPUT_BASE_DIR, 'interpretability_plots_v2')

    # 直接生成 HTML，不生成 placeholder 图片
    html_path = create_global_force_html(
        oof_data,
        output_dir=output_dir_h,
        n_samples=280,
        n_features=25
    )

    print(f"  [OK] 面板H HTML 已生成: {html_path}")
    print(f"  [INFO] 请在浏览器中打开 HTML 文件查看交互式图表")

    # ============================================================
    # 步骤4：打印总结信息
    # ============================================================
    print("\n" + "="*70)
    print("[完成] 所有单面板独立图表已导出")
    print("="*70)

    print("\n[验收标准]")
    print(f"  ✓ Base value不是0：范围[{np.nanmin(oof_data['base_values_oof']):.4f}, {np.nanmax(oof_data['base_values_oof']):.4f}]")
    print(f"  ✓ Base value均值：{np.nanmean(oof_data['base_values_oof']):.4f}")
    print(f"  ✓ OOF数据完整性：{np.sum(~np.isnan(oof_data['oof_prob']))}/{len(oof_data['oof_prob'])}样本有效")
    print(f"  ✓ Consistent features：{len(oof_data['feature_names'])}个")
    print(f"  ✓ SHAP输出空间：{oof_data['shap_output_space']}（log-odds margin）")
    print(f"  ✓ 正类标注：Malignant (Cancer)")
    print(f"   列对齐验证通过：无缺失值")

    print(f"\n所有图表已保存到: {output_dir}")
    print("\n生成的文件列表:")
    print("  • Fig_A_SHAP_Summary_Beeswarm.{png,svg} - 面板A")
    print("  • Fig_B_Decision_Plot.{png,svg} - 面板B（高亮F/G对抗性病例）")
    print("  • Fig_C_Grouped_Feature_Importance.{png,svg} - 面板C（Top 20）")
    if enable_lime and X_train_full is not None and feature_names_full is not None and LIME_AVAILABLE:
        print("  • Fig_E_LIME_HighRiskMalignant.{png,svg} - 面板E")
    print("  • Fig_F_ForceStrip_BenignCase.{png,svg} - 面板F（对抗性良性病例，Top-15特征）")
    print("  • Fig_G_ForceStrip_HighRiskMalignant.{png,svg} - 面板G（对抗性恶性病例，Top-15特征）")
    print("  • Fig_H_GlobalForce_请截图后旋转90度.html - 面板H（可交互HTML）")
    print("  • ForceStrip_F_Sample*_请截图后旋转90度.html - 面板F（如果matplotlib失败）")
    print("  • ForceStrip_G_Sample*_请截图后旋转90度.html - 面板G（如果matplotlib失败）")

    print("\n[改进说明]")
    print("  ✓ 样本选择：使用高确信度样本（接近决策边界但非极端）")
    print("    - Benign: p∈[0.10, 0.15]（低概率范围）")
    print("    - Malignant: p∈[0.85, 0.90]（高概率范围）")
    print("  ✓ ⚠️ Force Strip 优化：特征名换行处理，减少宽度占用")
    print("    - 第一行：特征名=特征值")
    print("    - 第二行：(SHAP=贡献值)")
    print("    - 减少约 30-40% 宽度")
    print("  ✓ 画布尺寸全面优化（确保所有内容可见，可自行裁剪）：")
    print("    - 面板A: 12×{max(10, n_features*0.4)} 英寸")
    print("    - 面板B: 16×12 英寸")
    print("    - 面板C: 14×12 英寸")
    print("    - 面板F/G: 28×18 英寸（超大画布，换行后增加高度）")
    print("  ✓ ⚠️ 关键修复：Force Strip 顶部 'higher'/'lower' 标签不再被裁剪")
    print("    - 使用 bbox_inches='tight' 保存完整 figure")
    print("    - 调整边距 top=0.92, bottom=0.08 为标签留出空间")
    print("  ✓ 面板H优化：只生成 HTML 文件（可交互），不生成占位图")


# =============================================================================
# 模块测试
# =============================================================================

if __name__ == "__main__":
    print("\n解释性图表模块 v2.0")
    print("功能：SHAP特征分析拼图（A-H面板）\n")
    print("请通过main_v2.py运行完整分析")
