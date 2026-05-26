# -*- coding: utf-8 -*-
"""
Integrated modeling pipeline with 10-fold cross-validation.

Performs within-fold LASSO feature selection followed by multi-model
training (LR, RF, SVM, ANN, XGBoost) with bootstrap confidence intervals,
calibration curves, and decision curve analysis.
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
import warnings
from datetime import datetime
import time

# sklearn相关
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    brier_score_loss,
    confusion_matrix, classification_report
)
from scipy.stats import uniform, randint, loguniform  # ✅ v2.5添加：log空间采样

# XGBoost和SHAP
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False
        XGBOOST_AVAILABLE = False
except ImportError:
    XGBOOST_AVAILABLE = False
    SHAP_AVAILABLE = False

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings('ignore')

# =============================================================================
# 参数配置区（在此集中修改参数）
# =============================================================================

class ModelConfig:
    """建模参数配置类 - 所有可调参数集中在此"""

    # 交叉验证参数
    N_SPLITS = 10

    # LASSO特征筛选
    USE_LASSO_FEATURE_SELECTION = True
    LASSO_RELAXED = True
    LASSO_CV_FOLDS = 10
    LASSO_MAX_ITER = 10000  # 从2000增加到10000，提高收敛稳定性

    # LASSO参数范围：(log10起始, log10结束, 点数)

    LASSO_ALPHAS_TONGUE = (-4, 2, 100)      # 舌象单独
    LASSO_ALPHAS_PULSE = (-4, 2, 100)       # 脉象单独
    LASSO_ALPHAS_STANDARD = (-4, 2, 100)     # 标准模式
    LASSO_ALPHAS_RELAXED = (-4, 2, 100)     # 放宽模式
    LASSO_ALPHAS_COMBINED = (-4, 2, 100)    # 融合模式（与单独模式统一）


    # 智能调参参数 - 平衡版（降低CPU占用）
    # v2.7优化：精简RF参数空间，大幅降低调参时间
    SMART_TUNE_N_ITER = 15                   # 默认值（向后兼容）
    SMART_TUNE_N_ITER_SIMPLE = 30            # Logistic Regression (50→30, -40%)
    SMART_TUNE_N_ITER_COMPLEX = 80           # ANN (200→80, -60%)
    SMART_TUNE_N_ITER_RF = 40                # Random Forest (150→80→40, -73%)
    SMART_TUNE_N_ITER_XGB = 60               # XGBoost (150→80→60, -60%)
    SMART_TUNE_N_ITER_SVM = 50               # SVM (100→50, -50%)
    SMART_TUNE_CV_FOLDS = 2                  # 内层CV折数 (3→2, -33%)
    SMART_TUNE_SCORING = 'roc_auc'

    # Bootstrap参数（略微降低）
    BOOTSTRAP_N = 1000                        # 1000→500 (-50%, 置信区间变化不大)
    BOOTSTRAP_ALPHA = 0.05

    # 校准参数
    CALIBRATION_N_BINS = 10
    CALIBRATION_STRATEGY = 'quantile'

    # 决策曲线参数
    DCA_THRESHOLDS_NUM = 100

    # 图形参数
    FIGURE_DPI = 600
    SAVE_FORMATS = ['jpg', 'pdf', 'svg']
    FIGURE_SIZE_SINGLE = (10, 8)
    FIGURE_SIZE_COMBINED = (16, 12)
    FIGURE_SIZE_CONFUSION = (28, 12)

    # 字体大小
    FONT_SIZE_TITLE = 14
    FONT_SIZE_LABEL = 12
    FONT_SIZE_TICK = 10
    FONT_SIZE_LEGEND = 10

    # 文件命名前缀
    MODELING_PREFIX = "Modeling_Results"


# =============================================================================
# 代码实现
# =============================================================================


class IntegratedModelingV2:
    """
    整合建模类

    核心功能：
    - LASSO特征筛选（每折内统一筛选）
    - 多模型训练和评估
    - 智能调参机制
    """

    def __init__(self, feature_mode='combined', models_to_run='all'):
        """
        初始化整合建模器

        Parameters:
        -----------
        feature_mode : str, default='combined'
            特征模式 ('tongue'/'pulse'/'combined')
        models_to_run : str or list, default='all'
            要运行的模型列表
        """
        self.feature_mode = feature_mode
        self.models_to_run = models_to_run

        # 结果存储
        self.results = {}
        self.X = None
        self.y = None
        self.feature_names = None

        # 打印数据文件信息
        self._print_data_info()

        # 模型配置
        self.model_configs = self._get_model_configs()

    def _print_data_info(self):
        """打印数据文件信息"""
        print("\n" + "="*70)
        print("【数据信息】")
        print("="*70)
        print(f"数据文件: {config.RAW_DATA_PATH}")
        print(f"目标变量: {config.TARGET_COLUMN}")
        print(f"标签含义: {config.TARGET_LABELS}")
        print("="*70)

    def _get_model_configs(self):
        """获取模型配置"""
        configs = {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'short_name': 'LR',
                'estimator': LogisticRegression,
                'use_scaler': True,
                'color': '#1F77B4'  # 蓝色
            },
            'random_forest': {
                'name': 'Random Forest',
                'short_name': 'RF',
                'estimator': RandomForestClassifier,
                'use_scaler': False,
                'color': '#FF7F0E'  # 橙色
            },
            'svm': {
                'name': 'SVM',
                'short_name': 'SVM',
                'estimator': SVC,
                'use_scaler': True,
                'color': '#2CA02C'  # 绿色
            },
            'ann': {
                'name': 'ANN',
                'short_name': 'ANN',
                'estimator': MLPClassifier,
                'use_scaler': True,
                'color': '#D62728'  # 红色
            },
            'xgboost': {
                'name': 'XGBoost',
                'short_name': 'XGB',
                'estimator': xgb.XGBClassifier if XGBOOST_AVAILABLE else None,
                'use_scaler': False,
                'color': '#9467BD'  # 紫色
            }
        }
        return configs

    def light_auto_tune(self, model_name, base_model, X_train, y_train, data_info, n_iter=None):
        """
        智能自动调参函数

        根据数据特性自适应调整参数空间：
        - 检测小数据集（n_samples < 200）
        - 检测类别不平衡（class_ratio < 0.3 或 > 0.7）
        - 使用RandomizedSearchCV进行搜索
        - 自动添加class_weight处理不平衡

        v2.5改进：根据模型复杂度自动分配n_iter（如果未指定）
        """
        """
        智能自动调参函数

        根据数据特性自适应调整参数空间：
        - 检测小数据集（n_samples < 200）
        - 检测类别不平衡（class_ratio < 0.3 或 > 0.7）
        - 使用RandomizedSearchCV进行搜索
        - 自动添加class_weight处理不平衡
        """
        start_time = time.time()
        n_samples = data_info['n_samples']
        n_features = data_info['n_features']
        class_ratio = data_info['class_ratio']

        # 根据模型复杂度自动分配n_iter（v2.5改进）
        # v2.5优化：RF和XGBoost使用更高的调参次数
        if n_iter is None:
            if model_name == 'Logistic Regression':
                n_iter = ModelConfig.SMART_TUNE_N_ITER_SIMPLE
            elif model_name == 'ANN':
                n_iter = ModelConfig.SMART_TUNE_N_ITER_COMPLEX
            elif model_name == 'Random Forest':
                n_iter = ModelConfig.SMART_TUNE_N_ITER_RF    # ✅ RF专用：150
            elif model_name == 'XGBoost':
                n_iter = ModelConfig.SMART_TUNE_N_ITER_XGB   # ✅ XGB专用：150
            elif model_name == 'SVM':
                n_iter = ModelConfig.SMART_TUNE_N_ITER_SVM
            else:
                n_iter = ModelConfig.SMART_TUNE_N_ITER  # 默认值

        is_small_dataset = n_samples < 200
        is_imbalanced = class_ratio < 0.3 or class_ratio > 0.7

        param_distributions = {}

        # 根据模型类型配置参数空间（v2.5全面优化版）
        if model_name == 'Logistic Regression':
            # ========== Logistic Regression 优化 ==========
            # 优化点：
            # 1. C改用loguniform（关键参数）
            # 2. 优化solver选择（lbfgs快速，saga适合大数据）
            # 3. 降低max_iter到合理范围
            # 4. 简化penalty为l2（更稳定）
            param_distributions = {
                'C': loguniform(1e-4, 100),           # ✅ log空间采样
                'penalty': ['l2'],                     # ✅ l2更稳定
                'solver': ['lbfgs', 'saga'],          # ✅ lbfgs快，saga稳
                'max_iter': [5000, 10000],            # ✅ 合理范围
                'tol': [1e-4, 1e-5],
                'random_state': [config.RANDOM_SEED]
            }

            # 只在不平衡时添加class_weight
            if is_imbalanced:
                param_distributions['class_weight'] = ['balanced', None]

        elif model_name == 'Random Forest':
            # ========== Random Forest 快速优化版 (v2.7) ==========
            # 优化策略：
            # 1. 使用离散值而非randint（缩小搜索空间）
            # 2. 移除不重要的criterion（gini/entropy差异小）
            # 3. 精简max_features选项（保留最重要的）
            # 4. 固定min_samples到常用值

            if is_small_dataset:
                param_distributions = {
                    # 离散值：重点测试常用数量
                    'n_estimators': [100, 150, 200, 300],        # 4个关键值
                    'max_depth': [5, 8, 10, 12, None],           # 5个关键值
                    'min_samples_split': [2, 5, 10],              # 3个关键值
                    'min_samples_leaf': [1, 2, 4],                # 3个关键值
                    'max_features': ['sqrt', 'log2'],             # 2个最重要
                    'random_state': [config.RANDOM_SEED]
                }
                # 组合空间: 4×5×3×3×2 = 360 (远小于原连续空间)
            else:
                # 大数据集配置
                param_distributions = {
                    'n_estimators': [200, 300, 400, 500],
                    'max_depth': [8, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.7],
                    'random_state': [config.RANDOM_SEED]
                }

            # 不平衡数据处理（精简选项）
            if is_imbalanced:
                param_distributions['class_weight'] = ['balanced', None]  # 2个选项即可

        elif model_name == 'SVM':
            # ========== SVM 全面优化 ==========
            # 优化点：
            # 1. C改用loguniform（关键参数）
            # 2. gamma改用loguniform（关键参数）
            # 3. 添加class_weight处理不平衡
            # 4. 移除混合的gamma选项（避免warning）
            param_distributions = {
                'C': loguniform(0.01, 1000),            # ✅ log空间：正则化强度
                'gamma': loguniform(1e-4, 10),          # ✅ log空间：RBF核宽度
                'kernel': ['rbf'],
                'probability': [True],
                'random_state': [config.RANDOM_SEED]
            }

            # 不平衡数据处理（SVM对不平衡敏感）
            if is_imbalanced:
                param_distributions['class_weight'] = ['balanced', None]

        elif model_name == 'ANN':
            # ========== ANN 优化 ==========
            # 优化点：
            # 1. 简化hidden_layer_sizes（避免过深）
            # 2. alpha改用loguniform（L2正则化）
            # 3. learning_rate_init改用loguniform
            # 4. 移除logistic激活函数（太慢）
            # 5. 只用adam solver（最稳定）
            param_distributions = {
                'hidden_layer_sizes': [(20,), (30,), (50,), (100,), (30, 20), (50, 30)],  # ✅ 简化
                'activation': ['relu', 'tanh'],          # ✅ 移除logistic
                'solver': ['adam'],                      # ✅ 只用adam（最稳定）
                'alpha': loguniform(1e-6, 1),            # ✅ log空间：L2正则化
                'learning_rate_init': loguniform(1e-4, 0.1),  # ✅ log空间
                'batch_size': [16, 32, 64, 128],        # ✅ 扩大范围
                'max_iter': [5000, 10000],              # ✅ 降低（配合早停）
                'early_stopping': [True],
                'validation_fraction': [0.1, 0.2],      # ✅ 扩大范围
                'n_iter_no_change': [10, 20],          # ✅ 扩大范围
                'random_state': [config.RANDOM_SEED]
            }

        elif model_name == 'XGBoost':
            # ========== XGBoost 重点优化 ==========
            # 优化点：
            # 1. learning_rate改用loguniform（关键参数）
            # 2. 大幅扩大n_estimators范围
            # 3. 添加gamma（最小分裂增益）
            # 4. 添加reg_alpha和reg_lambda（正则化）
            # 5. 扩大subsample和colsample范围
            # 6. 扩大min_child_weight范围
            if is_small_dataset:
                param_distributions = {
                    'max_depth': randint(3, 8),
                    'learning_rate': loguniform(0.005, 0.3),    # ✅ log空间
                    'n_estimators': randint(100, 500),           # ✅ 扩大范围
                    'min_child_weight': randint(1, 10),           # ✅ 扩大范围
                    'gamma': uniform(0, 0.5),                     # ✅ 新增：最小分裂增益
                    'subsample': [0.7, 0.8, 0.9, 1.0],          # ✅ 扩大范围
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],   # ✅ 扩大范围
                    'random_state': [config.RANDOM_SEED]
                }
            else:
                # 大数据集：更全面的配置
                param_distributions = {
                    'max_depth': randint(3, 12),                   # ✅ 扩大范围
                    'learning_rate': loguniform(0.005, 0.3),    # ✅ log空间
                    'n_estimators': randint(200, 1000),           # ✅ 大幅扩大
                    'min_child_weight': randint(1, 10),           # ✅ 扩大范围
                    'gamma': uniform(0, 1),                       # ✅ 新增：最小分裂增益
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],     # ✅ 更大范围
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # ✅ 更大范围
                    'reg_alpha': uniform(0, 1),                  # ✅ 新增：L1正则化
                    'reg_lambda': uniform(0.5, 2),               # ✅ 新增：L2正则化
                    'random_state': [config.RANDOM_SEED]
                }

            # 不平衡数据处理（计算scale_pos_weight）
            if is_imbalanced:
                scale_pos_weight = (1 - class_ratio) / class_ratio
                # 添加多个选项供选择
                param_distributions['scale_pos_weight'] = [1, scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2]

        if not param_distributions:
            return base_model

        cv_folds = 3 if n_samples >= 100 else 2

        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='roc_auc',
            random_state=config.RANDOM_SEED,
            n_jobs=min(4, os.cpu_count()),  # 限制最多4核并行，防止CPU过载导致卡顿
            verbose=0,
            error_score=np.nan
        )

        random_search.fit(X_train, y_train)

        # v2.5改进：显示实际使用的n_iter和最佳参数
        best_params = random_search.best_params_
        print(f"    n_iter={n_iter}, 最佳AUC: {random_search.best_score_:.4f}, 耗时: {time.time()-start_time:.1f}秒")

        # 对重点模型（RF, XGBoost）显示关键参数
        if model_name in ['Random Forest', 'XGBoost']:
            key_params = []
            if 'n_estimators' in best_params:
                key_params.append(f"n_estimators={best_params['n_estimators']}")
            if 'max_depth' in best_params:
                key_params.append(f"max_depth={best_params['max_depth']}")
            if 'learning_rate' in best_params:
                key_params.append(f"lr={best_params['learning_rate']:.4f}")
            if 'min_child_weight' in best_params:
                key_params.append(f"min_child_weight={best_params['min_child_weight']}")
            if key_params:
                print(f"    关键参数: {', '.join(key_params)}")

        return random_search.best_estimator_

    def _extract_feature_importance(self, model, model_name, X_train, selected_features):
        """
        提取特征重要性

        Parameters:
        -----------
        model : object
            训练好的模型
        model_name : str
            模型名称
        X_train : array-like
            训练集特征
        selected_features : list
            选中的特征名称列表

        Returns:
        --------
        importance : array-like
            特征重要性数组
        """
        try:
            if model_name == 'Logistic Regression':
                # 线性回归：系数绝对值
                importance = np.abs(model.coef_[0])
            elif model_name == 'Random Forest':
                # 随机森林：Gini重要性
                importance = model.feature_importances_
            elif model_name == 'SVM':
                # SVM：系数绝对值
                if hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])
                else:
                    # 如果没有coef_（如某些核函数），返回均匀重要性
                    importance = np.ones(len(selected_features))
            elif model_name == 'ANN':
                # ANN：第一层权重的平均绝对值
                if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
                    importance = np.mean(np.abs(model.coefs_[0]), axis=1)
                else:
                    importance = np.ones(len(selected_features))
            elif model_name == 'XGBoost':
                # XGBoost：增益重要性
                importance = model.feature_importances_
            else:
                # 默认：均匀重要性
                importance = np.ones(len(selected_features))

            # 确保重要性长度与特征数量一致
            if len(importance) != len(selected_features):
                # 如果不一致，进行归一化处理
                if len(importance) < len(selected_features):
                    # 填充0
                    padded = np.zeros(len(selected_features))
                    padded[:len(importance)] = importance
                    importance = padded
                else:
                    # 截断
                    importance = importance[:len(selected_features)]

            return importance

        except Exception as e:
            print(f"      [WARNING] 提取特征重要性时出错: {str(e)}")
            # 返回均匀重要性
            return np.ones(len(selected_features))

    def bootstrap_auc_ci(self, y_true, y_proba, n_bootstraps=1000, alpha=0.05):
        """
        计算AUC的bootstrap置信区间

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_proba : array-like
            预测概率
        n_bootstraps : int
            bootstrap重采样次数
        alpha : float
            显著性水平

        Returns:
        --------
        tuple : (auc_lower, auc_upper)
            AUC的95%置信区间下限和上限
        """
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        if len(y_true) != len(y_proba):
            return np.nan, np.nan

        if len(np.unique(y_true)) < 2:
            return np.nan, np.nan

        # 使用固定随机种子确保可重复性
        rng = np.random.RandomState(config.RANDOM_SEED)
        n_samples = len(y_true)
        auc_values = []

        for i in range(n_bootstraps):
            try:
                # 有放回抽样
                indices = rng.choice(n_samples, n_samples, replace=True)
                bs_y_true = y_true[indices]
                bs_y_proba = y_proba[indices]

                # 检查bootstrap样本是否有足够的类别
                if len(np.unique(bs_y_true)) < 2:
                    continue

                # 计算bootstrap样本的AUC
                bs_auc = roc_auc_score(bs_y_true, bs_y_proba)
                auc_values.append(bs_auc)

            except Exception:
                continue

        # 确保有足够的有效样本
        if len(auc_values) < 10:
            return np.nan, np.nan

        try:
            # 计算置信区间
            auc_values = np.array(auc_values)
            auc_lower = np.percentile(auc_values, alpha/2 * 100)
            auc_upper = np.percentile(auc_values, (1-alpha/2) * 100)
            return auc_lower, auc_upper
        except Exception:
            return np.nan, np.nan

    def bootstrap_brier_ci(self, y_true, y_proba, n_bootstraps=1000, alpha=0.05):
        """
        计算Brier score的bootstrap置信区间

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_proba : array-like
            预测概率
        n_bootstraps : int
            bootstrap重采样次数
        alpha : float
            显著性水平

        Returns:
        --------
        tuple : (brier_lower, brier_upper)
            Brier score的95%置信区间下限和上限
        """
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        if len(y_true) != len(y_proba):
            return np.nan, np.nan

        if len(y_true) == 0:
            return np.nan, np.nan

        # 使用固定随机种子确保可重复性
        rng = np.random.RandomState(config.RANDOM_SEED)
        n_samples = len(y_true)
        brier_values = []

        for i in range(n_bootstraps):
            try:
                # 有放回抽样
                indices = rng.choice(n_samples, n_samples, replace=True)
                bs_y_true = y_true[indices]
                bs_y_proba = y_proba[indices]

                # 检查bootstrap样本是否有足够的类别
                if len(np.unique(bs_y_true)) < 2:
                    continue

                # 计算bootstrap样本的Brier score
                bs_brier = brier_score_loss(bs_y_true, bs_y_proba)
                brier_values.append(bs_brier)

            except Exception:
                continue

        # 确保有足够的有效样本
        if len(brier_values) < 10:
            return np.nan, np.nan

        try:
            # 计算置信区间
            brier_values = np.array(brier_values)
            brier_lower = np.percentile(brier_values, alpha/2 * 100)
            brier_upper = np.percentile(brier_values, (1-alpha/2) * 100)
            return brier_lower, brier_upper
        except Exception:
            return np.nan, np.nan

    def bootstrap_ap_ci(self, y_true, y_proba, n_bootstraps=1000, alpha=0.05):
        """
        计算Average Precision (AP)的bootstrap置信区间

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_proba : array-like
            预测概率
        n_bootstraps : int
            bootstrap重采样次数
        alpha : float
            显著性水平

        Returns:
        --------
        tuple : (ap_lower, ap_upper)
            Average Precision的95%置信区间下限和上限
        """
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        if len(y_true) != len(y_proba):
            return np.nan, np.nan

        if len(y_true) == 0:
            return np.nan, np.nan

        # 使用固定随机种子确保可重复性
        rng = np.random.RandomState(config.RANDOM_SEED)
        n_samples = len(y_true)
        ap_values = []

        for i in range(n_bootstraps):
            try:
                # 有放回抽样
                indices = rng.choice(n_samples, n_samples, replace=True)
                bs_y_true = y_true[indices]
                bs_y_proba = y_proba[indices]

                # 检查bootstrap样本是否有足够的类别
                if len(np.unique(bs_y_true)) < 2:
                    continue

                # 计算bootstrap样本的Average Precision
                bs_ap = average_precision_score(bs_y_true, bs_y_proba)
                ap_values.append(bs_ap)

            except Exception:
                continue

        # 确保有足够的有效样本
        if len(ap_values) < 10:
            return np.nan, np.nan

        try:
            # 计算置信区间
            ap_values = np.array(ap_values)
            ap_lower = np.percentile(ap_values, alpha/2 * 100)
            ap_upper = np.percentile(ap_values, (1-alpha/2) * 100)
            return ap_lower, ap_upper
        except Exception:
            return np.nan, np.nan

    def calculate_calibration_intercept_slope(self, y_true, y_proba):
        """
        计算校准的截距和斜率

        通过对预测logit进行逻辑回归得到校准参数
        intercept ≈ 0 表示无偏差，slope ≈ 1 表示适当校准

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_proba : array-like
            预测概率

        Returns:
        --------
        tuple : (intercept, slope)
            校准截距和斜率
        """
        # 输入验证
        if len(y_true) != len(y_proba):
            return np.nan, np.nan

        if len(np.unique(y_true)) < 2:
            return np.nan, np.nan

        # 避免log(0)的情况
        y_proba_clipped = np.clip(y_proba, 1e-10, 1-1e-10)

        # 计算预测的logit
        logit_pred = np.log(y_proba_clipped / (1 - y_proba_clipped))
        logit_pred = logit_pred.reshape(-1, 1)

        # 使用逻辑回归拟合校准参数，使用固定随机种子
        try:
            lr = LogisticRegression(
                fit_intercept=True,
                random_state=config.RANDOM_SEED,
                max_iter=1000
            )
            lr.fit(logit_pred, y_true)

            intercept = lr.intercept_[0]
            slope = lr.coef_[0][0]

        except Exception:
            return np.nan, np.nan

        return intercept, slope

    def calculate_ece(self, y_true, y_proba, n_bins=10):
        """
        计算Expected Calibration Error (ECE)

        ECE是各分箱校准误差的加权平均，权重为样本比例

        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_proba : array-like
            预测概率
        n_bins : int
            分箱数量

        Returns:
        --------
        tuple : (ece_value, bin_details)
            ECE值和分箱详细信息
        """
        # 输入验证
        if len(y_true) != len(y_proba):
            return np.nan, []

        if len(y_true) < n_bins:
            n_bins = len(y_true) // 2

        try:
            # 分箱
            y_true_array = np.array(y_true)
            y_proba_array = np.array(y_proba)

            # 使用quantile分箱
            bin_edges = np.percentile(y_proba_array, np.linspace(0, 100, n_bins + 1))
            bin_indices = np.digitize(y_proba_array, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            ece = 0.0
            bin_details = []

            for i in range(n_bins):
                mask = bin_indices == i
                n_in_bin = np.sum(mask)

                if n_in_bin > 0:
                    # 该分箱的权重（样本比例）
                    weight = n_in_bin / len(y_true_array)

                    # 计算该分箱的平均预测概率和真实阳性率
                    bin_prob_pred = np.mean(y_proba_array[mask])
                    bin_prob_true = np.mean(y_true_array[mask])
                    calibration_error = np.abs(bin_prob_pred - bin_prob_true)

                    # 累加到ECE
                    ece += weight * calibration_error

                    # 记录分箱详细信息
                    bin_details.append({
                        'bin': i+1,
                        'n_samples': n_in_bin,
                        'weight': weight,
                        'prob_pred': bin_prob_pred,
                        'prob_true': bin_prob_true,
                        'calibration_error': calibration_error
                    })

            return ece, bin_details

        except Exception:
            return np.nan, []

    def load_data(self, X, y, feature_names):
        """
        加载数据并打印特征信息

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            特征矩阵
        y : array-like, shape (n_samples,)
            标签向量
        feature_names : list
            特征名称列表
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.feature_names = list(feature_names)

        print(f"\n[OK] 数据加载完成")
        print(f"  - 样本数: {len(self.y)}")
        print(f"  - 特征数: {len(self.feature_names)}")
        print(f"  - 正类比例: {self.y.mean():.3f}")

        # 打印特征列表
        self._print_feature_list()

        return self

    def _print_feature_list(self):
        """打印特征列表"""
        print(f"\n" + "-"*70)
        print(f"【特征列表 - {self.feature_mode.upper()}模态】")
        print("-"*70)
        print(f"共 {len(self.feature_names)} 个特征：")

        for i, feat in enumerate(self.feature_names, 1):
            # 判断是舌象还是脉象特征
            if feat in config.TONGUE_FEATURES:
                feature_type = "[舌象]"
            elif feat in config.PULSE_FEATURES:
                feature_type = "[脉象]"
            else:
                feature_type = "[其他]"

            print(f"  {i:2d}. {feat:20s} {feature_type}")

        print("-"*70)

    def _get_feature_indices(self, feature_type):
        """
        获取特定类型特征的索引

        Parameters:
        -----------
        feature_type : str
            'tongue' 或 'pulse'

        Returns:
        --------
        indices : ndarray
            特征索引数组
        """
        if feature_type == 'tongue':
            reference_list = config.TONGUE_FEATURES
        elif feature_type == 'pulse':
            reference_list = config.PULSE_FEATURES
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        # 找到当前特征中匹配的索引
        indices = []
        for i, feat_name in enumerate(self.feature_names):
            if feat_name in reference_list:
                indices.append(i)

        return np.array(indices)

    def run_cross_validation(self):
        """运行10折交叉验证"""
        print(f"\n10折交叉验证 - {self.feature_mode}模态")

        if self.feature_mode == 'combined':
            print("策略: 分层LASSO（舌象和脉象独立筛选）")
        else:
            print("策略: 标准LASSO特征筛选")

        # 确定要运行的模型
        if self.models_to_run == 'all':
            models_to_run = [k for k in self.model_configs.keys()
                           if k != 'xgboost' or XGBOOST_AVAILABLE]
        else:
            models_to_run = self.models_to_run if isinstance(self.models_to_run, list) else [self.models_to_run]

        # 外层10折CV
        outer_cv = StratifiedKFold(
            n_splits=ModelConfig.N_SPLITS,
            shuffle=True,
            random_state=config.RANDOM_SEED
        )

        # 初始化所有模型的结果存储
        for model_key in models_to_run:
            self.results[model_key] = {
                'fold_aucs': [],
                'fold_aps': [],
                'fold_sensitivities': [],
                'fold_specificities': [],
                'fold_f1s': [],
                'fold_probs': [],
                'fold_y_true': [],
                'fold_y_pred': [],  # 新增：保存类别预测
                'fold_selected_features': [],  # 每折选中的特征
                'fold_confusion_matrices': [],  # 新增：混淆矩阵
                'fold_feature_importance': [],  # 新增：特征重要性
                'fold_best_params': [],  # 新增：最佳参数
                # SHAP数据（仅XGBoost）
                'fold_shap_values': [],  # 新增：SHAP值
                'fold_X_tests': [],  # 新增：测试集特征（用于SHAP聚合）
                'fold_base_values': [],  # 新增：每折的base value（expected_value）
                'fold_sample_ids': []  # 新增：每折的样本原始索引（用于OOF对齐）
            }

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(self.X, self.y)):
            print(f"\n--- Fold {fold_idx+1}/10 ---")

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

            # LASSO特征筛选
            selected_mask, selected_features = self._lasso_feature_selection(
                X_train, y_train, fold_idx+1
            )

            X_train_selected = X_train[:, selected_mask]
            X_test_selected = X_test[:, selected_mask]

            print(f"LASSO选中 {len(selected_features)} 个特征")

            for model_key in models_to_run:
                self.results[model_key]['fold_selected_features'].append(selected_features)

            for model_key in models_to_run:
                model_config = self.model_configs[model_key]

                if model_config['use_scaler']:
                    scaler = StandardScaler()
                    X_train_model = scaler.fit_transform(X_train_selected)
                    X_test_model = scaler.transform(X_test_selected)
                else:
                    X_train_model = X_train_selected
                    X_test_model = X_test_selected

                data_info = {
                    'n_samples': len(X_train_model),
                    'n_features': X_train_model.shape[1],
                    'class_ratio': y_train.mean()
                }

                print(f"\n{model_config['name']}")
                base_model = model_config['estimator']()
                best_model = self.light_auto_tune(
                    model_config['name'],
                    base_model,
                    X_train_model,
                    y_train,
                    data_info,
                    n_iter=None  # v2.5改进：使用模型复杂度自动分配的n_iter
                )

                metrics = self._evaluate_model(best_model, X_test_model, y_test)

                importance = self._extract_feature_importance(
                    best_model, model_config['name'], X_train_model, selected_features
                )

                if model_key == 'xgboost' and SHAP_AVAILABLE:
                    try:
                        explainer = shap.TreeExplainer(best_model)
                        shap_values = explainer.shap_values(X_test_model)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

                        # 获取 base value（expected_value）
                        base_value = explainer.expected_value
                        if isinstance(base_value, list):
                            base_value = base_value[1] if len(base_value) == 2 else base_value[0]

                        # 保存 SHAP 数据和 base value
                        self.results[model_key]['fold_shap_values'].append(shap_values)
                        self.results[model_key]['fold_X_tests'].append(X_test_model)
                        self.results[model_key]['fold_base_values'].append(base_value)

                        # 保存样本原始索引（用于 OOF 对齐）
                        self.results[model_key]['fold_sample_ids'].append(test_idx)
                    except Exception as e:
                        print(f"      [WARNING] SHAP计算失败: {str(e)}")
                        # 如果失败，保存默认值以确保长度一致
                        self.results[model_key]['fold_shap_values'].append(None)
                        self.results[model_key]['fold_X_tests'].append(None)
                        self.results[model_key]['fold_base_values'].append(None)
                        self.results[model_key]['fold_sample_ids'].append(test_idx)

                self.results[model_key]['fold_aucs'].append(metrics['auc'])
                self.results[model_key]['fold_aps'].append(metrics['ap'])
                self.results[model_key]['fold_sensitivities'].append(metrics['sensitivity'])
                self.results[model_key]['fold_specificities'].append(metrics['specificity'])
                self.results[model_key]['fold_f1s'].append(metrics['f1'])
                self.results[model_key]['fold_probs'].append(metrics['y_pred_proba'])
                self.results[model_key]['fold_y_true'].append(metrics['y_true'])
                self.results[model_key]['fold_y_pred'].append(metrics['y_pred'])
                self.results[model_key]['fold_confusion_matrices'].append(metrics['confusion_matrix'])
                self.results[model_key]['fold_feature_importance'].append(importance)
                self.results[model_key]['fold_best_params'].append({
                    'model': model_config['name'],
                    'params': best_model.get_params() if hasattr(best_model, 'get_params') else {}
                })

            # 本折性能汇总
            print(f"\n{'模型':20s} {'AUC':<6s} {'AP':<6s}")
            for model_key in models_to_run:
                model_name = self.model_configs[model_key]['name']
                auc = self.results[model_key]['fold_aucs'][-1]
                ap = self.results[model_key]['fold_aps'][-1]
                print(f"{model_name:20s} {auc:<6.3f} {ap:<6.3f}")

        # ================================================================
        # 步骤3：统计所有折的结果
        # ================================================================
        self._summarize_cv_results(models_to_run)

        return self.results

    def _lasso_feature_selection(self, X_train, y_train, fold_idx):
        """
        使用LASSO进行特征筛选（v2.4增强版）

        新增功能：
        1. 支持完全禁用LASSO（使用全特征）
        2. 支持放宽LASSO参数（保留更多特征）
        3. 根据feature_mode自动选择策略

        策略选择：
        - 如果ModelConfig.USE_LASSO_FEATURE_SELECTION = False: 跳过LASSO，返回全特征
        - 如果ModelConfig.USE_LASSO_FEATURE_SELECTION = True:
          - tongue/pulse: 标准LASSO（已放宽）
          - combined: 分层LASSO（舌象和脉象独立筛选+进一步放宽惩罚）

        Parameters:
        -----------
        X_train : array-like
            训练集特征
        y_train : array-like
            训练集标签
        fold_idx : int
            折数索引

        Returns:
        --------
        selected_mask : array-like, bool
            特征选择掩码
        selected_features : list
            选中的特征名称列表
        """
        # 检查是否禁用LASSO特征选择
        if not ModelConfig.USE_LASSO_FEATURE_SELECTION:
            print(f"\n  [INFO] Fold {fold_idx}: LASSO特征选择已禁用（使用全部特征）")
            print(f"  [INFO] 当前特征数量: {X_train.shape[1]}")

            # 返回全特征掩码
            selected_mask = np.ones(X_train.shape[1], dtype=bool)
            selected_features = self.feature_names.copy()

            print(f"  [OK] 使用全部 {len(selected_features)} 个特征")

            return selected_mask, selected_features

        # 判断是否使用分层LASSO
        if self.feature_mode == 'combined':
            # 融合模态：使用分层LASSO
            return self._lasso_feature_selection_stratified(X_train, y_train, fold_idx)
        else:
            # 单一模态：使用标准LASSO
            return self._lasso_feature_selection_standard(X_train, y_train, fold_idx)

    def _lasso_feature_selection_standard(self, X_train, y_train, fold_idx):
        """
        标准LASSO特征筛选（用于单一模态）v2.4增强版

        改进：
        - 使用放宽的alpha范围（logspace(-6, 2, 200)）
        - 支持进一步放宽（logspace(-7, 2, 200)）
        - 提升特征保留率

        Parameters:
        -----------
        X_train : array-like
            训练集特征
        y_train : array-like
            训练集标签
        fold_idx : int
            折数索引

        Returns:
        --------
        selected_mask : array-like, bool
            特征选择掩码
        selected_features : list
            选中的特征名称列表
        """
        # 步骤1：标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        print(f"  [DEBUG] Fold {fold_idx}: 标准化完成")
        print(f"  [DEBUG]   - 原始数据范围: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"  [DEBUG]   - 标准化后范围: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")

        # 步骤2：根据特征模式和配置选择alpha范围（v2.5改进 - 按模态定制）
        # 优先使用按模态定制的范围
        if self.feature_mode == 'tongue':
            alphas = np.logspace(*ModelConfig.LASSO_ALPHAS_TONGUE)
            mode_desc = "舌象LASSO (alpha下限: 1e-6)"
        elif self.feature_mode == 'pulse':
            alphas = np.logspace(*ModelConfig.LASSO_ALPHAS_PULSE)
            mode_desc = "脉象LASSO (alpha下限: 1e-6)"
        elif ModelConfig.LASSO_RELAXED:
            # 放宽模式：logspace(-8, 2, 300)
            alphas = np.logspace(*ModelConfig.LASSO_ALPHAS_RELAXED)
            mode_desc = "放宽LASSO (alpha下限: 1e-8)"
        else:
            # 标准放宽模式：logspace(-7, 2, 300)
            alphas = np.logspace(*ModelConfig.LASSO_ALPHAS_STANDARD)
            mode_desc = "标准LASSO (alpha下限: 1e-7)"

        print(f"  [DEBUG]   - Lambda范围: [{alphas.min():.2e}, {alphas.max():.2e}]")
        print(f"  [DEBUG]   - Alpha点数: {len(alphas)}")
        print(f"  [INFO]    - 使用 {mode_desc}")

        # 步骤3：LassoCV
        lasso_cv = LassoCV(
            alphas=alphas,
            cv=10,
            random_state=config.RANDOM_SEED,
            max_iter=ModelConfig.LASSO_MAX_ITER  # 使用配置而非硬编码
        )

        lasso_cv.fit(X_train_scaled, y_train)

        print(f"  [DEBUG]   - 最优Lambda: {lasso_cv.alpha_:.6f}")

        # 选择非零系数的特征
        selected_mask = lasso_cv.coef_ != 0

        # 如果没有选中任何特征，选择系数绝对值最大的前5个
        if selected_mask.sum() == 0:
            print(f"  [WARNING] Fold {fold_idx}: LASSO未选中任何特征，选择Top 5")
            top5_idx = np.argsort(np.abs(lasso_cv.coef_))[-5:]
            selected_mask = np.zeros_like(selected_mask, dtype=bool)
            selected_mask[top5_idx] = True
        else:
            # 打印选中的特征数量和系数信息
            n_selected = selected_mask.sum()
            print(f"  [OK] Fold {fold_idx}: LASSO选中 {n_selected} 个特征")
            print(f"  [DEBUG]   - 非零系数范围: [{lasso_cv.coef_[selected_mask].min():.6f}, {lasso_cv.coef_[selected_mask].max():.6f}]")
            print(f"  [DEBUG]   - 系数绝对值最大: {np.abs(lasso_cv.coef_).max():.6f}")

        selected_features = [self.feature_names[i]
                            for i in range(len(self.feature_names))
                            if selected_mask[i]]

        return selected_mask, selected_features

    def _lasso_feature_selection_stratified(self, X_train, y_train, fold_idx):
        """
        分层LASSO特征筛选（用于融合模态）

        策略：舌象和脉象独立筛选，然后拼接
        - 每个模态使用放宽参数（logspace(-8, 2, 300)）
        - 完全信任LASSO的数据驱动选择

        Parameters:
        -----------
        X_train : array-like
            训练集特征
        y_train : array-like
            训练集标签
        fold_idx : int
            折数索引

        Returns:
        --------
        selected_mask : array-like, bool
            特征选择掩码（完整长度）
        selected_features : list
            选中的特征名称列表
        """
        # 获取舌象和脉象特征索引
        tongue_indices = self._get_feature_indices('tongue')
        pulse_indices = self._get_feature_indices('pulse')

        print(f"  分层LASSO: 舌象{len(tongue_indices)}个, 脉象{len(pulse_indices)}个")

        # 打印LASSO参数范围
        alphas_combined = np.logspace(
            ModelConfig.LASSO_ALPHAS_COMBINED[0],
            ModelConfig.LASSO_ALPHAS_COMBINED[1],
            ModelConfig.LASSO_ALPHAS_COMBINED[2]
        )
        print(f"  [INFO] LASSO参数范围: log10({ModelConfig.LASSO_ALPHAS_COMBINED[0]}, {ModelConfig.LASSO_ALPHAS_COMBINED[1]}, {ModelConfig.LASSO_ALPHAS_COMBINED[2]}点)")
        print(f"  [INFO] Lambda范围: [{alphas_combined.min():.2e}, {alphas_combined.max():.2e}], 共{len(alphas_combined)}个alpha值")

        # 舌象筛选
        scaler_tongue = StandardScaler()
        X_tongue = X_train[:, tongue_indices]
        X_tongue_scaled = scaler_tongue.fit_transform(X_tongue)

        lasso_tongue = LassoCV(
            alphas=alphas_combined,
            cv=10,
            max_iter=ModelConfig.LASSO_MAX_ITER,  # 使用配置而非硬编码
            random_state=config.RANDOM_SEED
        )
        lasso_tongue.fit(X_tongue_scaled, y_train)

        print(f"    舌象最优Lambda: {lasso_tongue.alpha_:.6f}")

        tongue_mask_in_modality = lasso_tongue.coef_ != 0
        tongue_selected = tongue_indices[tongue_mask_in_modality]

        print(f"    舌象选中 {len(tongue_selected)} 个特征")

        # 脉象筛选
        scaler_pulse = StandardScaler()
        X_pulse = X_train[:, pulse_indices]
        X_pulse_scaled = scaler_pulse.fit_transform(X_pulse)

        lasso_pulse = LassoCV(
            alphas=alphas_combined,
            cv=10,
            max_iter=ModelConfig.LASSO_MAX_ITER,  # 使用配置而非硬编码
            random_state=config.RANDOM_SEED
        )
        lasso_pulse.fit(X_pulse_scaled, y_train)

        print(f"    脉象最优Lambda: {lasso_pulse.alpha_:.6f}")

        pulse_mask_in_modality = lasso_pulse.coef_ != 0
        pulse_selected = pulse_indices[pulse_mask_in_modality]

        print(f"    脉象选中 {len(pulse_selected)} 个特征")

        # 拼接选定的特征
        selected_indices = np.concatenate([tongue_selected, pulse_selected])

        print(f"  总计: {len(selected_indices)} 个特征 (舌象{len(tongue_selected)}+脉象{len(pulse_selected)})")

        # 构建完整掩码（与原代码兼容）
        selected_mask = np.zeros(X_train.shape[1], dtype=bool)
        selected_mask[selected_indices] = True

        # 获取特征名称
        selected_features = [self.feature_names[i] for i in selected_indices]

        print(f"  [OK] Fold {fold_idx}: 分层LASSO完成，共选中 {len(selected_features)} 个特征")
        print(f"  特征列表: {', '.join(selected_features)}")

        return selected_mask, selected_features

    def _evaluate_model(self, model, X_test, y_test):
        """
        评估模型性能（完整版 - 包含置信区间和校准指标）

        Parameters:
        -----------
        model : object
            训练好的模型
        X_test : array-like
            测试集特征
        y_test : array-like
            测试集标签

        Returns:
        --------
        metrics : dict
            性能指标字典（包含基础指标、置信区间、校准指标）
        """
        # 预测概率
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            # SVC没有predict_proba时用decision_function
            y_pred_proba = model.decision_function(X_test)
            # 归一化到[0,1]
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / \
                          (y_pred_proba.max() - y_pred_proba.min())

        # 预测类别
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # ========== 基础指标 ==========
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        try:
            tn, fp, fn, tp = cm.ravel()
        except ValueError:
            # 处理单类情况
            if cm.shape == (1, 1):
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                raise

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Brier Score
        brier_score = brier_score_loss(y_test, y_pred_proba)

        # ========== 置信区间（Bootstrap） ==========
        try:
            auc_lower, auc_upper = self.bootstrap_auc_ci(y_test, y_pred_proba)
            brier_lower, brier_upper = self.bootstrap_brier_ci(y_test, y_pred_proba)
            ap_lower, ap_upper = self.bootstrap_ap_ci(y_test, y_pred_proba)
        except Exception as e:
            print(f"      [WARNING] 计算置信区间时出错: {str(e)}")
            auc_lower, auc_upper = np.nan, np.nan
            brier_lower, brier_upper = np.nan, np.nan
            ap_lower, ap_upper = np.nan, np.nan

        # ========== 校准指标 ==========
        try:
            calib_intercept, calib_slope = self.calculate_calibration_intercept_slope(
                y_test, y_pred_proba
            )
            ece, _ = self.calculate_ece(y_test, y_pred_proba)
        except Exception as e:
            print(f"      [WARNING] 计算校准指标时出错: {str(e)}")
            calib_intercept, calib_slope = np.nan, np.nan
            ece = np.nan

        # ========== 整合所有指标 ==========
        metrics = {
            # 基础分类指标
            'auc': auc,
            'ap': ap,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'brier_score': brier_score,

            # 置信区间
            'auc_ci_lower': auc_lower,
            'auc_ci_upper': auc_upper,
            'ap_ci_lower': ap_lower,
            'ap_ci_upper': ap_upper,
            'brier_ci_lower': brier_lower,
            'brier_ci_upper': brier_upper,

            # 校准指标
            'calibration_intercept': calib_intercept,
            'calibration_slope': calib_slope,
            'ece': ece,

            # 预测结果（用于后续绘图）
            'confusion_matrix': cm,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'y_true': y_test
        }

        return metrics

    def _summarize_cv_results(self, models_to_run):
        """汇总10折CV结果"""
        print(f"\n{'='*60}")
        print(f"{'模型':20s} {'AUC':<10s} {'Std':<8s} {'95% CI':<15s} {'AP':<8s}")
        print("-"*60)

        for model_key in models_to_run:
            cv_results = self.results[model_key]

            mean_auc = np.mean(cv_results['fold_aucs'])
            std_auc = np.std(cv_results['fold_aucs'])
            ci_lower = mean_auc - 1.96 * std_auc / np.sqrt(10)
            ci_upper = mean_auc + 1.96 * std_auc / np.sqrt(10)
            mean_ap = np.mean(cv_results['fold_aps'])

            model_name = self.model_configs[model_key]['short_name']

            print(f"{model_name:20s} {mean_auc:<10.3f} {std_auc:<8.3f} "
                  f"[{ci_lower:.3f}, {ci_upper:.3f}] {mean_ap:<8.3f}")

        print("="*60)

    def generate_feature_frequency_table(self, output_dir, feature_mode):
        """
        生成特征出现频率表（49特征 × 10折）

        这是一个关键表格，用于说明：
        1. 哪些是"一致特征"（10/10折都出现）
        2. 每个特征在各折中的出现情况
        3. 最终纳入模型的是哪些特征

        Parameters:
        -----------
        output_dir : str
            输出目录
        feature_mode : str
            特征模式名称
        """
        print(f"\n{'='*70}")
        print("【生成特征出现频率表】")
        print("="*70)

        # 只分析XGBoost的特征选择结果
        if 'xgboost' not in self.results:
            print("  [SKIP] XGBoost未运行，无法生成特征频率表")
            return None

        fold_features = self.results['xgboost']['fold_selected_features']

        # 收集所有可能的特征（49个特征）
        all_possible_features = set()
        for features in fold_features:
            all_possible_features.update(features)

        all_possible_features = sorted(list(all_possible_features))

        # 构建49特征 × 10折的矩阵
        frequency_matrix = np.zeros((len(all_possible_features), 10), dtype=int)

        for fold_idx, features in enumerate(fold_features):
            for feat_idx, feat in enumerate(all_possible_features):
                if feat in features:
                    frequency_matrix[feat_idx, fold_idx] = 1

        # 统计每个特征的出现频率
        feature_appear_count = {}
        for features in fold_features:
            for feat in features:
                feature_appear_count[feat] = feature_appear_count.get(feat, 0) + 1

        # 创建DataFrame
        freq_df = pd.DataFrame(
            frequency_matrix,
            index=all_possible_features,
            columns=[f'Fold{i+1}' for i in range(10)]
        )

        # 添加统计列
        freq_df['Total_Appearances'] = freq_df.sum(axis=1)
        freq_df['Appearance_Rate'] = freq_df['Total_Appearances'].apply(
            lambda x: f'{x}/10 ({x*10}%)'
        )
        freq_df['Stability'] = freq_df['Total_Appearances'].apply(
            lambda x: 'Consistent (10/10)' if x == 10 else f'Variable ({x}/10)'
        )
        freq_df['Included_in_Model'] = freq_df['Total_Appearances'].apply(
            lambda x: 'Yes' if x >= 5 else 'No'  # 至少5折出现才纳入
        )

        # 重新排列列：将统计列放在前面
        stats_cols = ['Total_Appearances', 'Appearance_Rate', 'Stability', 'Included_in_Model']
        fold_cols = [f'Fold{i+1}' for i in range(10)]
        freq_df = freq_df[stats_cols + fold_cols]

        # 按出现频率降序排列
        freq_df = freq_df.sort_values('Total_Appearances', ascending=False)

        # 导出表格
        timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)
        excel_file = os.path.join(output_dir,
                                   f'Table_Feature_Frequency_Matrix_{feature_mode}_{timestamp}.xlsx')
        freq_df.to_excel(excel_file, index=True, engine='openpyxl')
        print(f"  [OK] 特征频率矩阵表已保存: {os.path.basename(excel_file)}")

        # 同时导出CSV
        csv_file = os.path.join(output_dir,
                               f'Table_Feature_Frequency_Matrix_{feature_mode}_{timestamp}.csv')
        freq_df.to_csv(csv_file, index=True, encoding='utf-8-sig')
        print(f"  [OK] 特征频率矩阵表已保存: {os.path.basename(csv_file)}")

        # 统计信息
        n_stable = (freq_df['Total_Appearances'] == 10).sum()
        n_included = (freq_df['Total_Appearances'] >= 5).sum()
        n_all_features = len(all_possible_features)

        print(f"\n  特征统计:")
        print(f"    - 总特征数: {n_all_features}")
        print(f"    - 一致特征(10/10): {n_stable}个")
        print(f"    - 纳入模型特征(≥5/10): {n_included}个")

        # 打印前20个特征
        print(f"\n  Top 20特征出现频率:")
        print(f"  {'特征':<20s} {'出现次数':<10s} {'出现率':<12s} {'一致性':<20s} {'纳入模型':<15s}")
        print("  " * 80)
        for _, row in freq_df.head(20).iterrows():
            print(f"  {row.name:<20s} {row['Total_Appearances']:<10d} "
                  f"{row['Appearance_Rate']:<12s} {row['Stability']:<20s} "
                  f"{row['Included_in_Model']:<15s}")

        print(f"\n  说明:")
        print(f"    - Consistent (10/10): 在所有10折中均被选中")
        print(f"    - Included_in_Model=Yes: 至少在5折中出现，纳入最终模型")
        print(f"    - 该表用于论文中说明特征选择的一致性")

        return freq_df

    def save_results_to_excel(self, output_dir, feature_mode):
        """
        保存完整结果到Excel文件（基础指标+高级指标两张表）

        Parameters:
        -----------
        output_dir : str
            输出目录
        feature_mode : str
            特征模式名称
        """
        timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)

        # 创建Excel writer
        filepath = os.path.join(
            output_dir,
            f'{ModelConfig.MODELING_PREFIX}_{feature_mode}_{timestamp}.xlsx'
        )

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # ========== 表1：基础指标（AUC + 传统分类指标） ==========
            basic_data = []

            for model_key, cv_results in self.results.items():
                model_name = self.model_configs[model_key]['name']

                # 计算平均指标
                mean_auc = np.mean(cv_results['fold_aucs'])
                std_auc = np.std(cv_results['fold_aucs'])

                # 计算AUC的95%置信区间（bootstrap）
                all_probs = np.concatenate(cv_results['fold_probs'])
                all_y_true = np.concatenate(cv_results['fold_y_true'])
                auc_lower, auc_upper = self.bootstrap_auc_ci(all_y_true, all_probs)

                mean_ap = np.mean(cv_results['fold_aps'])
                mean_sensitivity = np.mean(cv_results['fold_sensitivities'])
                mean_specificity = np.mean(cv_results['fold_specificities'])
                mean_f1 = np.mean(cv_results['fold_f1s'])

                basic_data.append({
                    'Model': model_name,
                    'AUC': f'{mean_auc:.3f}',
                    'AUC_Std': f'{std_auc:.3f}',
                    'AUC_95%_CI_Lower': f'{auc_lower:.3f}' if not np.isnan(auc_lower) else 'N/A',
                    'AUC_95%_CI_Upper': f'{auc_upper:.3f}' if not np.isnan(auc_upper) else 'N/A',
                    'AP': f'{mean_ap:.3f}',
                    'Sensitivity': f'{mean_sensitivity:.3f}',
                    'Specificity': f'{mean_specificity:.3f}',
                    'F1-Score': f'{mean_f1:.3f}'
                })

            basic_df = pd.DataFrame(basic_data)
            basic_df.to_excel(writer, sheet_name='Basic_Metrics', index=False)

            # ========== 表2：高级指标（置信区间和校准指标） ==========
            advanced_data = []

            for model_key, cv_results in self.results.items():
                model_name = self.model_configs[model_key]['name']

                # 合并所有折的数据用于计算高级指标
                all_probs = np.concatenate(cv_results['fold_probs'])
                all_y_true = np.concatenate(cv_results['fold_y_true'])

                # Brier Score及置信区间
                brier_lower, brier_upper = self.bootstrap_brier_ci(all_y_true, all_probs)
                brier_score = brier_score_loss(all_y_true, all_probs)

                # AP置信区间
                ap_lower, ap_upper = self.bootstrap_ap_ci(all_y_true, all_probs)

                # 校准指标
                calib_intercept, calib_slope = self.calculate_calibration_intercept_slope(
                    all_y_true, all_probs
                )
                ece, _ = self.calculate_ece(all_y_true, all_probs)

                advanced_data.append({
                    'Model': model_name,
                    'Brier_Score': f'{brier_score:.3f}',
                    'Brier_95%_CI_Lower': f'{brier_lower:.3f}' if not np.isnan(brier_lower) else 'N/A',
                    'Brier_95%_CI_Upper': f'{brier_upper:.3f}' if not np.isnan(brier_upper) else 'N/A',
                    'AP_95%_CI_Lower': f'{ap_lower:.3f}' if not np.isnan(ap_lower) else 'N/A',
                    'AP_95%_CI_Upper': f'{ap_upper:.3f}' if not np.isnan(ap_upper) else 'N/A',
                    'Calibration_Intercept': f'{calib_intercept:.3f}' if not np.isnan(calib_intercept) else 'N/A',
                    'Calibration_Slope': f'{calib_slope:.3f}' if not np.isnan(calib_slope) else 'N/A',
                    'ECE': f'{ece:.3f}' if not np.isnan(ece) else 'N/A'
                })

            advanced_df = pd.DataFrame(advanced_data)
            advanced_df.to_excel(writer, sheet_name='Advanced_Metrics', index=False)

            # ========== 表3：每折详细结果 ==========
            fold_data = []

            for model_key, cv_results in self.results.items():
                model_name = self.model_configs[model_key]['short_name']

                for fold_idx in range(len(cv_results['fold_aucs'])):
                    fold_data.append({
                        'Model': model_name,
                        'Fold': fold_idx + 1,
                        'AUC': cv_results['fold_aucs'][fold_idx],
                        'AP': cv_results['fold_aps'][fold_idx],
                        'Sensitivity': cv_results['fold_sensitivities'][fold_idx],
                        'Specificity': cv_results['fold_specificities'][fold_idx],
                        'F1': cv_results['fold_f1s'][fold_idx]
                    })

            fold_df = pd.DataFrame(fold_data)
            fold_df.to_excel(writer, sheet_name='Fold_Results', index=False)

        print(f"\n[OK] 结果已保存到Excel: {filepath}")
        print(f"  - 基础指标表（Basic_Metrics）：{len(basic_df)}个模型")
        print(f"  - 高级指标表（Advanced_Metrics）：{len(advanced_df)}个模型")
        print(f"  - 每折详细结果（Fold_Results）：{len(fold_df)}条记录")

        return filepath

    def generate_comparison_plots(self, output_dir):
        """
        生成多模型对比图（类似BMC脚本）

        包括：
        - ROC曲线对比
        - PR曲线对比
        """
        print(f"\n{'='*70}")
        print("【生成对比图】")
        print("="*70)

        # 1. ROC曲线对比图
        self._plot_roc_comparison(output_dir)

        # 2. PR曲线对比图
        self._plot_pr_comparison(output_dir)

        print(f"\n[OK] 所有对比图已保存")

    def _plot_roc_comparison(self, output_dir):
        """绘制ROC曲线对比图"""
        fig, ax = plt.subplots(figsize=ModelConfig.FIGURE_SIZE_SINGLE)

        for model_key, model_config in self.model_configs.items():
            if model_key not in self.results:
                continue

            # 拼接所有折的预测结果
            all_probs = np.concatenate(self.results[model_key]['fold_probs'])
            all_y_true = np.concatenate(self.results[model_key]['fold_y_true'])

            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(all_y_true, all_probs)
            auc = np.mean(self.results[model_key]['fold_aucs'])

            # 绘制ROC曲线
            ax.plot(fpr, tpr,
                   color=model_config['color'],
                   lw=2,
                   label=f"{model_config['short_name']} (AUC={auc:.3f})")

        # 对角线
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=ModelConfig.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=ModelConfig.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title('ROC Curve Comparison (10-Fold CV)',
                    fontsize=ModelConfig.FONT_SIZE_TITLE, fontweight='bold')
        ax.legend(fontsize=ModelConfig.FONT_SIZE_LEGEND, loc='lower right')
        ax.grid(True, alpha=0.3)

        # 保存
        for fmt in ModelConfig.SAVE_FORMATS:
            filepath = os.path.join(output_dir, f'roc_comparison.{fmt}')
            plt.savefig(filepath, dpi=ModelConfig.FIGURE_DPI, bbox_inches='tight')
        plt.close()

    def _plot_pr_comparison(self, output_dir):
        """绘制PR曲线对比图"""
        fig, ax = plt.subplots(figsize=ModelConfig.FIGURE_SIZE_SINGLE)

        for model_key, model_config in self.model_configs.items():
            if model_key not in self.results:
                continue

            # 拼接所有折的预测结果
            all_probs = np.concatenate(self.results[model_key]['fold_probs'])
            all_y_true = np.concatenate(self.results[model_key]['fold_y_true'])

            # 计算PR曲线
            precision, recall, _ = precision_recall_curve(all_y_true, all_probs)
            ap = np.mean(self.results[model_key]['fold_aps'])

            # 绘制PR曲线
            ax.plot(recall, precision,
                   color=model_config['color'],
                   lw=2,
                   label=f"{model_config['short_name']} (AP={ap:.3f})")

        # 基线
        baseline = np.mean([np.concatenate(self.results[m]['fold_y_true']).mean()
                           for m in self.results.keys()])
        ax.axhline(y=baseline, color='k', linestyle='--', lw=1, label='Baseline')

        ax.set_xlabel('Recall', fontsize=ModelConfig.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=ModelConfig.FONT_SIZE_LABEL, fontweight='bold')
        ax.set_title('Precision-Recall Curve Comparison (10-Fold CV)',
                    fontsize=ModelConfig.FONT_SIZE_TITLE, fontweight='bold')
        ax.legend(fontsize=ModelConfig.FONT_SIZE_LEGEND, loc='lower left')
        ax.grid(True, alpha=0.3)

        # 保存
        for fmt in ModelConfig.SAVE_FORMATS:
            filepath = os.path.join(output_dir, f'pr_comparison.{fmt}')
            plt.savefig(filepath, dpi=ModelConfig.FIGURE_DPI, bbox_inches='tight')
        plt.close()

    def print_feature_importance_analysis(self, output_dir, feature_mode):
        """打印特征重要性分析报告并生成图形"""
        print(f"\n{'='*70}")
        print("【特征重要性分析】")
        print("="*70)

        for model_key in self.results.keys():
            model_name = self.model_configs[model_key]['name']
            short_name = self.model_configs[model_key]['short_name']

            if not self.results[model_key]['fold_feature_importance']:
                continue

            # 使用字典聚合特征重要性（处理不同折选择不同特征的情况）
            feature_importance_dict = {}

            for fold_idx in range(10):
                fold_features = self.results[model_key]['fold_selected_features'][fold_idx]
                fold_importance = self.results[model_key]['fold_feature_importance'][fold_idx]

                # 将该折的每个特征的重要性添加到字典
                for feat, imp in zip(fold_features, fold_importance):
                    if feat not in feature_importance_dict:
                        feature_importance_dict[feat] = []
                    feature_importance_dict[feat].append(imp)

            # 计算每个特征的平均重要性（只在出现该特征的折中平均）
            avg_feature_importance = {
                feat: np.mean(imp_list)
                for feat, imp_list in feature_importance_dict.items()
            }

            # 按重要性排序
            ranked_features = sorted(
                avg_feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # 打印结果
            print(f"\n{model_name} 特征重要性排序 (Top 10):")
            print(f"{'排名':<6s} {'特征':<20s} {'重要性':<10s} {'出现次数':<8s}")
            print("-"*50)
            for rank, (feature, score) in enumerate(ranked_features[:10], 1):
                appear_count = len(feature_importance_dict[feature])
                print(f"{rank:<6d} {feature:<20s} {score:<10.4f} {appear_count:<8d}/10")

            # ========== 绘制特征重要性条形图 ==========
            top_features = ranked_features[:15]  # Top 15
            features, scores = zip(*top_features)
            features = list(features)
            scores = list(scores)

            fig, ax = plt.subplots(figsize=ModelConfig.FIGURE_SIZE_SINGLE)
            bars = ax.barh(features, scores, color='skyblue', edgecolor='black', alpha=0.8)
            ax.invert_yaxis()
            ax.set_xlabel('Importance Score', fontweight='bold', fontsize=12)
            ax.set_title(f'Top 15 Feature Importance - {short_name}',
                        fontweight='bold', fontsize=14, pad=20)
            ax.grid(True, alpha=0.3, axis='x')

            # 添加数值标签
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax.text(score + max(scores)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', va='center', ha='left', fontsize=9)

            plt.tight_layout()

            # 保存到输出目录
            for fmt in ModelConfig.SAVE_FORMATS:
                filepath = os.path.join(output_dir,
                                       f'feature_importance_{short_name}.{fmt}')
                plt.savefig(filepath, dpi=ModelConfig.FIGURE_DPI, bbox_inches='tight')
            plt.close()

            print(f"  [OK] {short_name} 特征重要性图已保存")

    def generate_shap_analysis(self, unified_dir, feature_mode):
        """生成SHAP分析（仅XGBoost）- 只分析一致特征"""
        if 'xgboost' not in self.results or not SHAP_AVAILABLE:
            print("\n[SKIP] SHAP分析未运行（XGBoost未运行或SHAP未安装）")
            return None

        print(f"\n{'='*70}")
        print("【SHAP可解释性分析（XGBoost）】")
        print("="*70)

        # 找出在所有折中都出现的一致特征
        fold_features = self.results['xgboost']['fold_selected_features']
        feature_appear_count = {}

        for features in fold_features:
            for feat in features:
                feature_appear_count[feat] = feature_appear_count.get(feat, 0) + 1

        # 只选择在所有折（10折）中都出现的特征
        consistent_features = [feat for feat, count in feature_appear_count.items()
                              if count == 10]

        if not consistent_features:
            print("\n  [WARNING] 没有在所有折中都出现的特征，无法生成聚合SHAP图")
            print("  [INFO] 将为每一折单独生成SHAP图")

            # 为每一折单独生成SHAP图
            for fold_idx in range(10):
                try:
                    fold_shap_values = self.results['xgboost']['fold_shap_values'][fold_idx]
                    fold_X_test = self.results['xgboost']['fold_X_tests'][fold_idx]
                    fold_features = self.results['xgboost']['fold_selected_features'][fold_idx]

                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(fold_shap_values, fold_X_test,
                                    feature_names=fold_features,
                                    show=False)
                    plt.title(f'SHAP Summary Plot - Fold {fold_idx+1} ({len(fold_features)} features)',
                             fontweight='bold', fontsize=14)
                    plt.tight_layout()

                    for fmt in ModelConfig.SAVE_FORMATS:
                        filepath = os.path.join(unified_dir, f'shap_summary_fold{fold_idx+1}.{fmt}')
                        plt.savefig(filepath, dpi=ModelConfig.FIGURE_DPI, bbox_inches='tight')
                    plt.close()

                    print(f"  [OK] Fold {fold_idx+1} SHAP图已保存 ({len(fold_features)} features)")
                except Exception as e:
                    print(f"  [ERROR] Fold {fold_idx+1} SHAP图生成失败: {str(e)}")
                    continue

            return unified_dir

        # ============================================================
        # 只分析一致特征（10折中100%出现）
        # ============================================================
        print(f"\n【一致特征SHAP分析】")
        print(f"  - 一致特征数量: {len(consistent_features)}/{len(feature_appear_count)}")
        print(f"  - 定义：在所有10折交叉验证中均被LASSO选中的特征")
        print(f"  - 说明：这些特征在不同数据子集上表现一致，适合用于SHAP解释")

        # 收集一致特征的SHAP值
        consistent_shap_list = []
        consistent_X_test_list = []

        for fold_idx in range(10):
            fold_features = self.results['xgboost']['fold_selected_features'][fold_idx]
            fold_shap_values = self.results['xgboost']['fold_shap_values'][fold_idx]
            fold_X_test = self.results['xgboost']['fold_X_tests'][fold_idx]

            # 找到一致特征在该折中的索引
            consistent_indices = [i for i, feat in enumerate(fold_features) if feat in consistent_features]

            # 提取一致特征的SHAP值和数据
            consistent_shap_list.append(fold_shap_values[:, consistent_indices])
            consistent_X_test_list.append(fold_X_test[:, consistent_indices])

        # 合并所有折的一致特征SHAP数据
        all_shap_values = np.concatenate(consistent_shap_list, axis=0)
        all_X_test = np.concatenate(consistent_X_test_list, axis=0)

        print(f"  - 总样本量: {all_X_test.shape[0]}")
        print(f"  - 特征数量: {len(consistent_features)}")

        # 创建SHAP Summary Plot
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(all_shap_values, all_X_test,
                            feature_names=consistent_features,
                            show=False)
            plt.title(f'SHAP Summary Plot (XGBoost - {len(consistent_features)} Consistent Features)',
                     fontweight='bold', fontsize=14)
            plt.tight_layout()

            # 保存到统一目录
            for fmt in ModelConfig.SAVE_FORMATS:
                filepath = os.path.join(unified_dir, f'shap_summary.{fmt}')
                plt.savefig(filepath, dpi=ModelConfig.FIGURE_DPI, bbox_inches='tight')
            plt.close()

            print(f"  [OK] SHAP Summary Plot已保存")
        except Exception as e:
            print(f"  [ERROR] 生成SHAP Summary Plot失败: {str(e)}")

        # 计算SHAP统计信息
        mean_abs_shap = np.abs(all_shap_values).mean(axis=0)
        mean_shap = all_shap_values.mean(axis=0)
        std_shap = all_shap_values.std(axis=0)
        max_shap = all_shap_values.max(axis=0)
        min_shap = all_shap_values.min(axis=0)

        # 创建特征重要性列表
        feature_importance = list(zip(consistent_features, mean_abs_shap, mean_shap, std_shap, max_shap, min_shap))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # 打印Top 10
        print(f"\n【表12：XGBoost一致特征SHAP分析】")
        print(f"XGBoost SHAP特征重要性 (Top 10):")
        print(f"{'排名':<6s} {'特征':<20s} {'Mean|SHAP|':<12s}")
        print("-"*40)
        for rank, (feature, imp_abs, _, _, _, _) in enumerate(feature_importance[:10], 1):
            print(f"{rank:<6d} {feature:<20s} {imp_abs:<12.4f}")

        # 生成表12：完整的SHAP统计表格
        shap_table_data = []
        for rank, (feature, imp_abs, imp_mean, imp_std, imp_max, imp_min) in enumerate(feature_importance, 1):
            shap_table_data.append({
                'Rank': rank,
                'Feature': feature,
                'Appearance_Rate': '10/10 (100%)',  # 明确标注是一致特征
                'Mean_Abs_SHAP': round(imp_abs, 4),
                'Mean_SHAP': round(imp_mean, 4),
                'Std_SHAP': round(imp_std, 4),
                'Max_SHAP': round(imp_max, 4),
                'Min_SHAP': round(imp_min, 4)
            })

        shap_df = pd.DataFrame(shap_table_data)

        # 导出Excel
        timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)
        excel_file = os.path.join(unified_dir, f'Table12_XGBoost_ConsistentFeatures_SHAP_{timestamp}.xlsx')
        shap_df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"  [OK] 表12已保存: {os.path.basename(excel_file)}")

        # 同时导出CSV
        csv_file = os.path.join(unified_dir, f'Table12_XGBoost_ConsistentFeatures_SHAP_{timestamp}.csv')
        shap_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"  [OK] 表12已保存: {os.path.basename(csv_file)}")

        # 打印完整表格信息
        print(f"\n表12: 基于一致特征的XGBoost模型SHAP分析 (共{len(shap_df)}个一致特征)")
        print("="*100)
        print(f"{'Rank':<6s} {'Feature':<20s} {'Appearance':<15s} {'Mean|SHAP|':<12s} {'Mean_SHAP':<12s}")
        print("-"*100)
        for _, row in shap_df.head(10).iterrows():
            print(f"{row['Rank']:<6d} {row['Feature']:<20s} {row['Appearance_Rate']:<15s} "
                  f"{row['Mean_Abs_SHAP']:<12.4f} {row['Mean_SHAP']:<12.4f}")
        print("="*100)

        # 生成特征出现频率补充表（可选）
        print(f"\n【补充：所有特征的出现频率】")
        all_features_sorted = sorted(feature_appear_count.items(),
                                     key=lambda x: x[1], reverse=True)

        freq_table_data = []
        for feat, freq in all_features_sorted:
            freq_table_data.append({
                'Feature': feat,
                'Appearance_Frequency': freq,
                'Appearance_Rate': f'{freq}/10 ({freq*10}%)',
                'Consistency': 'Consistent' if freq == 10 else 'Variable'
            })

        freq_df = pd.DataFrame(freq_table_data)
        freq_file = os.path.join(unified_dir, f'Supplement_Table_Feature_Consistency_{timestamp}.xlsx')
        freq_df.to_excel(freq_file, index=False, engine='openpyxl')
        print(f"  [OK] 特征一致性补充表已保存: {os.path.basename(freq_file)}")
        print(f"      总特征数: {len(all_features_sorted)}")
        print(f"      一致特征数: {len(consistent_features)}")

        return unified_dir


    def create_10fold_confusion_matrix(self, unified_dir, feature_mode):
        """创建XGBoost 10折混淆矩阵拼接图（2×5布局）"""
        if 'xgboost' not in self.results:
            print("\n[SKIP] 10折混淆矩阵图未生成（XGBoost未运行）")
            return None

        print(f"\n{'='*70}")
        print("【生成10折混淆矩阵拼接图（XGBoost）】")
        print("="*70)

        # 收集所有折的混淆矩阵
        fold_cms = self.results['xgboost']['fold_confusion_matrices']
        fold_y_preds = self.results['xgboost']['fold_y_pred']
        fold_y_trues = self.results['xgboost']['fold_y_true']

        # 计算整体混淆矩阵（基于所有折的合并数据）
        all_y_pred = np.concatenate(fold_y_preds)
        all_y_true = np.concatenate(fold_y_trues)
        overall_cm = confusion_matrix(all_y_true, all_y_pred)

        # 从整体混淆矩阵计算性能指标（与Excel表格计算方式一致）
        tn, fp, fn, tp = overall_cm.ravel()
        overall_accuracy = (tp + tn) / overall_cm.sum()
        overall_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        overall_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # 创建2×5子图
        fig, axes = plt.subplots(2, 5, figsize=ModelConfig.FIGURE_SIZE_CONFUSION)
        fig.suptitle('XGBoost 10-Fold Confusion Matrices',
                    fontweight='bold', fontsize=16, y=0.98)

        for fold_idx in range(10):
            ax = axes.flatten()[fold_idx]
            cm = fold_cms[fold_idx]

            # 计算该折的准确率
            fold_tp, fold_tn = cm[1, 1], cm[0, 0]
            fold_accuracy = (fold_tp + fold_tn) / cm.sum()

            # 绘制热图
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.8)
            ax.set_title(f'Fold {fold_idx+1}\nAcc={fold_accuracy:.3f}',
                        fontsize=14, fontweight='bold')

            # 添加数值标签（超大字体）
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, f'{cm[i, j]}',
                                 ha="center", va="center",
                                 color="white" if cm[i, j] > cm.max() / 2 else "black",
                                 fontsize=24, fontweight='bold')

            ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['0', '1'])
            ax.set_yticklabels(['0', '1'])

        # 添加整体性能信息
        fig.text(0.5, 0.02,
                f'Overall Performance (10-Fold CV): Accuracy={overall_accuracy:.3f}, '
                f'Sensitivity={overall_sensitivity:.3f}, Specificity={overall_specificity:.3f}, '
                f'Precision={overall_precision:.3f}',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

        # 调整子图间距（增大行间距避免重叠）
        plt.subplots_adjust(hspace=0.55, wspace=0.3)
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        # 保存到统一目录
        for fmt in ModelConfig.SAVE_FORMATS:
            filepath = os.path.join(unified_dir, f'xgboost_10fold_confusion_matrix.{fmt}')
            plt.savefig(filepath, dpi=ModelConfig.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        print(f"  [OK] 10折混淆矩阵拼接图已保存")
        print(f"  - Overall Accuracy: {overall_accuracy:.3f}")
        print(f"  - Overall Sensitivity: {overall_sensitivity:.3f}")
        print(f"  - Overall Specificity: {overall_specificity:.3f}")
        print(f"  - Overall Precision: {overall_precision:.3f}")

        return unified_dir

    def create_calibration_curves(self, unified_dir, feature_mode):
        """创建校准曲线（类似BMC脚本）"""
        from sklearn.calibration import calibration_curve

        print(f"\n{'='*70}")
        print("【生成校准曲线】")
        print("="*70)

        # 定义颜色和标记样式
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']

        fig, ax = plt.subplots(figsize=ModelConfig.FIGURE_SIZE_SINGLE)

        for idx, model_key in enumerate(self.results.keys()):
            if model_key not in self.results:
                continue

            model_config = self.model_configs[model_key]
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            # 合并所有折的数据
            all_probs = np.concatenate(self.results[model_key]['fold_probs'])
            all_y_true = np.concatenate(self.results[model_key]['fold_y_true'])

            # 计算校准曲线
            prob_true, prob_pred = calibration_curve(
                all_y_true, all_probs, n_bins=10, strategy='quantile'
            )

            # 绘制校准曲线
            ax.plot(prob_pred, prob_true, marker=marker, color=color,
                   lw=2.5, markersize=8, label=model_config['short_name'],
                   alpha=0.8, markeredgecolor='white', markeredgewidth=1)

        # 完美校准线
        ax.plot([0, 1], [0, 1], 'k--', lw=2.5, alpha=0.7, label='Perfect calibration')

        # 设置坐标轴范围
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # 添加参考区域
        ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')

        # 设置标签和标题
        ax.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=12)
        ax.set_title('Calibration Curves', fontweight='bold', fontsize=14, pad=20)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存到统一目录
        for fmt in ModelConfig.SAVE_FORMATS:
            filepath = os.path.join(unified_dir, f'calibration_curves.{fmt}')
            plt.savefig(filepath, dpi=ModelConfig.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        print(f"  [OK] 校准曲线已保存")

        return unified_dir

    def calculate_decision_curve(self, y_true, y_proba):
        """计算决策曲线"""
        thresholds = np.linspace(0, 1, 100)
        net_benefits = []

        for pt in thresholds:
            y_pred_thresh = (np.array(y_proba) >= pt).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            n = len(y_true)
            net_benefit = (tp/n) - (fp/n)*(pt/(1-pt)) if pt != 1 else 0
            net_benefits.append(net_benefit)

        return thresholds, net_benefits

    def create_decision_curves(self, unified_dir, feature_mode):
        """创建决策曲线分析"""
        print(f"\n{'='*70}")
        print("【生成决策曲线】")
        print("="*70)

        fig, ax = plt.subplots(figsize=ModelConfig.FIGURE_SIZE_SINGLE)

        # 定义颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, model_key in enumerate(self.results.keys()):
            if model_key not in self.results:
                continue

            model_config = self.model_configs[model_key]
            color = colors[idx % len(colors)]

            # 合并所有折的数据
            all_probs = np.concatenate(self.results[model_key]['fold_probs'])
            all_y_true = np.concatenate(self.results[model_key]['fold_y_true'])

            # 计算决策曲线
            thresholds, net_benefits = self.calculate_decision_curve(all_y_true, all_probs)

            # 绘制决策曲线
            ax.plot(thresholds, net_benefits, lw=2.5, color=color,
                   label=model_config['short_name'])

        # 添加基准线
        ax.plot([0, 1], [0, 0], 'k--', alpha=0.5, label='Treat None', linewidth=1.5)
        thresholds = np.linspace(0, 1, 100)
        all_treat_benefits = [t - (1-t)*(pt/(1-pt)) if pt != 1 else t
                              for pt, t in zip(thresholds, thresholds)]
        ax.plot(thresholds, all_treat_benefits, 'k:', alpha=0.5,
               label='Treat All', linewidth=1.5)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([-0.1, 0.5])
        ax.set_xlabel('Threshold Probability', fontweight='bold', fontsize=12)
        ax.set_ylabel('Net Benefit', fontweight='bold', fontsize=12)
        ax.set_title('Decision Curve Analysis', fontweight='bold', fontsize=14)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存到统一目录
        for fmt in ModelConfig.SAVE_FORMATS:
            filepath = os.path.join(unified_dir, f'decision_curves.{fmt}')
            plt.savefig(filepath, dpi=ModelConfig.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        print(f"  [OK] 决策曲线已保存")

        return unified_dir

    def create_combined_2x2_figure(self, unified_dir, feature_mode):
        """创建2×2拼接图：(A)ROC + (B)PR + (C)决策曲线 + (D)校准曲线"""
        from sklearn.calibration import calibration_curve

        print(f"\n{'='*70}")
        print("【生成2×2拼接图（ROC+PR+DCA+Calibration）】")
        print("="*70)

        # 创建2×2子图
        fig, axes = plt.subplots(2, 2, figsize=ModelConfig.FIGURE_SIZE_COMBINED)
        fig.suptitle(f'Model Performance Analysis - {feature_mode.upper()}',
                    fontweight='bold', fontsize=16, y=0.98)

        # 统一样式
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        linewidth = 2.5

        # ========== (A) ROC曲线 ==========
        ax_roc = axes[0, 0]
        for idx, (model_key, model_config) in enumerate(self.model_configs.items()):
            if model_key not in self.results:
                continue

            color = colors[idx % len(colors)]

            # 合并所有折的数据
            all_probs = np.concatenate(self.results[model_key]['fold_probs'])
            all_y_true = np.concatenate(self.results[model_key]['fold_y_true'])

            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(all_y_true, all_probs)
            auc = np.mean(self.results[model_key]['fold_aucs'])

            ax_roc.plot(fpr, tpr, lw=linewidth, color=color,
                      label=f"{model_config['short_name']} (AUC={auc:.3f})")

        ax_roc.plot([0, 1], [0, 1], 'gray', linestyle='--', alpha=0.8, lw=2)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
        ax_roc.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
        ax_roc.set_title('ROC Curves', fontweight='bold', fontsize=12)
        ax_roc.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax_roc.grid(True, alpha=0.3)

        # 添加A标记
        ax_roc.text(0.02, 0.98, 'A', transform=ax_roc.transAxes, fontsize=18,
                   fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', alpha=0.9),
                   zorder=10)

        # ========== (B) PR曲线 ==========
        ax_pr = axes[0, 1]
        for idx, (model_key, model_config) in enumerate(self.model_configs.items()):
            if model_key not in self.results:
                continue

            color = colors[idx % len(colors)]

            # 合并所有折的数据
            all_probs = np.concatenate(self.results[model_key]['fold_probs'])
            all_y_true = np.concatenate(self.results[model_key]['fold_y_true'])

            # 计算PR曲线
            precision, recall, _ = precision_recall_curve(all_y_true, all_probs)
            ap = np.mean(self.results[model_key]['fold_aps'])

            ax_pr.plot(recall, precision, lw=linewidth, color=color,
                      label=f"{model_config['short_name']} (AP={ap:.3f})")

        # 基线
        baseline = np.mean([np.concatenate(self.results[m]['fold_y_true']).mean()
                           for m in self.results.keys()])
        ax_pr.axhline(y=baseline, color='k', linestyle='--', lw=1, label='Baseline')

        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Recall', fontweight='bold', fontsize=11)
        ax_pr.set_ylabel('Precision', fontweight='bold', fontsize=11)
        ax_pr.set_title('Precision-Recall Curves', fontweight='bold', fontsize=12)
        ax_pr.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax_pr.grid(True, alpha=0.3)

        # 添加B标记
        ax_pr.text(0.02, 0.98, 'B', transform=ax_pr.transAxes, fontsize=18,
                   fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', alpha=0.9),
                   zorder=10)

        # ========== (C) 决策曲线 ==========
        ax_dc = axes[1, 0]
        for idx, (model_key, model_config) in enumerate(self.model_configs.items()):
            if model_key not in self.results:
                continue

            color = colors[idx % len(colors)]

            # 合并所有折的数据
            all_probs = np.concatenate(self.results[model_key]['fold_probs'])
            all_y_true = np.concatenate(self.results[model_key]['fold_y_true'])

            # 计算决策曲线
            thresholds, net_benefits = self.calculate_decision_curve(all_y_true, all_probs)

            ax_dc.plot(thresholds, net_benefits, lw=linewidth, color=color,
                      label=model_config['short_name'])

        # 添加基准线
        ax_dc.plot([0, 1], [0, 0], 'k--', alpha=0.5, label='Treat None', linewidth=1.5)
        thresholds = np.linspace(0, 1, 100)
        all_treat_benefits = [t - (1-t)*(pt/(1-pt)) if pt != 1 else t
                              for pt, t in zip(thresholds, thresholds)]
        ax_dc.plot(thresholds, all_treat_benefits, 'k:', alpha=0.5,
                  label='Treat All', linewidth=1.5)

        ax_dc.set_xlim([0.0, 1.0])
        ax_dc.set_ylim([-0.1, 0.5])
        ax_dc.set_xlabel('Threshold Probability', fontweight='bold', fontsize=11)
        ax_dc.set_ylabel('Net Benefit', fontweight='bold', fontsize=11)
        ax_dc.set_title('Decision Curve Analysis', fontweight='bold', fontsize=12)
        ax_dc.legend(loc='upper right', fontsize=7, framealpha=0.9)
        ax_dc.grid(True, alpha=0.3)

        # 添加C标记
        ax_dc.text(0.02, 0.98, 'C', transform=ax_dc.transAxes, fontsize=18,
                   fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', alpha=0.9),
                   zorder=10)

        # ========== (D) 校准曲线 ==========
        ax_cal = axes[1, 1]
        for idx, (model_key, model_config) in enumerate(self.model_configs.items()):
            if model_key not in self.results:
                continue

            color = colors[idx % len(colors)]

            # 合并所有折的数据
            all_probs = np.concatenate(self.results[model_key]['fold_probs'])
            all_y_true = np.concatenate(self.results[model_key]['fold_y_true'])

            # 计算校准曲线
            prob_true, prob_pred = calibration_curve(
                all_y_true, all_probs, n_bins=10, strategy='quantile'
            )

            ax_cal.plot(prob_pred, prob_true, 's-', lw=linewidth, color=color,
                       label=model_config['short_name'], markersize=5, alpha=0.8)

        ax_cal.plot([0, 1], [0, 1], 'k--', lw=2.5, alpha=0.7, label='Perfect calibration')
        ax_cal.set_xlim([0.0, 1.0])
        ax_cal.set_ylim([0.0, 1.05])
        ax_cal.set_xlabel('Mean Predicted Probability', fontweight='bold', fontsize=11)
        ax_cal.set_ylabel('Fraction of Positives', fontweight='bold', fontsize=11)
        ax_cal.set_title('Calibration Curves', fontweight='bold', fontsize=12)
        ax_cal.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax_cal.grid(True, alpha=0.3)

        # 添加D标记
        ax_cal.text(0.02, 0.98, 'D', transform=ax_cal.transAxes, fontsize=18,
                   fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', alpha=0.9),
                   zorder=10)

        # 调整子图间距
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # 保存到统一目录
        for fmt in ModelConfig.SAVE_FORMATS:
            filepath = os.path.join(unified_dir, f'combined_analysis_ABCD.{fmt}')
            plt.savefig(filepath, dpi=ModelConfig.FIGURE_DPI, bbox_inches='tight')
        plt.close()

        print(f"  [OK] 2×2拼接图已保存")

        return unified_dir

    def generate_interpretability_plots(self, modeler_tongue, modeler_pulse, output_dir):
        """
        生成主文级解释性图表

        生成两张图：
        1. 多模态互补总结图（性能森林图 + 模态贡献分解图）
        2. SHAP解释性组合图（Top3 dependence plots + waterfall plot）

        Parameters:
        -----------
        modeler_tongue, modeler_pulse: IntegratedModelingV2实例
            舌-only和脉-only的建模器对象
        output_dir: str
            输出目录
        """
        from modules.interpretability_plots import (
            generate_modality_complementarity_plot,
            generate_shap_interpretability_combo
        )

        print(f"\n{'='*70}")
        print("【生成主文级解释性图表】")
        print("="*70)

        # 检查必需的缓存数据
        self._validate_shap_cache()

        # 生成图1：多模态互补总结图
        print("\n[图1] 多模态互补总结图")
        generate_modality_complementarity_plot(
            modeler_tongue,
            modeler_pulse,
            self,  # fusion modeler
            output_dir
        )

        # 生成图2：SHAP解释性组合图
        print("\n[图2] SHAP解释性组合图")
        generate_shap_interpretability_combo(
            self,  # fusion modeler only
            output_dir
        )

        print(f"\n{'='*70}")
        print("[完成] 主文级解释性图表已生成")
        print("="*70)

    def _validate_shap_cache(self):
        """
        验证SHAP缓存数据的完整性

        Raises:
        --------
        ValueError: 如果必需的缓存数据缺失
        """
        if 'xgboost' not in self.results:
            raise ValueError("XGBoost模型未运行，无法生成解释性图表")

        required_keys = ['fold_shap_values', 'fold_X_tests', 'fold_selected_features',
                        'fold_y_true', 'fold_probs']

        for key in required_keys:
            if key not in self.results['xgboost']:
                raise ValueError(f"缺少必需的缓存数据: {key}")

        # 检查consistent features数量
        fold_features = self.results['xgboost']['fold_selected_features']
        feature_appear_count = {}
        for features in fold_features:
            for feat in features:
                feature_appear_count[feat] = feature_appear_count.get(feat, 0) + 1

        n_consistent = sum(1 for count in feature_appear_count.values() if count == 10)

        if n_consistent < 3:
            raise ValueError(f"Consistent features数量不足（当前{n_consistent}个，需要≥3个），"
                            "无法生成Top3 dependence plots")

        print(f"  [OK] SHAP缓存数据验证通过（{n_consistent}个consistent features）")


# =============================================================================
# 便捷函数
# =============================================================================

def run_integrated_modeling_v2(data_loader, feature_mode='combined', models_to_run='all'):
    """运行整合建模分析"""
    X, y, feature_names = data_loader.prepare_modeling_data(feature_mode=feature_mode)

    modeler = IntegratedModelingV2(
        feature_mode=feature_mode,
        models_to_run=models_to_run
    )
    modeler.load_data(X, y, feature_names)

    modeler.run_cross_validation()

    # 创建统一的输出目录（简单命名）
    timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)
    unified_dir = os.path.join(
        config.OUTPUT_BASE_DIR,
        'modeling_results_v2',
        f'{feature_mode}_results_{timestamp}'
    )
    os.makedirs(unified_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"输出目录: {unified_dir}")
    print('='*70)

    # 生成可视化和分析（所有文件都保存到统一目录）
    modeler.generate_comparison_plots(unified_dir)
    modeler.print_feature_importance_analysis(unified_dir, feature_mode)
    modeler.generate_shap_analysis(unified_dir, feature_mode)
    modeler.create_10fold_confusion_matrix(unified_dir, feature_mode)
    modeler.create_calibration_curves(unified_dir, feature_mode)
    modeler.create_decision_curves(unified_dir, feature_mode)
    modeler.create_combined_2x2_figure(unified_dir, feature_mode)
    modeler.generate_feature_frequency_table(unified_dir, feature_mode)  # 新增：特征频率表
    modeler.save_results_to_excel(unified_dir, feature_mode)

    print(f"\n{'='*70}")
    print(f"完成！所有结果已保存到: {unified_dir}")
    print('='*70)

    return modeler


# =============================================================================
# 主程序测试
# =============================================================================

if __name__ == "__main__":
    print("\n整合建模系统 v2.0")
    print("功能: 智能调参 + LASSO特征筛选 + Bootstrap置信区间 + 校准指标\n")
    print("请通过main_v2.py运行完整分析")
