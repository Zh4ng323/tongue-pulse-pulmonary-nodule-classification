# -*- coding: utf-8 -*-
"""
Data loading and quality checking module.
Supports CSV and Excel formats with automatic feature detection.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
import sys

# 添加父目录到路径以导入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings('ignore')


class DataLoader:
    """
    数据加载器类

    功能：
    - 加载Excel数据
    - 基本数据检查
    - 自动特征分组
    - 数据统计摘要
    """

    def __init__(self, data_path=None, target_col=None):
        """
        初始化数据加载器

        Parameters:
        -----------
        data_path : str, optional
            数据文件路径，默认使用config中的配置
        target_col : str, optional
            目标列名，默认使用config中的配置
        """
        self.data_path = data_path or config.RAW_DATA_PATH
        self.target_col = target_col or config.TARGET_COLUMN

        self.data = None
        self.X = None  # 特征
        self.y = None  # 标签

        self.feature_groups = {
            'tongue': [],      # 舌象特征
            'pulse': [],       # 脉象特征
            'other': []        # 其他特征
        }

        self.groups = {
            'benign': None,    # 良性结节组（Group=0）
            'cancer': None     # 肺癌组（Group=1）
        }

    def load_data(self, sheet_name="Sheet1"):
        """
        加载数据文件（支持 .csv / .xlsx / .xls）

        Parameters:
        -----------
        sheet_name : str, default="Sheet1"
            Excel工作表名称（仅对xlsx/xls格式有效）

        Returns:
        --------
        self : DataLoader
            返回自身，支持链式调用
        """
        print("\n" + "="*60)
        print("【模块1】数据加载与基本检查")
        print("="*60)

        try:
            print(f"\n正在读取文件: {self.data_path}")
            file_ext = os.path.splitext(self.data_path)[1].lower()
            if file_ext == '.csv':
                self.data = pd.read_csv(self.data_path)
            elif file_ext in ('.xlsx', '.xls'):
                self.data = pd.read_excel(self.data_path, sheet_name=sheet_name)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}，支持 .csv/.xlsx/.xls")
            print(f"[OK] 数据加载成功")
            print(f"  - 样本数: {self.data.shape[0]}")
            print(f"  - 总列数: {self.data.shape[1]}")

            # 检查目标列是否存在
            if self.target_col not in self.data.columns:
                raise ValueError(f"目标列 '{self.target_col}' 不存在于数据中")

            return self

        except Exception as e:
            print(f"[X] 数据加载失败: {str(e)}")
            raise

    def check_data_quality(self):
        """
        数据质量检查

        检查项：
        - 缺失值统计
        - 数据类型
        - 目标变量分布
        - 重复样本

        Returns:
        --------
        self : DataLoader
        """
        print("\n" + "-"*60)
        print("数据质量检查")
        print("-"*60)

        # 1. 缺失值检查
        print("\n【1】缺失值统计")
        missing_stats = self.data.isnull().sum()
        total_missing = missing_stats.sum()

        if total_missing == 0:
            print("  [OK] 无缺失值")
        else:
            missing_cols = missing_stats[missing_stats > 0]
            print(f"  [WARNING] 发现 {total_missing} 个缺失值")
            print("\n  缺失值详情:")
            for col, count in missing_cols.items():
                pct = count / len(self.data) * 100
                print(f"    - {col}: {count} ({pct:.2f}%)")

            print("\n  Samples with missing values will be excluded in subsequent analyses.")

        # 2. 数据类型检查
        print("\n【2】数据类型统计")
        dtype_counts = self.data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  - {dtype}: {count}列")

        # 3. 目标变量分布
        print("\n【3】目标变量分布")
        self.y = self.data[self.target_col]
        value_counts = self.y.value_counts().sort_index()

        print(f"  目标列: {self.target_col}")
        for value, count in value_counts.items():
            label = config.TARGET_LABELS.get(value, f"类别{value}")
            pct = count / len(self.y) * 100
            print(f"  - {label} (Group={value}): {count} ({pct:.2f}%)")

        # 4. 重复样本检查
        print("\n【4】重复样本检查")
        n_duplicates = self.data.duplicated().sum()
        if n_duplicates == 0:
            print("  [OK] 无重复样本")
        else:
            print(f"  [WARNING] 发现 {n_duplicates} 个重复样本")
            print("  Please verify data entry.")

        # 5. 基本统计信息
        print("\n【5】数值型特征统计")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        print(f"  数值型特征数: {len(numeric_cols)}")

        if len(numeric_cols) > 0:
            print("\n  基本统计量（前5个特征）:")
            stats = self.data[numeric_cols[:5]].describe()
            print(stats.round(3).to_string())

        return self

    def auto_detect_features(self):
        """
        自动识别舌/脉特征

        识别规则：
        - 舌象特征：精确匹配config.TONGUE_FEATURES列表
        - 脉象特征：精确匹配config.PULSE_FEATURES列表
        - 其他特征：既不在舌象也不在脉象列表中的特征

        Returns:
        --------
        self : DataLoader
        """
        print("\n" + "-"*60)
        print("特征自动识别")
        print("-"*60)

        # 获取所有特征列（排除目标列）
        all_cols = self.data.columns.tolist()
        feature_cols = [col for col in all_cols if col != self.target_col]

        # 转换为小写进行匹配（处理大小写不一致）
        tongue_features_lower = [f.lower() for f in config.TONGUE_FEATURES]
        pulse_features_lower = [f.lower() for f in config.PULSE_FEATURES]

        # 识别舌象特征（精确匹配）
        tongue_features = []
        for col in feature_cols:
            if col.lower() in tongue_features_lower:
                tongue_features.append(col)

        # 识别脉象特征（精确匹配）
        pulse_features = []
        for col in feature_cols:
            if col.lower() in pulse_features_lower:
                pulse_features.append(col)

        # 其他特征
        other_features = [col for col in feature_cols
                        if col not in tongue_features and col not in pulse_features]

        # 保存结果
        self.feature_groups['tongue'] = tongue_features
        self.feature_groups['pulse'] = pulse_features
        self.feature_groups['other'] = other_features

        # 输出结果
        print(f"\n[OK] 特征识别完成")
        print(f"  - 舌象特征: {len(tongue_features)}个")
        print(f"  - 脉象特征: {len(pulse_features)}个")
        print(f"  - 其他特征: {len(other_features)}个")
        print(f"  - 总计: {len(feature_cols)}个")

        # 显示所有识别的特征
        if len(tongue_features) > 0:
            print(f"\n  舌象特征列表:")
            for i, feat in enumerate(tongue_features, 1):
                print(f"    {i}. {feat}")

        if len(pulse_features) > 0:
            print(f"\n  脉象特征列表:")
            for i, feat in enumerate(pulse_features, 1):
                print(f"    {i}. {feat}")

        if len(other_features) > 0:
            print(f"\n  其他特征: {other_features}")

        return self

    def split_by_group(self):
        """
        按Group列分组数据

        Returns:
        --------
        self : DataLoader
        """
        print("\n" + "-"*60)
        print("数据分组")
        print("-"*60)

        # 良性结节组
        self.groups['benign'] = self.data[self.data[self.target_col] == 0].copy()
        print(f"\n[OK] 良性结节组 (Group=0): {len(self.groups['benign'])} 样本")

        # 肺癌组
        self.groups['cancer'] = self.data[self.data[self.target_col] == 1].copy()
        print(f"[OK] 肺癌组 (Group=1): {len(self.groups['cancer'])} 样本")

        # 检查分组
        total_samples = len(self.groups['benign']) + len(self.groups['cancer'])
        if total_samples != len(self.data):
            print(f"\n[WARNING] 警告: 分组样本数 ({total_samples}) 与总样本数 ({len(self.data)}) 不一致")

        return self

    def prepare_modeling_data(self, feature_mode='combined'):
        """
        准备建模数据

        Parameters:
        -----------
        feature_mode : str, default='combined'
            特征模式
            - 'tongue': 仅舌象特征
            - 'pulse': 仅脉象特征
            - 'combined': 舌+脉联合特征

        Returns:
        --------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            标签向量
        selected_features : list
            选中的特征列表
        """
        print("\n" + "-"*60)
        print(f"准备建模数据 (模式: {feature_mode})")
        print("-"*60)

        # 根据模式选择特征
        if feature_mode == 'tongue':
            selected_features = self.feature_groups['tongue']
        elif feature_mode == 'pulse':
            selected_features = self.feature_groups['pulse']
        elif feature_mode == 'combined':
            selected_features = (self.feature_groups['tongue'] +
                               self.feature_groups['pulse'])
        else:
            raise ValueError(f"未知的特征模式: {feature_mode}")

        # 检查是否有特征被选中
        if len(selected_features) == 0:
            raise ValueError(f"模式 '{feature_mode}' 下没有可用特征")

        # 准备数据
        self.X = self.data[selected_features].copy()
        self.y = self.data[self.target_col]

        print(f"\n[OK] 数据准备完成")
        print(f"  - 样本数: {len(self.X)}")
        print(f"  - 特征数: {len(selected_features)}")
        print(f"  - 正类比例: {self.y.mean():.3f}")

        return self.X, self.y, selected_features

    def get_summary(self):
        """
        获取数据摘要

        Returns:
        --------
        summary : dict
            数据摘要字典
        """
        summary = {
            'n_samples': len(self.data),
            'n_features': len(self.data.columns) - 1,  # 减去目标列
            'n_tongue_features': len(self.feature_groups['tongue']),
            'n_pulse_features': len(self.feature_groups['pulse']),
            'n_other_features': len(self.feature_groups['other']),
            'n_benign': len(self.groups['benign']) if self.groups['benign'] is not None else 0,
            'n_cancer': len(self.groups['cancer']) if self.groups['cancer'] is not None else 0,
            'target_col': self.target_col,
            'data_path': self.data_path
        }

        return summary

    def save_summary_report(self, output_dir=None):
        """
        保存数据摘要报告

        Parameters:
        -----------
        output_dir : str, optional
            输出目录，默认使用config中的配置
        """
        if output_dir is None:
            output_dir = config.OUTPUT_BASE_DIR

        timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)
        report_file = os.path.join(output_dir, f"data_summary_report_{timestamp}.txt")

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("数据摘要报告\n")
                f.write("="*60 + "\n\n")

                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据路径: {self.data_path}\n")
                f.write(f"目标列名: {self.target_col}\n\n")

                summary = self.get_summary()

                f.write("-"*60 + "\n")
                f.write("基本信息\n")
                f.write("-"*60 + "\n")
                f.write(f"总样本数: {summary['n_samples']}\n")
                f.write(f"总特征数: {summary['n_features']}\n")
                f.write(f"舌象特征: {summary['n_tongue_features']}个\n")
                f.write(f"脉象特征: {summary['n_pulse_features']}个\n")
                f.write(f"其他特征: {summary['n_other_features']}个\n\n")

                f.write("-"*60 + "\n")
                f.write("分组信息\n")
                f.write("-"*60 + "\n")
                f.write(f"良性结节组: {summary['n_benign']}例\n")
                f.write(f"肺癌组: {summary['n_cancer']}例\n\n")

                f.write("-"*60 + "\n")
                f.write("特征列表\n")
                f.write("-"*60 + "\n")

                f.write("\n【舌象特征】\n")
                for i, feat in enumerate(self.feature_groups['tongue'], 1):
                    f.write(f"  {i}. {feat}\n")

                f.write("\n【脉象特征】\n")
                for i, feat in enumerate(self.feature_groups['pulse'], 1):
                    f.write(f"  {i}. {feat}\n")

                if len(self.feature_groups['other']) > 0:
                    f.write("\n【其他特征】\n")
                    for i, feat in enumerate(self.feature_groups['other'], 1):
                        f.write(f"  {i}. {feat}\n")

                f.write("\n" + "="*60 + "\n")
                f.write("报告结束\n")
                f.write("="*60 + "\n")

            print(f"\n[OK] 数据摘要报告已保存: {report_file}")

        except Exception as e:
            print(f"\n[X] 保存报告失败: {str(e)}")


# =============================================================================
# 便捷函数
# =============================================================================

def load_and_check(data_path=None, target_col=None):
    """
    一键加载和检查数据

    Parameters:
    -----------
    data_path : str, optional
        数据文件路径
    target_col : str, optional
        目标列名

    Returns:
    --------
    loader : DataLoader
        数据加载器对象
    """
    loader = DataLoader(data_path, target_col)
    loader.load_data()
    loader.check_data_quality()
    loader.auto_detect_features()
    loader.split_by_group()
    loader.save_summary_report()

    return loader


# =============================================================================
# 主程序测试
# =============================================================================

if __name__ == "__main__":
    # 测试数据加载功能
    print("\n" + "="*60)
    print("模块1测试：数据加载与检查")
    print("="*60)

    try:
        loader = load_and_check()

        # 测试准备建模数据
        print("\n" + "="*60)
        print("测试准备建模数据")
        print("="*60)

        X, y, features = loader.prepare_modeling_data(feature_mode='combined')
        print(f"\n[OK] 测试完成")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  特征数: {len(features)}")

    except Exception as e:
        print(f"\n[X] 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
