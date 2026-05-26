# -*- coding: utf-8 -*-
"""
Cross-modal correlation network visualization.

Generates bipartite network graphs for tongue-pulse correlations
and differential network comparisons between groups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from datetime import datetime
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# =============================================================================
# 工具函数：Fisher r-to-z 转换
# =============================================================================

def fisher_r_to_z(r):
    """
    Fisher r-to-z 转换

    Parameters:
    -----------
    r : float or ndarray
        相关系数

    Returns:
    --------
    float or ndarray
        z值 (arctanh(r))
    """
    # Clip r to avoid infinite values
    r_clipped = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r_clipped)


def compute_fisher_z_test(r1, r2, n1, n2):
    """
    Fisher z-test: 检验两个相关系数是否有显著差异

    Parameters:
    -----------
    r1, r2 : float
        两组相关系数
    n1, n2 : int
        两组样本量

    Returns:
    --------
    Z_stat : float
        Z统计量
    p_value : float
        双侧P值
    se : float
        标准误
    """
    # Fisher z转换
    z1 = fisher_r_to_z(r1)
    z2 = fisher_r_to_z(r2)

    # 标准误
    se = np.sqrt(1/(n1-3) + 1/(n2-3))

    # Z统计量
    Z_stat = (z1 - z2) / se

    # 双侧P值
    p_value = 2 * (1 - stats.norm.cdf(abs(Z_stat)))

    return Z_stat, p_value, se


# =============================================================================
# 布局函数：固定双部布局
# =============================================================================

def build_bipartite_pos(tongue_nodes, pulse_nodes, order='config'):
    """
    构建固定双部布局 (bipartite layout)

    舌象节点固定在左侧 (x=0), 脉象节点固定在右侧 (x=1)
    y坐标按固定顺序排列,保证所有网络图节点位置一致

    Parameters:
    -----------
    tongue_nodes : list
        舌象特征节点列表
    pulse_nodes : list
        脉象特征节点列表
    order : str, default='config'
        y坐标排序方式:
        - 'config': 按config.TONGUE_FEATURES/PULSE_FEATURES的原始顺序
        - 'sorted': 按特征名称字母顺序排序
        - 'appearance': 按当前列表顺序(不做排序)

    Returns:
    --------
    pos : dict
        节点位置字典 {node: (x, y)}
    """
    # 确定y坐标顺序
    if order == 'config':
        # 按配置文件的顺序
        tongue_order = [f for f in config.TONGUE_FEATURES if f in tongue_nodes]
        pulse_order = [f for f in config.PULSE_FEATURES if f in pulse_nodes]
    elif order == 'sorted':
        # 按字母顺序
        tongue_order = sorted(tongue_nodes)
        pulse_order = sorted(pulse_nodes)
    else:  # 'appearance'
        # 按出现顺序
        tongue_order = list(tongue_nodes)
        pulse_order = list(pulse_nodes)

    # 构建位置字典
    pos = {}

    # 舌象节点: x=0, y均匀分布
    n_tongue = len(tongue_order)
    for i, node in enumerate(tongue_order):
        y = i / max(1, n_tongue - 1)  # 归一化到[0, 1]
        pos[node] = (0, y)

    # 脉象节点: x=1, y均匀分布
    n_pulse = len(pulse_order)
    for i, node in enumerate(pulse_order):
        y = i / max(1, n_pulse - 1)  # 归一化到[0, 1]
        pos[node] = (1, y)

    return pos


# =============================================================================
# 核心函数1: 跨模态相关网络图
# =============================================================================

def plot_crossmodal_network(corr_matrix, q_matrix, tongue_features, pulse_features,
                            group_name, output_path,
                            r_thr=None, q_thr=None,
                            figsize=(14, 10)):
    """
    绘制跨模态相关网络图 (Fig9-NET)

    双部网络: 左侧=舌象特征, 右侧=脉象特征
    边: q < q_thr 且 |r| >= r_thr 的舌×脉相关
    边颜色: 正相关=Teal, 负相关=Coral
    边粗: 与 |r| 成正比
    节点大小: 与度成正比

    阈值说明（双层体系）：
    - r_thr 默认使用 config.BACKBONE_R_THR (0.25)
      这是"骨架阈值"，展示核心结构
    - q_thr 默认使用 config.Q_THR (0.05)
      所有图表统一的显著性阈值

    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        舌×脉相关系数矩阵 (rows=舌, cols=脉)
    q_matrix : pd.DataFrame
        舌×脉q值矩阵 (FDR校正后)
    tongue_features : list
        舌象特征列表
    pulse_features : list
        脉象特征列表
    group_name : str
        组名 ('Benign' 或 'Cancer')
    output_path : str
        输出路径 (不含扩展名)
    r_thr : float, optional
        相关系数阈值, 默认使用config.BACKBONE_R_THR (0.25)
    q_thr : float, optional
        q值阈值, 默认使用config.Q_THR (0.05)
    figsize : tuple, default=(14, 10)
        图形尺寸
    """
    # 使用默认阈值（骨架阈值）
    if r_thr is None:
        r_thr = config.BACKBONE_R_THR
    if q_thr is None:
        q_thr = config.Q_THR

    print(f"\n[Fig9-NET] 正在生成 {group_name} 组跨模态网络图...")
    print(f"  骨架阈值: |r|>={r_thr}, q<{q_thr}")
    print(f"  说明: 展示核心耦合关系（严谨筛选）")

    # 创建空图
    G = nx.Graph()

    # 添加节点 (带类型属性)
    for feat in tongue_features:
        G.add_node(feat, bipartite=0, node_type='tongue')

    for feat in pulse_features:
        G.add_node(feat, bipartite=1, node_type='pulse')

    # 筛选并添加边
    edges_data = []
    for tongue_feat in tongue_features:
        if tongue_feat not in corr_matrix.index:
            continue

        for pulse_feat in pulse_features:
            if pulse_feat not in corr_matrix.columns:
                continue

            r = corr_matrix.loc[tongue_feat, pulse_feat]
            q = q_matrix.loc[tongue_feat, pulse_feat]

            # 跳过NaN值
            if pd.isna(r) or pd.isna(q):
                continue

            # 筛选条件: q < q_thr 且 |r| >= r_thr
            if q < q_thr and abs(r) >= r_thr:
                edges_data.append({
                    'tongue': tongue_feat,
                    'pulse': pulse_feat,
                    'r': r,
                    'q': q
                })

                # 添加边
                G.add_edge(tongue_feat, pulse_feat,
                          weight=abs(r),
                          sign=1 if r > 0 else -1,
                          r_value=r)

    # 计算节点度数 (用于节点大小)
    degrees = dict(G.degree())

    # 构建固定布局
    pos = build_bipartite_pos(tongue_features, pulse_features, order='config')

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 如果没有边,显示提示信息
    if len(edges_data) == 0:
        ax.text(0.5, 0.5, f'No edges pass thresholds\n(|r|>={r_thr}, q<{q_thr})',
                ha='center', va='center', fontsize=16,
                transform=ax.transAxes, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'Cross-Modal Correlation Network - {group_name}\n(No significant edges)',
                     fontsize=16, fontweight='bold')
    else:
        # ========== 绘制边 ==========
        for edge in edges_data:
            tongue = edge['tongue']
            pulse = edge['pulse']
            r = edge['r']

            # 边颜色
            edge_color = config.NET_POSITIVE_COLOR if r > 0 else config.NET_NEGATIVE_COLOR

            # 边宽 (与 |r| 成正比)
            edge_width = config.NET_EDGE_MIN_WIDTH + \
                        (abs(r) - r_thr) / (1 - r_thr) * (config.NET_EDGE_MAX_WIDTH - config.NET_EDGE_MIN_WIDTH)
            edge_width = np.clip(edge_width, config.NET_EDGE_MIN_WIDTH, config.NET_EDGE_MAX_WIDTH)

            # 绘制边
            x0, y0 = pos[tongue]
            x1, y1 = pos[pulse]
            ax.plot([x0, x1], [y0, y1],
                   color=edge_color, linewidth=edge_width, alpha=0.6, zorder=1)

        # ========== 绘制节点 ==========
        # 节点大小 (与度成正比)
        max_degree = max(degrees.values()) if degrees else 1
        for node, degree in degrees.items():
            if degree == 0:
                continue

            # 归一化度数并映射到节点大小
            normalized_degree = degree / max_degree
            node_size = config.NET_NODE_MIN_SIZE + \
                       normalized_degree * (config.NET_NODE_MAX_SIZE - config.NET_NODE_MIN_SIZE)

            # 节点颜色
            node_color = config.NET_TONGUE_NODE_COLOR if G.nodes[node]['node_type'] == 'tongue' else config.NET_PULSE_NODE_COLOR

            # 绘制节点
            x, y = pos[node]
            ax.scatter(x, y, s=node_size, c=node_color, alpha=0.8,
                      edgecolors='black', linewidth=1.5, zorder=2)

            # 添加节点标签 (所有有连接的节点都要标签)
            # 使用固定左右偏移避免重叠：舌象标签在左侧，脉象标签在右侧
            node_type = G.nodes[node]['node_type']
            if node_type == 'tongue':
                # 舌象节点：标签在节点左侧
                label_x = x - 0.05
                ha = 'right'
            else:
                # 脉象节点：标签在节点右侧
                label_x = x + 0.05
                ha = 'left'

            ax.text(label_x, y, node, fontsize=9, ha=ha, va='center',
                   fontweight='bold', zorder=3, color='black')

        # ========== 添加图例 ==========
        # 正相关边
        ax.plot([], [], color=config.NET_POSITIVE_COLOR, linewidth=3, label='Positive (r>0)')
        # 负相关边
        ax.plot([], [], color=config.NET_NEGATIVE_COLOR, linewidth=3, label='Negative (r<0)')
        # 舌象节点
        ax.scatter([], [], s=200, c=config.NET_TONGUE_NODE_COLOR, edgecolors='black', linewidth=1.5, label='Tongue')
        # 脉象节点
        ax.scatter([], [], s=200, c=config.NET_PULSE_NODE_COLOR, edgecolors='black', linewidth=1.5, label='Pulse')

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9, bbox_to_anchor=(1.18, 1.0))

        # 设置坐标轴（扩展左侧以容纳舌象标签，右侧以容纳图例）
        ax.set_xlim(-0.25, 1.15)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Tongue Features', 'Pulse Features'], fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 标题
        edge_count = len(edges_data)
        node_count = sum(1 for d in degrees.values() if d > 0)

        # 如果边数很少，添加注释
        if edge_count <= 5:
            title_note = f'\nNote: Only {edge_count} edge(s) passed the stringent thresholds (|r|>={r_thr}, q<{q_thr}).'
        else:
            title_note = ''

        ax.set_title(f'Cross-Modal Correlation Network - {group_name}\n'
                    f'({node_count} nodes, {edge_count} edges, |r|>={r_thr}, q<{q_thr}){title_note}',
                    fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    # 保存 (PNG + SVG)
    for fmt in ['png', 'svg']:
        filepath = f'{output_path}.{fmt}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] 已保存: {os.path.basename(filepath)}")

    plt.close()

    return G, edges_data


# =============================================================================
# 核心函数2: 差异网络图
# =============================================================================

def plot_differential_network(benign_corr, cancer_corr,
                             benign_q, cancer_q,
                             tongue_features, pulse_features,
                             n_benign, n_cancer,
                             output_path,
                             dr_thr=None, q_diff_thr=None,
                             require_maxr=None,
                             figsize=(14, 10)):
    """
    绘制差异网络图 (Fig9-DIFF)

    只保留经Fisher z-test检验显著的边
    筛选条件（需同时满足）：
    1. q_diff < q_diff_thr（差异统计显著）
    2. |Δr| >= dr_thr（差异量足够大）
    3. max(|r_benign|, |r_cancer|) >= require_maxr（两边不能都太弱）

    边颜色: Δr>0 (Cancer相关更强) = Teal, Δr<0 = Coral
    边粗: 与 |Δr| 成正比
    节点布局: 与Fig9-NET完全一致 (保证可比性)

    阈值说明（双层体系）：
    - dr_thr 默认使用 config.DIFF_DR_THR (0.20)
      差异量阈值
    - q_diff_thr 默认使用 config.DIFF_Q_THR (0.05)
      差异显著性阈值
    - require_maxr 默认使用 config.DIFF_REQUIRE_MAXR (0.25)
      保护机制：避免"两个都很弱但差异刚好0.2"的尴尬边

    Parameters:
    -----------
    benign_corr, cancer_corr : pd.DataFrame
        两组舌×脉相关系数矩阵
    benign_q, cancer_q : pd.DataFrame
        两组舌×脉q值矩阵 (FDR校正后)
    tongue_features : list
        舌象特征列表
    pulse_features : list
        脉象特征列表
    n_benign, n_cancer : int
        两组样本量
    output_path : str
        输出路径 (不含扩展名)
    dr_thr : float, optional
        相关系数差异阈值, 默认使用config.DIFF_DR_THR (0.20)
    q_diff_thr : float, optional
        差异显著性q值阈值, 默认使用config.DIFF_Q_THR (0.05)
    require_maxr : float, optional
        两边最大相关系数保护阈值
        默认使用config.DIFF_REQUIRE_MAXR (0.25)
        确保差异边的生物学意义
    figsize : tuple, default=(14, 10)
        图形尺寸
    """
    # 使用默认阈值
    if dr_thr is None:
        dr_thr = config.DIFF_DR_THR
    if q_diff_thr is None:
        q_diff_thr = config.DIFF_Q_THR
    if require_maxr is None:
        require_maxr = config.DIFF_REQUIRE_MAXR

    print(f"\n[Fig9-DIFF] 正在生成差异网络图 (Cancer vs Benign)...")
    print(f"  阈值: |Δr|>={dr_thr}, q_diff<{q_diff_thr}")
    print(f"  保护: max(|r_benign|, |r_cancer|)>={require_maxr}")
    print(f"  说明: 只保留生物学意义的差异边")

    # 计算Fisher z-test
    diff_results = []
    p_values = []

    for tongue_feat in tongue_features:
        if tongue_feat not in benign_corr.index or tongue_feat not in cancer_corr.index:
            continue

        for pulse_feat in pulse_features:
            if pulse_feat not in benign_corr.columns or pulse_feat not in cancer_corr.columns:
                continue

            r_benign = benign_corr.loc[tongue_feat, pulse_feat]
            r_cancer = cancer_corr.loc[tongue_feat, pulse_feat]

            # 跳过NaN
            if pd.isna(r_benign) or pd.isna(r_cancer):
                continue

            # Fisher z-test
            Z_stat, p_value, se = compute_fisher_z_test(r_benign, r_cancer, n_benign, n_cancer)

            dr = r_cancer - r_benign  # Δr = r_cancer - r_benign

            diff_results.append({
                'tongue': tongue_feat,
                'pulse': pulse_feat,
                'r_benign': r_benign,
                'r_cancer': r_cancer,
                'dr': dr,
                'Z': Z_stat,
                'p': p_value
            })

            if not np.isnan(p_value):
                p_values.append(p_value)

    # BH-FDR校正
    from statsmodels.stats.multitest import multipletests

    if len(p_values) > 0:
        rejected, q_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    else:
        q_corrected = []

    # 回填q值
    for i, result in enumerate(diff_results):
        if i < len(q_corrected):
            result['q'] = q_corrected[i]
        else:
            result['q'] = np.nan

    # 筛选显著差异边（三重条件）
    significant_edges = []
    for edge in diff_results:
        # 条件1: 统计显著
        if pd.isna(edge['q']) or edge['q'] >= q_diff_thr:
            continue

        # 条件2: 差异量足够大
        if abs(edge['dr']) < dr_thr:
            continue

        # 条件3: 保护机制（两边不能都太弱）
        # max(|r_benign|, |r_cancer|) >= require_maxr
        if max(abs(edge['r_benign']), abs(edge['r_cancer'])) < require_maxr:
            continue

        # 三个条件都满足
        significant_edges.append(edge)

    print(f"  总边数: {len(diff_results)}, 显著差异边: {len(significant_edges)}")

    # 创建图
    G = nx.Graph()

    # 添加节点
    for feat in tongue_features:
        G.add_node(feat, bipartite=0, node_type='tongue')

    for feat in pulse_features:
        G.add_node(feat, bipartite=1, node_type='pulse')

    # 添加显著差异边
    for edge in significant_edges:
        G.add_edge(edge['tongue'], edge['pulse'],
                  weight=abs(edge['dr']),
                  sign=1 if edge['dr'] > 0 else -1,
                  dr_value=edge['dr'])

    # 计算节点度数
    degrees = dict(G.degree())

    # 构建布局 (与Fig9-NET完全一致)
    pos = build_bipartite_pos(tongue_features, pulse_features, order='config')

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)

    if len(significant_edges) == 0:
        # 无显著差异边
        ax.text(0.5, 0.5, f'No differential edges pass thresholds\n(|Δr|>={dr_thr}, q<{q_diff_thr})',
                ha='center', va='center', fontsize=16,
                transform=ax.transAxes, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Differential Cross-Modal Network\n(Cancer vs Benign)\n(No significant differences)',
                     fontsize=16, fontweight='bold')
    else:
        # 绘制边
        for edge in significant_edges:
            tongue = edge['tongue']
            pulse = edge['pulse']
            dr = edge['dr']

            # 边颜色: Δr>0 (Cancer更强) = Teal, Δr<0 = Coral
            edge_color = config.NET_POSITIVE_COLOR if dr > 0 else config.NET_NEGATIVE_COLOR

            # 边宽 (与 |Δr| 成正比)
            edge_width = config.NET_EDGE_MIN_WIDTH + \
                        (abs(dr) - dr_thr) / (2 - dr_thr) * (config.NET_EDGE_MAX_WIDTH - config.NET_EDGE_MIN_WIDTH)
            edge_width = np.clip(edge_width, config.NET_EDGE_MIN_WIDTH, config.NET_EDGE_MAX_WIDTH)

            x0, y0 = pos[tongue]
            x1, y1 = pos[pulse]
            ax.plot([x0, x1], [y0, y1],
                   color=edge_color, linewidth=edge_width, alpha=0.6, zorder=1)

        # 绘制节点
        max_degree = max(degrees.values()) if degrees else 1
        for node, degree in degrees.items():
            if degree == 0:
                continue

            normalized_degree = degree / max_degree
            node_size = config.NET_NODE_MIN_SIZE + \
                       normalized_degree * (config.NET_NODE_MAX_SIZE - config.NET_NODE_MIN_SIZE)

            node_color = config.NET_TONGUE_NODE_COLOR if G.nodes[node]['node_type'] == 'tongue' else config.NET_PULSE_NODE_COLOR

            x, y = pos[node]
            ax.scatter(x, y, s=node_size, c=node_color, alpha=0.8,
                      edgecolors='black', linewidth=1.5, zorder=2)

            # 添加节点标签 (所有有连接的节点都要标签)
            # 使用固定左右偏移避免重叠：舌象标签在左侧，脉象标签在右侧
            node_type = G.nodes[node]['node_type']
            if node_type == 'tongue':
                # 舌象节点：标签在节点左侧
                label_x = x - 0.05
                ha = 'right'
            else:
                # 脉象节点：标签在节点右侧
                label_x = x + 0.05
                ha = 'left'

            ax.text(label_x, y, node, fontsize=9, ha=ha, va='center',
                   fontweight='bold', zorder=3, color='black')

        # 图例
        ax.plot([], [], color=config.NET_POSITIVE_COLOR, linewidth=3, label='Δr>0 (Cancer > Benign)')
        ax.plot([], [], color=config.NET_NEGATIVE_COLOR, linewidth=3, label='Δr<0 (Cancer < Benign)')
        ax.scatter([], [], s=200, c=config.NET_TONGUE_NODE_COLOR, edgecolors='black', linewidth=1.5, label='Tongue')
        ax.scatter([], [], s=200, c=config.NET_PULSE_NODE_COLOR, edgecolors='black', linewidth=1.5, label='Pulse')

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9, bbox_to_anchor=(1.18, 1.0))

        # 坐标轴（扩展左侧以容纳舌象标签，右侧以容纳图例）
        ax.set_xlim(-0.25, 1.15)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Tongue Features', 'Pulse Features'], fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 标题
        edge_count = len(significant_edges)
        node_count = sum(1 for d in degrees.values() if d > 0)

        # 如果边数很少，添加注释
        if edge_count <= 5:
            title_note = f'\nNote: Only {edge_count} differential edge(s) passed thresholds (|Δr|>={dr_thr}, q<{q_diff_thr}).'
        else:
            title_note = ''

        ax.set_title(f'Differential Cross-Modal Network (Cancer vs Benign)\n'
                    f'({node_count} nodes, {edge_count} edges, |Δr|>={dr_thr}, q<{q_diff_thr}){title_note}',
                    fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    # 保存
    for fmt in ['png', 'svg']:
        filepath = f'{output_path}.{fmt}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] 已保存: {os.path.basename(filepath)}")

    plt.close()

    return G, significant_edges


# =============================================================================
# 核心函数3: 导出网络统计表
# =============================================================================

def export_network_stats(benign_edges, cancer_edges, diff_edges,
                        n_benign, n_cancer,
                        output_path):
    """
    导出网络统计表

    Parameters:
    -----------
    benign_edges, cancer_edges : list of dict
        两组的边数据
    diff_edges : list of dict
        差异显著的边数据
    n_benign, n_cancer : int
        两组样本量
    output_path : str
        输出Excel路径 (不含扩展名)
    """
    print(f"\n[Table] 正在导出网络统计表...")

    with pd.ExcelWriter(f'{output_path}.xlsx', engine='openpyxl') as writer:
        # ========== Sheet1: 网络汇总统计 ==========
        summary_data = {
            'Metric': [
                'Sample Size (Benign)',
                'Sample Size (Cancer)',
                'Number of Edges (Benign)',
                'Number of Edges (Cancer)',
                'Number of Differential Edges',
                'Mean |r| (Benign)',
                'Mean |r| (Cancer)',
                'Network Density (Benign)',
                'Network Density (Cancer)'
            ],
            'Value': [
                n_benign,
                n_cancer,
                len(benign_edges),
                len(cancer_edges),
                len(diff_edges),
                np.mean([abs(e['r']) for e in benign_edges]) if benign_edges else 0,
                np.mean([abs(e['r']) for e in cancer_edges]) if cancer_edges else 0,
                len(benign_edges) / (34 * 14) if n_benign > 0 else 0,  # 34舌*14脉
                len(cancer_edges) / (34 * 14) if n_cancer > 0 else 0
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Network_Summary', index=False)

        # ========== Sheet2: Benign组边列表 ==========
        if benign_edges:
            benign_df = pd.DataFrame(benign_edges)
            benign_df = benign_df.sort_values('r', key=abs, ascending=False)
            benign_df.to_excel(writer, sheet_name='Benign_Edges', index=False)

        # ========== Sheet3: Cancer组边列表 ==========
        if cancer_edges:
            cancer_df = pd.DataFrame(cancer_edges)
            cancer_df = cancer_df.sort_values('r', key=abs, ascending=False)
            cancer_df.to_excel(writer, sheet_name='Cancer_Edges', index=False)

        # ========== Sheet4: 差异边列表 ==========
        if diff_edges:
            diff_df = pd.DataFrame(diff_edges)
            diff_df = diff_df.sort_values('dr', key=abs, ascending=False)
            diff_df.to_excel(writer, sheet_name='Differential_Edges', index=False)

    print(f"  [OK] 已保存: {os.path.basename(output_path)}.xlsx")


# =============================================================================
# 主程序测试
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("网络图模块测试")
    print("="*70)
    print("\n此模块需要在simple_correlation.py中集成使用")
    print("提供以下函数:")
    print("  1. plot_crossmodal_network() - 绘制跨模态相关网络图")
    print("  2. plot_differential_network() - 绘制差异网络图")
    print("  3. export_network_stats() - 导出网络统计表")
    print("="*70)
