# -*- coding: utf-8 -*-
"""
Enhanced heatmap and clustermap plotting functions.

Provides scatter-based correlation heatmaps and cross-modal clustermaps
with FDR-annotated significance markers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# =============================================================================
# 全局配色和工具函数
# =============================================================================

def create_coral_teal_colormap_vivid():
    """
    创建"橘红-白-青绿" (Coral-White-Teal) 配色方案（高饱和度版）

    Returns:
    --------
    LinearSegmentedColormap
        自定义颜色映射
    """
    colors = ['#FF4500', '#FFFFFF', '#009688']  # 深珊瑚红 → 白色 → 深青绿
    nodes = [0.0, 0.5, 1.0]

    cmap = mcolors.LinearSegmentedColormap.from_list('coral_teal_vivid',
                                                       list(zip(nodes, colors)))
    return cmap

# 创建全局colormap实例
CORAL_TEAL_CMAP_VIVID = create_coral_teal_colormap_vivid()

# =============================================================================
# 视觉编码映射函数（点大小和透明度）
# =============================================================================

def compute_point_size(r, s_min=40, s_max=260, r_scale=0.6):
    """
    根据相关系数计算点大小（旧版，保留兼容）
    """
    abs_r = abs(r)
    size = s_min + (s_max - s_min) * (abs_r / r_scale)
    return min(size, s_max)


def compute_point_size_power(r, s_min=40, s_max=260, r_scale=0.6, power=1.0):
    """
    根据相关系数计算点大小（幂映射版 - 用于Fig7/8/9）

    Parameters:
    -----------
    r : float
        相关系数
    s_min : float
        最小点大小
    s_max : float
        最大点大小
    r_scale : float
        相关系数缩放因子（|r|达到此值时达到最大大小）
    power : float
        幂次（>1增强对比度，<1平滑过渡）

    Returns:
    --------
    float
        点大小
    """
    abs_r = abs(r)
    # 幂映射 + clip
    x = min(abs_r / r_scale, 1.0)  # clip到[0,1]
    size = s_min + (x ** power) * (s_max - s_min)
    return size


def compute_point_alpha(r, a_min=0.15, a_max=0.95, r_scale=0.6, power=1.0):
    """
    根据相关系数计算点透明度（旧版，保留兼容）
    """
    abs_r = abs(r)
    normalized = (abs_r / r_scale) ** power
    alpha = a_min + normalized * (a_max - a_min)
    return min(alpha, a_max)


def compute_point_alpha_power(r, a_min=0.15, a_max=0.95, r_scale=0.6, power=1.0):
    """
    根据相关系数计算点透明度（幂映射版 - 用于Fig7/8/9）

    Parameters:
    -----------
    r : float
        相关系数
    a_min : float
        最小透明度（弱相关）
    a_max : float
        最大透明度（强相关）
    r_scale : float
        相关系数缩放因子
    power : float
        幂次（>1增强对比度，<1平滑过渡）

    Returns:
    --------
    float
        透明度（0-1之间）
    """
    abs_r = abs(r)
    # 幂映射 + clip
    x = min(abs_r / r_scale, 1.0)  # clip到[0,1]
    alpha = a_min + (x ** power) * (a_max - a_min)
    return alpha


def get_cluster_boundaries(linkage, n_clusters, n_leaves):
    """
    根据层次聚类结果获取簇边界

    Parameters:
    -----------
    linkage : ndarray
        层次聚类linkage矩阵
    n_clusters : int
        簇数量
    n_leaves : int
        叶子节点数量

    Returns:
    --------
    list
        每个簇的边界索引列表，例如 [(0, 5), (5, 12), (12, 18), (18, 24)]
    """
    from scipy.cluster.hierarchy import fcluster

    # 获取聚类标签
    cluster_labels = fcluster(linkage, n_clusters, criterion='maxclust')

    # 按照叶子顺序重新标记
    boundaries = []
    current_label = cluster_labels[0]
    start_idx = 0

    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] != current_label:
            boundaries.append((start_idx, i))
            start_idx = i
            current_label = cluster_labels[i]

    # 添加最后一个簇
    boundaries.append((start_idx, len(cluster_labels)))

    return boundaries


def get_cluster_colors(n_clusters):
    """
    生成簇的颜色（用于color strip）

    Parameters:
    -----------
    n_clusters : int
        簇数量

    Returns:
    --------
    list
        颜色列表
    """
    # 使用Tableau 20配色（美观且易区分）
    tableau_colors = [
        '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
        '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'
    ]
    return tableau_colors[:n_clusters]


def get_significance_stars_q(q):
    """
    根据q值（FDR校正后的P值）返回星号标记

    Parameters:
    -----------
    q : float
        q值（Benjamini-Hochberg FDR校正后的P值，即校正后的P值）

    Returns:
    --------
    str
        星号标记 ('***', '**', '*', 或 '')
    """
    if pd.isna(q):
        return ''
    elif q < 0.001:
        return '***'
    elif q < 0.01:
        return '**'
    elif q < 0.05:
        return '*'
    else:
        return ''


# =============================================================================
# 函数1：图7/8 - 相关性热图（舌×舌、脉×脉）
# =============================================================================

def plot_corr_heatmap_improved(corr_matrix, q_matrix, title, output_path,
                              figsize=None, feature_type='tongue',
                              power_size=1.6, power_alpha=1.3,
                              cell_scale=1.0,
                              show_extreme_marks=False,
                              upper_triangle_only=True):
    """
    绘制相关性热图（舌×舌、脉×脉）- 主文图版

    样式：
    - 所有格子：圆点全画，用大小+透明度编码|r|（幂映射）
    - 不显示数字、不显示星号
    - 配色：橘红(负)→白(0)→青绿(正)
    - 自相关矩阵默认只显示上三角（避免重复信息）

    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        相关系数矩阵
    q_matrix : pd.DataFrame
        q值矩阵（FDR校正后的P值）
    title : str
        图表标题
    output_path : str
        输出路径（不含扩展名）
    figsize : tuple, optional
        图形尺寸
    feature_type : str
        特征类型 ('tongue' 或 'pulse')
    power_size : float, default=1.6
        大小幂次（>1增强对比度）
    power_alpha : float, default=1.3
        透明度幂次
    cell_scale : float, default=1.0
        额外放大系数（仅用于脉象，让圆点更饱满）
    show_extreme_marks : bool, default=False
        是否为|r|>=0.80的圆点加黑描边
    upper_triangle_only : bool, default=True
        是否只显示上三角（仅用于自相关矩阵，避免重复信息）
        注意：由于Y轴反转，这里"上三角"实际保留的是视觉上的左下三角（从左下到右上）
    """
    
    # 根据特征类型设置参数
    if feature_type == 'tongue':
        # Fig7: 舌象（34×34），使用强对比度
        s_min = 40
        s_max = 260
        r_scale = 0.6
        a_min = 0.22  # 提高到0.22避免空白感
        a_max = 0.95
    else:
        # Fig8: 脉象（15×15），自适应放大
        n_vars = len(corr_matrix)
        base_s_min = 40
        base_s_max = 260

        # 维度缩放：保持34×34的相对大小比例
        scale = max(1.0, 34 / n_vars)
        # 额外放大：让圆点更饱满
        s_min = base_s_min * scale * cell_scale
        s_max = base_s_max * scale * cell_scale
        r_scale = 0.6
        a_min = 0.22
        a_max = 0.95
        # 脉象使用温和的指数
        power_size = 1.2
        power_alpha = 1.1
    
    # 设置图形尺寸
    if figsize is None:
        figsize = (14, 12)
    
    n_vars = len(corr_matrix)
    
    # 设置样式
    plt.rcParams['font.size'] = 13
    plt.rcParams['font.family'] = config.FONT_ENGLISH
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 准备数据
    corr_values = corr_matrix.values
    
    # 绘制所有圆点（全画，不筛选）
    for i in range(n_vars):
        for j in range(n_vars):
            # 自相关矩阵：跳过对角线和上三角（如果启用三角模式）
            # 只保留下三角（j < i），在Y轴反转的情况下视觉上为左下角区域
            # 特征名：Y轴标签在左侧，X轴标签在底部
            if upper_triangle_only:
                if j >= i:  # 跳过对角线和上三角（j >= i，视觉上的右上区域）
                    continue
            else:
                if i == j:  # 仅跳过对角线
                    continue

            r = corr_values[i, j]

            if pd.isna(r):
                continue

            # 幂映射计算大小和透明度
            size = compute_point_size_power(r, s_min, s_max, r_scale, power_size)
            alpha = compute_point_alpha_power(r, a_min, a_max, r_scale, power_alpha)

            # 极端相关标记（可选）
            if show_extreme_marks and abs(r) >= 0.80:
                edge_colors = 'black'
                line_width = 0.6
            else:
                edge_colors = 'none'
                line_width = 0
            
            # 绘制圆点
            ax.scatter(j, i, s=size,
                      c=[CORAL_TEAL_CMAP_VIVID((r + 1) / 2)],
                      alpha=alpha,
                      edgecolors=edge_colors,
                      linewidths=line_width,
                      marker='o')
    
    # 网格线
    for i in range(n_vars + 1):
        ax.axhline(i - 0.5, color='#CCCCCC', linewidth=0.5, zorder=0)
        ax.axvline(i - 0.5, color='#CCCCCC', linewidth=0.5, zorder=0)
    
    # 坐标轴设置
    ax.set_xlim(-0.5, n_vars - 0.5)
    ax.set_ylim(n_vars - 0.5, -0.5)
    ax.set_xticks(range(n_vars))
    ax.set_yticks(range(n_vars))
    ax.set_xticklabels(corr_matrix.columns, fontsize=13, rotation=45,
                       ha='right', fontweight='bold')
    ax.set_yticklabels(corr_matrix.index, fontsize=13, rotation=0, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # 保持正圆
    ax.set_aspect('equal')
    
    # 添加色条
    sm = plt.cm.ScalarMappable(cmap=CORAL_TEAL_CMAP_VIVID,
                               norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation Coefficient (r)',
                   rotation=270, labelpad=20,
                   fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)
    
    # 标题
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # 白色背景
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # 保存：只保存PNG和SVG
    for fmt in ['png', 'svg']:
        filepath = f'{output_path}.{fmt}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] 已保存: {os.path.basename(filepath)}")
    
    plt.close()

# =============================================================================
# 函数2：图9 - 跨模态聚类图（舌×脉）
# =============================================================================

def plot_clustermap_improved_v4(corr_matrix, q_matrix, title, output_path, figsize=None,
                               cross_star_thr=None, q_thr=None,
                               show_module_boundaries=False):
    """
    绘制聚类图（舌×脉，跨模态）- 主文图版

    样式：
    - 所有格子：圆点全画，用大小+透明度编码|r|（幂映射，power<1）
    - 星号：仅在 q < q_thr 且 |r| >= cross_star_thr 时显示
    - 必须使用scatter和ax.set_aspect('auto')

    阈值说明（双层体系）：
    - cross_star_thr 默认使用 config.OVERVIEW_R_THR (0.15)
      这是"概览阈值"，用于展示整体模式
    - q_thr 默认使用 config.Q_THR (0.05)
      所有图表统一的显著性阈值

    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        相关系数矩阵（舌象×脉象）
    q_matrix : pd.DataFrame
        q值矩阵（FDR校正后的P值）
    title : str
        图表标题
    output_path : str
        输出路径（不含扩展名）
    figsize : tuple, optional
        图形尺寸
    cross_star_thr : float, optional
        星号门槛（|r| >= 此值且q < q_thr才显示星号）
        默认使用 config.OVERVIEW_R_THR (0.15)
    q_thr : float, optional
        显著性阈值
        默认使用 config.Q_THR (0.05)
    show_module_boundaries : bool, default=False
        是否显示模块边界线
    """

    # 从config读取默认阈值（如果未提供）
    if cross_star_thr is None:
        cross_star_thr = config.OVERVIEW_R_THR
    if q_thr is None:
        q_thr = config.Q_THR

    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    import matplotlib.gridspec as gridspec
    import warnings

    n_rows = len(corr_matrix)
    n_cols = len(corr_matrix.columns)
    
    # 单元格尺寸设置
    col_width = 0.40
    row_height = 0.28
    heatmap_width = n_cols * col_width
    heatmap_height = n_rows * row_height
    
    # 树状图尺寸
    tree_width_ratio = 0.18
    tree_height_ratio = 0.12
    tree_width = heatmap_width * tree_width_ratio
    tree_height = heatmap_height * tree_height_ratio
    
    # 间距和色条
    gap_width = 1.0  # 增大间距，让热图与色条分离更明显
    cbar_width = 0.3
    fig_width = heatmap_width + tree_width + gap_width + cbar_width
    fig_height = heatmap_height + tree_height
    figsize = (fig_width, fig_height)
    
    print(f"  [INFO] 聚类图: {figsize}, 矩阵: {n_rows}×{n_cols}")
    
    # 计算聚类
    row_linkage = linkage(corr_matrix, method='average', metric='euclidean')
    col_linkage = linkage(corr_matrix.T, method='average', metric='euclidean')
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    
    # GridSpec布局
    gs = gridspec.GridSpec(2, 4,
                          width_ratios=[tree_width, heatmap_width, gap_width, cbar_width],
                          height_ratios=[tree_height, heatmap_height],
                          wspace=0.0, hspace=0.0,
                          figure=fig)
    
    # --- 行树状图（左侧） ---
    ax_row_dendro = fig.add_subplot(gs[1, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dendro_row = dendrogram(
            row_linkage, orientation='left',
            no_labels=True,
            link_color_func=lambda x: '#666666',
            ax=ax_row_dendro
        )
    ax_row_dendro.axis('off')
    ax_row_dendro.invert_yaxis()  # ★ 让它和热图的“y反转”方向一致

    # --- 列树状图（顶部） ---
    ax_col_dendro = fig.add_subplot(gs[0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dendro_col = dendrogram(
            col_linkage, orientation='top',
            no_labels=True,
            link_color_func=lambda x: '#666666',
            ax=ax_col_dendro
        )
    ax_col_dendro.axis('off')

    # ★ 用 dendrogram 实际使用的 leaves 顺序（保证树与热图严格一致）
    row_order = dendro_row['leaves']
    col_order = dendro_col['leaves']

    # --- 热图轴 ---
    ax_heatmap = fig.add_subplot(gs[1, 1])

    corr_sorted = corr_matrix.iloc[row_order, col_order]
    q_sorted = q_matrix.iloc[row_order, col_order]

    # ★ 明确坐标范围 + y 反转（对齐格子）
    ax_heatmap.set_xlim(-0.5, n_cols - 0.5)
    ax_heatmap.set_ylim(n_rows - 0.5, -0.5)

    # --- 网格线（让每个格子看得见） ---
    for k in range(n_cols + 1):
        ax_heatmap.axvline(k - 0.5, color='#DDDDDD', linewidth=0.6, zorder=0)
    for k in range(n_rows + 1):
        ax_heatmap.axhline(k - 0.5, color='#DDDDDD', linewidth=0.6, zorder=0)

    # --- 圆点参数（你原来的设置保留） ---
    s_min, s_max, r_scale = 40, 260, 0.6
    a_min, a_max = 0.30, 0.95
    power_size, power_alpha = 0.9, 0.9

    # --- 画点（scatter） ---
    for i in range(n_rows):
        for j in range(n_cols):
            r = corr_sorted.iloc[i, j]
            if pd.isna(r):
                continue
            size = compute_point_size_power(r, s_min, s_max, r_scale, power_size)
            alpha = compute_point_alpha_power(r, a_min, a_max, r_scale, power_alpha)

            ax_heatmap.scatter(
                j, i, s=size,
                c=[CORAL_TEAL_CMAP_VIVID((r + 1) / 2)],
                alpha=alpha,
                edgecolors='none',
                marker='o',
                zorder=2
            )

    # --- 星号（修复：annotate 必须 offset points，否则会全堆到(0,0)） ---
    star_fontsize = 9
    for i in range(n_rows):
        for j in range(n_cols):
            r = corr_sorted.iloc[i, j]
            q = q_sorted.iloc[i, j]
            if pd.isna(r) or pd.isna(q):
                continue
            if (q < q_thr) and (abs(r) >= cross_star_thr):
                stars = get_significance_stars_q(q)
                if stars:
                    ax_heatmap.annotate(
                        stars,
                        xy=(j, i), xycoords='data',
                        xytext=(0, 0), textcoords='offset points',  # ★关键
                        ha='center', va='center',
                        color='black',
                        fontsize=star_fontsize,
                        fontweight='bold',
                        clip_on=True,
                        zorder=3
                    )

    # --- 坐标标签：Y 轴放到右侧（你想要的效果） ---
    ax_heatmap.set_xticks(range(n_cols))
    ax_heatmap.set_yticks(range(n_rows))
    ax_heatmap.set_xticklabels(corr_sorted.columns, fontsize=11, rotation=45, ha='right')
    ax_heatmap.set_yticklabels(corr_sorted.index, fontsize=11)

    ax_heatmap.yaxis.set_ticks_position('right')
    ax_heatmap.yaxis.set_label_position('right')
    ax_heatmap.tick_params(axis='y', labelright=True, labelleft=False)

    ax_heatmap.set_xlabel('')
    ax_heatmap.set_ylabel('')
    ax_heatmap.set_aspect('auto')

    # --- 色条：必须用独立 cax，不要用 ax=ax_heatmap（否则会挤热图导致树/格子错位） ---
    ax_cbar = fig.add_subplot(gs[1, 3])
    sm = plt.cm.ScalarMappable(cmap=CORAL_TEAL_CMAP_VIVID, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.set_label('Pearson Correlation Coefficient (r)', rotation=270, labelpad=20,
                fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)

    # ★不要 tight_layout（会再改布局导致对齐问题），用 subplots_adjust 控边距
    plt.subplots_adjust(left=0.03, right=0.96, top=0.93, bottom=0.10)
    # 标题
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 白色背景
    ax_heatmap.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # 保存：只保存PNG和SVG
    for fmt in ['png', 'svg']:
        filepath = f'{output_path}.{fmt}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  [OK] 已保存: {os.path.basename(filepath)}")
    
    plt.close()
