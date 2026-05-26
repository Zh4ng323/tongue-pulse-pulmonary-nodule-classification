# -*- coding: utf-8 -*-
"""
Pulmonary nodule classification - Integrated analysis system.

Modules:
  1. Pearson correlation analysis
  2. Canonical correlation analysis (CCA)
  3. Integrated modeling (10-fold CV with LASSO + 5 ML models)
  4. SHAP interpretability analysis
"""

import sys
import os
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 修复Windows终端编码问题（必须在其他导入之前）
import encoding_fix
encoding_fix.fix_windows_encoding()

import config
from modules.data_loader import load_and_check
from modules.cca_analysis import GroupedCCAAnalyzer
from modules.simple_correlation import SimpleCorrelationAnalyzer
from modules.integrated_modeling_v2 import run_integrated_modeling_v2


def print_banner():
    """打印系统横幅"""
    print("\n" + "="*70)
    print(" "*10 + "肺结节良恶性识别 - 整合分析系统")
    print("="*70)
    print(f"版本: {config.VERSION}")
    print(f"更新日期: {config.VERSION_DATE}")
    print(f"作者: {', '.join(config.AUTHORS)}")
    print("="*70)


def print_main_menu():
    """打印主菜单"""
    print("\n" + "="*70)
    print("【主菜单】请选择分析类型")
    print("="*70)
    print(" 1. 简单相关性分析（Pearson相关）")
    print("     - 舌象×舌象、脉象×脉象相关性热图")
    print("     - 舌象×脉象聚类热图")
    print("     - 分组分析：良性结节组 vs 肺癌组")
    print()
    print(" 2. 典型相关分析（CCA）")
    print("     - 分析舌象与脉象特征的典型相关性")
    print("     - 分组分析：良性结节组 vs 肺癌组")
    print()
    print(" 3. 整合建模分析（10折交叉验证）")
    print("     - 折叠内统一LASSO特征筛选")
    print("     - 所有模型使用相同特征训练（公平对比）")
    print("     - 生成多模型对比图（ROC/PR曲线）")
    print()
    print(" 4. 一键执行简单相关 + CCA分析")
    print("     - 一次性完成简单相关性分析和CCA分析")
    print("     - 无需重复选择，自动依次执行")
    print()
    print(" 5. SHAP 特征分析拼图")
    print("     - 基于融合模态建模结果，导出关键单面板独立图表")
    print("     - 使用OOF数据（10-fold out-of-fold预测与SHAP）")
    print()
    print(" 0. 退出")
    print("="*70)


def get_user_choice(prompt, options):
    """获取用户选择"""
    while True:
        try:
            choice = input(f"\n{prompt}").strip()
            if choice in options:
                return choice
            else:
                print(f"\n[!] 无效选项，请选择: {', '.join(options)}")
        except KeyboardInterrupt:
            print("\n\n[INFO] 用户中断操作")
            sys.exit(0)


def get_feature_mode():
    """获取特征模式选择"""
    print("\n" + "-"*70)
    print("请选择特征维度")
    print("-"*70)
    print(" 1. 舌象特征（34个）")
    print(" 2. 脉象特征（15个）")
    print(" 3. 舌+脉联合特征（49个）")

    choice = get_user_choice("请输入选项 (1/2/3): ", ['1', '2', '3'])

    mode_map = {
        '1': 'tongue',
        '2': 'pulse',
        '3': 'combined'
    }

    return mode_map[choice]


def run_simple_correlation_analysis(data_loader):
    """运行简单相关性分析"""
    print("\n" + "="*70)
    print("【模块1.5】简单相关性分析（Pearson相关）")
    print("="*70)

    try:
        # 创建简单相关性分析器（构造时传入数据，自动分组）
        print("\n正在初始化相关性分析器...")
        corr_analyzer = SimpleCorrelationAnalyzer(
            data=data_loader.data,
            target_col=data_loader.target_col
        )

        # 运行完整相关性分析（自动识别特征、分析所有组、生成图表）
        print("\n正在运行相关性分析...")
        output_dir = os.path.join(config.OUTPUT_BASE_DIR, 'simple_correlation_results')
        corr_analyzer.run_full_analysis(output_dir=output_dir)

        print("\n" + "="*70)
        print("[OK] 简单相关性分析完成！")
        print("="*70)
        print(f"\n结果已保存到: {output_dir}")

        return True

    except Exception as e:
        print(f"\n[!] 相关性分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_cca_analysis(data_loader):
    """运行CCA分析"""
    print("\n" + "="*70)
    print("【模块2】典型相关分析（CCA）")
    print("="*70)

    try:
        # 创建CCA分析器（构造时传入数据，自动分组）
        print("\n正在初始化CCA分析器...")
        cca_analyzer = GroupedCCAAnalyzer(
            data=data_loader.data,
            target_col=data_loader.target_col
        )

        # 运行完整CCA分析（自动识别特征、分析所有组、生成图表）
        print("\n正在运行CCA分析...")
        output_dir = os.path.join(config.OUTPUT_BASE_DIR, 'cca_results')
        cca_analyzer.run_full_analysis(output_dir=output_dir)

        print("\n" + "="*70)
        print("[OK] CCA分析完成！")
        print("="*70)
        print(f"\n结果已保存到: {output_dir}")

        return True

    except Exception as e:
        print(f"\n[!] CCA分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_integrated_analysis(data_loader):
    """运行整合建模分析"""
    print("\n" + "="*70)
    print("【模块3】整合建模分析")
    print("="*70)

    try:
        # 选择特征模式
        feature_mode = get_feature_mode()
        print(f"\n[OK] 已选择特征模式: {feature_mode}")

        # 说明改进点
        print(f"\n【方法学说明】")
        print(f" 本系统采用改进的交叉验证设计：")
        print(f" for fold in 10_folds:")
        print(f"      1. 划分训练集和测试集")
        print(f"      2. LASSO特征筛选（在训练集上，只做一次）")
        print(f"      3. 所有模型用选定的特征训练")
        print(f"      4. 在测试集上评估所有模型")
        print(f" ")
        print(f" 优点：")
        print(f"    ✓ 公平对比：所有模型使用相同特征")
        print(f"    ✓ 避免数据泄露：特征选择在训练集上进行")
        print(f"    ✓ 符合真实场景：先选定特征，再尝试不同模型")
        print()

        confirm = input(f"\n是否继续？: ").strip().lower()
        if confirm != 'y':
            print("\n[INFO] 用户取消操作")
            return False

        # 运行整合建模
        print("\n正在运行10折交叉验证...")
        print("这可能需要较长时间（请耐心等待）...")
        print("-"*70)

        modeler = run_integrated_modeling_v2(
            data_loader=data_loader,
            feature_mode=feature_mode,
            models_to_run='all'  # 运行所有模型进行对比
        )

        print("\n" + "="*70)
        print("[OK] 整合建模分析完成！")
        print("="*70)

        # 输出主要结果摘要
        print("\n主要结果摘要:")
        for model_key, results in modeler.results.items():
            model_name_cn = modeler.model_configs[model_key]['name']
            mean_auc = np.mean(results['fold_aucs'])
            std_auc = np.std(results['fold_aucs'])
            print(f"\n  {model_name_cn}:")
            print(f"    - Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")

        print(f"\n结果已保存到: {os.path.join(config.OUTPUT_BASE_DIR, 'modeling_results_v2')}")

        return modeler  # 返回modeler对象以便后续使用

    except Exception as e:
        print(f"\n[!] 整合建模分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_correlation_and_cca(data_loader):
    """一次性运行简单相关 + CCA分析"""
    print("\n" + "="*70)
    print("【一键执行】简单相关性分析 + 典型相关分析")
    print("="*70)
    print("\n正在依次执行以下分析:")
    print("  1. 简单相关性分析（Pearson相关）")
    print("  2. 典型相关分析（CCA）")
    print()

    # 步骤1：简单相关性分析
    print("\n" + "="*70)
    print("步骤 1/2: 简单相关性分析（Pearson相关）")
    print("="*70)

    success_corr = run_simple_correlation_analysis(data_loader)

    if success_corr:
        print("\n" + "="*70)
        print("[OK] 简单相关性分析完成！")
        print("="*70)
        print("\n正在执行CCA分析...")
        success_cca = run_cca_analysis(data_loader)

        if success_cca:
            print("\n" + "="*70)
            print("[OK] CCA分析完成！")
            print("="*70)
            print("\n✅ 所有分析已成功完成！")
            print(f"\n结果已保存到: {config.OUTPUT_BASE_DIR}")
            print("  - simple_correlation_results/ (简单相关)")
            print("  - cca_results/ (典型相关)")
            return True
        else:
            print("\n" + "="*70)
            print("[!] CCA分析失败，跳过")
            print("="*70)
            return False
    else:
        print("\n" + "="*70)
        print("[!] 简单相关性分析失败，跳过CCA分析")
        print("="*70)
        return False


def run_shap_feature_analysis_v2_only(data_loader):
    """仅生成 SHAP 特征分析拼图 v2.0（基于已完成的建模结果）"""
    print("\n" + "="*70)
    print("【仅生成 SHAP 特征分析拼图 v2.0】")
    print("="*70)

    try:
        # 选择特征模式
        feature_mode = get_feature_mode()
        print(f"\n[OK] 已选择特征模式: {feature_mode}")

        print("\n【说明】")
        print("  此选项将：")
        print("  1. 运行融合模态的XGBoost建模（如果尚未运行）")
        print("  2. 生成SHAP特征分析拼图 v2.0（A-H面板）")
        print("  3. 使用OOF数据（10-fold out-of-fold预测与SHAP）")
        print("  4. Base value来自TreeExplainer.expected_value（不是0）")
        print()

        confirm = input("是否继续？: ").strip().lower()
        if confirm != 'y':
            print("\n[INFO] 用户取消操作")
            return False

        # 运行融合模态建模
        print("\n" + "="*70)
        print("步骤 1/2: 融合模态建模")
        print("="*70)
        modeler_fusion = run_integrated_modeling_v2(
            data_loader=data_loader,
            feature_mode='combined',
            models_to_run=['xgboost']
        )
        print("[OK] 融合模态完成")

        # 生成 SHAP 特征分析拼图 v2.0
        print("\n" + "="*70)
        print("步骤 2/2: 生成 SHAP 特征分析拼图 v2.0")
        print("="*70)

        from modules.interpretability_plots_v2 import generate_fig_shap_feature_analysis

        output_dir = os.path.join(config.OUTPUT_BASE_DIR, 'interpretability_plots_v2')
        os.makedirs(output_dir, exist_ok=True)

        # 获取完整的训练数据（用于LIME，如果需要）
        X_full, y_full, feature_names_full = data_loader.prepare_modeling_data(feature_mode='combined')

        generate_fig_shap_feature_analysis(
            modeler_fusion,
            output_dir,
            X_train_full=X_full,
            feature_names_full=feature_names_full
        )

        print("\n" + "="*70)
        print("[OK] SHAP 特征分析拼图 v2.0 生成完成！")
        print("="*70)

        # 验证输出（更新为新的文件名）
        print("\n检查生成的文件:")
        expected_files = [
            'Fig_A_SHAP_Summary_Beeswarm.png',
            'Fig_A_SHAP_Summary_Beeswarm.svg',
            'Fig_B_Decision_Plot.png',
            'Fig_B_Decision_Plot.svg',
            'Fig_C_Grouped_Feature_Importance.png',
            'Fig_C_Grouped_Feature_Importance.svg',
            'Fig_F_ForceStrip_BenignCase.png',
            'Fig_F_ForceStrip_BenignCase.svg',
            'Fig_G_ForceStrip_HighRiskMalignant.png',
            'Fig_G_ForceStrip_HighRiskMalignant.svg',
            'Fig_H_GlobalForce_Placeholder.png',
            'Fig_H_GlobalForce_Placeholder.svg'
        ]

        all_exist = True
        for filename in expected_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                filesize = os.path.getsize(filepath) / 1024  # KB
                print(f"  ✓ {filename} ({filesize:.1f} KB)")
            else:
                print(f"  ✗ {filename} (未找到)")
                all_exist = False

        if all_exist:
            print(f"\n所有图表已保存到: {output_dir}")
            return True
        else:
            print("\n[!] 部分文件未生成，请检查错误信息")
            return False

    except Exception as e:
        print(f"\n[!] 生成 SHAP 特征分析拼图 v2.0 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 打印横幅
    print_banner()

    # 加载数据
    print("\n" + "="*70)
    print("【模块1】数据加载与基本检查")
    print("="*70)

    try:
        data_loader = load_and_check()
    except Exception as e:
        print(f"\n[!] 数据加载失败: {str(e)}")
        print("\n请检查数据文件路径和格式")
        return

    # 主循环
    while True:
        print_main_menu()

        choice = get_user_choice("请输入选项 (0-5): ", ['0', '1', '2', '3', '4', '5'])

        if choice == '0':
            print("\n感谢使用整合分析系统 v2.1！")
            print("再见！")
            break

        if choice == '1':
            # 简单相关性分析
            run_simple_correlation_analysis(data_loader)

        elif choice == '2':
            # CCA分析
            run_cca_analysis(data_loader)

        elif choice == '3':
            # 整合建模分析（v2.1改进版）
            run_integrated_analysis(data_loader)

        elif choice == '4':
            # 【一键执行】简单相关 + CCA分析
            run_correlation_and_cca(data_loader)

        elif choice == '5':
            # [生成 SHAP 特征分析拼图 v2.0]
            run_shap_feature_analysis_v2_only(data_loader)

        # 询问是否继续
        print("\n" + "-"*70)
        continue_choice = input("是否返回主菜单继续其他分析？: ").strip().lower()
        if continue_choice != 'y':
            print("\n感谢使用整合分析系统 v2.1！")
            print("再见！")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
