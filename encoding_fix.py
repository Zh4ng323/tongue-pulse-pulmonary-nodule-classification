# -*- coding: utf-8 -*-
"""
Windows terminal encoding fix for UTF-8 output.

Usage:
    import encoding_fix
    encoding_fix.fix_windows_encoding()
"""

import sys
import io

# 全局标志，确保只修复一次
_ENCODING_FIXED = False

def fix_windows_encoding():
    """
    修复Windows终端的编码问题

    功能：
    - 将标准输出(stdout)和错误输出(stderr)重定向为UTF-8编码
    - 自动检测Windows平台，不影响其他操作系统
    - 使用errors='replace'避免编码错误导致程序崩溃
    - 使用全局标志确保只修复一次，避免重复重定向

    返回：
    - bool: 是否成功修复
    """
    global _ENCODING_FIXED

    # 如果已经修复过，直接返回
    if _ENCODING_FIXED:
        return True

    if sys.platform == 'win32':
        try:
            # 检查stdout是否已经是我们想要的类型
            if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding != 'utf-8':
                # 创建新的UTF-8编码的stdout
                new_stdout = io.TextIOWrapper(
                    sys.stdout.buffer,
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=True
                )
                # 保留原始stdout的引用
                new_stdout.original_stdout = getattr(sys.stdout, 'original_stdout', sys.stdout)
                sys.stdout = new_stdout

            # 检查stderr
            if hasattr(sys.stderr, 'buffer') and sys.stderr.encoding != 'utf-8':
                # 创建新的UTF-8编码的stderr
                new_stderr = io.TextIOWrapper(
                    sys.stderr.buffer,
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=True
                )
                # 保留原始stderr的引用
                new_stderr.original_stderr = getattr(sys.stderr, 'original_stderr', sys.stderr)
                sys.stderr = new_stderr

            _ENCODING_FIXED = True
            return True
        except Exception:
            # 如果修复失败，静默处理
            return False

    return False


def restore_original_encoding():
    """
    恢复原始编码（调试用）

    注意：
    - 一般不需要调用此函数
    - 仅在调试时可能需要
    """
    global _ENCODING_FIXED

    if hasattr(sys.stdout, 'original_stdout'):
        sys.stdout = sys.stdout.original_stdout
    if hasattr(sys.stderr, 'original_stderr'):
        sys.stderr = sys.stderr.original_stderr

    _ENCODING_FIXED = False
