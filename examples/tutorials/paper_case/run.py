#!/usr/bin/env python3
"""自动准备 CPU 环境，并执行整套教学或启动 JupyterLab。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
ENV = ROOT / '.venv-tutorial'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=['check', 'lab'], default='check',
                        help='check：执行四本 notebook（默认）；lab：打开交互服务')
    parser.add_argument('--offline', action='store_true', help='复用已安装环境，不访问软件源')
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        parser.error('请使用 Python 3.10–3.13 启动。')
    if not (ROOT.parents[2] / 'src/dpeva/sampling/manager.py').is_file():
        parser.error('缺少 src/dpeva，请克隆完整 DP-EVA 仓库。')
    python = ENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if args.offline and not python.is_file():
        parser.error('环境尚未创建。请先联网运行一次本命令（不加 --offline）。')
    if not python.is_file():
        print('创建专用 CPU 环境 .venv-tutorial/ …', flush=True)
        venv.EnvBuilder(with_pip=True).create(ENV)
    if not args.offline:
        print('安装/检查 requirements.txt 中的依赖 …', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                        '-r', str(ROOT / 'requirements.txt')], check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    # Write environment details alongside all other disposable runtime outputs.
    work = ROOT / 'work'
    work.mkdir(exist_ok=True)
    packages = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    (work / 'environment-installed.txt').write_text(packages, encoding='utf-8')
    if args.mode == 'check':
        print('执行数据核验和四本 notebook；结果保存到 work/ …', flush=True)
        subprocess.run([str(python), str(ROOT / 'scripts/verify.py')], cwd=ROOT, check=True)
    else:
        print('请打开终端显示的本地链接，依次运行 01–04；Ctrl+C 停止服务。', flush=True)
        subprocess.run([str(python), '-m', 'jupyter', 'lab', '--no-browser',
                        '--ServerApp.ip=127.0.0.1', f'--ServerApp.root_dir={ROOT}'],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f'启动未完成（退出码 {error.returncode}），请根据上方错误修复后重试。', file=sys.stderr)
        raise SystemExit(error.returncode)
    except KeyboardInterrupt:
        raise SystemExit(130)
