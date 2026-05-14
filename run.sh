#!/bin/bash
# 车辆视觉检索系统启动脚本
cd "$(dirname "$0")"
source venv/bin/activate
python main.py --mode gui
