# config_loader.py
import os
import yaml

def load_agent_config(agent_name: str) -> dict:
    """动态读取并解析 config/ 目录下的 yaml 配置文件"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config", f"{agent_name}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 关键配置文件缺失: {config_path}，请检查项目组织结构。")
        
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)