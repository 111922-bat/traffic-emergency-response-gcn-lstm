#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据版本控制工具

此脚本提供简单的数据集版本管理功能，用于跟踪数据集的变化，
记录数据集元信息，并支持数据集的版本切换和标记。
"""

import os
import json
import shutil
import hashlib
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any


def calculate_file_hash(file_path: str) -> str:
    """
    计算文件的MD5哈希值，用于验证文件完整性
    
    Args:
        file_path: 文件路径
    
    Returns:
        str: 文件的MD5哈希值
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    获取文件的元数据
    
    Args:
        file_path: 文件路径
    
    Returns:
        Dict: 包含文件元数据的字典
    """
    stat_info = os.stat(file_path)
    return {
        "size": stat_info.st_size,
        "mtime": datetime.datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
        "atime": datetime.datetime.fromtimestamp(stat_info.st_atime).isoformat(),
        "ctime": datetime.datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
        "hash": calculate_file_hash(file_path)
    }


class DataVersionControl:
    """
    数据版本控制类
    提供数据集版本管理、版本记录、版本切换等功能
    """
    
    def __init__(self, data_dir: str, version_file: str = ".dvc/versions.json"):
        """
        初始化数据版本控制器
        
        Args:
            data_dir: 数据集目录
            version_file: 版本信息存储文件路径
        """
        self.data_dir = os.path.abspath(data_dir)
        self.version_file = os.path.join(os.path.dirname(data_dir), version_file)
        self.versions_dir = os.path.join(os.path.dirname(data_dir), ".dvc/versions")
        
        # 确保必要的目录存在
        os.makedirs(os.path.dirname(self.version_file), exist_ok=True)
        os.makedirs(self.versions_dir, exist_ok=True)
        
        # 加载或创建版本信息
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """
        加载版本信息
        
        Returns:
            Dict: 版本信息字典
        """
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"警告: 无法解析版本文件 {self.version_file}，将创建新的版本文件")
        
        # 返回默认版本结构
        return {
            "current_version": None,
            "versions": {},
            "created_at": datetime.datetime.now().isoformat(),
            "last_modified": datetime.datetime.now().isoformat()
        }
    
    def _save_versions(self):
        """
        保存版本信息到文件
        """
        self.versions["last_modified"] = datetime.datetime.now().isoformat()
        with open(self.version_file, "w", encoding="utf-8") as f:
            json.dump(self.versions, f, indent=2, ensure_ascii=False)
    
    def _scan_dataset(self) -> Dict[str, Dict[str, Any]]:
        """
        扫描数据集目录，获取所有文件的元数据
        
        Returns:
            Dict: 文件路径到元数据的映射
        """
        files_metadata = {}
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                # 跳过版本控制相关文件
                if file.endswith(".gitignore") or ".dvc" in root:
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.data_dir)
                
                try:
                    files_metadata[relative_path] = get_file_metadata(file_path)
                except Exception as e:
                    print(f"警告: 无法获取文件 {relative_path} 的元数据: {e}")
        
        return files_metadata
    
    def create_version(self, version_name: str, description: str = "") -> Dict[str, Any]:
        """
        创建新的数据集版本
        
        Args:
            version_name: 版本名称
            description: 版本描述
        
        Returns:
            Dict: 版本信息
        """
        # 检查版本名称是否已存在
        if version_name in self.versions["versions"]:
            raise ValueError(f"版本名称 '{version_name}' 已存在")
        
        # 扫描当前数据集状态
        files_metadata = self._scan_dataset()
        
        # 创建版本快照目录
        version_dir = os.path.join(self.versions_dir, version_name)
        os.makedirs(version_dir, exist_ok=True)
        
        # 创建版本信息
        version_info = {
            "name": version_name,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "files": files_metadata,
            "file_count": len(files_metadata),
            "total_size": sum(m["size"] for m in files_metadata.values())
        }
        
        # 保存版本信息
        version_file = os.path.join(version_dir, "version_info.json")
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)
        
        # 更新主版本记录
        self.versions["versions"][version_name] = {
            "created_at": version_info["created_at"],
            "description": description,
            "file_count": version_info["file_count"],
            "total_size": version_info["total_size"]
        }
        
        # 设置为当前版本
        self.versions["current_version"] = version_name
        self._save_versions()
        
        print(f"已创建版本: {version_name}")
        print(f"文件数量: {version_info['file_count']}")
        print(f"总大小: {version_info['total_size']} 字节")
        
        return version_info
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        列出所有版本
        
        Returns:
            List: 版本信息列表
        """
        versions_list = []
        for name, info in self.versions["versions"].items():
            version_info = {
                "name": name,
                "created_at": info["created_at"],
                "description": info["description"],
                "file_count": info["file_count"],
                "total_size": info["total_size"],
                "is_current": name == self.versions["current_version"]
            }
            versions_list.append(version_info)
        
        # 按创建时间排序
        versions_list.sort(key=lambda x: x["created_at"], reverse=True)
        
        # 打印版本列表
        print("数据集版本列表:")
        print("-" * 80)
        print(f"{'版本名':<20} {'创建时间':<25} {'文件数':<10} {'大小(KB)':<10} {'当前?':<5} {'描述':<20}")
        print("-" * 80)
        
        for v in versions_list:
            size_kb = f"{v['total_size'] / 1024:.1f}"
            is_current = "✓" if v["is_current"] else ""
            print(f"{v['name']:<20} {v['created_at'][:19]:<25} {v['file_count']:<10} {size_kb:<10} {is_current:<5} {v['description'][:30]}")
        
        return versions_list
    
    def get_version_info(self, version_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取指定版本的详细信息
        
        Args:
            version_name: 版本名称，如果为None则获取当前版本
        
        Returns:
            Dict: 版本详细信息
        """
        if version_name is None:
            version_name = self.versions["current_version"]
            if version_name is None:
                raise ValueError("没有设置当前版本")
        
        if version_name not in self.versions["versions"]:
            raise ValueError(f"版本 '{version_name}' 不存在")
        
        # 加载版本详细信息
        version_dir = os.path.join(self.versions_dir, version_name)
        version_file = os.path.join(version_dir, "version_info.json")
        
        if not os.path.exists(version_file):
            raise FileNotFoundError(f"版本文件不存在: {version_file}")
        
        with open(version_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        比较两个版本之间的差异
        
        Args:
            version1: 第一个版本名称
            version2: 第二个版本名称
        
        Returns:
            Dict: 版本差异信息
        """
        # 获取两个版本的详细信息
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)
        
        # 获取文件集合
        files1 = set(info1["files"].keys())
        files2 = set(info2["files"].keys())
        
        # 计算差异
        added_files = files2 - files1
        removed_files = files1 - files2
        common_files = files1 & files2
        modified_files = set()
        
        # 检查共同文件是否有修改
        for file in common_files:
            if info1["files"][file]["hash"] != info2["files"][file]["hash"]:
                modified_files.add(file)
        
        # 构建差异报告
        diff_report = {
            "version1": version1,
            "version2": version2,
            "added_files": sorted(list(added_files)),
            "removed_files": sorted(list(removed_files)),
            "modified_files": sorted(list(modified_files)),
            "added_count": len(added_files),
            "removed_count": len(removed_files),
            "modified_count": len(modified_files),
            "unchanged_count": len(common_files) - len(modified_files)
        }
        
        # 打印差异报告
        print(f"\n版本比较: {version1} → {version2}")
        print("-" * 80)
        print(f"新增文件: {diff_report['added_count']}")
        for f in diff_report['added_files'][:5]:  # 只显示前5个
            print(f"  + {f}")
        if len(diff_report['added_files']) > 5:
            print(f"  ... 还有 {len(diff_report['added_files']) - 5} 个文件")
        
        print(f"\n删除文件: {diff_report['removed_count']}")
        for f in diff_report['removed_files'][:5]:
            print(f"  - {f}")
        if len(diff_report['removed_files']) > 5:
            print(f"  ... 还有 {len(diff_report['removed_files']) - 5} 个文件")
        
        print(f"\n修改文件: {diff_report['modified_count']}")
        for f in diff_report['modified_files'][:5]:
            print(f"  ~ {f}")
        if len(diff_report['modified_files']) > 5:
            print(f"  ... 还有 {len(diff_report['modified_files']) - 5} 个文件")
        
        print(f"\n未变更文件: {diff_report['unchanged_count']}")
        print("-" * 80)
        
        return diff_report
    
    def create_tag(self, version_name: str, tag_name: str, description: str = ""):
        """
        为指定版本创建标签
        
        Args:
            version_name: 版本名称
            tag_name: 标签名称
            description: 标签描述
        """
        if version_name not in self.versions["versions"]:
            raise ValueError(f"版本 '{version_name}' 不存在")
        
        # 确保tags字段存在
        if "tags" not in self.versions:
            self.versions["tags"] = {}
        
        # 检查标签是否已存在
        if tag_name in self.versions["tags"]:
            raise ValueError(f"标签 '{tag_name}' 已存在")
        
        # 创建标签
        self.versions["tags"][tag_name] = {
            "version": version_name,
            "created_at": datetime.datetime.now().isoformat(),
            "description": description
        }
        
        self._save_versions()
        print(f"已为版本 '{version_name}' 创建标签: {tag_name}")
    
    def list_tags(self):
        """
        列出所有标签
        """
        if "tags" not in self.versions or not self.versions["tags"]:
            print("没有可用的标签")
            return
        
        print("数据集标签列表:")
        print("-" * 80)
        print(f"{'标签名':<20} {'关联版本':<20} {'创建时间':<25} {'描述':<30}")
        print("-" * 80)
        
        # 按创建时间排序
        tags = sorted(self.versions["tags"].items(), 
                     key=lambda x: x[1]["created_at"], 
                     reverse=True)
        
        for name, info in tags:
            print(f"{name:<20} {info['version']:<20} {info['created_at'][:19]:<25} {info['description'][:30]}")
    
    def export_version_info(self, output_file: str):
        """
        导出版本信息到文件
        
        Args:
            output_file: 输出文件路径
        """
        export_info = {
            "dataset_dir": self.data_dir,
            "export_time": datetime.datetime.now().isoformat(),
            "current_version": self.versions["current_version"],
            "versions": self.versions["versions"],
            "tags": self.versions.get("tags", {})
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_info, f, indent=2, ensure_ascii=False)
        
        print(f"版本信息已导出到: {output_file}")


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="数据集版本控制工具")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="数据集目录路径")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 创建版本命令
    create_parser = subparsers.add_parser("create", help="创建新版本")
    create_parser.add_argument("version_name", type=str, help="版本名称")
    create_parser.add_argument("--description", type=str, default="", help="版本描述")
    
    # 列出版本命令
    subparsers.add_parser("list", help="列出所有版本")
    
    # 显示版本信息命令
    info_parser = subparsers.add_parser("info", help="显示指定版本信息")
    info_parser.add_argument("version_name", type=str, nargs="?", help="版本名称，不指定则显示当前版本")
    
    # 比较版本命令
    compare_parser = subparsers.add_parser("compare", help="比较两个版本")
    compare_parser.add_argument("version1", type=str, help="第一个版本")
    compare_parser.add_argument("version2", type=str, help="第二个版本")
    
    # 创建标签命令
    tag_parser = subparsers.add_parser("tag", help="为版本创建标签")
    tag_parser.add_argument("version_name", type=str, help="版本名称")
    tag_parser.add_argument("tag_name", type=str, help="标签名称")
    tag_parser.add_argument("--description", type=str, default="", help="标签描述")
    
    # 列出标签命令
    subparsers.add_parser("tags", help="列出所有标签")
    
    # 导出版本信息命令
    export_parser = subparsers.add_parser("export", help="导出版本信息")
    export_parser.add_argument("output_file", type=str, help="输出文件路径")
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_arguments()
    
    # 创建版本控制器实例
    dvc = DataVersionControl(args.data_dir)
    
    # 根据命令执行相应操作
    if args.command == "create":
        dvc.create_version(args.version_name, args.description)
    elif args.command == "list":
        dvc.list_versions()
    elif args.command == "info":
        info = dvc.get_version_info(args.version_name)
        print(f"\n版本: {info['name']}")
        print(f"描述: {info['description']}")
        print(f"创建时间: {info['created_at']}")
        print(f"文件数量: {info['file_count']}")
        print(f"总大小: {info['total_size'] / 1024:.1f} KB")
        print(f"\n文件列表:")
        for i, (file_path, metadata) in enumerate(info["files"].items(), 1):
            size_kb = metadata["size"] / 1024
            print(f"  {i:3d}. {file_path} ({size_kb:.1f} KB)")
            # 只显示前20个文件
            if i >= 20 and len(info["files"]) > 20:
                print(f"  ... 还有 {len(info["files"]) - 20} 个文件")
                break
    elif args.command == "compare":
        dvc.compare_versions(args.version1, args.version2)
    elif args.command == "tag":
        dvc.create_tag(args.version_name, args.tag_name, args.description)
    elif args.command == "tags":
        dvc.list_tags()
    elif args.command == "export":
        dvc.export_version_info(args.output_file)
    else:
        print("请指定命令，使用 -h 查看帮助")


if __name__ == "__main__":
    main()