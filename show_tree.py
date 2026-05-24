import subprocess
import os
from collections import defaultdict

def get_git_files():
    """获取所有 git 会追踪的文件（自动遵守 .gitignore）"""
    result = subprocess.run(
        ['git', 'ls-files', '--others', '--cached', '--exclude-standard'],
        capture_output=True, text=True, encoding='utf-8'
    )
    return result.stdout.strip().split('\n')

def build_tree(files):
    """把文件列表构建成嵌套字典"""
    tree = lambda: defaultdict(tree)
    root = tree()
    for f in files:
        parts = f.replace('\\', '/').split('/')
        node = root
        for part in parts:
            node = node[part]
    return root

def print_tree(node, prefix=''):
    entries = sorted(node.keys(), key=lambda x: (not bool(node[x]), x.lower()))
    # 文件夹优先（有子节点的）；按字母序
    for i, name in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = '└── ' if is_last else '├── '
        print(prefix + connector + name)
        if node[name]:  # 是文件夹
            extension = '    ' if is_last else '│   '
            print_tree(node[name], prefix + extension)

if __name__ == '__main__':
    files = get_git_files()
    files = [f for f in files if f]  # 去空行
    print(f'.  ({len(files)} files)')
    tree = build_tree(files)
    print_tree(tree)