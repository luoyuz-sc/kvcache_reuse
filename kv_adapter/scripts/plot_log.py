#!/usr/bin/env python3
import re
import ast
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_file(path):
    with open(path, 'r') as f:
        s = f.read()
    # 找到类似 `list: [1.23, 4.56, ...]` 的部分
    m = re.search(r"list:\s*(\[[^\]]*\])", s, re.S)
    if not m:
        raise RuntimeError("没有在日志中找到 'list: [...]' 部分")
    lst = ast.literal_eval(m.group(1))
    return lst


def main():
    parser = argparse.ArgumentParser(description='Plot scatter from log file')
    parser.add_argument('--input', '-i', default='test_attn1.log', help='输入日志文件路径')
    parser.add_argument('--output', '-o', default='test_attn1_scatter.png', help='输出图片路径')
    args = parser.parse_args()

    data = parse_file(args.input)
    x = list(range(len(data)))
    y = data

    plt.figure(figsize=(10,5))
    plt.scatter(x, y, s=30, alpha=0.7, label='values')
    plt.plot(x, y, alpha=0.3)
    plt.xlabel('Index')
    plt.ylabel('Loss(CE)')
    plt.title(f'Scatter plot of values from {args.input}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print('Saved', args.output)


if __name__ == '__main__':
    main()
