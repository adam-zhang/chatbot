#!/usr/bin/env python3
"""
把 chatbot 保存的 JSON 历史文件转换成 Markdown。

用法:
    python3 json2md.py chat.json                     # 输出到 chat.md
    python3 json2md.py chat.json -o output.md        # 指定输出文件
    python3 json2md.py chat.json --title "我的对话"   # 自定义标题
    python3 json2md.py chat.json --show-system       # 显示 system 消息
    python3 json2md.py chat.json --stdout            # 输出到标准输出
    python3 json2md.py *.json --outdir md/           # 批量转换到 md/ 目录
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any


def load_history(path: str) -> List[Dict[str, Any]]:
    """加载并校验 JSON 历史文件"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path}: 顶层应为消息列表，实际是 {type(data).__name__}")

    for i, m in enumerate(data):
        if not isinstance(m, dict):
            raise ValueError(f"{path}: 第 {i} 条消息不是对象")
        if "role" not in m:
            raise ValueError(f"{path}: 第 {i} 条消息缺少 role")
        # content 可能为 None（比如某些工具调用），这里统一成空串
        if m.get("content") is None:
            m["content"] = ""

    return data


def escape_md(text: str) -> str:
    """转义 Markdown 中可能破坏结构的内联字符（保守处理）"""
    # 不转义内容里的 # * ` 等，因为聊天内容里这些很常见，转义反而难看
    # 只处理可能被误认为 HTML 的尖括号，避免 < > 影响渲染
    return text.replace("<", "&lt;").replace(">", "&gt;")


def format_message(msg: Dict[str, Any], show_system: bool) -> str:
    """把一条消息格式化成 Markdown 片段"""
    role = msg.get("role", "unknown")
    content = msg.get("content", "") or ""

    if role == "system":
        if not show_system:
            return ""
        return f"### ⚙️ System\n\n{content}\n\n"

    if role == "user":
        return f"### 🧑 用户\n\n{content}\n\n"

    if role == "assistant":
        return f"### 🤖 助手\n\n{content}\n\n"

    # 其它角色（如 tool）也一并输出，便于查看完整历史
    return f"### {role}\n\n{content}\n\n"


def convert(
    messages: List[Dict[str, Any]],
    title: str = None,
    source: str = None,
    show_system: bool = False,
) -> str:
    """把消息列表转换成 Markdown 文本"""
    lines: List[str] = []

    # ---- 标题 ----
    if title is None:
        title = os.path.splitext(os.path.basename(source or "对话"))[0]
    lines.append(f"# {title}")
    lines.append("")

    # ---- 元信息 ----
    if source:
        lines.append(f"- 来源文件: `{source}`")
    lines.append(f"- 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total = len(messages)
    users = sum(1 for m in messages if m.get("role") == "user")
    assistants = sum(1 for m in messages if m.get("role") == "assistant")
    systems = sum(1 for m in messages if m.get("role") == "system")
    lines.append(f"- 消息总数: {total}（system {systems} / user {users} / assistant {assistants}）")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 正文 ----
    rendered_count = 0
    for msg in messages:
        block = format_message(msg, show_system)
        if block:
            lines.append(block)
            rendered_count += 1

    if rendered_count == 0:
        lines.append("_(没有可显示的消息)_\n")

    return "\n".join(lines)


def default_output_path(input_path: str, outdir: str = None) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0] + ".md"
    if outdir:
        return os.path.join(outdir, base)
    return os.path.join(os.path.dirname(input_path) or ".", base)


def main():
    parser = argparse.ArgumentParser(
        description="把 chatbot 的 JSON 历史文件转换成 Markdown"
    )
    parser.add_argument("inputs", nargs="+", help="输入的 JSON 文件（可多个）")
    parser.add_argument("-o", "--output", help="输出文件（仅单个输入时有效）")
    parser.add_argument("--outdir", help="批量转换时的输出目录")
    parser.add_argument("--title", help="Markdown 标题（仅单个输入时有效）")
    parser.add_argument(
        "--show-system", action="store_true", help="显示 system 消息"
    )
    parser.add_argument(
        "--stdout", action="store_true", help="输出到标准输出而不写文件"
    )
    args = parser.parse_args()

    if args.output and len(args.inputs) > 1:
        print("错误: -o 只能在单个输入时使用，多个输入请用 --outdir", file=sys.stderr)
        sys.exit(1)

    if args.title and len(args.inputs) > 1:
        print("错误: --title 只能在单个输入时使用", file=sys.stderr)
        sys.exit(1)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    for path in args.inputs:
        try:
            messages = load_history(path)
        except FileNotFoundError:
            print(f"跳过: 文件不存在 {path}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"跳过: {path}: {e}", file=sys.stderr)
            continue

        md = convert(
            messages,
            title=args.title,
            source=path,
            show_system=args.show_system,
        )

        if args.stdout:
            print(md)
            continue

        if args.output:
            out_path = args.output
        else:
            out_path = default_output_path(path, args.outdir)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)

        print(f"已生成: {out_path}")


if __name__ == "__main__":
    main() 
