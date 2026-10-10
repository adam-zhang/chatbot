#!/usr/bin/env python3.9
"""
DeepSeek API 命令行 Chatbot (基于 requests, 无 Rust 编译依赖)
支持多轮对话、流式输出、历史记录保存
"""

import os
import sys
import json
import argparse
from typing import List, Dict

import requests


class DeepSeekChatbot:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        system_prompt: str = "You are a helpful assistant.",
        stream: bool = True,
        temperature: float = 1.0,
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.stream = stream
        self.temperature = temperature
        self.timeout = timeout
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        """发送消息并获取回复"""
        self.messages.append({"role": "user", "content": user_input})

        try:
            if self.stream:
                response_text = self._stream_response()
            else:
                response_text = self._blocking_response()

            self.messages.append({"role": "assistant", "content": response_text})
            return response_text
        except Exception as e:
            # 失败时回滚用户消息，保持历史干净
            self.messages.pop()
            raise e

    def clear_history(self, system_prompt: str = "You are a helpful assistant."):
        self.messages = [{"role": "system", "content": system_prompt}]

    def save_history(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def load_history(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.messages = json.load(f)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _payload(self, stream: bool) -> Dict:
        return {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "stream": stream,
        }

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _blocking_response(self) -> str:
        """非流式请求"""
        resp = requests.post(
            self._endpoint(),
            headers=self._headers(),
            json=self._payload(stream=False),
            timeout=self.timeout,
        )
        self._raise_for_status(resp)
        data = resp.json()
        content = data["choices"][0]["message"]["content"] or ""
        print(content)
        return content

    def _stream_response(self) -> str:
        """流式请求，逐块打印"""
        resp = requests.post(
            self._endpoint(),
            headers=self._headers(),
            json=self._payload(stream=True),
            stream=True,
            timeout=self.timeout,
        )
        self._raise_for_status(resp)

        chunks: List[str] = []
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = event.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                print(piece, end="", flush=True)
                chunks.append(piece)

        print()  # 收尾换行
        return "".join(chunks)

    @staticmethod
    def _raise_for_status(resp: requests.Response):
        """把 HTTP 错误转换成更友好的提示"""
        if resp.status_code < 400:
            return
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(
            f"API 请求失败 (HTTP {resp.status_code}): {detail}"
        )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
HELP_TEXT = """
可用命令:
  /help          显示帮助
  /clear         清空对话历史
  /save <file>   保存对话历史到文件
  /load <file>   从文件加载对话历史
  /history       显示当前对话历史条数
  /exit 或 /quit 退出
"""


def print_help():
    print(HELP_TEXT)


def main():
    parser = argparse.ArgumentParser(description="DeepSeek 命令行 Chatbot")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help="DeepSeek API Key (默认读取环境变量 DEEPSEEK_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        choices=["deepseek-chat", "deepseek-reasoner"],
        help="使用的模型",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="API 基础地址",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="系统提示词",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=120, help="请求超时秒数")
    parser.add_argument(
        "--no-stream", action="store_true", help="禁用流式输出"
    )
    parser.add_argument(
        "--save-log",
        default=None,
        help="退出时自动保存对话到指定文件",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "错误: 未提供 API Key。请设置环境变量 DEEPSEEK_API_KEY "
            "或使用 --api-key 参数。",
            file=sys.stderr,
        )
        sys.exit(1)

    bot = DeepSeekChatbot(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        system_prompt=args.system,
        stream=not args.no_stream,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    print(f"DeepSeek Chatbot 已启动 [模型: {args.model}]")
    print("输入 /help 查看命令，输入 /exit 退出。\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # ---- 命令处理 ----
        if user_input in ("/exit", "/quit"):
            print("再见！")
            break
        elif user_input == "/help":
            print_help()
            continue
        elif user_input == "/clear":
            bot.clear_history(args.system)
            print("对话历史已清空。")
            continue
        elif user_input == "/history":
            print(f"当前历史消息数: {len(bot.messages)}")
            continue
        elif user_input.startswith("/save "):
            path = user_input[6:].strip()
            try:
                bot.save_history(path)
                print(f"已保存到 {path}")
            except Exception as e:
                print(f"保存失败: {e}", file=sys.stderr)
            continue
        elif user_input.startswith("/load "):
            path = user_input[6:].strip()
            try:
                bot.load_history(path)
                print(f"已从 {path} 加载 {len(bot.messages)} 条消息。")
            except FileNotFoundError:
                print(f"文件不存在: {path}")
            except Exception as e:
                print(f"加载失败: {e}", file=sys.stderr)
            continue

        # ---- 正常对话 ----
        print("<<< ", end="", flush=True)
        try:
            bot.chat(user_input)
        except Exception as e:
            print(f"\n[错误] {e}", file=sys.stderr)

    # 退出时自动保存
    if args.save_log:
        try:
            bot.save_history(args.save_log)
            print(f"对话已保存到 {args.save_log}")
        except Exception as e:
            print(f"保存失败: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
