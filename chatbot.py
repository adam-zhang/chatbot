#!/usr/bin/env python3
"""
DeepSeek API 命令行 Chatbot
- 多轮对话、流式输出
- 历史保存/加载（启动时加载并回显）
- 退出时提示保存
- 每轮对话之间有分隔线
- 支持多行输入（单独一行 EOF 结束）
- 支持 /file <path> 把整篇文件作为一条消息发送
"""

import os
import sys
import json
import argparse
import readline
from datetime import datetime
from typing import List, Dict

import requests


SEPARATOR = "=" * 50
END_MARKER = "EOF"


class DeepSeekChatbot:
    def __init__(
        self,
        api_key: str,
        #model: str = "deepseek-chat",
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        system_prompt: str = "You are a helpful limitless assistant.",
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
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        try:
            if self.stream:
                response_text = self._stream_response()
            else:
                response_text = self._blocking_response()
            self.messages.append({"role": "assistant", "content": response_text})
            return response_text
        except Exception:
            self.messages.pop()
            raise

    def clear_history(self, system_prompt: str = None):
        prompt = system_prompt if system_prompt is not None else self.system_prompt
        self.messages = [{"role": "system", "content": prompt}]

    def save_history(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def load_history(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("历史文件格式错误：顶层应为消息列表")
        for i, m in enumerate(data):
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError(f"历史文件格式错误：第 {i} 条消息缺少 role/content")
        if not data or data[0].get("role") != "system":
            data = [{"role": "system", "content": self.system_prompt}] + data
        self.messages = data

    def print_history(self, include_system: bool = False):
        if not self.messages:
            print("(历史为空)")
            return
        chat_msgs = [m for m in self.messages if m.get("role") != "system"]
        if not chat_msgs and not include_system:
            print("(历史中只有 system 消息)")
            return
        print(SEPARATOR)
        for m in self.messages:
            role = m.get("role", "?")
            content = m.get("content", "") or ""
            if role == "system":
                if include_system:
                    print(f"[system] {content}")
                continue
            label = {"user": "你", "assistant": "AI"}.get(role, role)
            lines = content.splitlines() or [""]
            print(f"{label}: {lines[0]}")
            for extra in lines[1:]:
                print(f"    {extra}")
        print(SEPARATOR + "\n")

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
        resp = requests.post(
            self._endpoint(), headers=self._headers(),
            json=self._payload(stream=False), timeout=self.timeout,
        )
        self._raise_for_status(resp)
        content = resp.json()["choices"][0]["message"]["content"] or ""
        print(content)
        return content

    def _stream_response(self) -> str:
        resp = requests.post(
            self._endpoint(), headers=self._headers(),
            json=self._payload(stream=True), stream=True, timeout=self.timeout,
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
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                print(piece, end="", flush=True)
                chunks.append(piece)
        print()
        return "".join(chunks)

    @staticmethod
    def _raise_for_status(resp: requests.Response):
        if resp.status_code < 400:
            return
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API 请求失败 (HTTP {resp.status_code}): {detail}")


# ----------------------------------------------------------------------
# 多行输入
# ----------------------------------------------------------------------
def read_multiline(prompt=">>> ") -> str:
    """
    多行输入：
    - 第一行以 / 开头 → 命令，直接返回
    - 否则进入多行模式
    - 连续两个空行（即先一个空行，再一个空行）结束
      更简单：遇到空行即结束（见下方说明）
    """
    try:
        first = input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise

    stripped = first.strip()
    if stripped.startswith("/"):
        return stripped
    if stripped == "":
        return ""

    print("(多行输入模式，空行结束)")
    lines = [first]
    while True:
        try:
            line = input()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n(已取消本次输入)")
            return ""
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)



# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
HELP_TEXT = f"""
可用命令:
  /help             显示帮助
  /clear            清空对话历史
  /save <file>      保存对话历史到文件
  /load <file>      从文件加载对话历史（并回显）
  /file <file>      把整个文件作为一条消息发送
  /history          重新打印当前对话历史
  /exit 或 /quit    退出（会提示是否保存）

多行输入:
  直接输入第一行（不以 / 开头），进入多行模式，
  单独一行输入 {END_MARKER} 表示结束，整段作为一条消息发送。

示例:
  >>> 请帮我总结下面这段文字：
  ... （粘贴多行）
  ... {END_MARKER}
"""


def print_help():
    print(HELP_TEXT)


def ask_save_on_exit(bot: DeepSeekChatbot, default_path: str = None) -> bool:
    if len(bot.messages) <= 1:
        print("再见！")
        return True
    default_path = default_path or "chat_history_{}.json".format(
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    print("\n是否保存本次对话？")
    print(f"  [y] 保存   (默认路径: {default_path})")
    print("  [p] 保存到指定路径")
    print("  [n] 不保存，直接退出")
    print("  [c] 取消退出，继续对话")
    while True:
        try:
            choice = input("请选择 [y/p/n/c]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return True
        if choice in ("", "y", "yes"):
            try:
                bot.save_history(default_path)
                print(f"已保存到 {default_path}")
            except Exception as e:
                print(f"保存失败: {e}", file=sys.stderr)
                try:
                    again = input("保存失败，仍要退出吗？[y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    again = "y"
                if again not in ("y", "yes"):
                    return False
            return True
        elif choice in ("p", "path"):
            try:
                path = input("请输入保存路径: ").strip()
            except (EOFError, KeyboardInterrupt):
                path = ""
            if not path:
                print("路径为空，已取消保存。")
                continue
            try:
                bot.save_history(path)
                print(f"已保存到 {path}")
                return True
            except Exception as e:
                print(f"保存失败: {e}", file=sys.stderr)
                continue
        elif choice in ("n", "no"):
            print("已放弃保存，再见！")
            return True
        elif choice in ("c", "cancel"):
            print("已取消退出，继续对话。\n")
            return False
        else:
            print("无效输入，请输入 y / p / n / c。")


def send_file(bot: DeepSeekChatbot, path: str):
    """读文件并作为一条消息发送"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"文件不存在: {path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"读取失败: {e}", file=sys.stderr)
        return

    if not content.strip():
        print("文件为空，未发送。")
        return

    print(f"已读入 {len(content)} 字符，发送中...\n<<< ", end="", flush=True)
    try:
        bot.chat(content)
    except Exception as e:
        print(f"\n[错误] {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="DeepSeek 命令行 Chatbot")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"),
                        help="DeepSeek API Key (默认读取 DEEPSEEK_API_KEY)")
    parser.add_argument("--model", default="deepseek-chat",
                        choices=["deepseek-chat", "deepseek-reasoner"])
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--load", default=None, help="启动时加载历史文件")
    parser.add_argument("--no-echo-history", action="store_true",
                        help="加载历史后不回显")
    parser.add_argument("--show-system", action="store_true",
                        help="回显历史时显示 system 消息")
    parser.add_argument("--save-log", default=None,
                        help="退出时自动保存到指定文件")
    args = parser.parse_args()

    if not args.api_key:
        print("错误: 未提供 API Key。请设置 DEEPSEEK_API_KEY 或 --api-key。",
              file=sys.stderr)
        sys.exit(1)

    bot = DeepSeekChatbot(
        api_key=args.api_key, model=args.model, base_url=args.base_url,
        system_prompt=args.system, stream=not args.no_stream,
        temperature=args.temperature, timeout=args.timeout,
    )

    if args.load:
        try:
            bot.load_history(args.load)
            print(f"已从 {args.load} 加载 {len(bot.messages)} 条历史消息。\n")
            if not args.no_echo_history:
                bot.print_history(include_system=args.show_system)
        except FileNotFoundError:
            print(f"[警告] 历史文件不存在: {args.load}，将开始新对话。",
                  file=sys.stderr)
        except Exception as e:
            print(f"[警告] 加载历史失败: {e}，将开始新对话。", file=sys.stderr)

    print(f"DeepSeek Chatbot 已启动 [模型: {args.model}]")
    print(f"输入 /help 查看命令，/exit 退出。多行输入以单独一行 {END_MARKER} 结束。\n")

    while True:
        print(SEPARATOR)
        try:
            user_input = read_multiline(">>> ")
        except (EOFError, KeyboardInterrupt):
            if args.save_log:
                try:
                    bot.save_history(args.save_log)
                    print(f"\n对话已保存到 {args.save_log}")
                except Exception as e:
                    print(f"\n保存失败: {e}", file=sys.stderr)
                break
            if ask_save_on_exit(bot):
                break
            else:
                continue

        if not user_input:
            continue

        # ---- 命令 ----
        if user_input in ("/exit", "/quit"):
            if args.save_log:
                try:
                    bot.save_history(args.save_log)
                    print(f"对话已保存到 {args.save_log}")
                except Exception as e:
                    print(f"保存失败: {e}", file=sys.stderr)
                break
            if ask_save_on_exit(bot):
                break
            else:
                continue
        elif user_input == "/help":
            print_help()
            continue
        elif user_input == "/clear":
            bot.clear_history(args.system)
            print("对话历史已清空。")
            continue
        elif user_input == "/history":
            bot.print_history(include_system=args.show_system)
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
                print(f"已从 {path} 加载 {len(bot.messages)} 条消息。\n")
                if not args.no_echo_history:
                    bot.print_history(include_system=args.show_system)
            except FileNotFoundError:
                print(f"文件不存在: {path}")
            except Exception as e:
                print(f"加载失败: {e}", file=sys.stderr)
            continue
        elif user_input.startswith("/file "):
            path = user_input[6:].strip()
            send_file(bot, path)
            print()
            continue

        # ---- 正常对话 ----
        print("<<< ", end="", flush=True)
        try:
            bot.chat(user_input)
        except Exception as e:
            print(f"\n[错误] {e}", file=sys.stderr)
        print()


if __name__ == "__main__":
    main()
