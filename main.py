from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path

from raven_agent.app import AppRuntime
from raven_agent.channels import run_client
from raven_agent.config import load_config

import logging
logging.basicConfig(level=logging.WARNING)  # 设置日志级别为 WARNING 或更高

logging.getLogger("raven_agent.proactive").setLevel(logging.INFO)
logging.getLogger("raven_agent.peer").setLevel(logging.INFO)
logging.getLogger("raven_agent.plugins").setLevel(logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    输入:
        无。

    输出:
        配置好子命令的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(prog="main.py", description="raven-agent 入口")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="启动 IPC Server 模式")

    dash_parser = sub.add_parser("dashboard", help="启动 Dashboard API 服务")
    sub.add_parser("backup", help="手动触发一次全量数据库备份")
    dash_parser.add_argument(
        "--host",
        default=None,
        help="HTTP 监听地址；缺省时读取 config.toml 的 [dashboard].host",
    )
    dash_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP 监听端口；缺省时读取 config.toml 的 [dashboard].port",
    )

    cli_parser = sub.add_parser("cli", help="启动 IPC Client 模式")
    cli_parser.add_argument(
        "--socket",
        default=None,
        help="服务端地址；缺省时读取 config.toml 的 [channels].socket",
    )
    group = cli_parser.add_mutually_exclusive_group()
    group.add_argument(
        "-c",
        "--continue",
        dest="continue_latest",
        action="store_true",
        help="继续最近一次 CLI session",
    )
    group.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="列出历史 CLI session 并手动选择",
    )
    return parser


async def run_serve() -> None:
    """启动 IPC Server。

    输入:
        无。

    输出:
        None。
    """
    config = load_config("config.toml")
    if not config.channels.ipc_enabled:
        config = replace(
            config,
            channels=replace(config.channels, ipc_enabled=True, cli_enabled=False),
        )
    app = AppRuntime.create(config, workspace=Path(".raven"), allowed_dir=Path.cwd())
    await app.start()
    try:
        await app.run_serve_loop()
    finally:
        await app.stop()

async def run_dashboard(args: argparse.Namespace) -> None:
    """启动 Dashboard API 独立服务。

    输入:
        args: 解析后的 dashboard 子命令参数。

    输出:
        None。阻塞直到 Ctrl+C。
    """
    config = load_config("config.toml")
    if args.host is not None or args.port is not None:
        host = args.host or config.dashboard.host
        port = args.port or config.dashboard.port
        config = replace(
            config,
            dashboard=replace(config.dashboard, host=host, port=port),
        )
    # 确保至少 CLI 关闭——dashboard 模式不需要交互式 CLI
    config = replace(
        config,
        channels=replace(config.channels, cli_enabled=False, ipc_enabled=False),
    )
    # Dashboard 强制开启
    config = replace(config, dashboard=replace(config.dashboard, enabled=True))

    app = AppRuntime.create(config, workspace=Path(".raven"), allowed_dir=Path.cwd())
    await app.start()
    print(
        f"Raven Agent Dashboard API 已启动\n"
        f"  地址: http://{config.dashboard.host}:{config.dashboard.port}\n"
        f"  API 文档: http://{config.dashboard.host}:{config.dashboard.port}/docs\n"
        f"按 Ctrl+C 停止服务"
    )
    try:
        # 保持运行直到收到停止信号
        while not app._stopped:
            await asyncio.sleep(1)
    finally:
        await app.stop()

async def run_backup(args: argparse.Namespace) -> None:
    """执行一次全量数据库备份并退出。

    输入:
        args: 解析后的命令行参数（未使用）。

    输出:
        None。打印备份结果后退出。
    """
    from pathlib import Path
    from raven_agent.plugins.builtins.observe.backup import backup_databases

    workspace = Path(".raven")
    results = backup_databases(workspace)
    print("数据库备份完成：")
    for name, path in results.items():
        status = path if path else "（源文件不存在，已跳过）"
        print(f"  {name}: {status}")

def run_cli(args: argparse.Namespace) -> None:
    """启动 IPC Client，按参数决定 session 选择模式。

    输入:
        args: 解析后的 cli 子命令参数。

    输出:
        None。
    """
    config = load_config("config.toml")
    socket_path = args.socket or config.channels.socket
    if args.resume:
        mode = "resume"
    elif args.continue_latest:
        mode = "continue"
    else:
        mode = "new"
    run_client(socket_path, mode=mode)


def main() -> None:
    """程序入口。

    输入:
        sys.argv 命令行参数。

    输出:
        None。
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "serve":
            asyncio.run(run_serve())
        elif args.command == "cli":
            run_cli(args)
        elif args.command == "dashboard":
            asyncio.run(run_dashboard(args))
        elif args.command == "backup":
            asyncio.run(run_backup(args))
        else:
            parser.print_help()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()