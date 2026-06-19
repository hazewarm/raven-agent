"""插件 Dashboard 面板的编译与 HTTP 托管。

本模块负责：
- 发现插件目录下的 dashboard_panel*.ts 文件
- 用 esbuild 编译为 JS
- 提供 HTTP 路由托管编译产物（JS + CSS）
- 提供 /api/dashboard/plugins 发现端点

输入:
    plugins_root: 用户插件目录的 Path。

输出:
    setup_plugin_panels(app, plugins_root, project_root) — 在 FastAPI app 上注册路由。
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger("dashboard.plugins")

# 待编译插件列表（esbuild 还未就绪时暂存）
_pending_plugins: list[tuple[Path, Path]] = []
_pending_plugins_lock = threading.Lock()


# ── esbuild 命令查找 ─────────────────────────────────────────────────

def _esbuild_command(project_root: Path) -> list[str] | None:
    """查找 esbuild 可执行命令。

    优先级:
        1. 项目本地 node_modules/.bin/esbuild
        2. 系统全局 npx + esbuild

    输入:
        project_root: 项目根目录。

    输出:
        命令列表（可直接传给 subprocess），或 None（不可用）。
    """
    bin_name = "esbuild.cmd" if os.name == "nt" else "esbuild"
    local_bin = project_root / "node_modules" / ".bin" / bin_name
    if local_bin.exists():
        return [str(local_bin)]
    if os.name == "nt":
        cmd_bin = shutil.which("cmd.exe") or shutil.which("cmd")
        npx_bin = shutil.which("npx.cmd") or shutil.which("npx")
        if cmd_bin and npx_bin:
            return [cmd_bin, "/d", "/s", "/c", "npx", "--yes", "esbuild"]
        return None
    npx_bin = shutil.which("npx")
    if npx_bin:
        return [npx_bin, "--yes", "esbuild"]
    return None


# ── 编译单个插件的面板 ────────────────────────────────────────────────

def _build_plugin_panels_js(project_root: Path, plugin_dir: Path) -> None:
    """编译某个插件目录下所有 dashboard_panel*.ts 文件。

    跳过已是最新的——如果 .js 的 mtime ≥ .ts 的 mtime，不重复编译。

    输入:
        project_root: 项目根目录。
        plugin_dir: 插件目录（如 plugins/default_memory）。

    输出:
        None。编译产物写入 plugin_dir 下同名 .js 文件。
    """
    esbuild_cmd: list[str] | None = None
    for ts_path in sorted(plugin_dir.glob("dashboard_panel*.ts")):
        js_path = ts_path.with_suffix(".js")
        if js_path.exists() and js_path.stat().st_mtime >= ts_path.stat().st_mtime:
            continue
        if esbuild_cmd is None:
            esbuild_cmd = _esbuild_command(project_root)
        if esbuild_cmd is None:
            # esbuild 不可用 → 暂存到待编译列表
            with _pending_plugins_lock:
                _pending_plugins.append((project_root, plugin_dir))
            return
        _run_esbuild(esbuild_cmd, ts_path, js_path,
                     f"{plugin_dir.name}/{ts_path.stem}")


def _run_esbuild(
    cmd: list[str], ts_path: Path, js_path: Path, name: str,
) -> None:
    """执行 esbuild 编译单个 .ts 文件。

    输入:
        cmd: esbuild 命令列表。
        ts_path: TypeScript 源文件路径。
        js_path: 输出 JS 文件路径。
        name: 日志显示名称。

    输出:
        None。
    """
    try:
        result = subprocess.run(
            [
                *cmd,
                str(ts_path),
                f"--outfile={js_path}",
                "--bundle",
                "--format=iife",
                "--platform=browser",
                "--target=es2021",
                "--minify",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "esbuild 编译 %s 失败:\n%s",
                name, result.stderr[:500],
            )
        else:
            logger.info("esbuild 编译完成: %s", name)
    except Exception:
        logger.exception("esbuild 编译 %s 异常", name)


# ── 异步编译待处理插件 ──────────────────────────────────────────────

async def _compile_pending_plugins_async() -> None:
    """异步编译所有待处理的插件面板。

    在 FastAPI lifespan 中调用。首次启动时 esbuild 可能还未
    安装（npx 需要网络），此时插件面板被放入 _pending_plugins。
    lifespan 阶段再次尝试。

    输入:
        无。

    输出:
        None。
    """
    with _pending_plugins_lock:
        if not _pending_plugins:
            return
        pending = _pending_plugins.copy()
        _pending_plugins.clear()
    first_root = pending[0][0]

    logger.info("正在安装前端构建工具 (npx esbuild)...")
    esbuild_cmd = _esbuild_command(first_root)
    if esbuild_cmd is None:
        logger.warning(
            "esbuild 不可用（neither local install nor npx found），"
            "插件面板未编译"
        )
        return
    proc = await asyncio.create_subprocess_exec(
        *esbuild_cmd,
        "--version",
        cwd=str(first_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning(
            "npx esbuild 不可用 (%d)，插件面板未编译:\n%s",
            proc.returncode,
            stderr.decode("utf-8", errors="replace")[:500],
        )
        return
    version = stdout.decode("utf-8", errors="replace").strip()
    logger.info("npx esbuild 就绪 (%s)，开始编译插件面板...", version)
    for root, pdir in pending:
        for ts_path in sorted(pdir.glob("dashboard_panel*.ts")):
            js_path = ts_path.with_suffix(".js")
            if not (
                js_path.exists()
                and js_path.stat().st_mtime >= ts_path.stat().st_mtime
            ):
                _run_esbuild(
                    esbuild_cmd, ts_path, js_path,
                    f"{pdir.name}/{ts_path.stem}",
                )


# ── 插件目录解析（安全检查）────────────────────────────────────────

def _resolve_plugin_dir(plugins_root: Path, plugin_id: str) -> Path:
    """解析并校验插件目录。

    防止路径遍历攻击——确保解析后的路径在 plugins_root 下。

    输入:
        plugins_root: 插件根目录。
        plugin_id: 插件 ID（URL 中的路径段）。

    输出:
        插件目录的 Path。

    异常:
        HTTPException(404): 插件不存在或路径非法。
    """
    try:
        candidate = (plugins_root / plugin_id).resolve()
    except Exception:
        raise HTTPException(status_code=404, detail="plugin not found")
    root = plugins_root.resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(status_code=400, detail="invalid plugin id")
    return candidate


def _is_plugin_disabled(plugin_dir: Path) -> bool:
    """检查插件是否被禁用。

    输入:
        plugin_dir: 插件目录。

    输出:
        True 表示 plugin_dir/plugin.disabled 文件存在。
    """
    return (plugin_dir / "plugin.disabled").exists()


# ── 设置插件面板路由 ────────────────────────────────────────────────

def setup_plugin_panels(
    app: FastAPI,
    plugins_root: Path,
    project_root: Path,
) -> None:
    """在 FastAPI app 上注册插件面板相关路由。

    注册三个路由:
        GET /api/dashboard/plugins           — 列出所有插件的可用面板
        GET /plugins/{plugin_id}/{name}.js   — 托管编译后的 JS
        GET /plugins/{plugin_id}/{name}.css  — 托管面板 CSS

    输入:
        app: FastAPI 实例。
        plugins_root: 插件根目录。
        project_root: 项目根目录（用于查找 esbuild）。

    输出:
        None。
    """

    @app.get("/api/dashboard/plugins")
    def list_dashboard_plugins() -> list[dict[str, object]]:
        """列出所有可用插件的 Dashboard 面板。

        前端在启动时调用此端点，获取需要加载的插件面板 JS/CSS 列表。

        返回:
            [
              {
                "id": "default_memory",
                "panels": [
                  {
                    "name": "dashboard_panel",
                    "js_version": "1717200000000000000",
                    "has_css": true
                  }
                ]
              },
              ...
            ]
        """
        result: list[dict[str, object]] = []
        if not plugins_root.is_dir():
            return result

        for plugin_dir in sorted(plugins_root.iterdir()):
            if not plugin_dir.is_dir():
                continue
            if _is_plugin_disabled(plugin_dir):
                continue
            # 尝试编译
            _build_plugin_panels_js(project_root, plugin_dir)
            # 收集已编译的 .js 文件
            panels: list[dict[str, object]] = []
            for js_path in sorted(plugin_dir.glob("dashboard_panel*.js")):
                css_path = js_path.with_suffix(".css")
                panels.append({
                    "name": js_path.stem,
                    "js_version": str(js_path.stat().st_mtime_ns),
                    "has_css": css_path.exists(),
                })
            if panels:
                result.append({"id": plugin_dir.name, "panels": panels})

        return result

    @app.get("/plugins/{plugin_id}/{panel_name}.js")
    def get_plugin_panel_js(plugin_id: str, panel_name: str) -> FileResponse:
        """托管插件的 dashboard_panel JS 文件。

        输入:
            plugin_id: 插件 ID。
            panel_name: 面板名（不含 .js 扩展名）。

        输出:
            FileResponse — JavaScript 文件。

        异常:
            HTTPException(404): 面板不存在或插件被禁用。
        """
        if not panel_name.startswith("dashboard_panel"):
            raise HTTPException(status_code=404, detail="plugin panel not found")
        plugin_dir = _resolve_plugin_dir(plugins_root, plugin_id)
        if _is_plugin_disabled(plugin_dir):
            raise HTTPException(status_code=404, detail="plugin panel not found")
        _build_plugin_panels_js(project_root, plugin_dir)
        js_path = plugin_dir / f"{panel_name}.js"
        if not js_path.exists():
            raise HTTPException(status_code=404, detail="plugin panel not found")
        return FileResponse(js_path, media_type="application/javascript")

    @app.get("/plugins/{plugin_id}/{panel_name}.css")
    def get_plugin_panel_css(plugin_id: str, panel_name: str) -> FileResponse:
        """托管插件的 dashboard_panel CSS 文件。

        输入:
            plugin_id: 插件 ID。
            panel_name: 面板名（不含 .css 扩展名）。

        输出:
            FileResponse — CSS 文件。

        异常:
            HTTPException(404): 面板 CSS 不存在或插件被禁用。
        """
        if not panel_name.startswith("dashboard_panel"):
            raise HTTPException(status_code=404, detail="plugin panel css not found")
        plugin_dir = _resolve_plugin_dir(plugins_root, plugin_id)
        if _is_plugin_disabled(plugin_dir):
            raise HTTPException(status_code=404, detail="plugin panel css not found")
        css_path = plugin_dir / f"{panel_name}.css"
        if not css_path.exists():
            raise HTTPException(status_code=404, detail="plugin panel css not found")
        return FileResponse(css_path, media_type="text/css")


# ── 便捷入口：在 create_dashboard_app 中调用 ─────────────────────────

def register_plugin_panels(
    app: FastAPI,
    plugins_root: Path,
    project_root: Path,
) -> None:
    """注册插件面板相关路由。

    同时将延迟编译协程存入 app.state，供 lifespan 在事件循环就绪后调用。

    输入:
        app: FastAPI 实例。
        plugins_root: 插件根目录。
        project_root: 项目根目录。

    输出:
        None。
    """
    setup_plugin_panels(app, plugins_root, project_root)
    app.state._compile_pending = _compile_pending_plugins_async