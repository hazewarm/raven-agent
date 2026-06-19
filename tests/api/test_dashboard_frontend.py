"""Dashboard 前端接口测试 —— 静态文件、插件面板、首页。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def _make_fastapi_app(
    tmp_path: Path,
    *,
    with_static: bool = True,
    with_plugins: bool = True,
):
    """创建测试用的 FastAPI app。

    输入:
        tmp_path: 临时目录。
        with_static: 是否注册静态文件挂载。
        with_plugins: 是否注册插件面板路由。

    输出:
        (FastAPI app, TestClient)。
    """
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi 未安装")

    from raven_agent.api.dashboard import create_dashboard_app
    from raven_agent.memory import DisabledMemoryEngine
    from raven_agent.session import SessionManager
    from raven_agent.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions.db")
    sessions = SessionManager(store)
    memory_admin = DisabledMemoryEngine()

    static_dir = None
    plugins_root = None
    project_root = tmp_path

    if with_static:
        static_dir = tmp_path / "static" / "dashboard"
        static_dir.mkdir(parents=True)
        (static_dir / "index.html").write_text(
            '<!DOCTYPE html><html><head></head><body>'
            '<link rel="stylesheet" href="/assets/styles.css">'
            '<script src="/assets/app.js"></script>'
            '<div id="root"></div></body></html>'
        )
        # 创建虚拟的 app.js 和 styles.css
        (static_dir / "app.js").write_text("// mock app.js")
        (static_dir / "styles.css").write_text("/* mock styles.css */")

    if with_plugins:
        plugins_root = tmp_path / "plugins"
        plugins_root.mkdir(parents=True)
        test_plugin = plugins_root / "test_plugin"
        test_plugin.mkdir()
        # 模拟已编译的 dashboard_panel.js
        panel_js = test_plugin / "dashboard_panel.js"
        panel_js.write_text("// mock dashboard panel")
        panel_css = test_plugin / "dashboard_panel.css"
        panel_css.write_text("/* mock dashboard panel css */")

    app = create_dashboard_app(
        workspace=tmp_path,
        store=store,
        sessions=sessions,
        memory_admin=memory_admin,
        api_key="",
        project_root=project_root if with_plugins else None,
        plugins_root=plugins_root if with_plugins else None,
        static_dir=static_dir if with_static else None,
    )

    return app, TestClient(app)


class TestStaticFilesAndIndex:
    """验证静态文件托管和首页路由。"""

    @pytest.fixture
    def client(self, tmp_path):
        _, client = _make_fastapi_app(tmp_path, with_static=True, with_plugins=False)
        return client

    def test_index_returns_html(self, client):
        """访问 / 返回 index.html。"""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert '<div id="root">' in resp.text

    def test_index_has_cache_busting(self, client):
        """index.html 中的 JS/CSS 引用包含版本号。"""
        resp = client.get("/")
        assert "/assets/app.js?v=" in resp.text
        assert "/assets/styles.css?v=" in resp.text

    def test_assets_app_js(self, client):
        """/assets/app.js 返回 JavaScript。"""
        resp = client.get("/assets/app.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"]

    def test_assets_styles_css(self, client):
        """/assets/styles.css 返回 CSS。"""
        resp = client.get("/assets/styles.css")
        assert resp.status_code == 200
        assert "css" in resp.headers["content-type"]


class TestPluginPanels:
    """验证插件面板发现和托管。"""

    @pytest.fixture
    def client(self, tmp_path):
        _, client = _make_fastapi_app(tmp_path, with_static=False, with_plugins=True)
        return client

    def test_list_plugins(self, client):
        """GET /api/dashboard/plugins 返回面板列表。"""
        resp = client.get("/api/dashboard/plugins")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        plugin = data[0]
        assert "id" in plugin
        assert "panels" in plugin
        assert len(plugin["panels"]) >= 1

    def test_get_plugin_panel_js(self, client):
        """GET /plugins/{id}/{name}.js 返回 JS 文件。"""
        resp = client.get("/plugins/test_plugin/dashboard_panel.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"]

    def test_get_plugin_panel_css(self, client):
        """GET /plugins/{id}/{name}.css 返回 CSS 文件。"""
        resp = client.get("/plugins/test_plugin/dashboard_panel.css")
        assert resp.status_code == 200
        assert "css" in resp.headers["content-type"]

    def test_plugin_panel_not_found(self, client):
        """不存在的面板返回 404。"""
        resp = client.get("/plugins/test_plugin/nonexistent.js")
        assert resp.status_code == 404

    def test_plugin_panel_invalid_name(self, client):
        """不以 dashboard_panel 开头的面板名返回 404。"""
        resp = client.get("/plugins/test_plugin/evil.js")
        assert resp.status_code == 404

    def test_plugin_not_found(self, client):
        """不存在的插件返回 404。"""
        resp = client.get("/plugins/nonexistent/dashboard_panel.js")
        assert resp.status_code == 404


class TestDashboardAccessLogFilter:
    """验证 Dashboard 访问日志过滤器。"""

    def test_filter_dashboard_paths(self):
        """Dashboard 路径被识别。"""
        import logging
        from raven_agent.api.dashboard import (
            DashboardAccessLogFilter,
            _is_dashboard_access_record,
        )

        # 模拟 uvicorn.access 日志记录
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="%s - %s",
            args=(("127.0.0.1", 12345), "GET", "/api/dashboard/sessions"),
            exc_info=None,
        )
        assert _is_dashboard_access_record(record) is True

    def test_filter_non_dashboard_paths(self):
        """非 Dashboard 路径不被过滤器拦截。"""
        import logging
        from raven_agent.api.dashboard import (
            DashboardAccessLogFilter,
            _is_dashboard_access_record,
        )

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="%s - %s",
            args=(("127.0.0.1", 12345), "GET", "/healthz"),
            exc_info=None,
        )
        assert _is_dashboard_access_record(record) is False

    def test_filter_installed(self):
        """过滤器安装后存在于 uvicorn.access logger。"""
        import logging
        from raven_agent.api.dashboard import (
            DashboardAccessLogFilter,
            _install_dashboard_access_log_filter,
        )

        access_logger = logging.getLogger("uvicorn.access")
        before = any(
            isinstance(f, DashboardAccessLogFilter)
            for f in access_logger.filters
        )
        _install_dashboard_access_log_filter()
        after = any(
            isinstance(f, DashboardAccessLogFilter)
            for f in access_logger.filters
        )
        # 如果之前没有，安装后应该有
        assert after or before