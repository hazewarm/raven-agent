"""raven-agent Dashboard API 模块。

提供 RESTful HTTP API 用于查询和管理 session、message、
memory、proactive、observe 数据，以及触发运维操作。
同时托管前端静态资源和插件 Dashboard 面板。
"""

from raven_agent.api.dashboard import create_dashboard_app

__all__ = ["create_dashboard_app"]