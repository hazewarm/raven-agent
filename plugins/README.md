# 插件编写指南

一个插件是一个 `plugins/<name>/plugin.py`，继承 `Plugin` 并可选提供四种扩展：

| 扩展 | 作用 | 面向 |
| --- | --- | --- |
| `@tool` | 给 LLM 的新能力 | LLM |
| `@on_xxx` | 生命周期事件响应 | EventBus |
| `xxx_modules()` | 插入 Phase 链的结构化模块 | Pipeline |
| `@on_tool_pre/post/error` | 拦截/观察工具调用 | ToolExecutor |

---

## 1. 目录结构

```text
plugins/
└── my_plugin/
    ├── plugin.py            # 必需
    ├── manifest.yaml        # 可选
    ├── _conf_schema.json    # 可选
    ├── plugin_config.json   # 可选（不入库）
    └── plugin.disabled      # 可选，存在则跳过
```

启用 `config.toml`：

```toml
[plugins]
enabled = true
dirs = ["plugins"]
```

---

## 2. 最小骨架

```python
from raven_agent.plugins import Plugin


class MyPlugin(Plugin):
    name = "my_plugin"           # 必需（可被 manifest.yaml 覆盖）

    async def initialize(self) -> None:
        pass                     # 可选，加载后调用

    async def terminate(self) -> None:
        pass                     # 可选，停止时调用
```

PluginManager 加载后注入 `self.context`：

| 字段 | 说明 |
| --- | --- |
| `event_bus` | EventBus |
| `tool_registry` | ToolRegistry |
| `plugin_id` | 插件 ID |
| `plugin_dir` | 插件目录 Path |
| `kv_store` | 私有 KV 存储 |
| `config` | PluginConfig，无 schema 时为 None |
| `workspace` | workspace 根目录 |
| `session_manager` | SessionManager |
| `memory_engine` | MemoryEngine |

```python
# kv_store
self.context.kv_store.set("k", v)
self.context.kv_store.get("k", default=None)
self.context.kv_store.increment("k")          # → int

# config（需要 _conf_schema.json，否则为 None）
self.context.config.get("api_key", default="")
self.context.config.api_key                   # 属性访问
```

---

## 3. `@tool` — 给 LLM 的能力

```python
from raven_agent.plugins import tool

@tool(
    name,                           # str，必填。暴露给模型的名字
    *,
    risk="read-only",               # "read-only" | "write" | "external-side-effect"
    always_on=False,                # True=每轮默认可见；False=需 tool_search 解锁
    search_hint=None,               # str | None，搜索补充关键词
)
```

### 方法签名要求

```text
async def fn(self, event, param1: type1, param2: type2 = default) -> str:
    ...

self, event     — 必须前两个参数，否则 TypeError
event           — PluginToolEvent(plugin, context, tool_name, arguments)
其余参数        — LLM 传入，从类型注解 + docstring 的 Args 段自动推导 JSON Schema
                  (str→string, int→integer, float→number, bool→boolean,
                   list→array, dict→object，无默认值→required)
返回            — str
```

### 示例

```python
from raven_agent.plugins import Plugin, tool


class WeatherPlugin(Plugin):
    name = "weather"

    @tool("get_weather", risk="read-only", always_on=True, search_hint="天气 weather")
    async def get_weather(self, event, city: str, days: int = 1) -> str:
        """查询天气。

        Args:
            city: 城市名。
            days: 预报天数。
        """

        return f"{city} 未来 {days} 天：晴"
```

---

## 4. `@on_xxx` — EventBus 生命周期 handler

### 装饰器列表

| 装饰器 | 传入对象 | emit/observe | 用途 |
| --- | --- | --- | --- |
| `@on_turn_started()` | `TurnStarted` | emit | 入站打标 / 改写 inbound |
| `@on_before_turn()` | `BeforeTurnCtx` | emit | 命令拦截 / abort |
| `@on_before_reasoning()` | `BeforeReasoningCtx` | emit | 追加 hint / 禁用 section |
| `@on_prompt_render()` | `PromptRenderCtx` | emit | 追加 system section / hint |
| `@on_before_step()` | `BeforeStepCtx` | emit | step 前 hint / early_stop |
| `@on_after_step()` | `AfterStepCtx` | observe | 记录每步工具使用 |
| `@on_after_reasoning()` | `AfterReasoningCtx` | emit | 清理 reply / 追加 metadata |
| `@on_after_turn()` | `AfterTurnCtx` | observe | turn 后统计 |
| `@on_turn_completed()` | `TurnCompleted` | observe | 记忆整理 / 日志 |

### 规则

```text
所有装饰器可传 priority=int，越大越先执行。

emit 模式（turn_started / before_* / prompt_render / after_reasoning）：
  handler 必须 return ctx_or_event，用 return 来改写。

observe 模式（after_step / after_turn / turn_completed）：
  返回值被忽略，仅用于副作用。
```

### 示例

```python
from raven_agent.plugins import Plugin, on_before_turn, on_turn_completed


class MyPlugin(Plugin):
    name = "my"

    # — emit，必须 return —
    @on_before_turn()
    async def handle_ping(self, ctx):
        if ctx.content.strip() == "/ping":
            ctx.abort = True
            ctx.abort_reply = "pong"
        return ctx

    # — observe，return 被忽略 —
    @on_turn_completed()
    async def count(self, event):
        self.context.kv_store.increment("turns")
```

---

## 5. `xxx_modules()` — Phase module

用 `@on_xxx` handler 拿不到 `frame.input`（如 session），或需要跨模块依赖 / 精确插入位置时使用。

### 插件提供的钩子

```python
def before_turn_modules(self) -> list[object]: ...
def before_reasoning_modules(self) -> list[object]: ...
def prompt_render_modules(self) -> list[object]: ...
def before_step_modules(self) -> list[object]: ...
def after_step_modules(self) -> list[object]: ...
def after_reasoning_modules(self) -> list[object]: ...
def after_turn_modules(self) -> list[object]: ...
```

### module 写法

```text
每个 module 是一个对象：
  slot: str                     — 必需，全局唯一
  requires: tuple[str, ...]     — 可选，指向其他模块 slot 或其 produces 数据 slot
  produces: tuple[str, ...]     — 可选，声明额外产出的数据 slot 名
  async def run(self, frame) → frame

slot 命名建议：<插件名>.<功能>，如 "my.prompt"

run() 中：
  frame.input   — 阶段输入（如 before_reasoning 为 PassiveTurnState 含 session）
  frame.slots   — 模块间共享 dict，写 prefix slot 后由内置模块自动收集
  frame.output  — 一般不要写，由内置 return 模块处理
```

### 各阶段可用的 slot

| 阶段 | ctx slot | 可写 prefix slot（自动收集） |
| --- | --- | --- |
| before_turn | `before_turn:ctx` | `before_turn:extra_hint:*`、`before_turn:metadata:*` |
| before_reasoning | `before_reasoning:ctx` | `before_reasoning:extra_hint:*`、`before_reasoning:metadata:*` |
| prompt_render | `prompt_render:ctx` | `prompt_render:section_top:*`、`prompt_render:section_bottom:*`、`prompt_render:extra_hint:*` |
| before_step | `before_step:ctx` | `before_step:extra_hint:*` |
| after_step | `after_step:ctx` | `after_step:telemetry:*` |
| after_reasoning | `after_reasoning:ctx` | `after_reasoning:outbound_metadata:*` |
| after_turn | `after_turn:ctx` | `after_turn:telemetry:*` |

### 示例

```python
from raven_agent.plugins import Plugin


class PressureModule:
    """历史 > 50 条时注入提示。"""

    slot = "my.pressure"
    requires = ("prompt_render.build_ctx", "prompt_render:ctx")

    async def run(self, frame):
        session = frame.input.session          # PassiveTurnState.session
        if session and len(session.messages) > 50:
            frame.slots["prompt_render:section_bottom:my"] = (
                "历史较长，请优先总结。"
            )
        return frame


class MyPlugin(Plugin):
    name = "my"

    def prompt_render_modules(self):
        return [PressureModule()]
```

> 注意：`@on_prompt_render()` handler 拿到的 `PromptRenderCtx` **没有 session**；需要 session 时必须用 Phase module 从 `frame.input.session` 取。

---

## 6. `@on_tool_pre/post/error` — 工具调用拦截/观察

```python
@on_tool_pre(*, tool_name=None, priority=0)
@on_tool_post(*, tool_name=None, priority=0)
@on_tool_error(*, tool_name=None, priority=0)
```

```text
tool_name=None  → 匹配所有工具。
tool_name="x"   → 只匹配工具 x。
```

方法签名：

```text
async def hook(self, event) -> ...

event: PluginToolHookEvent
  字段: context / event / session_key / tool_name / arguments / call_id /
        metadata / result (成功时) / error (失败时)
```

### 返回值规则

```text
pre_tool_use:
  return None                  → 放行
  return dict                  → 用它替换工具参数
  return ToolHookOutcome(...)  → 可 deny

post_tool_use / post_tool_error:
  返回值被忽略，用于副作用。
  event.result / event.error  可读。
```

```python
from raven_agent.tools import ToolHookOutcome

ToolHookOutcome(
    decision="deny",         # "pass" | "deny"
    updated_arguments=None,  # 改写后的参数
    extra_message="",
    reason="",
)
```

### 示例

```python
from raven_agent.plugins import Plugin, on_tool_pre, on_tool_post
from raven_agent.tools import ToolHookOutcome


class MyPlugin(Plugin):
    name = "my"

    # 改写参数
    @on_tool_pre(tool_name="read_text_file")
    async def cap_read(self, event):
        args = dict(event.arguments)
        args.setdefault("max_chars", 2000)
        return args

    # 拒绝危险调用
    @on_tool_pre(tool_name="shell")
    async def guard_shell(self, event):
        if "rm -rf" in str(event.arguments.get("command", "")):
            return ToolHookOutcome(decision="deny", reason="禁止递归强删")
        return None

    # 观察结果
    @on_tool_post(tool_name="read_text_file")
    async def count_read(self, event):
        self.context.kv_store.increment("read_count")
```

---

## 7. manifest.yaml 与配置

```yaml
# manifest.yaml（可选，优先级高于类属性）
name: my_plugin
version: 0.1.0
desc: 描述
author: 作者
```

```json
// _conf_schema.json（可选，声明配置项与默认值）
{ "api_key": {"default": ""}, "max_results": {"default": 10} }

// plugin_config.json（可选，本地覆盖，不入库）
{ "api_key": "real-key", "max_results": 20 }
```

```python
# 读取（无 _conf_schema.json 时 self.context.config 为 None）
self.context.config.get("api_key")       # "real-key"
self.context.config.max_results          # 20
```

---

## 8. 综合示例

```python
from raven_agent.plugins import (
    Plugin,
    on_after_turn,
    on_before_turn,
    on_tool_post,
    tool,
)


class DemoPlugin(Plugin):
    name = "demo"

    # — @tool —
    @tool("demo_echo", risk="read-only", always_on=True, search_hint="复述 echo")
    async def demo_echo(self, event, text: str) -> str:
        """复述文本。Args: text: 要复述的文本。"""
        return f"demo:{text}"

    # — EventBus handler —
    @on_before_turn()
    async def handle_command(self, ctx):
        if ctx.content.strip() == "/demo":
            ctx.abort = True
            ctx.abort_reply = "demo ok"
        return ctx

    @on_after_turn()
    async def count(self, ctx):
        self.context.kv_store.increment("turns")

    # — tool hook —
    @on_tool_post(tool_name="demo_echo")
    async def record_echo(self, event):
        self.context.kv_store.set("last_tool", event.tool_name)
```

---

## 9. 决策速查

| 做什么 | 用 |
| --- | --- |
| 给 LLM 新能力 | `@tool` |
| 改写工具参数 | `@on_tool_pre` 返回 dict |
| 拒绝工具调用 | `@on_tool_pre` 返回 `ToolHookOutcome(decision="deny")` |
| 观察工具结果 | `@on_tool_post` / `@on_tool_error` |
| 改写 inbound | `@on_turn_started()` |
| 命令拦截 / 提前结束 | `@on_before_turn()` |
| 追加 hint | `@on_before_reasoning()` 或 `@on_before_step()` |
| 注入 system section | `@on_prompt_render()` 或 `prompt_render_modules()` |
| 清理回复 | `@on_after_reasoning()` |
| 需要 session.messages | `xxx_modules()`，从 `frame.input.session` 取 |
| 需要 requires/produces 依赖 | `xxx_modules()` |
| 统计 / 日志 | `@on_after_step()` / `@on_after_turn()` / `@on_turn_completed()` |

---

## 10. 常见坑

```text
@tool 方法前两个参数必须正好是 self, event，否则 TypeError。
emit handler 必须 return ctx/event；observe handler return 会被忽略。
EventBus handler 拿不到 session；需要 session 用 Phase module。
Phase module slot 必须全局唯一，建议加插件前缀。
没有 _conf_schema.json 时 self.context.config 为 None。
plugins.enabled 默认 false，不开启则完全不加载。
initialize() 抛异常会回滚整个插件的工具/hook/module 注册。
```
