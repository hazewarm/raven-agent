import React, { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, asPageResult, pageCount } from "./api";
import {
  encodePath,
  formatSessionKeyForTable,
  relativeTime,
  renderMarkdown,
  roleClass,
  shortTs,
  stripMarkdown,
} from "./format";
import { attachJsonViewers, installDashboardGlobals, jvPlaceholder, loadPluginAssets } from "./pluginRuntime";
import { PluginDetail } from "./PluginDetail";
import type {
  DashboardColumn,
  MessageRow,
  PageResult,
  PluginConfig,
  PluginDispatch,
  PluginState,
  SessionRow,
  SortOrder,
  ViewMode,
} from "./types";

type NavOpen = Record<string, boolean>;

// Creates a PluginDispatch bound to the given plugin + latest state getter.
function makeDispatch(
  plugin: PluginConfig,
  getState: () => PluginState | null,
  onSetState: (updater: (s: PluginState) => PluginState) => void,
  onActivate?: () => void,
): PluginDispatch {
  const fetchAndApply = async (
    nextFilters: Record<string, string>,
    nextSortBy: string,
    nextSortOrder: SortOrder,
  ): Promise<void> => {
    const state = getState();
    if (!state) return;
    const result = await plugin.fetchPage({ page: 1, pageSize: state.pageSize, filters: nextFilters, sortBy: nextSortBy, sortOrder: nextSortOrder });
    onSetState((s) => ({
      ...s,
      page: 1,
      total: result.total || 0,
      items: result.items || [],
      activeRowKey: null,
      activeDetail: null,
      filters: nextFilters,
      sortBy: nextSortBy,
      sortOrder: nextSortOrder,
    }));
  };

  const updateFilters = (updater: (filters: Record<string, string>) => Record<string, string>): void => {
    const state = getState();
    if (!state) return;
    void fetchAndApply(updater({ ...state.filters }), state.sortBy, state.sortOrder);
  };

  return {
    get filters() { return getState()?.filters ?? {}; },
    setFilter(key: string, value: string): void {
      updateFilters((filters) => ({ ...filters, [key]: value }));
    },
    clearFilter(key: string): void {
      updateFilters((filters) => {
        delete filters[key];
        return filters;
      });
    },
    setFilters(next: Record<string, string>): void {
      updateFilters((filters) => ({ ...filters, ...next }));
    },
    clearFilters(keys: string[]): void {
      updateFilters((filters) => {
        for (const key of keys) delete filters[key];
        return filters;
      });
    },
    get sortBy() { return getState()?.sortBy ?? ""; },
    get sortOrder() { return getState()?.sortOrder ?? "desc"; },
    setSort(key: string): void {
      const state = getState();
      if (!state) return;
      const nextOrder: SortOrder = state.sortBy === key && state.sortOrder === "desc" ? "asc" : "desc";
      void fetchAndApply(state.filters, key, nextOrder);
    },
    refresh(): void {
      const state = getState();
      if (!state) return;
      void fetchAndApply(state.filters, state.sortBy, state.sortOrder);
    },
    activate(): void {
      onActivate?.();
    },
  };
}

function App(): React.ReactElement {
  const [viewMode, setViewMode] = useState<ViewMode>("sessions");
  const [navOpen, setNavOpen] = useState<NavOpen>({ sessions: false });
  const [plugins, setPlugins] = useState<PluginConfig[]>([]);
  const [pluginState, setPluginState] = useState<Record<string, PluginState>>({});
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [sessionSearch, setSessionSearch] = useState("");
  const [sessionChannel, setSessionChannel] = useState("");
  const [activeSessionKey, setActiveSessionKey] = useState<string | null>(null);
  const [activeSession, setActiveSession] = useState<SessionRow | null>(null);
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [messageSearch, setMessageSearch] = useState("");
  const [messageRole, setMessageRole] = useState("");
  const [messagePage, setMessagePage] = useState(1);
  const [messageSortBy, setMessageSortBy] = useState("ts");
  const [messageSortOrder, setMessageSortOrder] = useState<SortOrder>("desc");
  const [totalMessages, setTotalMessages] = useState(0);
  const [activeMessage, setActiveMessage] = useState<MessageRow | null>(null);
  const [selectedMessageIds, setSelectedMessageIds] = useState<Set<string>>(new Set());
  const [hiddenPlugins, setHiddenPlugins] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);

  const messagePageSize = 25;
  const currentPluginId = viewMode.startsWith("plugin:") ? viewMode.slice(7) : "";
  const currentPlugin = plugins.find((plugin) => plugin.id === currentPluginId) ?? null;
  const currentPluginState = currentPluginId ? pluginState[currentPluginId] : null;

  const channels = useMemo(() => Array.from(new Set(sessions.map((session) => session.key.split(":")[0]).filter(Boolean))), [sessions]);

  const run = useCallback(async (work: () => Promise<void>) => {
    try {
      setError(null);
      await work();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, []);

  const loadSessions = useCallback(async () => {
    const params = new URLSearchParams();
    if (sessionSearch) params.set("q", sessionSearch);
    if (sessionChannel) params.set("channel", sessionChannel);
    params.set("page_size", "200");
    const payload = asPageResult(await api<PageResult<SessionRow>>(`/api/dashboard/sessions?${params.toString()}`));
    setSessions(payload.items);
    setActiveSession((current) => {
      if (!activeSessionKey) return current;
      return payload.items.find((session) => session.key === activeSessionKey) ?? null;
    });
  }, [activeSessionKey, sessionChannel, sessionSearch]);

  const loadMessages = useCallback(async () => {
    const params = new URLSearchParams();
    if (activeSessionKey) params.set("session_key", activeSessionKey);
    if (messageSearch) params.set("q", messageSearch);
    if (messageRole) params.set("role", messageRole);
    params.set("page", String(messagePage));
    params.set("page_size", String(messagePageSize));
    params.set("sort_by", messageSortBy);
    params.set("sort_order", messageSortOrder);
    const payload = asPageResult(await api<PageResult<MessageRow>>(`/api/dashboard/messages?${params.toString()}`));
    setMessages(payload.items);
    setTotalMessages(payload.total);
    setActiveMessage((current) => current && payload.items.some((item) => item.id === current.id) ? current : null);
  }, [activeSessionKey, messagePage, messageRole, messageSearch, messageSortBy, messageSortOrder]);

  const loadPluginPanel = useCallback(async (pluginId: string) => {
    const plugin = plugins.find((item) => item.id === pluginId);
    const state = pluginState[pluginId];
    if (!plugin || !state) return;
    const result = await plugin.fetchPage({ page: state.page, pageSize: state.pageSize, filters: state.filters, sortBy: state.sortBy, sortOrder: state.sortOrder });
    setPluginState((current) => ({
      ...current,
      [pluginId]: {
        ...current[pluginId],
        total: result.total || 0,
        items: result.items || [],
        activeRowKey: current[pluginId]?.activeRowKey && result.items.some((item) => String(item[plugin.rowKey] ?? "") === current[pluginId].activeRowKey)
          ? current[pluginId].activeRowKey
          : null,
        activeDetail: current[pluginId]?.activeRowKey && result.items.some((item) => String(item[plugin.rowKey] ?? "") === current[pluginId].activeRowKey)
          ? current[pluginId].activeDetail
          : null,
      },
    }));
  }, [pluginState, plugins]);

  const refreshCurrentView = useCallback(async () => {
    await loadSessions();
    if (viewMode.startsWith("plugin:")) {
      await loadPluginPanel(viewMode.slice(7));
    } else {
      await loadMessages();
    }
  }, [loadMessages, loadPluginPanel, loadSessions, viewMode]);

  useEffect(() => {
    const refresh = (): void => {
      void run(refreshCurrentView);
    };
    window.addEventListener("raven-dashboard-refresh", refresh);
    return () => window.removeEventListener("raven-dashboard-refresh", refresh);
  }, [refreshCurrentView, run]);

  useEffect(() => {
    installDashboardGlobals((plugin) => {
      setPlugins((current) => current.some((item) => item.id === plugin.id) ? current : [...current, plugin]);
      setPluginState((current) => current[plugin.id] ? current : {
        ...current,
        [plugin.id]: {
          page: 1,
          pageSize: plugin.pageSize || 25,
          total: 0,
          items: [],
          activeRowKey: null,
          activeDetail: null,
          filters: {},
          sortBy: plugin.defaultSortBy ?? "",
          sortOrder: plugin.defaultSortOrder ?? "desc",
          selectedIds: new Set(),
        },
      });
    });
    void loadPluginAssets();
  }, []);

  useEffect(() => {
    void run(async () => {
      await loadSessions();
      await loadMessages();
    });
  }, [loadMessages, loadSessions, run]);

  useEffect(() => {
    for (const plugin of plugins) {
      void run(async () => {
        const count = await plugin.getCount();
        if (count === null) {
          setHiddenPlugins((current) => ({ ...current, [plugin.id]: true }));
        } else {
          setHiddenPlugins((current) => ({ ...current, [plugin.id]: false }));
          setPluginState((current) => ({
            ...current,
            [plugin.id]: { ...current[plugin.id], total: count },
          }));
        }
      });
    }
  }, [plugins, run]);

  const focusView = useCallback((next: ViewMode): void => {
    setViewMode(next);
    setNavOpen((current) => ({ ...current, [next]: true }));
  }, []);

  const selectView = (next: ViewMode): void => {
    focusView(next);
    void run(async () => {
      if (next === "sessions") await loadMessages();
      else await loadPluginPanel(next.slice(7));
    });
  };

  const toggleNav = (kind: ViewMode): void => {
    if (viewMode !== kind) {
      selectView(kind);
      return;
    }
    setNavOpen((current) => ({ ...current, [kind]: !current[kind] }));
  };

  const sort = (key: string): void => {
    const flip = (currentKey: string, currentOrder: SortOrder): SortOrder => currentKey === key && currentOrder === "desc" ? "asc" : "desc";
    setMessageSortOrder(flip(messageSortBy, messageSortOrder));
    setMessageSortBy(key);
    setMessagePage(1);
  };

  useEffect(() => {
    if (viewMode === "sessions") void run(loadMessages);
  }, [loadMessages, run, viewMode]);

  const currentPageCount = currentPluginState
    ? pageCount(currentPluginState.total, currentPluginState.pageSize)
    : pageCount(totalMessages, messagePageSize);

  const currentPage = currentPluginState?.page ?? messagePage;

  const changePage = (delta: number): void => {
    if (currentPage + delta < 1 || currentPage + delta > currentPageCount) return;
    if (currentPluginId) {
      void run(async () => {
        const plugin = plugins.find((item) => item.id === currentPluginId);
        const state = pluginState[currentPluginId];
        if (!plugin || !state) return;
        const nextPage = state.page + delta;
        const result = await plugin.fetchPage({ page: nextPage, pageSize: state.pageSize, filters: state.filters, sortBy: state.sortBy, sortOrder: state.sortOrder });
        setPluginState((current) => ({
          ...current,
          [currentPluginId]: {
            ...current[currentPluginId],
            page: nextPage,
            total: result.total || 0,
            items: result.items || [],
            activeRowKey: null,
            activeDetail: null,
          },
        }));
      });
    }
    setMessagePage((page) => page + delta);
  };

  // Batch count: messages or plugin selectedIds
  const pluginBatchCount = currentPluginState?.selectedIds.size ?? 0;
  const batchCount = viewMode.startsWith("plugin:") ? pluginBatchCount : selectedMessageIds.size;

  // dispatch for current plugin (used in DetailPane and batch bar)
  const currentDispatch = currentPlugin && currentPluginState
    ? makeDispatch(
        currentPlugin,
        () => pluginState[currentPlugin.id] ?? null,
        (updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) })),
        () => focusView(`plugin:${currentPlugin.id}`),
      )
    : undefined;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <div className="brand-title">Raven Dashboard</div>
            <div className="brand-sub">Session / Memory Explorer</div>
          </div>
        </div>
        <TopbarFilters
          viewMode={viewMode}
          messageSearch={messageSearch}
          setMessageSearch={(value) => { setMessageSearch(value); setMessagePage(1); }}
          messageRole={messageRole}
          setMessageRole={(value) => { setMessageRole(value); setMessagePage(1); }}
          activeSessionKey={activeSessionKey}
          clearSession={() => { setActiveSessionKey(null); setActiveSession(null); setActiveMessage(null); setMessagePage(1); }}
          currentPlugin={currentPlugin}
          currentPluginState={currentPluginState}
          onSetPluginState={currentPlugin ? (updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) })) : undefined}
        />
        <div className="topbar-view">
          <div className="view-chip"><span>{viewLabel(viewMode, currentPlugin)}</span></div>
          {viewMode.startsWith("plugin:") && currentPlugin?.renderTopbarAction && currentPluginState && currentDispatch && (
            <PluginTopbarAction
              plugin={currentPlugin}
              pluginId={currentPlugin.id}
              state={currentPluginState}
              onSetState={(updater) => setPluginState((c) => ({ ...c, [currentPlugin.id]: updater(c[currentPlugin.id]) }))}
              onActivate={() => focusView(`plugin:${currentPlugin.id}`)}
            />
          )}
        </div>
      </header>

      <main className="workspace">
        <aside className="sessions-pane">
          <div className="pane-head">
            <div className="pane-kicker">Explorer</div>
            <div className="pane-title">
              {currentPlugin && currentPluginState
                ? (currentPlugin.countTitle ? currentPlugin.countTitle(currentPluginState.total) : `${currentPluginState.total} 条记录`)
                : `${sessions.length} 个会话`}
            </div>
          </div>
          <div className="filters-stack">
            <label className="search search-small">
              <span>⌕</span>
              <input type="text" placeholder="过滤 session" value={sessionSearch} onChange={(event) => setSessionSearch(event.target.value.trim())} />
            </label>
            <select value={sessionChannel} onChange={(event) => setSessionChannel(event.target.value)}>
              <option value="">全部 channel</option>
              {channels.map((channel) => <option key={channel} value={channel}>{channel}</option>)}
            </select>
          </div>
          <nav className="explorer-nav">
            <NavGroup label="Sessions" count={totalMessages} active={viewMode === "sessions"} open={!!navOpen.sessions} onToggle={() => toggleNav("sessions")}>
              <button className={`all-messages-row ${viewMode === "sessions" && !activeSessionKey ? "active" : ""}`} type="button" onClick={() => {
                setActiveSessionKey(null);
                setActiveSession(null);
                setActiveMessage(null);
                setMessagePage(1);
                selectView("sessions");
              }}>
                <span>全部消息</span><strong>{sessions.length}</strong>
              </button>
              <div className="session-list">
                {sessions.map((session) => (
                  <div key={session.key} className={`session-item ${activeSessionKey === session.key ? "active" : ""}`} onClick={() => {
                    setActiveSessionKey(session.key);
                    setActiveSession(session);
                    setActiveMessage(null);
                    setMessagePage(1);
                    selectView("sessions");
                  }}>
                    <div className="nav-item-row">
                      <span className="nav-type-dot memory-type-profile" />
                      <span className="nav-item-name mono">{formatSessionKeyForTable(session.key)}</span>
                      <span className="nav-item-count">{session.message_count}</span>
                    </div>
                    <div className="nav-item-desc">
                      <span>{relativeTime(session.updated_at)}</span>
                      <button
                        className="icon-btn danger"
                        type="button"
                        title="删除此 session"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!confirm(`确定删除 ${session.key} 及其所有消息？此操作不可恢复。`)) return;
                          void run(async () => {
                            await api(`/api/dashboard/sessions/${encodeURIComponent(session.key)}?cascade=true`, { method: "DELETE" });
                            await loadSessions();
                            if (activeSessionKey === session.key) {
                              setActiveSessionKey(null);
                              setActiveSession(null);
                              setActiveMessage(null);
                            }
                          });
                        }}
                      >×</button>
                    </div>
                  </div>
                ))}
              </div>
            </NavGroup>
          </nav>
        </aside>

        <section className="messages-pane">
          {batchCount > 0 && (
            <div className="batch-bar">
              <span>已选 {batchCount} 条</span>
              {viewMode.startsWith("plugin:") && currentPlugin?.batchActions && currentPluginState
                ? currentPlugin.batchActions.map((action: PluginBatchAction) => (
                    <button key={action.label} className={action.className} type="button" onClick={() => void run(async () => {
                      const ids = [...currentPluginState.selectedIds];
                      await action.run(ids);
                      setPluginState((c) => ({ ...c, [currentPlugin.id]: { ...c[currentPlugin.id], selectedIds: new Set() } }));
                      await loadPluginPanel(currentPlugin.id);
                    })}>{action.label}</button>
                  ))
                : <button className="danger-ghost" type="button" onClick={() => void run(async () => {
                    await api("/api/dashboard/messages/batch-delete", { method: "POST", body: JSON.stringify({ ids: [...selectedMessageIds] }) });
                    setSelectedMessageIds(new Set());
                    await refreshCurrentView();
                  })}>批量删除</button>
              }
              <button className="ghost" type="button" onClick={() => {
                if (viewMode.startsWith("plugin:") && currentPlugin) {
                  setPluginState((c) => ({ ...c, [currentPlugin.id]: { ...c[currentPlugin.id], selectedIds: new Set() } }));
                } else {
                  setSelectedMessageIds(new Set());
                }
              }}>取消选择</button>
            </div>
          )}
          <TableHead viewMode={viewMode} plugin={currentPlugin} pluginState={currentPluginState} messageSortBy={messageSortBy} messageSortOrder={messageSortOrder} onSort={sort} onPluginSort={currentDispatch ? (key) => currentDispatch.setSort(key) : undefined} />
          <div className="table-body">
            <Rows
              viewMode={viewMode}
              messages={messages}
              plugin={currentPlugin}
              pluginState={currentPluginState}
              selectedMessageIds={selectedMessageIds}
              activeMessage={activeMessage}
              onSelectMessage={setActiveMessage}
              onSelectPluginRow={(row) => {
                if (!currentPlugin || !currentPluginState) return;
                const key = String(row[currentPlugin.rowKey] ?? "");
                void run(async () => {
                  const detail = currentPlugin.fetchDetail ? await currentPlugin.fetchDetail(row) : row;
                  setPluginState((current) => ({ ...current, [currentPlugin.id]: { ...current[currentPlugin.id], activeRowKey: key, activeDetail: detail } }));
                });
              }}
              onTogglePluginRow={(id) => {
                if (!currentPlugin) return;
                setPluginState((c) => {
                  const ps = c[currentPlugin.id];
                  if (!ps) return c;
                  const next = new Set(ps.selectedIds);
                  if (next.has(id)) next.delete(id);
                  else next.add(id);
                  return { ...c, [currentPlugin.id]: { ...ps, selectedIds: next } };
                });
              }}
              setSelectedMessageIds={setSelectedMessageIds}
            />
          </div>
          <footer className="table-foot">
            <div>{tableMeta(viewMode, totalMessages, currentPlugin, currentPluginState)}</div>
            <div className="pager">
              <button className="ghost" type="button" disabled={currentPage <= 1} onClick={() => changePage(-1)}>‹</button>
              <span>{currentPage} / {currentPageCount}</span>
              <button className="ghost" type="button" disabled={currentPage >= currentPageCount} onClick={() => changePage(1)}>›</button>
            </div>
          </footer>
        </section>

        <aside className="detail-pane">
          <DetailPane
            viewMode={viewMode}
            activeSession={activeSession}
            activeMessage={activeMessage}
            plugin={currentPlugin}
            pluginState={currentPluginState}
            dispatch={currentDispatch}
          />
        </aside>
      </main>
      {error && <div className="modal-backdrop" onClick={() => setError(null)}><div className="modal"><div className="modal-title">请求失败</div><p>{error}</p><div className="modal-actions"><button className="primary" type="button" onClick={() => setError(null)}>关闭</button></div></div></div>}
    </div>
  );
}

function PluginNavBody(props: {
  plugin: PluginConfig;
  pluginId: string;
  state: PluginState;
  onSetState: (updater: (s: PluginState) => PluginState) => void;
  onActivate(): void;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const getState = useEffectEvent(() => props.state);
  const filtersKey = JSON.stringify(props.state.filters);

  useEffect(() => {
    if (ref.current && props.plugin.renderNavBody) {
      const dispatch = makeDispatch(props.plugin, getState, props.onSetState, props.onActivate);
      props.plugin.renderNavBody(ref.current, dispatch);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtersKey, props.onActivate, props.plugin, props.pluginId, props.state.sortBy, props.state.sortOrder, props.state.total]);

  return <div ref={ref} />;
}

function PluginFilters(props: {
  plugin: PluginConfig;
  pluginId: string;
  state: PluginState;
  onSetState: (updater: (s: PluginState) => PluginState) => void;
  onActivate(): void;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const getState = useEffectEvent(() => props.state);
  const filtersKey = JSON.stringify(props.state.filters);

  useEffect(() => {
    if (ref.current && props.plugin.renderFilters) {
      const dispatch = makeDispatch(props.plugin, getState, props.onSetState, props.onActivate);
      props.plugin.renderFilters(ref.current, dispatch);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtersKey, props.onActivate, props.plugin, props.pluginId, props.state.sortBy, props.state.sortOrder]);

  return <div ref={ref} />;
}

function PluginTopbarAction(props: {
  plugin: PluginConfig;
  pluginId: string;
  state: PluginState;
  onSetState: (updater: (s: PluginState) => PluginState) => void;
  onActivate(): void;
}): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const getState = useEffectEvent(() => props.state);
  const filtersKey = JSON.stringify(props.state.filters);

  useEffect(() => {
    if (ref.current && props.plugin.renderTopbarAction) {
      const dispatch = makeDispatch(props.plugin, getState, props.onSetState, props.onActivate);
      props.plugin.renderTopbarAction(ref.current, dispatch);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtersKey, props.onActivate, props.plugin, props.pluginId, props.state.sortBy, props.state.sortOrder]);

  return <div ref={ref} />;
}

function TopbarFilters(props: {
  viewMode: ViewMode;
  messageSearch: string;
  setMessageSearch(value: string): void;
  messageRole: string;
  setMessageRole(value: string): void;
  activeSessionKey: string | null;
  clearSession(): void;
  currentPlugin: PluginConfig | null;
  currentPluginState: PluginState | null;
  onSetPluginState?: (updater: (s: PluginState) => PluginState) => void;
}): React.ReactElement {
  return (
    <div className="topbar-filters">
      {props.viewMode.startsWith("plugin:") && props.currentPlugin?.renderFilters && props.currentPluginState && props.onSetPluginState
        ? <PluginFilters
            plugin={props.currentPlugin}
            pluginId={props.currentPlugin.id}
            state={props.currentPluginState}
            onSetState={props.onSetPluginState}
            onActivate={() => {}}
          />
        : (
          <div className="filter-row">
            <label className="search"><span>⌕</span><input type="text" placeholder="搜索消息内容" value={props.messageSearch} onChange={(event) => props.setMessageSearch(event.target.value.trim())} /></label>
            <select value={props.messageRole} onChange={(event) => props.setMessageRole(event.target.value)}>
              <option value="">全部 role</option><option value="user">user</option><option value="assistant">assistant</option><option value="system">system</option><option value="tool">tool</option>
            </select>
            {props.activeSessionKey && <Chip label="session" value={props.activeSessionKey} onClear={props.clearSession} />}
          </div>
        )
      }
    </div>
  );
}

function Chip(props: { label: string; value: string; onClear(): void }): React.ReactElement {
  return <div className="active-session-chip"><span>{props.label}</span><code>{props.value}</code><button type="button" onClick={props.onClear}>×</button></div>;
}

function NavGroup(props: { label: string; count: number; active: boolean; open: boolean; onToggle(): void; children: React.ReactNode }): React.ReactElement {
  return (
    <section className={`nav-group${props.active ? " active" : ""}${props.open ? " open" : ""}`}>
      <button className="nav-group-toggle" type="button" onClick={props.onToggle}>
        <span className="nav-group-caret">▸</span>
        <span className="nav-group-label">{props.label}</span>
        <span className="nav-group-count">{props.count}</span>
      </button>
      <div className={`nav-group-body${props.open ? " open" : ""}`}>
        <div className="nav-group-body-inner">{props.children}</div>
      </div>
    </section>
  );
}

function TableHead(props: {
  viewMode: ViewMode;
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  messageSortBy: string;
  messageSortOrder: SortOrder;
  onSort(key: string): void;
  onPluginSort?: (key: string) => void;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin) {
    const hasBatch = Boolean(props.plugin.batchActions?.length);
    const grid = (hasBatch ? "32px " : "") + gridTemplate(props.plugin.columns);
    const sortBy = props.pluginState?.sortBy ?? "";
    const sortOrder = props.pluginState?.sortOrder ?? "desc";
    return (
      <div className="table-head" style={{ gridTemplateColumns: grid }}>
        {hasBatch && <div />}
        {props.plugin.columns.map((col) => col.sortable && props.onPluginSort
          ? <SortHead key={col.key} label={col.label} active={sortBy === col.key} order={sortOrder} onClick={() => props.onPluginSort!(col.key)} />
          : <div key={col.key}>{col.label}</div>
        )}
      </div>
    );
  }
  return <div className="table-head mode-messages">
    <div />
    <SortHead label="Session Key" active={props.messageSortBy === "session_key"} order={props.messageSortOrder} onClick={() => props.onSort("session_key")} />
    <SortHead label="Seq" active={props.messageSortBy === "seq"} order={props.messageSortOrder} onClick={() => props.onSort("seq")} />
    <div>Content</div>
    <SortHead label="Timestamp" active={props.messageSortBy === "ts"} order={props.messageSortOrder} onClick={() => props.onSort("ts")} />
    <SortHead label="Role" active={props.messageSortBy === "role"} order={props.messageSortOrder} onClick={() => props.onSort("role")} />
    <div />
  </div>;
}

function SortHead(props: { label: string; active: boolean; order: SortOrder; onClick(): void }): React.ReactElement {
  return <button className={`table-sort-btn ${props.active ? "active" : ""}`} type="button" onClick={props.onClick}><span>{props.label}</span><span className="table-sort-arrow">{props.active ? props.order === "asc" ? "↑" : "↓" : ""}</span></button>;
}

function Rows(props: {
  viewMode: ViewMode;
  messages: MessageRow[];
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  selectedMessageIds: Set<string>;
  activeMessage: MessageRow | null;
  onSelectMessage(item: MessageRow): void;
  onSelectPluginRow(row: Record<string, unknown>): void;
  onTogglePluginRow(id: string): void;
  setSelectedMessageIds(value: Set<string>): void;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin && props.pluginState) {
    const hasBatch = Boolean(props.plugin.batchActions?.length);
    const grid = (hasBatch ? "32px " : "") + gridTemplate(props.plugin.columns);
    return <>{props.pluginState.items.length ? props.pluginState.items.map((item) => {
      const key = String(item[props.plugin!.rowKey] ?? "");
      const isSelected = props.pluginState!.selectedIds.has(key);
      return <div key={key} className={`table-row ${props.pluginState!.activeRowKey === key ? "active" : ""} ${isSelected ? "selected" : ""} ${props.plugin!.rowClass?.(item) ?? ""}`} style={{ gridTemplateColumns: grid }} onClick={() => props.onSelectPluginRow(item)}>
        {hasBatch && (
          <label className="checkbox-cell" onClick={(event) => event.stopPropagation()}>
            <input type="checkbox" checked={isSelected} onChange={() => props.onTogglePluginRow(key)} />
          </label>
        )}
        {props.plugin!.columns.map((col) => {
          const cellClass = columnCellClass(col);
          if (col.renderCell) {
            return <div key={col.key} className={cellClass} title={col.rawTitle ? String(item[col.key] ?? "") : undefined} dangerouslySetInnerHTML={{ __html: col.renderCell(item[col.key], item) }} />;
          }
          return <div key={col.key} className={cellClass} title={col.rawTitle ? String(item[col.key] ?? "") : undefined}>{formatPluginCell(props.plugin!, col, item)}</div>;
        })}
      </div>;
    }) : <div className="empty-state">{props.plugin.emptyMessage || "暂无记录。"}</div>}</>;
  }
  return <>{props.messages.map((item) =><div key={item.id} className={`table-row mode-messages ${props.activeMessage?.id === item.id ? "active" : ""} ${props.selectedMessageIds.has(item.id) ? "selected" : ""}`} onClick={() => props.onSelectMessage(item)}>
    <label className="checkbox-cell" onClick={(event) => event.stopPropagation()}><input type="checkbox" checked={props.selectedMessageIds.has(item.id)} onChange={(event) => toggleSet(item.id, event.target.checked, props.selectedMessageIds, props.setSelectedMessageIds)} /></label>
    <div className="cell-session mono" title={item.session_key}>{formatSessionKeyForTable(item.session_key)}</div>
    <div className="cell-seq mono">#{item.seq}</div>
    <div className="content-preview">{stripMarkdown(item.content)}</div>
    <div className="cell-time mono">{shortTs(item.ts)}</div>
    <div><span className={`role-pill ${roleClass(item.role)}`}>{item.role}</span></div>
    <div />
  </div>)}</>;
}

function DetailPane(props: {
  viewMode: ViewMode;
  activeSession: SessionRow | null;
  activeMessage: MessageRow | null;
  plugin: PluginConfig | null;
  pluginState: PluginState | null;
  dispatch?: PluginDispatch;
}): React.ReactElement {
  if (props.viewMode.startsWith("plugin:") && props.plugin) {
    return <PluginDetail plugin={props.plugin} item={props.pluginState?.activeDetail ?? null} dispatch={props.dispatch} />;
  }
  if (props.activeMessage) {
    const message = props.activeMessage;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">消息详情</div><div className="detail-subtext">{message.session_key} · #{message.seq}</div></div></div>
      <div className="detail-grid">
        {detailRow("role", <span className={`role-pill ${roleClass(message.role)}`}>{message.role}</span>)}
        {detailRow("time", <code>{message.ts}</code>)}
        {detailRow("id", <code>{message.id}</code>)}
      </div>
      <div className="detail-block"><div className="detail-label">Content</div><div className="detail-content" dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }} /></div>
      <div className="detail-block"><div className="detail-label">Extra</div><JsonTreeBlock data={message.extra} /></div>
      <div className="detail-block"><div className="detail-label">Tool Chain</div><JsonTreeBlock data={message.tool_chain} /></div>
    </div>;
  }
  if (props.activeSession) {
    const session = props.activeSession;
    return <div className="detail-wrap">
      <div className="detail-toolbar"><div><div className="detail-title">Session 详情</div><div className="detail-subtext">{session.key}</div></div></div>
      <div className="detail-grid">
        {detailRow("messages", <code>{session.message_count}</code>)}
        {detailRow("updated", <code>{session.updated_at}</code>)}
        {detailRow("last_consolidated", <code>{session.last_consolidated}</code>)}
      </div>
      <div className="detail-block"><div className="detail-label">Metadata</div><JsonTreeBlock data={session.metadata} /></div>
    </div>;
  }
  return <EmptyDetail text="点开消息、session 或 memory 后，这里会显示完整内容、字段和 JSON 信息。" />;
}

function EmptyDetail(props: { text: string }): React.ReactElement {
  return <div className="detail-empty"><div className="detail-empty-title">详情</div><div className="detail-empty-text">{props.text}</div></div>;
}

function detailRow(label: string, value: React.ReactNode): React.ReactElement {
  return <div className="detail-row"><div className="detail-row-label">{label}</div><div className="detail-row-val">{value}</div></div>;
}

function JsonTreeBlock(props: { data: unknown }): React.ReactElement {
  const ref = useRef<HTMLDivElement>(null);
  const payload = JSON.stringify(props.data ?? null);

  useEffect(() => {
    if (!ref.current) return;
    ref.current.innerHTML = jvPlaceholder(props.data);
    attachJsonViewers(ref.current);
  }, [payload, props.data]);

  return <div ref={ref} />;
}

function toggleSet(id: string, checked: boolean, source: Set<string>, update: (value: Set<string>) => void): void {
  const next = new Set(source);
  if (checked) next.add(id);
  else next.delete(id);
  update(next);
}

function gridTemplate(columns: DashboardColumn[]): string {
  return columns.map((col) => col.flex ? "1fr" : col.width ? `${col.width}px` : "auto").join(" ");
}

function formatPluginCell(plugin: PluginConfig, column: DashboardColumn, item: Record<string, unknown>): string {
  const value = item[column.key];
  const formatter = plugin.formatters?.[column.fmt || ""] ?? (window as Window & { RavenDashboard?: { _formatters: Record<string, (value: unknown, item?: Record<string, unknown>) => string> } }).RavenDashboard?._formatters[column.fmt || "text"];
  return formatter ? formatter(value, item) : String(value ?? "");
}

function columnCellClass(column: DashboardColumn): string {
  const classes = [column.cellClass ?? ""];
  if (column.align === "right") classes.push("align-right");
  return classes.filter(Boolean).join(" ");
}

function tableMeta(viewMode: ViewMode, totalMessages: number, plugin: PluginConfig | null, pluginState: PluginState | null): string {
  if (plugin && pluginState) return plugin.countTitle ? plugin.countTitle(pluginState.total) : `共 ${pluginState.total} 条`;
  return `共 ${totalMessages} 条`;
}

function viewLabel(viewMode: ViewMode, plugin: PluginConfig | null): string {
  if (plugin) return plugin.viewLabel || plugin.label;
  return "messages";
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
