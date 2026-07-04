import { useMemo, useState } from "react";

import type { SessionSummaryResponse } from "../api/client";
import { Icon } from "./Icon";

export function Sidebar({
  sessions,
  activeSessionId,
  onNewChat,
  onSelectSession,
  onRenameSession,
  onDeleteSession,
  collapsed,
  onToggle,
}: {
  sessions: SessionSummaryResponse[];
  activeSessionId: string;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
  onRenameSession: (id: string, title: string) => void;
  onDeleteSession: (id: string) => void;
  collapsed: boolean;
  onToggle: () => void;
}) {
  const [editingId, setEditingId] = useState("");
  const [draftTitle, setDraftTitle] = useState("");
  const groups = useMemo(() => {
    const now = new Date();
    const today = now.toISOString().slice(0, 10);
    const yesterday = new Date(now.getTime() - 86400000).toISOString().slice(0, 10);
    const result: { label: string; items: SessionSummaryResponse[] }[] = [];
    let currentLabel = "";
    let currentItems: SessionSummaryResponse[] = [];

    for (const session of sessions) {
      const date = session.updated_at.slice(0, 10);
      let label = date;
      if (date === today) label = "今天";
      else if (date === yesterday) label = "昨天";

      if (label !== currentLabel) {
        if (currentItems.length) result.push({ label: currentLabel, items: currentItems });
        currentLabel = label;
        currentItems = [];
      }
      currentItems.push(session);
    }
    if (currentItems.length) result.push({ label: currentLabel, items: currentItems });
    return result;
  }, [sessions]);

  return (
    <aside className={collapsed ? "sidebar collapsed" : "sidebar"} aria-label="会话导航">
      <div className="brand-row">
        <div className="brand-mark">F</div>
        <span>fintell</span>
        <button type="button" aria-label={collapsed ? "展开侧边栏" : "收起侧边栏"} onClick={onToggle}>
          <Icon name="menu" />
        </button>
      </div>
      <button className="new-chat" type="button" onClick={onNewChat}>
        <Icon name="plus" />
        <span>开启新对话</span>
      </button>
      <nav className="history-list" aria-label="历史对话">
        {groups.map((group) => (
          <div className="history-group" key={group.label}>
            <p>{group.label}</p>
            {group.items.map((session) => {
              const editing = editingId === session.id;
              return (
                <div
                  className={`history-item${session.id === activeSessionId ? " active" : ""}${editing ? " editing" : ""}`}
                  key={session.id}
                >
                  {editing ? (
                    <form
                      className="history-rename"
                      onSubmit={(event) => {
                        event.preventDefault();
                        const title = draftTitle.trim();
                        if (title) onRenameSession(session.id, title);
                        setEditingId("");
                      }}
                    >
                      <input
                        value={draftTitle}
                        autoFocus
                        onBlur={() => setEditingId("")}
                        onChange={(event) => setDraftTitle(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") setEditingId("");
                        }}
                      />
                    </form>
                  ) : (
                    <>
                      <button
                        className="history-title"
                        type="button"
                        onClick={() => onSelectSession(session.id)}
                      >
                        <span>{session.title}</span>
                      </button>
                      <div className="history-actions">
                        <button
                          type="button"
                          aria-label="重命名会话"
                          onClick={() => {
                            setEditingId(session.id);
                            setDraftTitle(session.title);
                          }}
                        >
                          <Icon name="pencil" />
                        </button>
                        <button
                          type="button"
                          aria-label="删除会话"
                          onClick={() => onDeleteSession(session.id)}
                        >
                          <Icon name="trash" />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        ))}
        {sessions.length === 0 && !collapsed && (
          <p className="history-empty">暂无对话</p>
        )}
      </nav>
    </aside>
  );
}
