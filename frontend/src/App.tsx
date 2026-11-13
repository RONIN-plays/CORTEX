import { useCallback, useEffect, useState } from "react";
import { api, type Health } from "./api";
import Chat from "./components/Chat";
import GraphView from "./components/GraphView";
import Library from "./components/Library";
import NoteDrawer from "./components/NoteDrawer";
import SearchView from "./components/SearchView";

type View = "ask" | "search" | "library" | "graph";

/** Which note the reader panel shows, and optionally the passage that led there. */
export interface OpenNote {
  docId: number;
  passage?: string;
}

const VIEWS: { id: View; label: string; icon: string }[] = [
  { id: "ask", label: "Ask", icon: "M4 5h16v11H9l-5 4z" },
  { id: "search", label: "Search", icon: "M10.5 4a6.5 6.5 0 1 0 4 11.6l4.4 4.4 1.5-1.5-4.4-4.4A6.5 6.5 0 0 0 10.5 4z" },
  { id: "library", label: "Library", icon: "M5 4h4v16H5zM10.5 4h4v16h-4zM16.5 6l3.5-1 3 14-3.5 1z" },
  { id: "graph", label: "Graph", icon: "M6 8a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zm12 4a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zM9 21a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zM8 6.5l8 2.5M16.5 11.5l-6 5.5M6.5 8L8.5 16" },
];

export default function App() {
  const [view, setView] = useState<View>("ask");
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [openNote, setOpenNote] = useState<OpenNote | null>(null);
  // Bumped whenever the library changes so views refetch.
  const [libraryVersion, setLibraryVersion] = useState(0);

  const refresh = useCallback(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        setOffline(false);
      })
      .catch(() => setOffline(true));
  }, []);

  useEffect(refresh, [refresh, libraryVersion]);

  const libraryChanged = useCallback(() => setLibraryVersion((v) => v + 1), []);

  return (
    <div className="app">
      <nav className="sidebar">
        <div className="brand">
          <span className="brand-mark" />
          Cortex
        </div>
        {VIEWS.map((v) => (
          <button
            key={v.id}
            className={`nav-item ${view === v.id ? "active" : ""}`}
            onClick={() => setView(v.id)}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d={v.icon} />
            </svg>
            <span>{v.label}</span>
          </button>
        ))}
        <div className="sidebar-status">
          {offline ? (
            <span className="status-bad">Server offline</span>
          ) : health ? (
            <>
              <span>
                {health.documents} {health.documents === 1 ? "note" : "notes"}
              </span>
              <span className={health.has_api_key ? "status-ok" : "status-warn"}>
                {health.has_api_key ? "Groq connected" : "No API key"}
              </span>
            </>
          ) : null}
        </div>
      </nav>

      <main className="main">
        {offline && (
          <div className="banner banner-bad">
            Cannot reach the Cortex server. Start the backend, then{" "}
            <button className="link" onClick={refresh}>
              retry
            </button>
            .
          </div>
        )}
        {/* Chat stays mounted so the conversation survives switching views. */}
        <div className="view" hidden={view !== "ask"}>
          <Chat
            hasKey={health?.has_api_key ?? true}
            empty={health?.documents === 0}
            onOpen={setOpenNote}
            goToLibrary={() => setView("library")}
          />
        </div>
        {view === "search" && <SearchView onOpen={setOpenNote} />}
        {view === "library" && (
          <Library version={libraryVersion} onChanged={libraryChanged} onOpen={setOpenNote} />
        )}
        {view === "graph" && <GraphView version={libraryVersion} onOpen={setOpenNote} />}
      </main>

      {openNote && (
        <NoteDrawer
          note={openNote}
          onOpen={setOpenNote}
          onClose={() => setOpenNote(null)}
          onDeleted={() => {
            setOpenNote(null);
            libraryChanged();
          }}
        />
      )}
    </div>
  );
}
