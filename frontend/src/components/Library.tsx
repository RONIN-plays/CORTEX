import { useEffect, useRef, useState } from "react";
import { api, type DocSummary, type IngestResult } from "../api";
import type { OpenNote } from "../App";

interface Props {
  version: number;
  onChanged: () => void;
  onOpen: (note: OpenNote) => void;
}

interface Notice {
  kind: "ok" | "bad";
  text: string;
}

function describe(result: IngestResult): Notice {
  const name = result.title ?? result.filename ?? "Item";
  if (result.error) return { kind: "bad", text: `${result.filename ?? name}: ${result.error}` };
  if (result.duplicate) return { kind: "ok", text: `"${name}" was already in your library.` };
  return { kind: "ok", text: `Saved "${name}".` };
}

export default function Library({ version, onChanged, onOpen }: Props) {
  const [docs, setDocs] = useState<DocSummary[] | null>(null);
  const [loadError, setLoadError] = useState("");
  const [notices, setNotices] = useState<Notice[]>([]);
  const [busy, setBusy] = useState("");
  const [dragging, setDragging] = useState(false);
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [filter, setFilter] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .documents()
      .then((d) => {
        setDocs(d);
        setLoadError("");
      })
      .catch((e: Error) => setLoadError(e.message));
  }, [version]);

  /** Run one save action, showing its outcome and refreshing the list. */
  async function save(label: string, action: () => Promise<IngestResult[]>, onSuccess?: () => void) {
    setBusy(label);
    setNotices([]);
    try {
      const results = await action();
      setNotices(results.map(describe));
      if (results.some((r) => !r.error)) onSuccess?.();
      onChanged();
    } catch (error) {
      setNotices([{ kind: "bad", text: error instanceof Error ? error.message : String(error) }]);
    } finally {
      setBusy("");
    }
  }

  const uploadFiles = (files: File[]) => {
    if (files.length) save(`Reading ${files.length} file${files.length > 1 ? "s" : ""}…`, () => api.upload(files));
  };

  const visible = docs?.filter((d) => d.title.toLowerCase().includes(filter.trim().toLowerCase()));

  return (
    <div className="page">
      <h1>Library</h1>

      <div className="add-grid">
        <div
          className={`card dropzone ${dragging ? "dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            uploadFiles([...e.dataTransfer.files]);
          }}
        >
          <h2>Upload files</h2>
          <p className="muted">Drop Markdown, text, PDF, HTML or image files here.</p>
          <button className="button" disabled={!!busy} onClick={() => fileInput.current?.click()}>
            Choose files
          </button>
          <input
            ref={fileInput}
            type="file"
            multiple
            hidden
            accept=".md,.markdown,.txt,.pdf,.html,.htm,.png,.jpg,.jpeg,.webp,.gif,.csv,.json,.rst"
            onChange={(e) => {
              uploadFiles([...(e.target.files ?? [])]);
              e.target.value = "";
            }}
          />
        </div>

        <form
          className="card"
          onSubmit={(e) => {
            e.preventDefault();
            save("Saving note…", async () => [await api.addNote(title, text)], () => {
              setTitle("");
              setText("");
            });
          }}
        >
          <h2>Write a note</h2>
          <input className="input" value={title} placeholder="Title" onChange={(e) => setTitle(e.target.value)} />
          <textarea
            className="input"
            rows={4}
            value={text}
            placeholder="Type or paste anything worth remembering"
            onChange={(e) => setText(e.target.value)}
          />
          <button className="button primary" disabled={!!busy || !text.trim()}>
            Save note
          </button>
        </form>

        <form
          className="card"
          onSubmit={(e) => {
            e.preventDefault();
            save("Fetching page…", async () => [await api.addUrl(url)], () => setUrl(""));
          }}
        >
          <h2>Save a web page</h2>
          <p className="muted">Cortex fetches the page and keeps its text.</p>
          <input
            className="input"
            type="url"
            value={url}
            placeholder="https://…"
            onChange={(e) => setUrl(e.target.value)}
          />
          <button className="button primary" disabled={!!busy || !url.trim()}>
            Save page
          </button>
        </form>
      </div>

      {busy && <div className="banner pulse">{busy}</div>}
      {notices.map((notice, i) => (
        <div key={i} className={`banner ${notice.kind === "bad" ? "banner-bad" : "banner-ok"}`}>
          {notice.text}
        </div>
      ))}

      <div className="list-head">
        <h2>{docs ? `${docs.length} ${docs.length === 1 ? "note" : "notes"}` : "Notes"}</h2>
        {docs && docs.length > 5 && (
          <input
            className="input small"
            value={filter}
            placeholder="Filter by title"
            onChange={(e) => setFilter(e.target.value)}
          />
        )}
      </div>
      {loadError && <div className="banner banner-bad">{loadError}</div>}
      {docs?.length === 0 && (
        <p className="muted">
          Nothing saved yet. Add a note above, or run <code>python -m eval.load_samples</code> in the backend folder to
          load 26 sample notes.
        </p>
      )}
      {visible && visible.length === 0 && docs!.length > 0 && <p className="muted">No titles match that filter.</p>}
      <div className="doc-list">
        {visible?.map((doc) => (
          <button key={doc.id} className="doc" onClick={() => onOpen({ docId: doc.id })}>
            <span className="tag">{doc.source_type}</span>
            <span className="doc-title">{doc.title}</span>
            <span className="muted doc-meta">
              {Math.max(1, Math.round(doc.chars / 5)).toLocaleString()} words ·{" "}
              {new Date(doc.created_at).toLocaleDateString()}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
