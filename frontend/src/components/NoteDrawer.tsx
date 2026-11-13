import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api, type DocDetail } from "../api";
import type { OpenNote } from "../App";

interface Props {
  note: OpenNote;
  onOpen: (note: OpenNote) => void;
  onClose: () => void;
  onDeleted: () => void;
}

export default function NoteDrawer({ note, onOpen, onClose, onDeleted }: Props) {
  const [doc, setDoc] = useState<DocDetail | null>(null);
  const [error, setError] = useState("");
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    let stale = false;
    setDoc(null);
    setError("");
    setConfirming(false);
    api
      .document(note.docId)
      .then((d) => !stale && setDoc(d))
      .catch((e: Error) => !stale && setError(e.message));
    return () => {
      stale = true;
    };
  }, [note.docId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const isLink = doc?.source?.match(/^https?:\/\//i);
  const rendersMarkdown = doc && ["markdown", "note", "web", "html", "image"].includes(doc.source_type);

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer">
        <div className="drawer-head">
          <div>
            <h2>{doc?.title ?? (error ? "Note unavailable" : "Loading…")}</h2>
            {doc && (
              <div className="muted drawer-meta">
                <span className="tag">{doc.source_type}</span>
                {isLink ? (
                  <a href={doc.source!} target="_blank" rel="noreferrer">
                    {doc.source}
                  </a>
                ) : (
                  doc.source
                )}
                <span>saved {new Date(doc.created_at).toLocaleDateString()}</span>
              </div>
            )}
          </div>
          <button className="button" onClick={onClose} aria-label="Close">
            Close
          </button>
        </div>

        <div className="drawer-body">
          {error && <div className="banner banner-bad">{error}</div>}
          {note.passage && doc && (
            <div className="passage">
              <div className="sources-label">Matched passage</div>
              {note.passage}
            </div>
          )}
          {doc &&
            (rendersMarkdown ? (
              <div className="note-content">
                <ReactMarkdown>{doc.content}</ReactMarkdown>
              </div>
            ) : (
              <div className="note-content plain">{doc.content}</div>
            ))}

          {doc && doc.related.length > 0 && (
            <div className="related">
              <div className="sources-label">Related notes</div>
              {doc.related.map((r) => (
                <button key={r.id} className="doc" onClick={() => onOpen({ docId: r.id })}>
                  <span className="doc-title">{r.title}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {doc && (
          <div className="drawer-foot">
            {confirming ? (
              <>
                <span>Delete this note permanently?</span>
                <button
                  className="button danger"
                  onClick={() =>
                    api
                      .deleteDocument(doc.id)
                      .then(onDeleted)
                      .catch((e: Error) => setError(e.message))
                  }
                >
                  Delete
                </button>
                <button className="button" onClick={() => setConfirming(false)}>
                  Cancel
                </button>
              </>
            ) : (
              <button className="button" onClick={() => setConfirming(true)}>
                Delete note
              </button>
            )}
          </div>
        )}
      </aside>
    </>
  );
}
