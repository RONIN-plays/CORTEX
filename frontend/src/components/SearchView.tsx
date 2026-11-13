import { useEffect, useState } from "react";
import { api, type Hit, type SearchMode } from "../api";
import type { OpenNote } from "../App";

const MODES: { id: SearchMode; label: string; hint: string }[] = [
  { id: "hybrid", label: "Hybrid", hint: "Meaning and keywords combined" },
  { id: "semantic", label: "Semantic", hint: "Matches by meaning, even with different words" },
  { id: "keyword", label: "Keyword", hint: "Matches the exact words" },
];

export default function SearchView({ onOpen }: { onOpen: (note: OpenNote) => void }) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Search as you type, after a short pause; ignore responses that arrive out of order.
  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setHits(null);
      setError("");
      return;
    }
    let stale = false;
    setLoading(true);
    const timer = setTimeout(() => {
      api
        .search(q, mode)
        .then((result) => {
          if (stale) return;
          setHits(result);
          setError("");
        })
        .catch((e: Error) => !stale && setError(e.message))
        .finally(() => !stale && setLoading(false));
    }, 250);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
  }, [query, mode]);

  return (
    <div className="page">
      <h1>Search</h1>
      <input
        className="input big"
        autoFocus
        value={query}
        placeholder="Search your notes by meaning or keyword"
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="segmented">
        {MODES.map((m) => (
          <button key={m.id} className={mode === m.id ? "active" : ""} title={m.hint} onClick={() => setMode(m.id)}>
            {m.label}
          </button>
        ))}
        <span className="muted segmented-hint">{MODES.find((m) => m.id === mode)!.hint}</span>
      </div>

      {error && <div className="banner banner-bad">{error}</div>}
      {!error && hits === null && !loading && (
        <p className="muted">
          Try describing what you remember, like "that thing about losing data when a server dies".
        </p>
      )}
      {!error && hits?.length === 0 && !loading && <p className="muted">No notes match that search.</p>}

      <div className={`results ${loading ? "dim" : ""}`}>
        {hits?.map((hit) => (
          <button key={hit.chunk_id} className="result" onClick={() => onOpen({ docId: hit.doc_id, passage: hit.text })}>
            <div className="result-head">
              <span className="result-title">{hit.title}</span>
              {hit.heading && hit.heading !== hit.title && <span className="muted">{hit.heading}</span>}
              {hit.page && <span className="muted">p. {hit.page}</span>}
              <span className="spacer" />
              {hit.semantic_rank && <span className="tag">meaning #{hit.semantic_rank}</span>}
              {hit.keyword_rank && <span className="tag">keyword #{hit.keyword_rank}</span>}
            </div>
            <div className="result-text">{hit.text}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
