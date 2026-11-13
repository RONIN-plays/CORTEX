import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { streamChat, type Hit } from "../api";
import type { OpenNote } from "../App";

interface Turn {
  question: string;
  answer: string;
  sources: Hit[];
  status: "streaming" | "done" | "error";
  error?: string;
  truncated?: boolean;
}

interface Props {
  hasKey: boolean;
  empty: boolean;
  onOpen: (note: OpenNote) => void;
  goToLibrary: () => void;
}

const SUGGESTIONS = [
  "What have I saved about caching?",
  "Summarize my notes on sleep and memory",
  "How does leader election work in Raft?",
];

/** Some models cite as "【2】" or "【2†source】"; rewrite those to the "[2]" form. */
function normalizeCitations(text: string): string {
  return text.replace(/【\s*(\d+)[^】]*】/g, "[$1]");
}

/** Turn "[2]" into a link the Markdown renderer hands back to us as a citation chip. */
function linkCitations(text: string, sourceCount: number): string {
  return text.replace(/\[(\d+)\](?!\()/g, (match, digits) => {
    const n = Number(digits);
    return n >= 1 && n <= sourceCount ? `[${n}](#cite-${n})` : match;
  });
}

function citedNumbers(text: string, sourceCount: number): number[] {
  const seen = new Set<number>();
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    const n = Number(match[1]);
    if (n >= 1 && n <= sourceCount) seen.add(n);
  }
  return [...seen].sort((a, b) => a - b);
}

export default function Chat({ hasKey, empty, onOpen, goToLibrary }: Props) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const busy = turns.at(-1)?.status === "streaming";

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [turns]);

  useEffect(() => () => abort.current?.abort(), []);

  const updateLast = (change: (turn: Turn) => Turn) =>
    setTurns((all) => all.map((turn, i) => (i === all.length - 1 ? change(turn) : turn)));

  async function ask(question: string) {
    question = question.trim();
    if (!question || busy) return;
    const history = turns
      .filter((t) => t.status === "done" && t.answer)
      .flatMap((t) => [
        { role: "user" as const, content: t.question },
        { role: "assistant" as const, content: t.answer },
      ]);
    setInput("");
    setTurns((all) => [...all, { question, answer: "", sources: [], status: "streaming" }]);
    const controller = new AbortController();
    abort.current = controller;
    try {
      await streamChat(
        question,
        history,
        (event) => {
          if (event.type === "sources") updateLast((t) => ({ ...t, sources: event.sources }));
          else if (event.type === "delta") updateLast((t) => ({ ...t, answer: t.answer + event.text }));
          else if (event.type === "done") updateLast((t) => ({ ...t, status: "done", truncated: event.truncated }));
          else updateLast((t) => ({ ...t, status: "error", error: event.message }));
        },
        controller.signal,
      );
      // A stream that closes without a final event was cut off.
      updateLast((t) =>
        t.status === "streaming" ? { ...t, status: "error", error: "The connection closed before the answer finished." } : t,
      );
    } catch (error) {
      updateLast((t) =>
        controller.signal.aborted
          ? { ...t, status: "done" }
          : { ...t, status: "error", error: error instanceof Error ? error.message : String(error) },
      );
    }
  }

  const openSource = (hit: Hit) => onOpen({ docId: hit.doc_id, passage: hit.text });

  return (
    <div className="chat">
      <div className="chat-scroll">
        {turns.length === 0 && (
          <div className="hero">
            <h1>Ask your notes anything</h1>
            <p className="muted">Answers come from what you have saved, with links back to the source.</p>
            {empty ? (
              <p>
                Your library is empty.{" "}
                <button className="link" onClick={goToLibrary}>
                  Add some notes
                </button>{" "}
                to get started.
              </p>
            ) : (
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="chip" onClick={() => ask(s)}>
                    {s}
                  </button>
                ))}
              </div>
            )}
            {!hasKey && (
              <div className="banner banner-warn">
                No Groq API key is set. You will see which notes match a question, but Cortex cannot write an
                answer until you add <code>GROQ_API_KEY</code> to the <code>.env</code> file and restart the
                server.
              </div>
            )}
          </div>
        )}

        {turns.map((turn, index) => {
          const answer = normalizeCitations(turn.answer);
          const cited = citedNumbers(answer, turn.sources.length);
          // With nothing cited (or no answer at all) show every retrieved passage.
          const shown = cited.length ? cited : turn.sources.map((_, i) => i + 1);
          return (
            <div key={index} className="turn">
              <div className="question">{turn.question}</div>
              <div className="answer">
                {answer ? (
                  <ReactMarkdown
                    components={{
                      a: ({ href, children }) => {
                        const cite = href?.match(/^#cite-(\d+)$/);
                        if (!cite) {
                          return (
                            <a href={href} target="_blank" rel="noreferrer">
                              {children}
                            </a>
                          );
                        }
                        const hit = turn.sources[Number(cite[1]) - 1];
                        return (
                          <button className="cite" title={hit.title} onClick={() => openSource(hit)}>
                            {cite[1]}
                          </button>
                        );
                      },
                    }}
                  >
                    {linkCitations(answer, turn.sources.length)}
                  </ReactMarkdown>
                ) : turn.status === "streaming" ? (
                  <span className="muted pulse">
                    {turn.sources.length ? "Reading your notes…" : "Searching your notes…"}
                  </span>
                ) : null}
                {turn.truncated && <p className="muted">The answer was cut off at the length limit.</p>}
                {turn.error && <div className="banner banner-warn">{turn.error}</div>}
              </div>
              {turn.status !== "streaming" && turn.sources.length > 0 && (
                <div className="sources">
                  <div className="sources-label">{cited.length ? "Sources" : "Closest notes"}</div>
                  {shown.map((n) => {
                    const hit = turn.sources[n - 1];
                    return (
                      <button key={n} className="source" onClick={() => openSource(hit)}>
                        <span className="cite">{n}</span>
                        <span className="source-body">
                          <span className="source-title">
                            {hit.title}
                            {hit.page ? ` · p. ${hit.page}` : hit.heading && hit.heading !== hit.title ? ` · ${hit.heading}` : ""}
                          </span>
                          <span className="source-text">{hit.text}</span>
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
        <div ref={bottom} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
      >
        <textarea
          value={input}
          rows={1}
          placeholder="Ask a question about your notes"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask(input);
            }
          }}
        />
        {busy ? (
          <button type="button" className="button" onClick={() => abort.current?.abort()}>
            Stop
          </button>
        ) : (
          <button type="submit" className="button primary" disabled={!input.trim()}>
            Ask
          </button>
        )}
        {turns.length > 0 && !busy && (
          <button type="button" className="button" onClick={() => setTurns([])}>
            New chat
          </button>
        )}
      </form>
    </div>
  );
}
