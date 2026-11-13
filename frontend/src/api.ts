export interface Health {
  documents: number;
  chunks: number;
  has_api_key: boolean;
  llm_model: string;
  embed_model: string;
  storage_is_temporary: boolean;
  max_upload_mb: number;
}

export interface DocSummary {
  id: number;
  title: string;
  source_type: string;
  source: string | null;
  created_at: string;
  chunk_count: number;
  chars: number;
}

export interface Related {
  id: number;
  title: string;
  similarity: number;
}

export interface DocDetail extends Omit<DocSummary, "chunk_count" | "chars"> {
  content: string;
  related: Related[];
}

export interface Hit {
  chunk_id: number;
  doc_id: number;
  title: string;
  source_type: string;
  text: string;
  heading: string | null;
  page: number | null;
  score: number;
  semantic_rank: number | null;
  keyword_rank: number | null;
}

export type SearchMode = "hybrid" | "semantic" | "keyword";

export interface GraphData {
  nodes: { id: number; title: string; source_type: string; chunk_count: number; links: number }[];
  links: { source: number; target: number; similarity: number }[];
}

export interface IngestResult {
  id?: number;
  title?: string;
  filename?: string;
  duplicate?: boolean;
  error?: string;
}

export type ChatEvent =
  | { type: "sources"; sources: Hit[] }
  | { type: "delta"; text: string }
  | { type: "done"; truncated: boolean }
  | { type: "error"; message: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new Error("Cannot reach the Cortex server. Check that the backend is running.");
  }
  if (response.status === 413) {
    throw new Error("This file is too large for the server to accept.");
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
    } catch {
      // The proxy answers with a non-JSON 5xx when the backend is down.
      if (response.status >= 500) detail = "The Cortex server is not responding. Check that the backend is running.";
    }
    throw new Error(detail);
  }
  return response.json();
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  documents: () => request<DocSummary[]>("/api/documents"),
  document: (id: number) => request<DocDetail>(`/api/documents/${id}`),
  deleteDocument: (id: number) => request<unknown>(`/api/documents/${id}`, { method: "DELETE" }),
  addNote: (title: string, text: string) => request<IngestResult>("/api/documents/text", json({ title, text })),
  addUrl: (url: string) => request<IngestResult>("/api/documents/url", json({ url })),
  /** One request per file, so a size limit or failure affects only that file. */
  upload: async (files: File[]) => {
    const results: IngestResult[] = [];
    for (const file of files) {
      const form = new FormData();
      form.append("files", file);
      try {
        results.push(...(await request<IngestResult[]>("/api/documents/upload", { method: "POST", body: form })));
      } catch (error) {
        results.push({ filename: file.name, error: error instanceof Error ? error.message : String(error) });
      }
    }
    return results;
  },
  search: (q: string, mode: SearchMode) =>
    request<Hit[]>(`/api/search?q=${encodeURIComponent(q)}&mode=${mode}&k=12`),
  graph: () => request<GraphData>("/api/graph"),
};

/** POST a question and call onEvent for each server-sent event until the stream ends. */
export async function streamChat(
  question: string,
  history: { role: "user" | "assistant"; content: string }[],
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch("/api/chat", { ...json({ question, history }), signal });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error("Cannot reach the Cortex server. Check that the backend is running.");
  }
  if (!response.ok || !response.body) {
    throw new Error(
      response.status >= 500
        ? "The Cortex server is not responding. Is the backend running?"
        : `Request failed (${response.status})`,
    );
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (frame.startsWith("data: ")) onEvent(JSON.parse(frame.slice(6)));
    }
  }
}
