import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { api, type GraphData } from "../api";
import type { OpenNote } from "../App";

interface Props {
  version: number;
  onOpen: (note: OpenNote) => void;
}

interface Node {
  id: number;
  title: string;
  links: number;
  x?: number;
  y?: number;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export default function GraphView({ version, onOpen }: Props) {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [error, setError] = useState("");
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [hovered, setHovered] = useState<number | null>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .graph()
      .then((g) => {
        setGraph(g);
        setError("");
      })
      .catch((e: Error) => setError(e.message));
  }, [version]);

  useEffect(() => {
    if (!box.current) return;
    const observer = new ResizeObserver(([entry]) =>
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    );
    observer.observe(box.current);
    return () => observer.disconnect();
  }, []);

  // The force layout mutates its input, so give it its own copy.
  const data = useMemo(
    () => (graph ? { nodes: graph.nodes.map((n) => ({ ...n })), links: graph.links.map((l) => ({ ...l })) } : null),
    [graph],
  );

  const neighbours = useMemo(() => {
    const map = new Map<number, Set<number>>();
    for (const link of graph?.links ?? []) {
      if (!map.has(link.source)) map.set(link.source, new Set());
      if (!map.has(link.target)) map.set(link.target, new Set());
      map.get(link.source)!.add(link.target);
      map.get(link.target)!.add(link.source);
    }
    return map;
  }, [graph]);

  const colors = useMemo(
    () => ({ accent: cssVar("--accent"), text: cssVar("--text"), muted: cssVar("--muted"), line: cssVar("--border") }),
    [],
  );

  const isLit = (id: number) => hovered === null || hovered === id || !!neighbours.get(hovered)?.has(id);

  return (
    <div className="page graph-page">
      <h1>Graph</h1>
      <p className="muted">
        Each dot is a note. Lines connect notes about similar things; Cortex finds these links on its own. Click a
        note to open it.
      </p>
      {error && <div className="banner banner-bad">{error}</div>}
      {graph && graph.nodes.length === 0 && <p className="muted">Add some notes and they will appear here.</p>}
      {graph && graph.nodes.length > 0 && graph.links.length === 0 && (
        <p className="muted">No links yet: none of your notes are similar enough to each other.</p>
      )}
      <div className="graph-box" ref={box}>
        {data && size.width > 0 && (
          <ForceGraph2D
            graphData={data}
            width={size.width}
            height={size.height}
            backgroundColor="rgba(0,0,0,0)"
            nodeRelSize={5}
            nodeVal={(n: Node) => 1 + n.links}
            nodeLabel={() => ""}
            linkColor={() => colors.line}
            linkWidth={(l: { similarity: number }) => 0.6 + Math.max(0, l.similarity) * 3}
            cooldownTicks={120}
            onNodeHover={(n: Node | null) => setHovered(n ? n.id : null)}
            onNodeClick={(n: Node) => onOpen({ docId: n.id })}
            nodeCanvasObjectMode={() => "replace"}
            nodeCanvasObject={(n: Node, ctx: CanvasRenderingContext2D, scale: number) => {
              const lit = isLit(n.id);
              const radius = 5 * Math.sqrt(1 + n.links);
              ctx.globalAlpha = lit ? 1 : 0.15;
              ctx.beginPath();
              ctx.arc(n.x!, n.y!, radius, 0, 2 * Math.PI);
              ctx.fillStyle = n.links ? colors.accent : colors.muted;
              ctx.fill();
              // Labels would overlap when zoomed far out; show them once there is room.
              if (scale > 0.9 || hovered === n.id) {
                const fontSize = 12 / scale;
                ctx.font = `${fontSize}px system-ui, sans-serif`;
                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                ctx.fillStyle = colors.text;
                ctx.fillText(n.title.length > 32 ? `${n.title.slice(0, 30)}…` : n.title, n.x!, n.y! + radius + 2 / scale);
              }
              ctx.globalAlpha = 1;
            }}
          />
        )}
      </div>
    </div>
  );
}
