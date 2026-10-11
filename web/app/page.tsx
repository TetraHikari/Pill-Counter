"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./page.module.css";
import ZoomPan from "./zoom-pan";

type Box = {
  x1: number; y1: number; x2: number; y2: number; conf: number | null; check: boolean;
  type?: string; type_conf?: number; type_unsure?: boolean; // from the pill-type classifier; "unknown" when unsure
};
type Result = {
  count: number; to_check: number; width: number; height: number; boxes: Box[]; seconds: number;
  types?: Record<string, number>;
};
type Outcome = { ok: true; result: Result } | { ok: false; error: string };
type State =
  | { kind: "idle" }
  | { kind: "counting" }
  | { kind: "done"; id: number; image: ImageBitmap; outcome: Outcome }
  | { kind: "error"; message: string };
type HistoryItem = {
  id: number; // Date.now() when saved
  count: number;
  image?: string; // ~900 px JPEG with the marks, for the grid and the detail view
  thumb?: string; // 360 px, items saved before the detail view existed
  toCheck?: number; types?: Record<string, number>; expected?: string; bad?: number; seconds?: number;
};

// Full phone resolution: the detector resizes to 1024 anyway, but the pill-type classifier crops each
// pill from this image and needs the detail (stamp, edges). A 12 MP JPEG is ~3-4 MB, under the 10 MB proxy limit.
const MAX_SIDE = 4032;
const HISTORY_KEY = "pilltally.history";
const HISTORY_MAX = 30;
const HISTORY_SIDE = 900;
const DOT_MIN_PX = 10; // smallest dot radius on screen (CSS px) before neighbours limit it
const COLORS = { ok: "#22c55e", check: "#f59e0b", bad: "#ef4444", unknown: "#8b5cf6" };
const pretty = (t: string) => (t === "unknown" ? "unknown form" : t.replace(/_/g, " "));
const isMismatch = (b: Box, expected: string) => !!expected && !!b.type && (b.type !== expected || !!b.type_unsure);
const checkText = (n: number) => (n ? `${n} orange pill${n === 1 ? "" : "s"} to check before accepting` : "All pills confident");

function mixText(types: Record<string, number> | undefined, expected: string, bad: number) {
  if (!types || Object.keys(types).length === 0) return null;
  const mix = Object.entries(types).sort((a, b) => b[1] - a[1]).map(([t, n]) => `${n} ${pretty(t)}`).join(" · ");
  if (!expected) return mix;
  return `${mix} · ${bad ? `${bad} may not be ${pretty(expected)} (red)` : `all look like ${pretty(expected)}`}`;
}

function whenText(ms: number) {
  const d = new Date(ms);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return d.toDateString() === new Date().toDateString()
    ? time
    : `${d.toLocaleDateString([], { day: "numeric", month: "short" })}, ${time}`;
}

async function shrink(file: File): Promise<{ blob: Blob; image: ImageBitmap }> {
  const original = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, MAX_SIDE / Math.max(original.width, original.height));
  const w = Math.round(original.width * scale);
  const h = Math.round(original.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d")!.drawImage(original, 0, 0, w, h);
  const blob = await new Promise<Blob>((ok, fail) =>
    canvas.toBlob((b) => (b ? ok(b) : fail(new Error("Could not encode photo"))), "image/jpeg", 0.9),
  );
  return { blob, image: await createImageBitmap(canvas) };
}

async function countPills(blob: Blob): Promise<Outcome> {
  const form = new FormData();
  form.append("file", blob, "photo.jpg");
  try {
    const res = await fetch("/api/count", { method: "POST", body: form });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      return { ok: false, error: detail?.detail ?? `The counting server answered ${res.status}. Is it running?` };
    }
    return { ok: true, result: await res.json() };
  } catch {
    return { ok: false, error: "Can't reach the server. Is the Mac on and on the same Wi-Fi?" };
  }
}

/** Reading order (rows top to bottom, then left to right), so the numbers are easy to follow when checking. */
function readingOrder(boxes: Box[]): Box[] {
  if (boxes.length === 0) return boxes;
  const hs = boxes.map((b) => b.y2 - b.y1).sort((a, b) => a - b);
  const band = hs[Math.floor(hs.length / 2)] * 0.6;
  const byY = [...boxes].sort((a, b) => a.y1 + a.y2 - (b.y1 + b.y2));
  const rows: Box[][] = [];
  for (const b of byY) {
    const cy = (b.y1 + b.y2) / 2;
    const row = rows[rows.length - 1];
    if (row && cy - (row[0].y1 + row[0].y2) / 2 < band) row.push(b);
    else rows.push([b]);
  }
  return rows.flatMap((r) => r.sort((a, b) => a.x1 + a.x2 - (b.x1 + b.x2)));
}

function loadHistory(): HistoryItem[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]");
  } catch {
    return [];
  }
}

/** Saves the list, dropping the oldest items until it fits; returns what was kept. */
function saveHistory(items: HistoryItem[]): HistoryItem[] {
  if (items.length === 0) {
    try {
      localStorage.removeItem(HISTORY_KEY);
    } catch {}
    return items;
  }
  for (let list = items; list.length > 0; list = list.slice(0, -1)) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
      return list;
    } catch {
      /* quota or blocked storage: try with fewer items */
    }
  }
  return items; // storage unusable (e.g. private mode): keep this session's list in memory only
}

/** Nearest other pill centre for each box: caps the dot so it doesn't spill onto neighbours. */
function nearestGaps(boxes: Box[]): number[] {
  const c = boxes.map((b) => [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2]);
  return c.map(([x, y], i) => {
    let best = Infinity;
    for (let j = 0; j < c.length; j++) if (j !== i) best = Math.min(best, Math.hypot(c[j][0] - x, c[j][1] - y));
    return best;
  });
}

const Icon = {
  gallery: (
    <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="15" height="13" rx="2.5" /><path d="M6 21h12.5A2.5 2.5 0 0 0 21 18.5V8" /><path d="m5 16 4-4 3 3 2-2 4 3" /><circle cx="13.5" cy="9" r="1.3" /></svg>
  ),
  history: (
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 12a8.5 8.5 0 1 0 2.5-6" /><path d="M3 4v4h4" /><path d="M12 8v4.5l3 2" /></svg>
  ),
  settings: (
    <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" /></svg>
  ),
  reset: (
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 0 0-14.3-4.9L4 8" /><path d="M4 3v5h5" /><path d="M4 13a8 8 0 0 0 14.3 4.9L20 16" /><path d="M20 21v-5h-5" /></svg>
  ),
  back: <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m15 5-7 7 7 7" /></svg>,
  trash: (
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16" /><path d="M9 7V4.5h6V7" /><path d="M6 7l1 13h10l1-13" /><path d="M10 11v5.5M14 11v5.5" /></svg>
  ),
  pills: (
    <svg viewBox="0 0 48 48" aria-hidden="true"><rect x="6" y="15" width="22" height="11" rx="5.5" transform="rotate(-30 17 20.5)" /><circle cx="33" cy="31" r="8" /><path d="M27.5 36.5 38.5 25.5" /></svg>
  ),
};

export default function Home() {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [marks, setMarks] = useState<"dots" | "boxes">("dots");
  const [trueCount, setTrueCount] = useState("");
  const [pillTypes, setPillTypes] = useState<string[]>([]);
  const [expected, setExpected] = useState(""); // "" = just count, any pill type
  const [sheet, setSheet] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [openId, setOpenId] = useState<number | null>(null); // history item shown large
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false); // first tap on Clear arms it, second clears
  const [stageWidth, setStageWidth] = useState(0); // redraw dots when the displayed photo size changes
  const [zoom, setZoom] = useState(1); // ... or when a pinch settles
  const savedFor = useRef<ImageBitmap | null>(null); // the photo whose thumbnail is already in history

  useEffect(() => {
    setHistory(loadHistory());
    // Re-check every 15 s: when the server restarts with a new classifier, the form list updates by itself.
    const load = () =>
      fetch("/api/health").then((r) => r.json()).then((h) => setPillTypes(h.pill_types ?? [])).catch(() => {});
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);
  const cameraRef = useRef<HTMLInputElement>(null);
  const libraryRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const ro = new ResizeObserver(([e]) => setStageWidth(Math.round(e.contentRect.width)));
    ro.observe(stage);
    return () => ro.disconnect();
  }, []);

  async function onFile(file: File | undefined) {
    if (!file) return;
    setState({ kind: "counting" });
    setTrueCount("");
    try {
      const { blob, image } = await shrink(file);
      const outcome = await countPills(blob);
      if (!outcome.ok) throw new Error(outcome.error);
      setZoom(1);
      setState({ kind: "done", id: Date.now(), image, outcome });
    } catch (e) {
      setState({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  const current = state.kind === "done" ? state.outcome : null;

  useEffect(() => {
    if (state.kind !== "done" || !canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = state.image.width;
    canvas.height = state.image.height;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(state.image, 0, 0);
    const outcome = state.outcome;
    if (!outcome.ok) return;
    const color = (b: Box) =>
      isMismatch(b, expected) ? COLORS.bad : b.check ? COLORS.check : b.type_unsure ? COLORS.unknown : COLORS.ok;
    const boxes = readingOrder(outcome.result.boxes);
    // Image pixels per on-screen CSS pixel, zoom included (clientWidth ignores the transform; reading it forces layout).
    const perCss = canvas.width / ((canvas.clientWidth || canvas.width) * zoom);
    if (marks === "boxes") {
      const lw = Math.max(1.5 * perCss, Math.round(Math.max(canvas.width, canvas.height) / 400));
      for (const b of boxes) {
        ctx.lineWidth = isMismatch(b, expected) ? lw * 2 : lw;
        ctx.strokeStyle = color(b);
        ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      }
    } else {
      // A numbered dot on every pill, sized to the pill, like a hand count you can follow. Far-away photos
      // get bigger dots so the numbers stay readable, but never past ~half the gap to the nearest pill.
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const gaps = nearestGaps(boxes);
      boxes.forEach((b, i) => {
        const own = Math.max(10, Math.min(b.x2 - b.x1, b.y2 - b.y1) * 0.3);
        const r = Math.max(own, Math.min(DOT_MIN_PX * perCss, gaps[i] * 0.46));
        const cx = (b.x1 + b.x2) / 2;
        const cy = (b.y1 + b.y2) / 2;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = color(b);
        ctx.fill();
        ctx.lineWidth = r * 0.14;
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.stroke();
        ctx.fillStyle = "#fff";
        const label = String(i + 1);
        ctx.font = `700 ${r * (label.length > 2 ? 0.82 : 1.05)}px system-ui, -apple-system, sans-serif`;
        ctx.fillText(label, cx, cy + r * 0.05);
      });
    }
    // Save each photo once (as first shown, with its marks) to the on-device history.
    if (savedFor.current !== state.image) {
      savedFor.current = state.image;
      const t = document.createElement("canvas");
      const s = Math.min(1, HISTORY_SIDE / Math.max(canvas.width, canvas.height));
      t.width = Math.round(canvas.width * s);
      t.height = Math.round(canvas.height * s);
      t.getContext("2d")!.drawImage(canvas, 0, 0, t.width, t.height);
      const r = outcome.result;
      const item: HistoryItem = {
        id: Date.now(), count: r.count, image: t.toDataURL("image/jpeg", 0.7),
        toCheck: r.to_check, types: r.types, expected, seconds: r.seconds,
        bad: r.boxes.filter((b) => isMismatch(b, expected)).length,
      };
      setHistory(saveHistory([item, ...loadHistory()].slice(0, HISTORY_MAX)));
    }
  }, [state, marks, expected, stageWidth, zoom]);

  const busy = state.kind === "counting";
  const truth = trueCount.trim() === "" ? null : Number(trueCount);
  const count = current?.ok ? current.result.count : null;
  const diff = count !== null && truth !== null && !Number.isNaN(truth) ? count - truth : null;

  function note() {
    if (!current) return null;
    if (!current.ok) return <p className={styles.noteError} role="alert">{current.error}</p>;
    const r = current.result;
    const bad = r.boxes.filter((b) => isMismatch(b, expected)).length;
    const mix = mixText(r.types, expected, bad);
    return (
      <>
        <p className={r.to_check ? styles.noteWarn : styles.noteOk}>
          {checkText(r.to_check)}
          <span className={styles.noteMuted}> · {r.seconds.toFixed(1)} s</span>
        </p>
        {mix && <p className={expected && bad ? styles.noteBad : styles.note}>{mix}</p>}
      </>
    );
  }

  const opened = openId === null ? null : history.find((h) => h.id === openId) ?? null;
  const openItem = (id: number | null) => {
    setOpenId(id);
    setConfirmDelete(false);
    setConfirmClear(false);
  };
  function clearHistory() {
    if (!confirmClear) return setConfirmClear(true);
    setHistory(saveHistory([]));
    setConfirmClear(false);
  }
  function deleteOpened() {
    if (!confirmDelete) return setConfirmDelete(true);
    setHistory(saveHistory(history.filter((h) => h.id !== openId)));
    openItem(null);
  }

  return (
    <main className={styles.app}>
      <header className={styles.top}>
        <button className={styles.iconBtn} aria-label="Start over" disabled={busy}
          onClick={() => { setState({ kind: "idle" }); setTrueCount(""); }}>
          {Icon.reset}
        </button>
        <h1 className={styles.title}>PillTally</h1>
        <button className={styles.iconBtn} aria-label="Settings" onClick={() => setSheet(true)}>{Icon.settings}</button>
      </header>

      <nav className={styles.chips} aria-label="Display options">
        <button className={styles.chip} onClick={() => setMarks(marks === "dots" ? "boxes" : "dots")}>
          {marks === "dots" ? "Numbers" : "Boxes"}
        </button>
        {pillTypes.length > 0 && (
          <button className={`${styles.chip} ${expected ? styles.chipOn : ""}`} onClick={() => setSheet(true)}>
            {expected ? pretty(expected) : "Any form"}
          </button>
        )}
      </nav>

      <section ref={stageRef} className={styles.stage} aria-live="polite">
        {state.kind === "done" && (
          <ZoomPan key={state.id} onZoom={setZoom}>
            <canvas ref={canvasRef} className={styles.photo} />
          </ZoomPan>
        )}
        {state.kind === "idle" && (
          <div className={styles.empty}>
            <span className={styles.emptyIcon}>{Icon.pills}</span>
            <p>Spread the pills in one layer on a plain tray.</p>
            <p className={styles.noteMuted}>Hold the phone at normal height with the whole tray in view, then tap the button.</p>
          </div>
        )}
        {busy && (
          <div className={styles.empty}>
            <span className={styles.spinner} aria-hidden="true" />
            <p>Counting…</p>
          </div>
        )}
        {state.kind === "error" && (
          <div className={styles.empty}>
            <p className={styles.noteError} role="alert">{state.message}</p>
          </div>
        )}
      </section>

      <div className={styles.readout}>
        <div className={styles.notes}>{state.kind === "done" && note()}</div>
        {count !== null && (
          <div className={styles.countRow}>
            <span className={styles.badge} aria-label={`${count} pills`}>
              {count}
            </span>
            {diff !== null && (
              <span className={diff === 0 ? styles.diffOk : styles.diffOff}>
                {diff === 0 ? "matches your count" : `${diff > 0 ? "+" : ""}${diff} vs your count`}
              </span>
            )}
          </div>
        )}
      </div>

      <footer className={styles.controls}>
        <button className={styles.roundBtn} aria-label="Choose from photos" disabled={busy} onClick={() => libraryRef.current?.click()}>
          {Icon.gallery}
        </button>
        <button className={styles.shutter} aria-label="Take photo" disabled={busy} onClick={() => cameraRef.current?.click()}>
          <span />
        </button>
        <button className={styles.roundBtn} aria-label="History" onClick={() => { setShowHistory(true); setConfirmClear(false); }}>{Icon.history}</button>
      </footer>

      <input ref={cameraRef} type="file" accept="image/*" capture="environment" hidden
        onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />
      <input ref={libraryRef} type="file" accept="image/*" hidden
        onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />

      {sheet && (
        <div className={styles.scrim} onClick={() => setSheet(false)}>
          <div className={styles.sheet} role="dialog" aria-label="Settings" onClick={(e) => e.stopPropagation()}>
            <div className={styles.grabber} />
            {pillTypes.length > 0 && (
              <label className={styles.field}>
                Pill form on this tray
                <select value={expected} onChange={(e) => setExpected(e.target.value)}>
                  <option value="">Any (just count)</option>
                  {pillTypes.map((t) => <option key={t} value={t}>{pretty(t)}</option>)}
                </select>
                <small>Pills of another form get a red mark.</small>
              </label>
            )}
            <label className={styles.field}>
              Your hand count (optional)
              <input type="number" inputMode="numeric" min={0} value={trueCount}
                onChange={(e) => setTrueCount(e.target.value)} placeholder="e.g. 60" />
              <small>Shows how far the count is off.</small>
            </label>
            <div className={styles.legend}>
              <span><i style={{ background: COLORS.ok }} /> confident</span>
              <span><i style={{ background: COLORS.check }} /> check this one</span>
              <span><i style={{ background: COLORS.bad }} /> other form</span>
              <span><i style={{ background: COLORS.unknown }} /> unknown form</span>
            </div>
            <button className={styles.done} onClick={() => setSheet(false)}>Done</button>
          </div>
        </div>
      )}

      {showHistory && !opened && (
        <div className={styles.historyPage} role="dialog" aria-label="History">
          <header className={styles.top}>
            <button className={styles.iconBtn} aria-label="Back" onClick={() => setShowHistory(false)}>{Icon.back}</button>
            <h1 className={styles.title}>History</h1>
            <button className={`${styles.textBtn} ${confirmClear ? styles.textBtnBad : ""}`}
              disabled={history.length === 0} onClick={clearHistory}>
              {confirmClear ? `Clear all ${history.length}` : "Clear"}
            </button>
          </header>
          {history.length === 0 ? (
            <p className={styles.historyEmpty}>Counts you take on this phone show up here.</p>
          ) : (
            <ul className={styles.grid}>
              {history.map((h) => (
                <li key={h.id}>
                  <button className={styles.gridItem} onClick={() => openItem(h.id)}
                    aria-label={`${h.count} pills, ${whenText(h.id)}. Open`}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={h.image ?? h.thumb} alt="" />
                    <span className={styles.badgeSmall}>{h.count}</span>
                    <time>{whenText(h.id)}</time>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {showHistory && opened && (
        <div className={`${styles.historyPage} ${styles.detail}`} role="dialog" aria-label={`Count from ${whenText(opened.id)}`}>
          <header className={styles.top}>
            <button className={styles.iconBtn} aria-label="Back to history" onClick={() => openItem(null)}>{Icon.back}</button>
            <h1 className={styles.title}>{whenText(opened.id)}</h1>
            {confirmDelete ? (
              <button className={`${styles.textBtn} ${styles.textBtnBad}`} onClick={deleteOpened}>Delete</button>
            ) : (
              <button className={styles.iconBtn} aria-label="Delete this count" onClick={deleteOpened}>{Icon.trash}</button>
            )}
          </header>
          <section className={styles.stage}>
            <ZoomPan key={opened.id}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img className={styles.photo} src={opened.image ?? opened.thumb} alt={`${opened.count} pills`} />
            </ZoomPan>
          </section>
          <div className={styles.readout}>
            <div className={styles.notes}>
              {opened.toCheck !== undefined && (
                <p className={opened.toCheck ? styles.noteWarn : styles.noteOk}>
                  {checkText(opened.toCheck)}
                  {opened.seconds !== undefined && <span className={styles.noteMuted}> · {opened.seconds.toFixed(1)} s</span>}
                </p>
              )}
              {mixText(opened.types, opened.expected ?? "", opened.bad ?? 0) && (
                <p className={opened.expected && opened.bad ? styles.noteBad : styles.note}>
                  {mixText(opened.types, opened.expected ?? "", opened.bad ?? 0)}
                </p>
              )}
            </div>
            <div className={styles.countRow}>
              <span className={styles.badge} aria-label={`${opened.count} pills`}>{opened.count}</span>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
