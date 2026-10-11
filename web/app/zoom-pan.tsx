"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import styles from "./page.module.css";

const MAX_ZOOM = 6;
const TAP_ZOOM = 2.5;
const HINT_KEY = "pilltally.zoomHintSeen";

function hintSeen() {
  try {
    return localStorage.getItem(HINT_KEY) === "1";
  } catch {
    return false;
  }
}

/**
 * Pinch, double-tap and drag zoom for one photo (its only child), without a library.
 * Unzoomed, one-finger drags still scroll the page; only two-finger gestures are taken over.
 * Transforms are written straight to the DOM so a 12 MP canvas doesn't re-render on every touch move.
 */
export default function ZoomPan({ children, onZoom }: { children: ReactNode; onZoom?: (z: number) => void }) {
  const outerRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const resetRef = useRef<() => void>(() => {});
  const [zoomed, setZoomed] = useState(false);
  const [hint, setHint] = useState(false); // shown until the first zoom, then never again on this phone
  const onZoomRef = useRef(onZoom);
  useEffect(() => {
    onZoomRef.current = onZoom;
  }, [onZoom]);

  useEffect(() => {
    const el = outerRef.current!;
    const box = innerRef.current!;
    setHint(!hintSeen());
    let view = { z: 1, x: 0, y: 0 };
    let reported = 1;
    let start: { z: number; x: number; y: number; d: number; mx: number; my: number } | null = null;
    let settle: ReturnType<typeof setTimeout> | undefined;

    function set(z: number, x: number, y: number, animate = false) {
      const W = el.clientWidth;
      const H = el.clientHeight;
      const photo = box.firstElementChild as HTMLElement | null;
      const ox = photo?.offsetLeft ?? 0;
      const oy = photo?.offsetTop ?? 0;
      const w = photo?.offsetWidth ?? W;
      const h = photo?.offsetHeight ?? H;
      z = Math.min(MAX_ZOOM, Math.max(1, z));
      // Keep the photo filling the frame on an axis where it's bigger than the frame, centered otherwise.
      const clamp = (t: number, size: number, off: number, frame: number) =>
        z * size <= frame ? (frame - z * size) / 2 - z * off : Math.min(-z * off, Math.max(frame - z * (off + size), t));
      x = clamp(x, w, ox, W);
      y = clamp(y, h, oy, H);
      view = { z, x, y };
      box.style.transition = animate ? "transform 0.22s ease-out" : "none";
      box.style.transform = z === 1 ? "" : `translate(${x}px, ${y}px) scale(${z})`;
      el.style.touchAction = z > 1 ? "none" : "";
      setZoomed(z > 1);
      if (z > 1) {
        setHint(false);
        try {
          localStorage.setItem(HINT_KEY, "1");
        } catch {}
      }
      // Tell the page once the zoom has settled (it redraws the marks for the new scale), not on every frame.
      clearTimeout(settle);
      settle = setTimeout(() => {
        if (!start && view.z !== reported) onZoomRef.current?.((reported = view.z));
      }, 250);
    }

    /** Zoom to z keeping the frame point (px, py) over the same spot of the photo. */
    function zoomAt(z: number, px: number, py: number, from = view, animate = false) {
      const nz = Math.min(MAX_ZOOM, Math.max(1, z));
      set(nz, px - (nz * (px - from.x)) / from.z, py - (nz * (py - from.y)) / from.z, animate);
    }

    function toggleAt(px: number, py: number) {
      if (view.z > 1) set(1, 0, 0, true);
      else zoomAt(TAP_ZOOM, px, py, view, true);
    }

    const local = (cx: number, cy: number) => {
      const r = el.getBoundingClientRect();
      return { x: cx - r.left, y: cy - r.top };
    };

    let tap: { x: number; y: number; t: number } | null = null;
    let lastTap: { x: number; y: number; t: number } | null = null;

    function begin(touches: TouchList) {
      if (touches.length >= 2) {
        const [a, b] = [touches[0], touches[1]];
        const m = local((a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2);
        start = { ...view, d: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) || 1, mx: m.x, my: m.y };
        tap = null;
      } else if (touches.length === 1) {
        const m = local(touches[0].clientX, touches[0].clientY);
        start = { ...view, d: 0, mx: m.x, my: m.y };
      }
    }

    function onTouchStart(e: TouchEvent) {
      box.style.transition = "none";
      if (e.touches.length === 1) tap = { x: e.touches[0].clientX, y: e.touches[0].clientY, t: e.timeStamp };
      begin(e.touches);
    }

    function onTouchMove(e: TouchEvent) {
      if (!start) return;
      if (tap && Math.hypot(e.touches[0].clientX - tap.x, e.touches[0].clientY - tap.y) > 10) tap = null;
      if (e.touches.length >= 2 && start.d) {
        e.preventDefault();
        const [a, b] = [e.touches[0], e.touches[1]];
        const m = local((a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2);
        const z = Math.min(MAX_ZOOM, Math.max(1, (start.z * Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)) / start.d));
        // The photo point that was under the fingers' midpoint follows the midpoint (pinch + pan together).
        set(z, m.x - (z * (start.mx - start.x)) / start.z, m.y - (z * (start.my - start.y)) / start.z);
      } else if (e.touches.length === 1 && view.z > 1) {
        e.preventDefault();
        const m = local(e.touches[0].clientX, e.touches[0].clientY);
        set(view.z, start.x + m.x - start.mx, start.y + m.y - start.my);
      }
    }

    function onTouchEnd(e: TouchEvent) {
      if (e.touches.length > 0) {
        begin(e.touches); // lifting one finger of a pinch continues as a pan
        return;
      }
      start = null;
      if (view.z < 1.05 && view.z > 1) set(1, 0, 0, true);
      if (tap && e.timeStamp - tap.t < 300) {
        if (lastTap && tap.t - lastTap.t < 400 && Math.hypot(tap.x - lastTap.x, tap.y - lastTap.y) < 30) {
          e.preventDefault(); // no synthetic click / dblclick after our double tap
          const p = local(tap.x, tap.y);
          toggleAt(p.x, p.y);
          lastTap = null;
        } else lastTap = tap;
      }
      tap = null;
    }

    // Desktop: double-click, trackpad pinch (ctrl+wheel) and mouse drag.
    function onDblClick(e: MouseEvent) {
      const p = local(e.clientX, e.clientY);
      toggleAt(p.x, p.y);
    }
    function onWheel(e: WheelEvent) {
      if (!e.ctrlKey) return;
      e.preventDefault();
      const p = local(e.clientX, e.clientY);
      zoomAt(view.z * Math.exp(-e.deltaY * 0.01), p.x, p.y);
    }
    let drag: { x: number; y: number; cx: number; cy: number } | null = null;
    function onPointerDown(e: PointerEvent) {
      if (e.pointerType !== "mouse" || view.z === 1) return;
      drag = { x: view.x, y: view.y, cx: e.clientX, cy: e.clientY };
      el.setPointerCapture(e.pointerId);
    }
    function onPointerMove(e: PointerEvent) {
      if (drag) set(view.z, drag.x + e.clientX - drag.cx, drag.y + e.clientY - drag.cy);
    }
    const onPointerUp = () => (drag = null);

    const ro = new ResizeObserver(() => set(view.z, view.x, view.y));
    ro.observe(el);
    resetRef.current = () => set(1, 0, 0, true);

    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove", onTouchMove, { passive: false });
    el.addEventListener("touchend", onTouchEnd, { passive: false });
    el.addEventListener("touchcancel", onTouchEnd);
    el.addEventListener("dblclick", onDblClick);
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    return () => {
      clearTimeout(settle);
      ro.disconnect();
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("touchend", onTouchEnd);
      el.removeEventListener("touchcancel", onTouchEnd);
      el.removeEventListener("dblclick", onDblClick);
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
    };
  }, []);

  return (
    <div ref={outerRef} className={`${styles.zoom} ${zoomed ? styles.zoomOn : ""}`}>
      <div ref={innerRef} className={styles.zoomInner}>{children}</div>
      {zoomed ? (
        <button className={styles.zoomReset} onClick={() => resetRef.current()}>Fit</button>
      ) : (
        hint && <span className={styles.zoomHint} aria-hidden="true">Pinch or double-tap to zoom</span>
      )}
    </div>
  );
}
