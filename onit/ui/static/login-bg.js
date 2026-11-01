/* Animated constellation backdrop for the login view.
   Runs only while #login is visible; static frame under prefers-reduced-motion. */
(() => {
  "use strict";

  const login = document.getElementById("login");
  const canvas = document.getElementById("login-canvas");
  if (!login || !canvas) return;

  const ctx = canvas.getContext("2d");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const LINK_DIST = 150;
  let nodes = [];
  let accent = "#c96442";
  let raf = 0;
  let w = 0, h = 0, dpr = 1;

  function readAccent() {
    accent = getComputedStyle(document.documentElement)
      .getPropertyValue("--accent").trim() || accent;
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = login.clientWidth;
    h = login.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
  }

  function seed() {
    const count = Math.min(70, Math.round((w * h) / 16000));
    nodes = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      r: 1 + Math.random() * 1.4,
    }));
  }

  function step() {
    for (const n of nodes) {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < 0 || n.x > w) n.vx *= -1;
      if (n.y < 0 || n.y > h) n.vy *= -1;
    }
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d = Math.hypot(dx, dy);
        if (d < LINK_DIST) {
          ctx.globalAlpha = (1 - d / LINK_DIST) * 0.14;
          ctx.strokeStyle = accent;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = accent;
    for (const n of nodes) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function loop() {
    step();
    draw();
    raf = requestAnimationFrame(loop);
  }

  function start() {
    if (raf) return;
    readAccent();
    resize();
    if (reducedMotion.matches) {
      draw();
      return;
    }
    raf = requestAnimationFrame(loop);
  }

  function stop() {
    cancelAnimationFrame(raf);
    raf = 0;
  }

  function sync() {
    if (login.hidden) stop();
    else start();
  }

  new MutationObserver(sync).observe(login, { attributes: true, attributeFilter: ["hidden"] });
  new MutationObserver(() => { readAccent(); if (!login.hidden) draw(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  window.addEventListener("resize", () => { if (!login.hidden) { resize(); draw(); } });
  reducedMotion.addEventListener("change", () => { stop(); sync(); });
  sync();
})();
