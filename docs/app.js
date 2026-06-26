/* ============================================================
   BOT / NOT — frontend engine
   Reads window.BOT_DATA (emitted by src/export.py) and renders the
   whole showcase: hero scan, charts, and the interactive explorer.
   No build step, no dependencies — just the DOM and a little SVG.
   ============================================================ */
(function () {
  "use strict";
  const D = window.BOT_DATA;
  if (!D) { console.error("BOT_DATA missing — run `python main.py`"); return; }

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const fmt = (n) => n.toLocaleString("en-US");
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  /* ── reveal on scroll ─────────────────────────────────── */
  const io = new IntersectionObserver(
    (entries) => entries.forEach((e) => { if (e.isIntersecting) e.target.classList.add("in-view"); }),
    { threshold: 0.12 }
  );
  $$(".reveal, .ba-card, .importance, .section").forEach((el) => io.observe(el));

  /* ── count-up numbers ─────────────────────────────────── */
  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    const decimals = parseInt(el.dataset.decimals || "0", 10);
    const suffix = el.dataset.suffix || "";
    const dur = 1300, t0 = performance.now();
    const isInt = decimals === 0 && Number.isInteger(target);
    function tick(now) {
      const p = clamp((now - t0) / dur, 0, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = target * eased;
      el.textContent = (isInt ? fmt(Math.round(val)) : val.toFixed(decimals)) + suffix;
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = (isInt ? fmt(target) : target.toFixed(decimals)) + suffix;
    }
    requestAnimationFrame(tick);
  }
  const countObs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting && !e.target.dataset.done) {
        e.target.dataset.done = "1"; countUp(e.target);
      }
    });
  }, { threshold: 0.5 });
  $$("[data-count]").forEach((el) => countObs.observe(el));

  /* ── footer timestamp ─────────────────────────────────── */
  $("#foot-generated").textContent = "Model output generated " + D.generated_at;

  /* ========================================================
     HERO — population scan
     A field of dots (humans + bots). A lime scanline sweeps
     left→right; bots flare coral as it passes them.
     ======================================================== */
  (function heroScan() {
    const canvas = $("#hero-canvas");
    const ctx = canvas.getContext("2d");
    const css = getComputedStyle(document.documentElement);
    const C = {
      bot: css.getPropertyValue("--bot").trim() || "#ff5a45",
      human: css.getPropertyValue("--human").trim() || "#5ec6d8",
      signal: css.getPropertyValue("--signal").trim() || "#d2f24a",
    };
    let W, H, dots = [], dpr = Math.min(window.devicePixelRatio || 1, 2);
    const COUNT = 150;

    function resize() {
      W = canvas.clientWidth; H = canvas.clientHeight;
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      dots = [];
      for (let i = 0; i < COUNT; i++) {
        dots.push({
          x: Math.random() * W, y: Math.random() * H,
          r: Math.random() * 1.6 + 1,
          bot: Math.random() < 0.21,
          vx: (Math.random() - 0.5) * 0.18, vy: (Math.random() - 0.5) * 0.18,
          flare: 0,
        });
      }
    }
    let scanX = 0;
    function frame() {
      ctx.clearRect(0, 0, W, H);
      scanX += W / 520;
      if (scanX > W + 60) scanX = -60;
      // scan beam
      const g = ctx.createLinearGradient(scanX - 70, 0, scanX + 8, 0);
      g.addColorStop(0, "rgba(210,242,74,0)");
      g.addColorStop(1, "rgba(210,242,74,0.16)");
      ctx.fillStyle = g; ctx.fillRect(scanX - 70, 0, 78, H);
      ctx.fillStyle = "rgba(210,242,74,0.5)"; ctx.fillRect(scanX, 0, 1.2, H);

      for (const d of dots) {
        d.x += d.vx; d.y += d.vy;
        if (d.x < 0 || d.x > W) d.vx *= -1;
        if (d.y < 0 || d.y > H) d.vy *= -1;
        if (d.bot && Math.abs(d.x - scanX) < 6) d.flare = 1;
        d.flare *= 0.96;
        let col, r = d.r;
        if (d.bot) {
          col = d.flare > 0.05 ? C.bot : "rgba(255,90,69,0.32)";
          r = d.r + d.flare * 2.2;
        } else {
          col = "rgba(94,198,216,0.32)";
        }
        ctx.beginPath(); ctx.arc(d.x, d.y, r, 0, Math.PI * 2);
        ctx.fillStyle = col; ctx.fill();
        if (d.bot && d.flare > 0.05) {
          ctx.beginPath(); ctx.arc(d.x, d.y, r + 4, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(255,90,69,${d.flare * 0.5})`; ctx.lineWidth = 1; ctx.stroke();
        }
      }
      requestAnimationFrame(frame);
    }
    resize(); frame();
    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(resize, 200); });
  })();

  /* ========================================================
     MODEL COMPARISON TABLE
     ======================================================== */
  (function comparison() {
    const tbody = $("#cmp-table tbody");
    const rows = D.model_comparison.slice();
    // find column winners
    const cols = ["roc_auc", "pr_auc", "bot_recall", "bot_f1"];
    const best = {};
    cols.forEach((c) => { best[c] = Math.max(...rows.map((r) => r[c])); });
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      if (r.name === D.selected_model) tr.className = "selected";
      const cell = (c) => {
        const v = r[c].toFixed(3);
        const win = r[c] === best[c] ? " win" : "";
        return `<td class="num${win}">${v}</td>`;
      };
      tr.innerHTML =
        `<td>${r.name}${r.name === D.selected_model ? '<span class="sel-mark">◆ selected</span>' : ""}</td>` +
        cols.map(cell).join("");
      tbody.appendChild(tr);
    });
  })();

  /* ========================================================
     CONFUSION MATRIX
     ======================================================== */
  (function confusion() {
    const c = D.confusion_matrix;
    const el = $("#confusion");
    // grid: corner, pred-human, pred-bot / true-human row / true-bot row
    el.innerHTML = `
      <div class="cf-corner"></div>
      <div class="cf-axis">pred · human</div>
      <div class="cf-axis">pred · bot</div>
      <div class="cf-axis v">true · human</div>
      <div class="cf-cell cf-tn"><span class="cf-n">${c.tn}</span><span class="cf-l">true neg</span></div>
      <div class="cf-cell cf-fp"><span class="cf-n">${c.fp}</span><span class="cf-l">false pos</span></div>
      <div class="cf-axis v">true · bot</div>
      <div class="cf-cell cf-fn"><span class="cf-n">${c.fn}</span><span class="cf-l">false neg</span></div>
      <div class="cf-cell cf-tp"><span class="cf-n">${c.tp}</span><span class="cf-l">true pos</span></div>`;
  })();

  /* ========================================================
     SVG CURVES (ROC + PR)
     ======================================================== */
  function lineChart(containerId, xs, ys, opts) {
    const w = 300, h = 230, pad = { l: 34, r: 12, t: 12, b: 30 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    const X = (v) => pad.l + v * iw;
    const Y = (v) => pad.t + (1 - v) * ih;
    let path = "", area = "";
    xs.forEach((x, i) => {
      const px = X(x), py = Y(ys[i]);
      path += (i ? "L" : "M") + px.toFixed(1) + " " + py.toFixed(1) + " ";
    });
    area = path + `L${X(xs[xs.length - 1]).toFixed(1)} ${Y(0).toFixed(1)} L${X(xs[0]).toFixed(1)} ${Y(0).toFixed(1)} Z`;
    const ticks = [0, 0.5, 1];
    const tickX = ticks.map((t) => `<text class="tick-txt" x="${X(t)}" y="${h - 14}" text-anchor="middle">${t}</text>`).join("");
    const tickY = ticks.map((t) => `<text class="tick-txt" x="${pad.l - 6}" y="${Y(t) + 3}" text-anchor="end">${t}</text>`).join("");
    const diag = opts.diagonal
      ? `<line class="diag" x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}"/>` : "";
    $(containerId).innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" role="img">
        <defs><linearGradient id="sig-grad-${containerId.replace('#','')}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="var(--signal)" stop-opacity="0.5"/>
          <stop offset="1" stop-color="var(--signal)" stop-opacity="0"/>
        </linearGradient></defs>
        <line class="axis-line" x1="${pad.l}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(0)}"/>
        <line class="axis-line" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${Y(0)}"/>
        ${diag}
        <path d="${area}" fill="url(#sig-grad-${containerId.replace('#','')})"/>
        <path class="curve" d="${path}"/>
        ${tickX}${tickY}
        <text class="axis-label" x="${pad.l + iw / 2}" y="${h - 2}" text-anchor="middle">${opts.xl}</text>
      </svg>`;
  }
  lineChart("#roc-chart", D.curves.roc.fpr, D.curves.roc.tpr, { diagonal: true, xl: "false positive rate" });
  lineChart("#pr-chart", D.curves.pr.recall, D.curves.pr.precision, { diagonal: false, xl: "recall" });
  $("#roc-auc-tag").textContent = "AUC " + D.metrics.roc_auc.toFixed(3);
  $("#pr-auc-tag").textContent = "AP " + D.metrics.pr_auc.toFixed(3);

  /* ========================================================
     FEATURE IMPORTANCE BARS
     ======================================================== */
  (function importance() {
    const wrap = $("#importance");
    const top = D.feature_importances.slice(0, 12);
    const max = Math.max(...top.map((f) => f.importance));
    top.forEach((f) => {
      const grp = (f.group || "").split(" ")[0]; // first word -> css class
      const row = document.createElement("div");
      row.className = "imp-row";
      row.innerHTML = `
        <span class="imp-name"><span class="imp-grp ${grp}">${f.group}</span>${f.label}</span>
        <div class="imp-track"><i class="imp-fill" style="--w:${(f.importance / max * 100).toFixed(1)}%"></i></div>
        <span class="imp-val">${f.importance.toFixed(3)}</span>
        <span class="imp-tip">${f.desc}</span>`;
      wrap.appendChild(row);
    });
    io.observe(wrap);
  })();

  /* ========================================================
     EXPLORER — beeswarm of all accounts + dossier
     ======================================================== */
  (function explorer() {
    const users = D.users;
    const swarm = $("#beeswarm");
    const featDefs = D.explorer_features;

    // population stats per explorer feature (for dossier bars + lean direction)
    const stats = {};
    featDefs.forEach((f) => {
      const vals = users.map((u) => u.features[f.name]);
      const humans = users.filter((u) => !u.is_bot).map((u) => u.features[f.name]);
      const bots = users.filter((u) => u.is_bot).map((u) => u.features[f.name]);
      const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
      stats[f.name] = {
        max: Math.max(...vals), min: Math.min(...vals),
        humanMean: mean(humans), botMean: mean(bots),
      };
    });

    // build bee elements
    const bees = users.map((u, i) => {
      const el = document.createElement("div");
      const wrong = u.is_bot !== u.predicted;
      el.className = "bee " + (u.is_bot ? "bot" : "human") + (wrong ? " wrong" : "");
      el.dataset.i = i;
      el.title = "@" + u.username;
      el.addEventListener("click", () => select(i));
      swarm.appendChild(el);
      return el;
    });

    // beeswarm layout: column-bucketed vertical stacking
    function layout() {
      const W = swarm.clientWidth, H = swarm.clientHeight;
      const colW = 9, spacing = 9, dotR = 4.5;
      const nCols = Math.max(1, Math.floor(W / colW));
      const buckets = Array.from({ length: nCols + 1 }, () => []);
      users.forEach((u, i) => {
        const ci = clamp(Math.round(u.bot_probability * nCols), 0, nCols);
        buckets[ci].push(i);
      });
      buckets.forEach((idxs, ci) => {
        const cx = (ci / nCols) * (W - dotR * 2) + dotR;
        const n = idxs.length;
        const fits = n * spacing <= H;
        const step = fits ? spacing : H / n;
        const startY = fits ? (H - n * spacing) / 2 + spacing / 2 : step / 2;
        // sort within column so same-truth dots cluster a touch (visual calm)
        idxs.sort((a, b) => users[a].is_bot - users[b].is_bot);
        idxs.forEach((i, k) => {
          bees[i].style.left = cx + "px";
          bees[i].style.top = (startY + k * step) + "px";
        });
      });
      // threshold line at p=0.5
      let thr = $(".bs-thresh-line", swarm.parentElement);
      if (!thr) {
        thr = document.createElement("div"); thr.className = "bs-thresh-line";
        swarm.appendChild(thr);
      }
      thr.style.left = (0.5 * (W - dotR * 2) + dotR) + "px";
    }

    // filtering + search
    let activeFilter = "all", query = "";
    function applyFilter() {
      users.forEach((u, i) => {
        const wrong = u.is_bot !== u.predicted;
        let show = true;
        if (activeFilter === "bot") show = u.is_bot === 1;
        else if (activeFilter === "human") show = u.is_bot === 0;
        else if (activeFilter === "wrong") show = wrong;
        if (show && query) show = u.username.toLowerCase().includes(query);
        bees[i].classList.toggle("dim", !show);
      });
    }
    $$("#exp-filters .chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        $$("#exp-filters .chip").forEach((c) => c.classList.remove("active"));
        chip.classList.add("active");
        activeFilter = chip.dataset.filter; applyFilter();
      });
    });
    $("#exp-search").addEventListener("input", (e) => {
      query = e.target.value.trim().toLowerCase(); applyFilter();
    });

    // dossier
    let activeIdx = null;
    function select(i) {
      if (activeIdx !== null) bees[activeIdx].classList.remove("active");
      activeIdx = i; bees[i].classList.add("active");
      renderDossier(users[i]);
    }
    function renderDossier(u) {
      const wrong = u.is_bot !== u.predicted;
      const pred = u.predicted === 1 ? "bot" : "human";
      const truth = u.is_bot === 1 ? "bot" : "human";
      const pct = (u.bot_probability * 100).toFixed(1);
      const feats = featDefs.map((f) => {
        const v = u.features[f.name];
        const s = stats[f.name];
        const span = s.max - s.min || 1;
        const w = clamp(((v - s.min) / span) * 100, 1, 100);
        const popX = clamp(((s.humanMean + s.botMean) / 2 - s.min) / span * 100, 0, 100);
        // does this value lean toward the bot population or the human one?
        const leanBot = Math.abs(v - s.botMean) < Math.abs(v - s.humanMean);
        const disp = Math.abs(v) >= 1000 ? fmt(Math.round(v)) : (Number.isInteger(v) ? v : v.toFixed(2));
        return `
          <div class="df">
            <span class="df-name">${f.label}</span>
            <span class="df-val">${disp}</span>
            <div class="df-bar"><i class="${leanBot ? "hi" : "lo"}" style="width:${w}%"></i>
              <span class="df-pop" style="left:${popX}%"></span></div>
          </div>`;
      }).join("");
      $("#dossier").innerHTML = `
        <div class="dossier-card">
          <div class="dos-top">
            <div class="dos-id">
              <h3>@${u.username}</h3>
              <div class="dos-meta">round ${u.dataset} · ground truth: ${truth}</div>
            </div>
            <div class="dos-verdict">
              <span class="verdict-badge ${pred}">predicted ${pred}</span>
              <div class="dos-prob"><span class="pnum" style="color:var(--${pred === 'bot' ? 'bot' : 'human'})">${pct}%</span>
                <span class="plabel">bot probability</span></div>
              <div class="dos-correct ${wrong ? "miss" : "ok"}">${wrong ? "✕ model was wrong" : "✓ model was right"}</div>
            </div>
          </div>
          <div class="probmeter"><i style="width:${pct}%"></i><span class="thr"></span></div>
          <div class="dos-feats">${feats}</div>
          <p style="font-family:var(--font-mono);font-size:11px;color:var(--bone-faint);margin-top:18px;letter-spacing:.04em">
            bars lean <span style="color:var(--bot)">coral</span> when a value sits closer to the bot population,
            <span style="color:var(--human)">teal</span> when closer to humans · lime tick = class midpoint
          </p>
        </div>`;
    }

    layout(); applyFilter();
    // pre-load the dossier with the most confident caught bot, so the section
    // is alive on arrival instead of showing an empty prompt.
    const caught = users
      .map((u, i) => ({ u, i }))
      .filter((o) => o.u.is_bot && o.u.predicted)
      .sort((a, b) => b.u.bot_probability - a.u.bot_probability)[0];
    if (caught) select(caught.i);

    let lt; window.addEventListener("resize", () => { clearTimeout(lt); lt = setTimeout(layout, 200); });
  })();
})();
