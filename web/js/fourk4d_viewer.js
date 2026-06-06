/**
 * ComfyUI-4K4D Viewer Extension
 * ================================
 * Registers the 4K4D viewer widget in ComfyUI.
 * Handles 4 tabs: Video Player, WebGL 3D Orbit, Split View, Iframe Export.
 *
 * Listens for server events:
 * - 4k4d.viewer.load: Load viewer data
 * - 4k4d.progress: Update progress bars
 * - 4k4d.quality_gate: Show quality gate results
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "4K4D.Viewer";

// Inject CSS inline (web/css/ is not loaded by ComfyUI's WEB_DIRECTORY)
const VIEWER_CSS = `
.fourk4d-viewer-container {
    background: #1a1a1a;
    border-radius: 8px;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.fourk4d-tabs {
    display: flex;
    background: #252525;
    border-bottom: 1px solid #333;
}
.fourk4d-tab {
    flex: 1;
    padding: 8px 12px;
    background: transparent;
    border: none;
    color: #888;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.2s;
}
.fourk4d-tab:hover {
    color: #ccc;
    background: #2a2a2a;
}
.fourk4d-tab.active {
    color: #ff8c00;
    background: #1a1a1a;
    border-bottom: 2px solid #ff8c00;
}
.fourk4d-tab-content {
    padding: 10px;
    min-height: 200px;
}
.fourk4d-tab-content video {
    display: block;
    margin: 0 auto;
    background: #000;
}
.fourk4d-status-container pre {
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.4;
}
.fourk4d-qgate-container {
    text-align: center;
    padding: 15px;
}
.fourk4d-gate-indicator {
    font-size: 48px;
    margin-bottom: 10px;
}
.fourk4d-progress {
    height: 4px;
    background: #333;
    border-radius: 2px;
    overflow: hidden;
    margin: 4px 0;
}
.fourk4d-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #ff8c00, #ffb84d);
    transition: width 0.3s ease;
}
.fourk4d-3d-controls {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 11px;
    color: #aaa;
}
.fourk4d-3d-controls button {
    background: #333;
    border: 1px solid #555;
    color: #ccc;
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
}
.fourk4d-3d-controls button:hover {
    background: #444;
    color: #fff;
}
.fourk4d-3d-controls input[type="range"] {
    flex: 1;
    accent-color: #ff8c00;
}
.fourk4d-3d-controls .frame-label {
    min-width: 70px;
    text-align: right;
}
.fourk4d-3d-info {
    color: #666;
    font-size: 10px;
    text-align: center;
    padding: 4px 0 0;
}
`;

// Inject styles once on load
if (!document.getElementById("fourk4d-viewer-styles")) {
    const styleEl = document.createElement("style");
    styleEl.id = "fourk4d-viewer-styles";
    styleEl.textContent = VIEWER_CSS;
    document.head.appendChild(styleEl);
}

app.registerExtension({
    name: EXTENSION_NAME,

    async setup() {
        console.log("[4K4D] Viewer extension loaded");

        // Listen for viewer load events
        api.addEventListener("4k4d.viewer.load", (event) => {
            const data = event.detail;
            console.log("[4K4D] Viewer load event:", data);
            this._handleViewerLoad(data);
        });

        // Listen for progress events
        api.addEventListener("4k4d.progress", (event) => {
            const { node, value, max, text } = event.detail;
            this._updateProgress(node, value, max, text);
        });

        // Listen for quality gate events
        api.addEventListener("4k4d.quality_gate", (event) => {
            const { passed, message } = event.detail;
            this._showQualityGateResult(passed, message);
        });
    },

    async nodeCreated(node) {
        if (node.comfyClass === "FourK4D_Viewer") {
            this._addViewerWidget(node);
        }
        if (node.comfyClass === "FourK4D_StatusMonitor") {
            this._addStatusWidget(node);
        }
        if (node.comfyClass === "FourK4D_QualityGate") {
            this._addQualityGateWidget(node);
        }
    },

    _addViewerWidget(node) {
        const container = document.createElement("div");
        container.className = "fourk4d-viewer-container";
        container.innerHTML = `
            <div class="fourk4d-tabs">
                <button class="fourk4d-tab active" data-tab="video">Video</button>
                <button class="fourk4d-tab" data-tab="webgl">3D View</button>
                <button class="fourk4d-tab" data-tab="split">Split</button>
                <button class="fourk4d-tab" data-tab="iframe">Export</button>
                <button class="fourk4d-tab" data-tab="live">Live 4D</button>
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-video">
                <p style="color:#888;text-align:center;padding:40px;">
                    Connect pipeline and execute to load viewer
                </p>
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-webgl" style="display:none;">
                <canvas id="fourk4d-3d-canvas" style="width:100%;height:520px;display:block;background:#1a1a2e;border-radius:4px;"></canvas>
                <div class="fourk4d-3d-controls" id="fourk4d-3d-controls" style="display:none;align-items:center;gap:6px;">
                    <button id="fourk4d-3d-play">Play</button>
                    <input type="range" id="fourk4d-3d-slider" min="0" max="0" value="0" step="1" style="flex:1;">
                    <span class="frame-label" id="fourk4d-3d-frame-label">Frame 0/0</span>
                    <button id="fourk4d-3d-autorotate" title="Toggle auto-rotate">&#x21BB;</button>
                    <button id="fourk4d-3d-reset" title="Reset view">&#x2316;</button>
                    <button id="fourk4d-3d-expand" title="Fullscreen">&#x26F6;</button>
                </div>
                <p class="fourk4d-3d-info">Drag: orbit &bull; Scroll: zoom &bull; Right-drag: pan &bull; &#x21BB; auto-rotate &bull; &#x26F6; fullscreen</p>
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-split" style="display:none;">
                <p style="color:#888;text-align:center;padding:40px;">Split view</p>
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-iframe" style="display:none;">
                <textarea readonly style="width:100%;height:100px;background:#2a2a2a;color:#fff;border:1px solid #444;font-family:monospace;font-size:12px;padding:8px;"></textarea>
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-live" style="display:none;">
                <canvas id="fourk4d-live-canvas" style="width:100%;height:360px;display:block;background:#000;border-radius:4px;cursor:grab;"></canvas>
                <div id="fourk4d-live-timeline" title="Click to scrub &middot; &#x25C6; = camera keyframe" style="position:relative;height:30px;margin-top:6px;background:#161616;border:1px solid #333;border-radius:4px;cursor:pointer;user-select:none;overflow:hidden;"></div>
                <div style="display:flex;align-items:center;gap:6px;margin-top:6px;flex-wrap:wrap;">
                    <button id="fourk4d-live-play" title="Build &amp; play the keyframed move">&#x25B6; Play</button>
                    <span id="fourk4d-live-frame" style="font-size:11px;color:#bbb;min-width:46px;text-align:center;">0/131</span>
                    <button id="fourk4d-live-addkey" title="Pin the current camera angle at this frame">&#x25C6; Add Key</button>
                    <button id="fourk4d-live-delkey" title="Delete the nearest camera keyframe">&#x2715; Del Key</button>
                    <button id="fourk4d-live-reset" title="Reset view">&#x2316;</button>
                    <span style="flex:1;"></span>
                    <button id="fourk4d-live-export" title="Render an MP4 of the performance with this camera move (unlocks after a Play buffers)" disabled style="opacity:0.5;">&#x2913; Export (needs Play)</button>
                </div>
                <div style="display:flex;align-items:center;gap:10px;margin-top:5px;">
                    <span id="fourk4d-live-status" style="font-size:11px;color:#bbb;">live: idle</span>
                    <span id="fourk4d-live-keycount" style="font-size:11px;color:#6cf;"></span>
                </div>
                <p class="fourk4d-3d-info" style="margin-top:5px;">Drag: orbit &bull; scroll: zoom &bull; click timeline: scrub &bull; &#x25C6; Add Key pins the angle at that frame &bull; Play previews the move &bull; Export renders it. Needs the render-server on :1024.</p>
            </div>
        `;

        // Tab switching
        container.querySelectorAll(".fourk4d-tab").forEach(btn => {
            btn.addEventListener("click", () => {
                container.querySelectorAll(".fourk4d-tab").forEach(b => b.classList.remove("active"));
                container.querySelectorAll(".fourk4d-tab-content").forEach(c => c.style.display = "none");
                btn.classList.add("active");
                const tabId = `fourk4d-${btn.dataset.tab}`;
                const tabEl = container.querySelector(`#${tabId}`);
                if (tabEl) tabEl.style.display = "block";
                if (btn.dataset.tab === "live" && !node._liveInit) {
                    node._liveInit = true;
                    this._initLive4D(container);
                }
            });
        });

        // Add as ComfyUI widget
        const widget = node.addDOMWidget("viewer", "custom", container, {
            serialize: false,
        });
        widget.computeSize = () => [node.size[0], 560];

        node._viewerContainer = container;
    },

    // Live 4D: stream the EasyVolcap render-server through ComfyUI's /4k4d/stream proxy
    _initLive4D(container) {
        const cv = container.querySelector("#fourk4d-live-canvas");
        const ctx = cv.getContext("2d");
        const statusEl = container.querySelector("#fourk4d-live-status");
        const keycountEl = container.querySelector("#fourk4d-live-keycount");
        const playBtn = container.querySelector("#fourk4d-live-play");
        const frameEl = container.querySelector("#fourk4d-live-frame");
        const resetBtn = container.querySelector("#fourk4d-live-reset");
        const addKeyBtn = container.querySelector("#fourk4d-live-addkey");
        const delKeyBtn = container.querySelector("#fourk4d-live-delkey");
        const exportBtn = container.querySelector("#fourk4d-live-export");
        const timelineEl = container.querySelector("#fourk4d-live-timeline");

        // config — calibrated to the capture dome
        const RW = 512, RH = 512, FL = RW * 0.82;   // lower res -> faster render readback/transfer
        const CENTER = [0.0, 0.0, 0.6], WUP = [0, 0, 1], NFRAMES = 131;
        const BOUNDS = [[-1.4, -1.4, -1.4], [1.4, 1.4, 1.7]];   // cull off-surface floaters
        const EASE = 0.22, HOME = { az: Math.PI / 2, el: 0.58, radius: 3.33 };
        // INTERACT = orbit/scrub -> render the current frame live (eased, no re-buffer);
        // PLAY = buffer 131 frames at the keyframed camera, play at 30fps.
        let cur = { ...HOME }, tgt = { ...HOME }, playhead = 0;
        let mode = "interact", playing = false, dirty = true;
        let keys = [], uAz = [], selKey = -1;       // camera keyframes (sorted by f) + unwrapped az

        // minimal zlib (single stored block) so the server's zlib.decompress() accepts our camera
        const adler32 = d => { let a = 1, b = 0; for (let i = 0; i < d.length; i++) { a = (a + d[i]) % 65521; b = (b + a) % 65521; } return ((b << 16) | a) >>> 0; };
        const zlibStore = str => {
            const d = new TextEncoder().encode(str), n = d.length, o = new Uint8Array(7 + n + 4);
            o[0] = 0x78; o[1] = 0x01; o[2] = 0x01; o[3] = n & 0xff; o[4] = (n >> 8) & 0xff; o[5] = (~n) & 0xff; o[6] = ((~n) >> 8) & 0xff;
            o.set(d, 7); const ad = adler32(d); o[7 + n] = (ad >>> 24) & 0xff; o[8 + n] = (ad >>> 16) & 0xff; o[9 + n] = (ad >>> 8) & 0xff; o[10 + n] = ad & 0xff;
            return o;
        };
        const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]], dot = (a, b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
        const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
        const norm = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/l, a[1]/l, a[2]/l]; };
        const clampEl = v => Math.max(-1.45, Math.min(1.45, v));
        const clampR = v => Math.max(2.0, Math.min(8.0, v));
        const catmull = (p0,p1,p2,p3,u) => { const u2=u*u, u3=u2*u; return 0.5*((2*p1)+(-p0+p2)*u+(2*p0-5*p1+4*p2-p3)*u2+(-p0+3*p1-3*p2+p3)*u3); };

        // orbit {az,el,radius} -> camera dict at normalized time t (look-at CENTER)
        const camAtView = (o, t) => {
            const ce=Math.cos(o.el), se=Math.sin(o.el), ca=Math.cos(o.az), sa=Math.sin(o.az);
            const eye=[CENTER[0]+o.radius*ce*ca, CENTER[1]+o.radius*ce*sa, CENTER[2]+o.radius*se];
            const fwd=norm(sub(CENTER,eye)), right=norm(cross(fwd,WUP)), down=cross(fwd,right);
            return { H: RH, W: RW, K: [[FL,0,RW/2],[0,FL,RH/2],[0,0,1]], R: [right, down, fwd],
                T: [[-dot(right,eye)],[-dot(down,eye)],[-dot(fwd,eye)]], n: 0.02, f: 100.0, t: t, v: 0.0,
                bounds: BOUNDS, mass: 0.1, moment_of_inertia: 0.1, movement_force: 1.0, movement_torque: 1.0,
                movement_speed: 1.0, origin: CENTER, world_up: WUP };
        };

        // camera keyframes -> interpolated orbit at performance frame f (Catmull-Rom through keys)
        const rebuildKeys = () => {
            keys.sort((a,b)=>a.f-b.f);
            uAz=[]; let prev=null;
            for (const k of keys) { let a=k.az; if(prev!==null){ while(a-prev>Math.PI)a-=2*Math.PI; while(a-prev<-Math.PI)a+=2*Math.PI; } uAz.push(a); prev=a; }
            renderMarkers();
        };
        const orbitAt = f => {
            const n=keys.length;
            if (n===0) return { az:cur.az, el:cur.el, radius:cur.radius };
            if (n===1) return { az:keys[0].az, el:keys[0].el, radius:keys[0].radius };
            const fc=Math.max(keys[0].f, Math.min(keys[n-1].f, f));
            let i=0; while (i<n-1 && keys[i+1].f<=fc) i++; if (i>n-2) i=n-2;
            const i0=Math.max(0,i-1), i3=Math.min(n-1,i+2), span=(keys[i+1].f-keys[i].f)||1, u=(fc-keys[i].f)/span;
            return { az: catmull(uAz[i0],uAz[i],uAz[i+1],uAz[i3],u),
                     el: clampEl(catmull(keys[i0].el,keys[i].el,keys[i+1].el,keys[i3].el,u)),
                     radius: clampR(catmull(keys[i0].radius,keys[i].radius,keys[i+1].radius,keys[i3].radius,u)) };
        };

        // ---- canvas ----
        const fit = () => { const r = cv.getBoundingClientRect(); cv.width = r.width || 800; cv.height = r.height || 360; };
        fit(); try { new ResizeObserver(() => fit()).observe(cv); } catch (e) {}
        const draw = bmp => {
            const s = Math.min(cv.width/bmp.width, cv.height/bmp.height), w = bmp.width*s, h = bmp.height*s;
            ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height);
            ctx.save(); ctx.translate((cv.width-w)/2, (cv.height-h)/2+h); ctx.scale(1, -1);  // server flips its render
            ctx.drawImage(bmp, 0, 0, w, h); ctx.restore();
        };

        // ---- timeline (playhead + keyframe markers) ----
        const pct = f => (NFRAMES<=1 ? 0 : (f/(NFRAMES-1))*100);
        timelineEl.innerHTML = '<div class="ph" style="position:absolute;top:0;bottom:0;width:2px;background:#48bb78;left:0;"></div>';
        const phEl = timelineEl.querySelector(".ph");
        const updatePlayhead = () => { phEl.style.left = pct(playhead)+"%"; };
        function renderMarkers() {
            timelineEl.querySelectorAll("[data-ki]").forEach(e => e.remove());
            keys.forEach((k,i)=>{ const d=document.createElement("div"); d.dataset.ki=i; d.title=`camera key @frame ${k.f}`;
                d.style.cssText=`position:absolute;top:50%;left:${pct(k.f)}%;width:11px;height:11px;margin:-6px 0 0 -6px;background:${i===selKey?'#f6ad55':'#6cf'};transform:rotate(45deg);border:1px solid #000;`;
                timelineEl.appendChild(d); });
            if (keycountEl) keycountEl.textContent = keys.length ? `${keys.length} cam key${keys.length>1?'s':''}` : "no keys (fixed angle)";
            updatePlayhead();
        }

        // ---- ws ping-pong + modes ----
        const proto = location.protocol === "https:" ? "wss" : "ws";
        const url = `${proto}://${location.host}/4k4d/stream`;
        const PRIME = 4;
        let cache = new Array(NFRAMES).fill(null), recvCount = 0, sentCount = 0, playIdx = 0;
        let active = false, playTimer = null, ws = null;
        const setStatus = (t, c) => { statusEl.textContent = t; statusEl.style.color = c || "#888"; };
        const sendCam = o => { if (ws && ws.readyState === 1) ws.send(zlibStore(JSON.stringify(o))); };
        const easeStep = () => { cur.az += (tgt.az-cur.az)*EASE; cur.el += (tgt.el-cur.el)*EASE; cur.radius *= Math.pow(tgt.radius/cur.radius, EASE); };
        const sendInteract = () => sendCam(camAtView(cur, playhead/(NFRAMES-1)));
        const sendBuffer = () => { const fi = sentCount%NFRAMES, t = fi/(NFRAMES-1); sentCount++; sendCam(camAtView(orbitAt(fi), t)); };
        const lockExport = () => { if (exportBtn) { exportBtn.disabled=true; exportBtn.style.opacity="0.5"; exportBtn.innerHTML="&#x2913; Export (needs Play)"; } };
        const unlockExport = () => { if (exportBtn) { exportBtn.disabled=false; exportBtn.style.opacity="1"; exportBtn.innerHTML="&#x2913; Export MP4"; } };

        function enterInteract() {   // orbit/scrub -> render current frame live (no re-buffer)
            mode="interact"; playing=false; dirty=true; playBtn.innerHTML="&#x25B6; Play";
            if (playTimer) { clearInterval(playTimer); playTimer=null; }
            lockExport();
            if (!active) { active=true; sendInteract(); }   // kick the ping-pong if idle
        }
        function startBuffer() {     // Play -> buffer the keyframed move, then play
            mode="buffer"; cache=new Array(NFRAMES).fill(null); recvCount=0; sentCount=0;
            if (playTimer) { clearInterval(playTimer); playTimer=null; }
            lockExport(); setStatus(`buffering 0/${NFRAMES}`);
            if (!active) { active=true; sendBuffer(); }
        }
        const startPlay = () => {    // cache full -> play at 30fps
            mode="play"; active=false; playing=true; dirty=false;
            playBtn.innerHTML="&#x23F8; Pause"; setStatus("playing 30fps", "#48bb78"); unlockExport();
            if (playTimer) clearInterval(playTimer);
            playTimer = setInterval(() => { if (!playing) return; const b = cache[playIdx]; if (b) draw(b);
                playhead = playIdx; frameEl.textContent = `${playIdx}/${NFRAMES}`; updatePlayhead();
                playIdx = (playIdx + 1) % NFRAMES; }, 1000 / 30);
        };

        // ---- controls ----
        let drag = false, lx = 0, ly = 0;
        // stopPropagation so ComfyUI/LiteGraph doesn't drag the NODE while we interact
        cv.addEventListener("pointerdown", e => { e.stopPropagation(); e.preventDefault(); drag=true; lx=e.clientX; ly=e.clientY; cv.setPointerCapture(e.pointerId); });
        cv.addEventListener("pointerup", () => { drag=false; });
        cv.addEventListener("pointermove", e => { if(!drag)return; e.stopPropagation(); tgt.az -= (e.clientX-lx)*0.01; tgt.el = clampEl(tgt.el+(e.clientY-ly)*0.01); lx=e.clientX; ly=e.clientY; enterInteract(); });
        cv.addEventListener("wheel", e => { e.preventDefault(); e.stopPropagation(); tgt.radius = clampR(tgt.radius*Math.exp(e.deltaY*0.001)); enterInteract(); }, { passive: false });

        let scrubbing = false;
        const scrubTo = e => { const r=timelineEl.getBoundingClientRect(); const x=Math.max(0,Math.min(1,(e.clientX-r.left)/(r.width||1)));
            playhead = Math.round(x*(NFRAMES-1)); frameEl.textContent = `${playhead}/${NFRAMES}`;
            if (keys.length>=2) tgt = orbitAt(playhead); enterInteract(); updatePlayhead(); };
        timelineEl.addEventListener("pointerdown", e => { e.stopPropagation(); e.preventDefault(); scrubbing=true; timelineEl.setPointerCapture(e.pointerId);
            const ki = e.target && e.target.dataset ? e.target.dataset.ki : undefined; if (ki!==undefined) { selKey=+ki; renderMarkers(); } scrubTo(e); });
        timelineEl.addEventListener("pointermove", e => { if(!scrubbing)return; e.stopPropagation(); scrubTo(e); });
        timelineEl.addEventListener("pointerup", () => { scrubbing=false; });

        playBtn.addEventListener("click", () => {
            if (mode==="play" && playing) { playing=false; playBtn.innerHTML="&#x25B6; Play"; setStatus("paused"); return; }
            if (mode==="play" && !playing && !dirty) { playing=true; playBtn.innerHTML="&#x23F8; Pause"; setStatus("playing 30fps","#48bb78"); return; }
            startBuffer();   // dirty / interacting -> (re)buffer the keyframed move then play
        });
        resetBtn.addEventListener("click", () => { tgt={...HOME}; enterInteract(); });
        addKeyBtn.addEventListener("click", () => {
            const k = { f:playhead, az:cur.az, el:cur.el, radius:cur.radius };
            const ex = keys.findIndex(x=>x.f===k.f); if (ex>=0) keys[ex]=k; else keys.push(k);
            rebuildKeys(); selKey = keys.findIndex(x=>x.f===k.f); dirty=true; renderMarkers();
            setStatus(`added key @${k.f} (${keys.length} total)`, "#6cf");
        });
        delKeyBtn.addEventListener("click", () => {
            if (!keys.length) return;
            let bi=0, bd=1e9; keys.forEach((k,i)=>{ const d=Math.abs(k.f-playhead); if(d<bd){ bd=d; bi=i; } });
            const f=keys[bi].f; keys.splice(bi,1); selKey=-1; rebuildKeys(); dirty=true;
            setStatus(`deleted key @${f} (${keys.length} left)`, "#6cf");
        });

        // ---- export: the keyframed move (>=2 keys) or the current fixed angle ----
        exportBtn.addEventListener("click", async () => {
            if (exportBtn.disabled) return;
            const RES = 1080;
            const cam = { H: RES, W: RES, K: [[RES*0.82,0,RES/2],[0,RES*0.82,RES/2],[0,0,1]], bounds: BOUNDS, nframes: NFRAMES, center: CENTER, world_up: WUP };
            if (keys.length>=2) cam.keyframes = keys.map(k=>({ f:k.f, az:k.az, el:k.el, radius:k.radius }));
            else { const base = camAtView(keys.length===1?keys[0]:cur, 0); cam.R = base.R; cam.T = base.T; }
            exportBtn.disabled=true; exportBtn.style.opacity="0.5"; exportBtn.innerHTML="&#x2913; Exporting&hellip;";
            playing=false; active=false; if (playTimer) { clearInterval(playTimer); playTimer=null; }  // free the render-server
            setStatus("export: starting…", "#f6ad55");
            let info;
            try { info = await (await fetch("/4k4d/export_video", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(cam) })).json(); }
            catch (e) { setStatus("export: request failed", "#e53e3e"); dirty=true; return; }
            const poll = setInterval(async () => {
                let txt=""; try { txt = await (await fetch(info.log_url + "&_=" + Date.now())).text(); } catch (e) {}
                const m = txt.match(/rendering \d+\/\d+/g); if (m) setStatus("export: " + m[m.length-1], "#f6ad55");
                if (/EXPORT DONE/.test(txt)) {
                    clearInterval(poll); setStatus("export: done ✓ — downloading", "#48bb78");
                    const a = document.createElement("a"); a.href = info.mp4_url; a.download = "dome_dancer_4d.mp4"; document.body.appendChild(a); a.click(); a.remove();
                    window.app.api.dispatchEvent(new CustomEvent("4k4d.viewer.load", { detail: { mp4_path: info.mp4_path, autoplay: true, loop: true } }));
                    dirty=true; setTimeout(() => playBtn.click(), 600);   // re-buffer + resume the live view
                } else if (/EXPORT (FAIL|ERROR)/.test(txt)) { clearInterval(poll); setStatus("export: failed", "#e53e3e"); dirty=true; }
            }, 1500);
        });

        // ---- connect ----
        const connect = () => {
            ws = new WebSocket(url); ws.binaryType = "arraybuffer";
            ws.onopen = () => { setStatus("buffering…"); startBuffer(); };
            ws.onclose = () => { setStatus("disconnected — retry", "#e53e3e"); active=false; if (playTimer) { clearInterval(playTimer); playTimer=null; } setTimeout(connect, 1500); };
            ws.onerror = () => { setStatus("starting render-server (loading model ~2 min)…", "#f6ad55"); };
            ws.onmessage = async ev => {
                let bmp = null; try { bmp = await createImageBitmap(new Blob([ev.data], { type: "image/jpeg" })); } catch (e) {}
                if (!active) return;
                if (mode === "interact") { if (bmp) draw(bmp); easeStep(); sendInteract(); return; }
                if (mode === "buffer") {
                    recvCount++; const idx = recvCount - PRIME - 1;
                    if (idx>=0 && idx<NFRAMES && bmp) { cache[idx]=bmp; draw(bmp); setStatus(`buffering ${idx+1}/${NFRAMES}`); }
                    else if (bmp) draw(bmp);
                    if (idx+1>=NFRAMES) { startPlay(); return; }
                    sendBuffer(); return;
                }
            };
        };
        rebuildKeys(); connect();
    },

    _addStatusWidget(node) {
        const container = document.createElement("div");
        container.className = "fourk4d-status-container";
        container.innerHTML = `
            <pre style="background:#1a1a1a;color:#0f0;padding:10px;font-size:11px;overflow:auto;max-height:300px;border-radius:4px;">
                Status monitor — execute to refresh
            </pre>
        `;
        node.addDOMWidget("status", "custom", container, { serialize: false });
    },

    _addQualityGateWidget(node) {
        const container = document.createElement("div");
        container.className = "fourk4d-qgate-container";
        container.innerHTML = `
            <div style="padding:10px;text-align:center;">
                <div class="fourk4d-gate-indicator" style="font-size:48px;">&#x23F3;</div>
                <p style="color:#888;">Quality Gate — waiting for execution</p>
            </div>
        `;
        node.addDOMWidget("quality_gate", "custom", container, { serialize: false });
        node._gateContainer = container;
    },

    _handleViewerLoad(data) {
        console.log("[4K4D] Loading viewer with data:", data);
        // Find the viewer node and update its content
        const nodes = app.graph._nodes;
        for (const node of nodes) {
            if (node.comfyClass === "FourK4D_Viewer" && node._viewerContainer) {
                // === VIDEO TAB ===
                const videoTab = node._viewerContainer.querySelector("#fourk4d-video");
                if (videoTab && (data.mp4_b64 || data.mp4_path)) {
                    const autoplay = data.autoplay ? "autoplay" : "";
                    const loop = data.loop ? "loop" : "";

                    // Use base64 data URI if available, fall back to custom /4k4d/view route
                    const src = data.mp4_b64
                        ? `data:video/mp4;base64,${data.mp4_b64}`
                        : `/4k4d/view?path=${encodeURIComponent(data.mp4_path)}`;

                    videoTab.innerHTML = `
                        <video ${autoplay} ${loop} controls
                               style="width:100%;max-height:350px;border-radius:4px;">
                            <source src="${src}" type="video/mp4">
                            Video not available
                        </video>
                    `;
                }

                // === 3D VIEW TAB ===
                this._init3DView(node, data);

                // === IFRAME/EXPORT TAB ===
                const iframeTab = node._viewerContainer.querySelector("#fourk4d-iframe textarea");
                if (iframeTab && data.mp4_path) {
                    const viewUrl = `${window.location.origin}/4k4d/view?path=${encodeURIComponent(data.mp4_path)}`;
                    iframeTab.value = `<iframe src="${viewUrl}" width="800" height="600" frameborder="0"></iframe>`;
                }
                break;
            }
        }
    },

    _init3DView(node, data) {
        const webglTab = node._viewerContainer.querySelector("#fourk4d-webgl");
        if (!webglTab) return;

        const plyUrls = data.ply_urls || [];
        if (plyUrls.length === 0) {
            console.log("[4K4D] No PLY URLs available for 3D view");
            webglTab.querySelector("canvas").style.display = "none";
            webglTab.querySelector(".fourk4d-3d-info").textContent = "No 3D data available — run pipeline to generate PLY files";
            return;
        }

        // Check if THREE is available
        if (typeof THREE === "undefined") {
            console.error("[4K4D] THREE.js not loaded — cannot initialize 3D viewer");
            webglTab.querySelector(".fourk4d-3d-info").textContent = "Error: Three.js not loaded";
            return;
        }

        // Deferred init: call stored factory functions if PLYLoader/OrbitControls not yet registered
        // (handles case where these scripts loaded before THREE.js)
        if (!THREE.PLYLoader && window.__4K4D_initPLYLoader) {
            window.__4K4D_initPLYLoader(THREE);
        }
        if (!THREE.OrbitControls && window.__4K4D_initOrbitControls) {
            window.__4K4D_initOrbitControls(THREE);
        }

        if (!THREE.PLYLoader) {
            console.error("[4K4D] THREE.PLYLoader not loaded — check PLYLoader.js");
            webglTab.querySelector(".fourk4d-3d-info").textContent = "Error: PLYLoader not loaded";
            return;
        }

        if (!THREE.OrbitControls) {
            console.error("[4K4D] THREE.OrbitControls not loaded — check OrbitControls.js");
            webglTab.querySelector(".fourk4d-3d-info").textContent = "Error: OrbitControls not loaded";
            return;
        }

        console.log(`[4K4D] Initializing 3D view with ${plyUrls.length} PLY files`);

        // Clean up previous renderer if any
        if (node._threeRenderer) {
            node._threeRenderer.dispose();
            node._threeAnimationId && cancelAnimationFrame(node._threeAnimationId);
        }

        const canvas = webglTab.querySelector("canvas");
        canvas.style.display = "block";

        // Get actual pixel dimensions
        const rect = canvas.getBoundingClientRect();
        const width = rect.width || 640;
        const height = rect.height || 520;
        canvas.width = width * window.devicePixelRatio;
        canvas.height = height * window.devicePixelRatio;
        canvas.style.width = width + "px";
        canvas.style.height = height + "px";

        // Three.js setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);

        const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 1000);
        camera.position.set(0, 0, 3);

        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.1;
        controls.rotateSpeed = 0.7;
        controls.zoomSpeed = 1.1;
        controls.panSpeed = 0.7;
        controls.screenSpacePanning = true;   // pan in screen plane (more intuitive)
        controls.minDistance = 0.05;
        controls.maxDistance = 50;
        controls.autoRotateSpeed = 2.0;
        if ("zoomToCursor" in controls) controls.zoomToCursor = true;

        // keep the renderer matched to the canvas size (node resize was blanking the view)
        if (node._threeResizeObserver) node._threeResizeObserver.disconnect();
        node._threeResizeObserver = new ResizeObserver(() => {
            const r = canvas.getBoundingClientRect();
            if (r.width > 1 && r.height > 1) {
                renderer.setSize(r.width, r.height, false);
                camera.aspect = r.width / r.height;
                camera.updateProjectionMatrix();
            }
        });
        node._threeResizeObserver.observe(canvas);

        // Lighting (for potential mesh rendering)
        scene.add(new THREE.AmbientLight(0xffffff, 0.8));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
        dirLight.position.set(5, 5, 5);
        scene.add(dirLight);

        // Frame management for temporal playback
        const frameGroup = new THREE.Group();
        scene.add(frameGroup);
        const loadedFrames = [];
        let currentFrame = 0;
        let isPlaying = false;
        let playInterval = null;
        let hasSetCamera = false;
        let sharedCenter = null;   // one center for ALL frames (fixes per-frame drift)

        // UI elements
        const controlsDiv = webglTab.querySelector("#fourk4d-3d-controls");
        const slider = webglTab.querySelector("#fourk4d-3d-slider");
        const frameLabel = webglTab.querySelector("#fourk4d-3d-frame-label");
        const playBtn = webglTab.querySelector("#fourk4d-3d-play");

        // Show controls if multiple frames
        if (plyUrls.length > 1) {
            controlsDiv.style.display = "flex";
            slider.max = plyUrls.length - 1;
            slider.value = 0;
            frameLabel.textContent = `Frame 1/${plyUrls.length}`;
        }

        // Load PLY files
        const loader = new THREE.PLYLoader();

        function loadFrame(index) {
            return new Promise((resolve, reject) => {
                if (loadedFrames[index]) {
                    resolve(loadedFrames[index]);
                    return;
                }

                loader.load(
                    plyUrls[index],
                    (geometry) => {
                        const hasColors = geometry.getAttribute && geometry.getAttribute("color");

                        // Compute bbox FIRST so every frame can size its points
                        // relative to its own cloud (the old code only sized frame 0,
                        // leaving all other frames at a tiny fixed 0.008 -> sparse dots).
                        geometry.computeBoundingBox();
                        const bbox = geometry.boundingBox;
                        const _sz = new THREE.Vector3(); bbox.getSize(_sz);
                        const _maxDim = Math.max(_sz.x, _sz.y, _sz.z) || 1;
                        const _cnt = geometry.getAttribute("position").count || 1;
                        // Bigger heuristic so sparse splats read as a solid figure.
                        const _psize = Math.max(0.004,
                            Math.min(_maxDim / Math.pow(_cnt, 1 / 3) * 1.4, _maxDim * 0.05));

                        const material = new THREE.PointsMaterial({
                            size: _psize,
                            vertexColors: hasColors ? true : false,
                            sizeAttenuation: true,
                            color: hasColors ? 0xffffff : 0xff8c00,
                        });

                        const points = new THREE.Points(geometry, material);
                        points.visible = false;

                        const center = new THREE.Vector3();
                        bbox.getCenter(center);
                        // anchor every frame to ONE shared center (frame 0's) so the dancer keeps
                        // its real motion but doesn't drift/jitter from per-frame bbox outliers
                        if (!sharedCenter) sharedCenter = center.clone();
                        points.position.set(-sharedCenter.x, -sharedCenter.y, -sharedCenter.z);

                        loadedFrames[index] = points;
                        frameGroup.add(points);

                        // Auto-fit camera on first loaded frame
                        if (!hasSetCamera) {
                            hasSetCamera = true;
                            const dist = _maxDim > 0 ? _maxDim * 1.8 : 3;
                            camera.position.set(dist * 0.5, dist * 0.3, dist);
                            camera.near = _maxDim * 0.001;
                            camera.far = _maxDim * 100;
                            camera.updateProjectionMatrix();
                            controls.target.set(0, 0, 0);
                            controls.update();
                            if (controls.saveState) controls.saveState();  // so the Reset button restores this fit
                            console.log(`[4K4D] 3D: ${_cnt} points, bbox=${_maxDim.toFixed(3)}, point size=${_psize.toFixed(4)}`);
                        }

                        resolve(points);
                    },
                    undefined,
                    (err) => {
                        console.error(`[4K4D] Failed to load PLY ${index}:`, err);
                        reject(err);
                    }
                );
            });
        }

        function showFrame(index) {
            // Hide all frames
            for (let i = 0; i < loadedFrames.length; i++) {
                if (loadedFrames[i]) loadedFrames[i].visible = false;
            }
            // Show target frame
            if (loadedFrames[index]) {
                loadedFrames[index].visible = true;
            }
            currentFrame = index;
            slider.value = index;
            frameLabel.textContent = `Frame ${index + 1}/${plyUrls.length}`;
        }

        // Load first frame immediately, then preload rest
        loadFrame(0).then(() => {
            showFrame(0);
            // Preload remaining frames in background
            for (let i = 1; i < plyUrls.length; i++) {
                loadFrame(i);
            }
        }).catch(err => {
            console.error("[4K4D] Failed to load initial PLY:", err);
            webglTab.querySelector(".fourk4d-3d-info").textContent = "Failed to load 3D data";
        });

        // Slider event
        slider.addEventListener("input", (e) => {
            const idx = parseInt(e.target.value);
            if (loadedFrames[idx]) {
                showFrame(idx);
            }
        });

        // Play/Pause button
        playBtn.addEventListener("click", () => {
            if (isPlaying) {
                isPlaying = false;
                playBtn.textContent = "Play";
                if (playInterval) clearInterval(playInterval);
            } else {
                isPlaying = true;
                playBtn.textContent = "Pause";
                playInterval = setInterval(() => {
                    const next = (currentFrame + 1) % plyUrls.length;
                    if (loadedFrames[next]) {
                        showFrame(next);
                    }
                }, 33); // ~30fps
            }
        });

        // Reset view
        const resetBtn = webglTab.querySelector("#fourk4d-3d-reset");
        if (resetBtn) resetBtn.addEventListener("click", () => { if (controls.reset) controls.reset(); });

        // Auto-rotate toggle
        const autoBtn = webglTab.querySelector("#fourk4d-3d-autorotate");
        if (autoBtn) autoBtn.addEventListener("click", () => {
            controls.autoRotate = !controls.autoRotate;
            autoBtn.style.background = controls.autoRotate ? "#3a7" : "";
        });

        // Fullscreen / expand
        function resizeRenderer() {
            const fs = document.fullscreenElement;
            const w = fs ? window.innerWidth : (canvas.getBoundingClientRect().width || 640);
            const h = fs ? window.innerHeight : (canvas.getBoundingClientRect().height || 520);
            if (w > 0 && h > 0) {
                renderer.setSize(w, h, false);
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
            }
        }
        const expandBtn = webglTab.querySelector("#fourk4d-3d-expand");
        if (expandBtn) expandBtn.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                const req = canvas.requestFullscreen || canvas.webkitRequestFullscreen;
                if (req) req.call(canvas);
            } else if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        });
        document.addEventListener("fullscreenchange", () => setTimeout(resizeRenderer, 60));

        // Animation loop
        function animate() {
            node._threeAnimationId = requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // Handle resize
        const resizeObserver = new ResizeObserver(() => {
            const newRect = canvas.getBoundingClientRect();
            const w = newRect.width || 640;
            const h = 350;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        });
        resizeObserver.observe(canvas.parentElement);

        // Store references for cleanup
        node._threeRenderer = renderer;
        node._threeScene = scene;
        node._threeControls = controls;
        node._threeResizeObserver = resizeObserver;

        // Update info text
        webglTab.querySelector(".fourk4d-3d-info").textContent =
            `Left-click: rotate \u2022 Scroll: zoom \u2022 Right-click: pan${plyUrls.length > 1 ? " \u2022 Use slider for temporal playback" : ""}`;
    },

    _updateProgress(nodeName, value, max, text) {
        // Could update a progress indicator on the relevant node
        console.log(`[4K4D] Progress: ${nodeName} ${value}/${max} — ${text}`);
    },

    _showQualityGateResult(passed, message) {
        const nodes = app.graph._nodes;
        for (const node of nodes) {
            if (node.comfyClass === "FourK4D_QualityGate" && node._gateContainer) {
                const indicator = node._gateContainer.querySelector(".fourk4d-gate-indicator");
                const text = node._gateContainer.querySelector("p");
                if (passed) {
                    indicator.textContent = "\u2705";
                    indicator.style.color = "#0f0";
                    text.textContent = "Quality Gate PASSED — Training may proceed";
                    text.style.color = "#0f0";
                } else {
                    indicator.textContent = "\u274C";
                    indicator.style.color = "#f00";
                    text.textContent = message || "Quality Gate FAILED — Training blocked";
                    text.style.color = "#f00";
                }
                break;
            }
        }
    },
});
