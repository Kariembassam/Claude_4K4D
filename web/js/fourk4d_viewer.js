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
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-video">
                <p style="color:#888;text-align:center;padding:40px;">
                    Connect pipeline and execute to load viewer
                </p>
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-webgl" style="display:none;">
                <canvas id="fourk4d-3d-canvas" style="width:100%;height:300px;"></canvas>
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-split" style="display:none;">
                <p style="color:#888;text-align:center;padding:40px;">Split view</p>
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-iframe" style="display:none;">
                <textarea readonly style="width:100%;height:100px;background:#2a2a2a;color:#fff;border:1px solid #444;font-family:monospace;font-size:12px;padding:8px;"></textarea>
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
            });
        });

        // Add as ComfyUI widget
        const widget = node.addDOMWidget("viewer", "custom", container, {
            serialize: false,
        });
        widget.computeSize = () => [node.size[0], 400];

        node._viewerContainer = container;
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

                // Update iframe tab with a working embed URL
                const iframeTab = node._viewerContainer.querySelector("#fourk4d-iframe textarea");
                if (iframeTab && data.mp4_path) {
                    const viewUrl = `${window.location.origin}/4k4d/view?path=${encodeURIComponent(data.mp4_path)}`;
                    iframeTab.value = `<iframe src="${viewUrl}" width="800" height="600" frameborder="0"></iframe>`;
                }
                break;
            }
        }
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
