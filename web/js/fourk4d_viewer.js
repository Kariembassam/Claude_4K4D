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
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-video">
                <p style="color:#888;text-align:center;padding:40px;">
                    Connect pipeline and execute to load viewer
                </p>
            </div>
            <div class="fourk4d-tab-content" id="fourk4d-webgl" style="display:none;">
                <canvas id="fourk4d-3d-canvas" style="width:100%;height:350px;display:block;"></canvas>
                <div class="fourk4d-3d-controls" id="fourk4d-3d-controls" style="display:none;">
                    <button id="fourk4d-3d-play">Play</button>
                    <input type="range" id="fourk4d-3d-slider" min="0" max="0" value="0" step="1">
                    <span class="frame-label" id="fourk4d-3d-frame-label">Frame 0/0</span>
                </div>
                <p class="fourk4d-3d-info">Left-click: rotate &bull; Scroll: zoom &bull; Right-click: pan</p>
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
        widget.computeSize = () => [node.size[0], 500];

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
        const height = 350;
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
        controls.dampingFactor = 0.12;
        controls.rotateSpeed = 0.8;
        controls.zoomSpeed = 1.2;
        controls.panSpeed = 0.8;

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
                        // Create point cloud material
                        const hasColors = geometry.getAttribute && geometry.getAttribute("color");
                        const material = new THREE.PointsMaterial({
                            size: 0.008,
                            vertexColors: hasColors ? true : false,
                            sizeAttenuation: true,
                            color: hasColors ? 0xffffff : 0xff8c00,
                        });

                        const points = new THREE.Points(geometry, material);
                        points.visible = false;

                        // Compute bounding box for centering
                        geometry.computeBoundingBox();
                        const bbox = geometry.boundingBox;
                        const center = new THREE.Vector3();
                        bbox.getCenter(center);

                        // Store centering offset
                        points.position.set(-center.x, -center.y, -center.z);

                        loadedFrames[index] = points;
                        frameGroup.add(points);

                        // Auto-fit camera on first loaded frame
                        if (!hasSetCamera) {
                            hasSetCamera = true;
                            const size = new THREE.Vector3();
                            bbox.getSize(size);
                            const maxDim = Math.max(size.x, size.y, size.z);
                            const dist = maxDim > 0 ? maxDim * 1.8 : 3;
                            camera.position.set(dist * 0.5, dist * 0.3, dist);
                            camera.near = maxDim * 0.001;
                            camera.far = maxDim * 100;
                            camera.updateProjectionMatrix();
                            controls.target.set(0, 0, 0);
                            controls.update();

                            // Adjust point size based on scale
                            const pointCount = geometry.getAttribute("position").count;
                            // Heuristic: smaller points for denser clouds
                            const idealSize = maxDim / Math.pow(pointCount, 1/3) * 0.5;
                            material.size = Math.max(0.001, Math.min(idealSize, maxDim * 0.02));

                            console.log(`[4K4D] 3D: ${pointCount} points, bbox size=${maxDim.toFixed(3)}, point size=${material.size.toFixed(4)}`);
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
