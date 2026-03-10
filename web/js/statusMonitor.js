/**
 * ComfyUI-4K4D Status Monitor Extension
 * Adds live status formatting to the StatusMonitor node.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "4K4D.StatusMonitor",

    async setup() {
        // Listen for progress updates to refresh status display
        api.addEventListener("4k4d.progress", (event) => {
            const { node, value, max, text } = event.detail;
            this._updateStatusNodes(node, value, max, text);
        });
    },

    _updateStatusNodes(nodeName, value, max, text) {
        const nodes = app.graph._nodes;
        for (const node of nodes) {
            if (node.comfyClass === "FourK4D_StatusMonitor") {
                const container = node.widgets?.find(w => w.name === "status");
                if (container?.element) {
                    const pre = container.element.querySelector("pre");
                    if (pre) {
                        const pct = max > 0 ? ((value / max) * 100).toFixed(1) : 0;
                        const bar = "█".repeat(Math.floor(pct / 5)) + "░".repeat(20 - Math.floor(pct / 5));
                        pre.textContent += `\n[${nodeName}] ${bar} ${pct}% — ${text}`;
                        pre.scrollTop = pre.scrollHeight;
                    }
                }
            }
        }
    },
});
