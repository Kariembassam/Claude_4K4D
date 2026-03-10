/**
 * OrbitControls for Three.js r128
 * =================================
 * Orbit/zoom/pan camera controls for 3D viewer.
 * Used by the 4K4D viewer for interactive point cloud exploration.
 *
 * Based on Three.js examples/jsm/controls/OrbitControls.js (r128)
 * Bundled for offline use on RunPod (no CDN access).
 */

(function (global) {
    'use strict';

    const THREE = global.THREE;

    if (!THREE) {
        console.warn('OrbitControls: THREE is not defined. Load three.min.js first.');
        return;
    }

    const STATE = {
        NONE: -1,
        ROTATE: 0,
        DOLLY: 1,
        PAN: 2,
        TOUCH_ROTATE: 3,
        TOUCH_PAN: 4,
        TOUCH_DOLLY_PAN: 5,
        TOUCH_DOLLY_ROTATE: 6
    };

    const EPS = 0.000001;
    const TWO_PI = 2 * Math.PI;

    class OrbitControls {

        constructor(camera, domElement) {
            this.camera = camera;
            this.domElement = domElement;

            // API
            this.enabled = true;
            this.target = new THREE.Vector3();

            this.minDistance = 0;
            this.maxDistance = Infinity;

            this.minZoom = 0;
            this.maxZoom = Infinity;

            this.minPolarAngle = 0;
            this.maxPolarAngle = Math.PI;

            this.minAzimuthAngle = -Infinity;
            this.maxAzimuthAngle = Infinity;

            this.enableDamping = false;
            this.dampingFactor = 0.05;

            this.enableZoom = true;
            this.zoomSpeed = 1.0;

            this.enableRotate = true;
            this.rotateSpeed = 1.0;

            this.enablePan = true;
            this.panSpeed = 1.0;
            this.screenSpacePanning = true;

            this.autoRotate = false;
            this.autoRotateSpeed = 2.0;

            this.enableKeys = true;
            this.keys = { LEFT: 37, UP: 38, RIGHT: 39, BOTTOM: 40 };

            // Internal state
            this._state = STATE.NONE;
            this._spherical = { radius: 1, phi: 0, theta: 0 };
            this._sphericalDelta = { radius: 0, phi: 0, theta: 0 };
            this._scale = 1;
            this._panOffset = new THREE.Vector3();
            this._zoomChanged = false;

            this._rotateStart = { x: 0, y: 0 };
            this._rotateEnd = { x: 0, y: 0 };
            this._rotateDelta = { x: 0, y: 0 };

            this._panStart = { x: 0, y: 0 };
            this._panEnd = { x: 0, y: 0 };
            this._panDelta = { x: 0, y: 0 };

            this._dollyStart = { x: 0, y: 0 };
            this._dollyEnd = { x: 0, y: 0 };
            this._dollyDelta = { x: 0, y: 0 };

            // Initialize spherical coordinates from camera position
            this._updateSpherical();

            // Bind event handlers
            this._onMouseDown = this._onMouseDown.bind(this);
            this._onMouseMove = this._onMouseMove.bind(this);
            this._onMouseUp = this._onMouseUp.bind(this);
            this._onMouseWheel = this._onMouseWheel.bind(this);
            this._onTouchStart = this._onTouchStart.bind(this);
            this._onTouchMove = this._onTouchMove.bind(this);
            this._onTouchEnd = this._onTouchEnd.bind(this);
            this._onContextMenu = this._onContextMenu.bind(this);

            // Attach listeners
            this.domElement.addEventListener('mousedown', this._onMouseDown, false);
            this.domElement.addEventListener('wheel', this._onMouseWheel, { passive: false });
            this.domElement.addEventListener('touchstart', this._onTouchStart, { passive: false });
            this.domElement.addEventListener('touchmove', this._onTouchMove, { passive: false });
            this.domElement.addEventListener('touchend', this._onTouchEnd, false);
            this.domElement.addEventListener('contextmenu', this._onContextMenu, false);

            this.update();
        }

        _updateSpherical() {
            const offset = new THREE.Vector3();
            offset.copy(this.camera.position).sub(this.target);
            this._spherical.radius = offset.length();
            if (this._spherical.radius < EPS) this._spherical.radius = EPS;
            this._spherical.theta = Math.atan2(offset.x, offset.z);
            this._spherical.phi = Math.acos(Math.max(-1, Math.min(1, offset.y / this._spherical.radius)));
        }

        update() {
            const offset = new THREE.Vector3();
            const position = this.camera.position;

            // Apply rotation delta
            this._spherical.theta += this._sphericalDelta.theta;
            this._spherical.phi += this._sphericalDelta.phi;

            // Auto-rotate
            if (this.autoRotate && this._state === STATE.NONE) {
                this._spherical.theta += (2 * Math.PI / 60 / 60) * this.autoRotateSpeed;
            }

            // Clamp phi
            this._spherical.phi = Math.max(this.minPolarAngle, Math.min(this.maxPolarAngle, this._spherical.phi));
            this._spherical.phi = Math.max(EPS, Math.min(Math.PI - EPS, this._spherical.phi));

            // Scale
            this._spherical.radius *= this._scale;
            this._spherical.radius = Math.max(this.minDistance, Math.min(this.maxDistance, this._spherical.radius));

            // Pan
            this.target.add(this._panOffset);

            // Convert spherical to cartesian
            offset.x = this._spherical.radius * Math.sin(this._spherical.phi) * Math.sin(this._spherical.theta);
            offset.y = this._spherical.radius * Math.cos(this._spherical.phi);
            offset.z = this._spherical.radius * Math.sin(this._spherical.phi) * Math.cos(this._spherical.theta);

            position.copy(this.target).add(offset);
            this.camera.lookAt(this.target.x, this.target.y, this.target.z);

            // Damping
            if (this.enableDamping) {
                this._sphericalDelta.theta *= (1 - this.dampingFactor);
                this._sphericalDelta.phi *= (1 - this.dampingFactor);
                this._panOffset.multiplyScalar(1 - this.dampingFactor);
            } else {
                this._sphericalDelta.theta = 0;
                this._sphericalDelta.phi = 0;
                this._panOffset.set(0, 0, 0);
            }

            this._scale = 1;
            this._zoomChanged = false;

            return false;
        }

        dispose() {
            this.domElement.removeEventListener('mousedown', this._onMouseDown, false);
            this.domElement.removeEventListener('wheel', this._onMouseWheel, false);
            this.domElement.removeEventListener('touchstart', this._onTouchStart, false);
            this.domElement.removeEventListener('touchmove', this._onTouchMove, false);
            this.domElement.removeEventListener('touchend', this._onTouchEnd, false);
            this.domElement.removeEventListener('contextmenu', this._onContextMenu, false);
            document.removeEventListener('mousemove', this._onMouseMove, false);
            document.removeEventListener('mouseup', this._onMouseUp, false);
        }

        reset() {
            this.target.set(0, 0, 0);
            this.camera.position.set(0, 0, 5);
            this._updateSpherical();
            this.update();
        }

        // Mouse handlers
        _onMouseDown(event) {
            if (!this.enabled) return;
            event.preventDefault();

            switch (event.button) {
                case 0: // left
                    if (this.enableRotate) {
                        this._rotateStart.x = event.clientX;
                        this._rotateStart.y = event.clientY;
                        this._state = STATE.ROTATE;
                    }
                    break;
                case 1: // middle
                    if (this.enableZoom) {
                        this._dollyStart.x = event.clientX;
                        this._dollyStart.y = event.clientY;
                        this._state = STATE.DOLLY;
                    }
                    break;
                case 2: // right
                    if (this.enablePan) {
                        this._panStart.x = event.clientX;
                        this._panStart.y = event.clientY;
                        this._state = STATE.PAN;
                    }
                    break;
            }

            if (this._state !== STATE.NONE) {
                document.addEventListener('mousemove', this._onMouseMove, false);
                document.addEventListener('mouseup', this._onMouseUp, false);
            }
        }

        _onMouseMove(event) {
            if (!this.enabled) return;
            event.preventDefault();

            const rect = this.domElement.getBoundingClientRect
                ? this.domElement.getBoundingClientRect()
                : { width: this.domElement.width || 800, height: this.domElement.height || 600 };

            switch (this._state) {
                case STATE.ROTATE:
                    this._rotateEnd.x = event.clientX;
                    this._rotateEnd.y = event.clientY;
                    this._rotateDelta.x = this._rotateEnd.x - this._rotateStart.x;
                    this._rotateDelta.y = this._rotateEnd.y - this._rotateStart.y;

                    this._sphericalDelta.theta -= 2 * Math.PI * this._rotateDelta.x / rect.width * this.rotateSpeed;
                    this._sphericalDelta.phi -= 2 * Math.PI * this._rotateDelta.y / rect.height * this.rotateSpeed;

                    this._rotateStart.x = this._rotateEnd.x;
                    this._rotateStart.y = this._rotateEnd.y;
                    break;

                case STATE.DOLLY:
                    this._dollyEnd.x = event.clientX;
                    this._dollyEnd.y = event.clientY;
                    this._dollyDelta.y = this._dollyEnd.y - this._dollyStart.y;

                    if (this._dollyDelta.y > 0) {
                        this._scale /= Math.pow(0.95, this.zoomSpeed);
                    } else if (this._dollyDelta.y < 0) {
                        this._scale *= Math.pow(0.95, this.zoomSpeed);
                    }

                    this._dollyStart.x = this._dollyEnd.x;
                    this._dollyStart.y = this._dollyEnd.y;
                    break;

                case STATE.PAN:
                    this._panEnd.x = event.clientX;
                    this._panEnd.y = event.clientY;
                    this._panDelta.x = this._panEnd.x - this._panStart.x;
                    this._panDelta.y = this._panEnd.y - this._panStart.y;

                    this._pan(this._panDelta.x, this._panDelta.y);

                    this._panStart.x = this._panEnd.x;
                    this._panStart.y = this._panEnd.y;
                    break;
            }

            this.update();
        }

        _onMouseUp() {
            this._state = STATE.NONE;
            document.removeEventListener('mousemove', this._onMouseMove, false);
            document.removeEventListener('mouseup', this._onMouseUp, false);
        }

        _onMouseWheel(event) {
            if (!this.enabled || !this.enableZoom) return;
            event.preventDefault();
            event.stopPropagation();

            if (event.deltaY < 0) {
                this._scale *= Math.pow(0.95, this.zoomSpeed);
            } else if (event.deltaY > 0) {
                this._scale /= Math.pow(0.95, this.zoomSpeed);
            }

            this.update();
        }

        _onTouchStart(event) {
            if (!this.enabled) return;
            event.preventDefault();

            switch (event.touches.length) {
                case 1:
                    if (this.enableRotate) {
                        this._rotateStart.x = event.touches[0].pageX;
                        this._rotateStart.y = event.touches[0].pageY;
                        this._state = STATE.TOUCH_ROTATE;
                    }
                    break;
                case 2:
                    if (this.enableZoom || this.enablePan) {
                        const dx = event.touches[0].pageX - event.touches[1].pageX;
                        const dy = event.touches[0].pageY - event.touches[1].pageY;
                        this._dollyStart.x = Math.sqrt(dx * dx + dy * dy);
                        this._panStart.x = (event.touches[0].pageX + event.touches[1].pageX) * 0.5;
                        this._panStart.y = (event.touches[0].pageY + event.touches[1].pageY) * 0.5;
                        this._state = STATE.TOUCH_DOLLY_PAN;
                    }
                    break;
            }
        }

        _onTouchMove(event) {
            if (!this.enabled) return;
            event.preventDefault();
            event.stopPropagation();

            const rect = this.domElement.getBoundingClientRect
                ? this.domElement.getBoundingClientRect()
                : { width: 800, height: 600 };

            switch (this._state) {
                case STATE.TOUCH_ROTATE:
                    this._rotateEnd.x = event.touches[0].pageX;
                    this._rotateEnd.y = event.touches[0].pageY;
                    this._rotateDelta.x = this._rotateEnd.x - this._rotateStart.x;
                    this._rotateDelta.y = this._rotateEnd.y - this._rotateStart.y;

                    this._sphericalDelta.theta -= 2 * Math.PI * this._rotateDelta.x / rect.width * this.rotateSpeed;
                    this._sphericalDelta.phi -= 2 * Math.PI * this._rotateDelta.y / rect.height * this.rotateSpeed;

                    this._rotateStart.x = this._rotateEnd.x;
                    this._rotateStart.y = this._rotateEnd.y;
                    break;

                case STATE.TOUCH_DOLLY_PAN:
                    if (this.enableZoom) {
                        const dx = event.touches[0].pageX - event.touches[1].pageX;
                        const dy = event.touches[0].pageY - event.touches[1].pageY;
                        const distance = Math.sqrt(dx * dx + dy * dy);

                        this._dollyEnd.x = distance;
                        const dollyScale = this._dollyStart.x / this._dollyEnd.x;
                        this._scale *= dollyScale;
                        this._dollyStart.x = this._dollyEnd.x;
                    }
                    if (this.enablePan) {
                        this._panEnd.x = (event.touches[0].pageX + event.touches[1].pageX) * 0.5;
                        this._panEnd.y = (event.touches[0].pageY + event.touches[1].pageY) * 0.5;
                        this._panDelta.x = this._panEnd.x - this._panStart.x;
                        this._panDelta.y = this._panEnd.y - this._panStart.y;
                        this._pan(this._panDelta.x, this._panDelta.y);
                        this._panStart.x = this._panEnd.x;
                        this._panStart.y = this._panEnd.y;
                    }
                    break;
            }

            this.update();
        }

        _onTouchEnd() {
            this._state = STATE.NONE;
        }

        _onContextMenu(event) {
            if (this.enabled) event.preventDefault();
        }

        _pan(deltaX, deltaY) {
            const offset = new THREE.Vector3();
            const position = this.camera.position;

            // Pan proportional to distance from target
            const targetDistance = position.distanceTo(this.target);
            const factor = targetDistance * this.panSpeed * 0.002;

            // Right vector
            offset.set(-deltaX * factor, deltaY * factor, 0);

            // Transform offset to camera space (simplified)
            const cameraDir = new THREE.Vector3();
            cameraDir.copy(this.target).sub(position).normalize();

            const right = new THREE.Vector3();
            right.set(-cameraDir.z, 0, cameraDir.x).normalize();

            const up = new THREE.Vector3(0, 1, 0);

            this._panOffset.add(right.multiplyScalar(-deltaX * factor));
            this._panOffset.add(up.multiplyScalar(deltaY * factor));
        }
    }

    THREE.OrbitControls = OrbitControls;

})(typeof window !== 'undefined' ? window : this);
