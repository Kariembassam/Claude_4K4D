/**
 * PLYLoader for Three.js r128
 * ============================
 * Loads PLY (Polygon File Format) point clouds and meshes.
 * Used by the 4K4D viewer to display reconstructed point clouds.
 *
 * Based on Three.js examples/jsm/loaders/PLYLoader.js (r128)
 * Bundled for offline use on RunPod (no CDN access).
 */

(function (global) {
    'use strict';

    function initPLYLoader(THREE) {
        if (!THREE) return;
        if (THREE.PLYLoader) return; // Already registered

    class PLYLoader extends THREE.Loader {

        constructor(manager) {
            super(manager);
        }

        load(url, onLoad, onProgress, onError) {
            const scope = this;
            const loader = new THREE.FileLoader(this.manager);
            loader.setPath(this.path);
            loader.setResponseType('arraybuffer');
            loader.load(url, function (data) {
                try {
                    onLoad(scope.parse(data));
                } catch (e) {
                    if (onError) {
                        onError(e);
                    } else {
                        console.error(e);
                    }
                }
            }, onProgress, onError);
        }

        parse(data) {
            let geometry;

            if (data instanceof ArrayBuffer) {
                geometry = this._parseBinary(data);
            } else {
                geometry = this._parseASCII(data);
            }

            return geometry;
        }

        _parseHeader(data) {
            const header = {
                comments: [],
                elements: [],
                headerLength: 0,
                format: ''
            };

            const lines = typeof data === 'string'
                ? data.split('\n')
                : this._textDecoder(data).split('\n');

            let currentElement = null;
            let lineIndex = 0;

            for (let i = 0; i < lines.length; i++) {
                let line = lines[i].trim();
                lineIndex += lines[i].length + 1;

                if (line === '') continue;

                const tokens = line.split(/\s+/);
                const keyword = tokens[0];

                switch (keyword) {
                    case 'ply':
                        break;
                    case 'format':
                        header.format = tokens[1];
                        break;
                    case 'comment':
                        header.comments.push(line.substring(8));
                        break;
                    case 'element':
                        currentElement = {
                            name: tokens[1],
                            count: parseInt(tokens[2]),
                            properties: []
                        };
                        header.elements.push(currentElement);
                        break;
                    case 'property':
                        if (currentElement) {
                            if (tokens[1] === 'list') {
                                currentElement.properties.push({
                                    type: 'list',
                                    countType: tokens[2],
                                    itemType: tokens[3],
                                    name: tokens[4]
                                });
                            } else {
                                currentElement.properties.push({
                                    type: tokens[1],
                                    name: tokens[2]
                                });
                            }
                        }
                        break;
                    case 'end_header':
                        header.headerLength = lineIndex;
                        return header;
                }
            }

            return header;
        }

        _textDecoder(buffer) {
            const decoder = new TextDecoder();
            const text = decoder.decode(buffer);
            const headerEnd = text.indexOf('end_header\n');
            if (headerEnd !== -1) {
                return text.substring(0, headerEnd + 11);
            }
            return text;
        }

        _parseASCII(data) {
            const text = typeof data === 'string' ? data : new TextDecoder().decode(data);
            const header = this._parseHeader(text);
            const geometry = new THREE.BufferGeometry();

            const vertexElement = header.elements.find(e => e.name === 'vertex');
            if (!vertexElement) return geometry;

            const lines = text.split('\n');
            let dataStart = 0;
            for (let i = 0; i < lines.length; i++) {
                if (lines[i].trim() === 'end_header') {
                    dataStart = i + 1;
                    break;
                }
            }

            const count = vertexElement.count;
            const properties = vertexElement.properties;

            const positions = new Float32Array(count * 3);
            const colors = new Float32Array(count * 3);
            const normals = new Float32Array(count * 3);
            let hasColors = false;
            let hasNormals = false;

            // Map property indices
            const propMap = {};
            properties.forEach((p, idx) => { propMap[p.name] = idx; });

            hasColors = ('red' in propMap) || ('r' in propMap);
            hasNormals = ('nx' in propMap);

            for (let i = 0; i < count; i++) {
                const lineIdx = dataStart + i;
                if (lineIdx >= lines.length) break;

                const values = lines[lineIdx].trim().split(/\s+/).map(Number);

                // Positions
                positions[i * 3] = values[propMap['x']] || 0;
                positions[i * 3 + 1] = values[propMap['y']] || 0;
                positions[i * 3 + 2] = values[propMap['z']] || 0;

                // Colors
                if (hasColors) {
                    const rKey = 'red' in propMap ? 'red' : 'r';
                    const gKey = 'green' in propMap ? 'green' : 'g';
                    const bKey = 'blue' in propMap ? 'blue' : 'b';
                    colors[i * 3] = (values[propMap[rKey]] || 0) / 255;
                    colors[i * 3 + 1] = (values[propMap[gKey]] || 0) / 255;
                    colors[i * 3 + 2] = (values[propMap[bKey]] || 0) / 255;
                }

                // Normals
                if (hasNormals) {
                    normals[i * 3] = values[propMap['nx']] || 0;
                    normals[i * 3 + 1] = values[propMap['ny']] || 0;
                    normals[i * 3 + 2] = values[propMap['nz']] || 0;
                }
            }

            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            if (hasColors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            if (hasNormals) geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));

            geometry.computeBoundingSphere();
            return geometry;
        }

        _parseBinary(data) {
            const header = this._parseHeader(data);
            const geometry = new THREE.BufferGeometry();

            const vertexElement = header.elements.find(e => e.name === 'vertex');
            if (!vertexElement) return geometry;

            const littleEndian = header.format === 'binary_little_endian';
            const dataView = new DataView(data, header.headerLength);
            const count = vertexElement.count;
            const properties = vertexElement.properties;

            const positions = new Float32Array(count * 3);
            const colors = new Float32Array(count * 3);
            const normals = new Float32Array(count * 3);
            let hasColors = false;
            let hasNormals = false;

            // Build property offsets
            const propInfo = [];
            let offset = 0;
            for (const prop of properties) {
                const size = this._typeSize(prop.type);
                propInfo.push({ name: prop.name, type: prop.type, offset: offset, size: size });
                offset += size;
                if (prop.name === 'red' || prop.name === 'r') hasColors = true;
                if (prop.name === 'nx') hasNormals = true;
            }
            const vertexSize = offset;

            const propLookup = {};
            propInfo.forEach(p => { propLookup[p.name] = p; });

            for (let i = 0; i < count; i++) {
                const baseOffset = i * vertexSize;

                positions[i * 3] = this._readProp(dataView, baseOffset, propLookup['x'], littleEndian);
                positions[i * 3 + 1] = this._readProp(dataView, baseOffset, propLookup['y'], littleEndian);
                positions[i * 3 + 2] = this._readProp(dataView, baseOffset, propLookup['z'], littleEndian);

                if (hasColors) {
                    const rProp = propLookup['red'] || propLookup['r'];
                    const gProp = propLookup['green'] || propLookup['g'];
                    const bProp = propLookup['blue'] || propLookup['b'];
                    colors[i * 3] = this._readProp(dataView, baseOffset, rProp, littleEndian) / 255;
                    colors[i * 3 + 1] = this._readProp(dataView, baseOffset, gProp, littleEndian) / 255;
                    colors[i * 3 + 2] = this._readProp(dataView, baseOffset, bProp, littleEndian) / 255;
                }

                if (hasNormals) {
                    normals[i * 3] = this._readProp(dataView, baseOffset, propLookup['nx'], littleEndian);
                    normals[i * 3 + 1] = this._readProp(dataView, baseOffset, propLookup['ny'], littleEndian);
                    normals[i * 3 + 2] = this._readProp(dataView, baseOffset, propLookup['nz'], littleEndian);
                }
            }

            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            if (hasColors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            if (hasNormals) geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));

            geometry.computeBoundingSphere();
            return geometry;
        }

        _readProp(dataView, baseOffset, propInfo, littleEndian) {
            if (!propInfo) return 0;
            const offset = baseOffset + propInfo.offset;
            switch (propInfo.type) {
                case 'float': case 'float32': return dataView.getFloat32(offset, littleEndian);
                case 'double': case 'float64': return dataView.getFloat64(offset, littleEndian);
                case 'int': case 'int32': return dataView.getInt32(offset, littleEndian);
                case 'uint': case 'uint32': return dataView.getUint32(offset, littleEndian);
                case 'short': case 'int16': return dataView.getInt16(offset, littleEndian);
                case 'ushort': case 'uint16': return dataView.getUint16(offset, littleEndian);
                case 'char': case 'int8': return dataView.getInt8(offset);
                case 'uchar': case 'uint8': return dataView.getUint8(offset);
                default: return 0;
            }
        }

        _typeSize(type) {
            switch (type) {
                case 'float': case 'float32': case 'int': case 'int32': case 'uint': case 'uint32': return 4;
                case 'double': case 'float64': return 8;
                case 'short': case 'int16': case 'ushort': case 'uint16': return 2;
                case 'char': case 'int8': case 'uchar': case 'uint8': return 1;
                default: return 4;
            }
        }
    }

        THREE.PLYLoader = PLYLoader;
    }

    // Try immediate init, or store for deferred init
    if (global.THREE) {
        initPLYLoader(global.THREE);
    }
    global.__4K4D_initPLYLoader = initPLYLoader;

})(typeof window !== 'undefined' ? window : this);
