const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
// Use the modernized PDF extractor instead
const pdfParse = require('pdf-extraction');

const DIMS = 16; // demo vectors
// =====================================================================
//  PRIORITY QUEUE (Needed for KD-Tree and HNSW)
// =====================================================================
class Heap {
    constructor(cmp) { this.data =[]; this.cmp = cmp; }
    push(v) { this.data.push(v); this.up(this.data.length - 1); }
    pop() {
        if (this.data.length === 0) return null;
        const top = this.data[0];
        const bot = this.data.pop();
        if (this.data.length > 0) { this.data[0] = bot; this.down(0); }
        return top;
    }
    top() { return this.data[0]; }
    size() { return this.data.length; }
    up(i) {
        while (i > 0) {
            let p = Math.floor((i - 1) / 2);
            if (this.cmp(this.data[i], this.data[p])) {
                [this.data[i], this.data[p]] = [this.data[p], this.data[i]];
                i = p;
            } else break;
        }
    }
    down(i) {
        let l = this.data.length;
        while (true) {
            let left = 2 * i + 1, right = 2 * i + 2, best = i;
            if (left < l && this.cmp(this.data[left], this.data[best])) best = left;
            if (right < l && this.cmp(this.data[right], this.data[best])) best = right;
            if (best !== i) {
                [this.data[i], this.data[best]] = [this.data[best], this.data[i]];
                i = best;
            } else break;
        }
    }
}

// =====================================================================
//  DISTANCE METRICS
// =====================================================================
const Distances = {
    euclidean: (a, b) => {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
        return Math.sqrt(s);
    },
    cosine: (a, b) => {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
        }
        if (na < 1e-9 || nb < 1e-9) return 1.0;
        return 1.0 - dot / (Math.sqrt(na) * Math.sqrt(nb));
    },
    manhattan: (a, b) => {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    }
};

function getDistFn(m) {
    if (m === "cosine") return Distances.cosine;
    if (m === "manhattan") return Distances.manhattan;
    return Distances.euclidean;
}

// =====================================================================
//  BRUTE FORCE
// =====================================================================
class BruteForce {
    constructor() { this.items =[]; }
    insert(v) { this.items.push(v); }
    knn(q, k, distFn) {
        let r = this.items.map(v => ({ dist: distFn(q, v.emb), id: v.id }));
        r.sort((a, b) => a.dist - b.dist);
        return r.slice(0, k);
    }
    remove(id) { this.items = this.items.filter(v => v.id !== id); }
}

// =====================================================================
//  KD-TREE
// =====================================================================
class KDNode {
    constructor(item) { this.item = item; this.left = null; this.right = null; }
}

class KDTree {
    constructor(dims) { this.root = null; this.dims = dims; }
    
    insert(v) {
        const ins = (n, v, d) => {
            if (!n) return new KDNode(v);
            let ax = d % this.dims;
            if (v.emb[ax] < n.item.emb[ax]) n.left = ins(n.left, v, d + 1);
            else n.right = ins(n.right, v, d + 1);
            return n;
        };
        this.root = ins(this.root, v, 0);
    }

    knn(q, k, distFn) {
        let heap = new Heap((a, b) => a.dist > b.dist); // Max-Heap
        
        const search = (n, d) => {
            if (!n) return;
            let dn = distFn(q, n.item.emb);
            if (heap.size() < k || dn < heap.top().dist) {
                heap.push({ dist: dn, id: n.item.id });
                if (heap.size() > k) heap.pop();
            }
            let ax = d % this.dims;
            let diff = q[ax] - n.item.emb[ax];
            let closer = diff < 0 ? n.left : n.right;
            let farther = diff < 0 ? n.right : n.left;
            
            search(closer, d + 1);
            if (heap.size() < k || Math.abs(diff) < heap.top().dist) {
                search(farther, d + 1);
            }
        };
        search(this.root, 0);
        
        let res =[];
        while (heap.size() > 0) res.push(heap.pop());
        return res.reverse();
    }

    rebuild(items) {
        this.root = null;
        for (let v of items) this.insert(v);
    }
}

// =====================================================================
//  HNSW
// =====================================================================
class HNSW {
    constructor(m = 16, efBuild = 200) {
        this.G = new Map();
        this.M = m; this.M0 = 2 * m; this.ef_build = efBuild;
        this.mL = 1.0 / Math.log(m);
        this.topLayer = -1;
        this.entryPt = -1;
    }

    randLevel() { return Math.floor(-Math.log(Math.random()) * this.mL); }

    searchLayer(q, ep, ef, lyr, distFn) {
        let vis = new Set([ep]);
        let cands = new Heap((a, b) => a.dist < b.dist); // Min-Heap
        let found = new Heap((a, b) => a.dist > b.dist); // Max-Heap

        let d0 = distFn(q, this.G.get(ep).item.emb);
        cands.push({ dist: d0, id: ep });
        found.push({ dist: d0, id: ep });

        while (cands.size() > 0) {
            let { dist: cd, id: cid } = cands.pop();
            if (found.size() >= ef && cd > found.top().dist) break;
            
            let node = this.G.get(cid);
            if (lyr >= node.nbrs.length) continue;
            
            for (let nid of node.nbrs[lyr]) {
                if (vis.has(nid) || !this.G.has(nid)) continue;
                vis.add(nid);
                let nd = distFn(q, this.G.get(nid).item.emb);
                if (found.size() < ef || nd < found.top().dist) {
                    cands.push({ dist: nd, id: nid });
                    found.push({ dist: nd, id: nid });
                    if (found.size() > ef) found.pop();
                }
            }
        }
        let res =[];
        while (found.size() > 0) res.push(found.pop());
        return res.reverse();
    }

    insert(item, distFn) {
        let id = item.id;
        let lvl = this.randLevel();
        let nbrs = Array.from({ length: lvl + 1 }, () =>[]);
        this.G.set(id, { item, maxLyr: lvl, nbrs });

        if (this.entryPt === -1) { this.entryPt = id; this.topLayer = lvl; return; }

        let ep = this.entryPt;
        for (let lc = this.topLayer; lc > lvl; lc--) {
            if (lc < this.G.get(ep).nbrs.length) {
                let W = this.searchLayer(item.emb, ep, 1, lc, distFn);
                if (W.length > 0) ep = W[0].id;
            }
        }
        for (let lc = Math.min(this.topLayer, lvl); lc >= 0; lc--) {
            let W = this.searchLayer(item.emb, ep, this.ef_build, lc, distFn);
            let maxM = (lc === 0) ? this.M0 : this.M;
            let sel = W.slice(0, maxM).map(w => w.id);
            this.G.get(id).nbrs[lc] = sel;

            for (let nid of sel) {
                if (!this.G.has(nid)) continue;
                let nNode = this.G.get(nid);
                if (nNode.nbrs.length <= lc) while(nNode.nbrs.length <= lc) nNode.nbrs.push([]);
                nNode.nbrs[lc].push(id);
                if (nNode.nbrs[lc].length > maxM) {
                    let ds = nNode.nbrs[lc].map(c => ({ dist: distFn(nNode.item.emb, this.G.get(c).item.emb), id: c }));
                    ds.sort((a, b) => a.dist - b.dist);
                    nNode.nbrs[lc] = ds.slice(0, maxM).map(x => x.id);
                }
            }
            if (W.length > 0) ep = W[0].id;
        }
        if (lvl > this.topLayer) { this.topLayer = lvl; this.entryPt = id; }
    }

    knn(q, k, ef, distFn) {
        if (this.entryPt === -1) return[];
        let ep = this.entryPt;
        for (let lc = this.topLayer; lc > 0; lc--) {
            if (lc < this.G.get(ep).nbrs.length) {
                let W = this.searchLayer(q, ep, 1, lc, distFn);
                if (W.length > 0) ep = W[0].id;
            }
        }
        let W = this.searchLayer(q, ep, Math.max(ef, k), 0, distFn);
        return W.slice(0, k);
    }

    remove(id) {
        if (!this.G.has(id)) return;
        for (let [nid, nd] of this.G.entries()) {
            nd.nbrs = nd.nbrs.map(layer => layer.filter(x => x !== id));
        }
        if (this.entryPt === id) {
            this.entryPt = -1;
            for (let nid of this.G.keys()) { if (nid !== id) { this.entryPt = nid; break; } }
        }
        this.G.delete(id);
    }
    
    getInfo() {
        let maxL = Math.max(this.topLayer + 1, 1);
        let info = {
            topLayer: this.topLayer, nodeCount: this.G.size,
            nodesPerLayer: new Array(maxL).fill(0),
            edgesPerLayer: new Array(maxL).fill(0),
            nodes: [], edges:[]
        };
        for (let [id, nd] of this.G.entries()) {
            info.nodes.push({ id, metadata: nd.item.metadata, category: nd.item.category, maxLyr: nd.maxLyr });
            for (let lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                info.nodesPerLayer[lc]++;
                if (lc < nd.nbrs.length) {
                    for (let nid of nd.nbrs[lc]) {
                        if (id < nid) {
                            info.edgesPerLayer[lc]++;
                            info.edges.push({ src: id, dst: nid, lyr: lc });
                        }
                    }
                }
            }
        }
        return info;
    }
}

// =====================================================================
//  DATABASES
// =====================================================================
class VectorDB {
    constructor(dims) {
        this.dims = dims;
        this.store = new Map();
        this.bf = new BruteForce();
        this.kdt = new KDTree(dims);
        this.hnsw = new HNSW(16, 200);
        this.nextId = 1;
        this.currentMetric = "cosine"; // Tracks what math the graph was built with
    }

    insert(meta, cat, emb) {
        let v = { id: this.nextId++, metadata: meta, category: cat, emb };
        this.store.set(v.id, v);
        this.bf.insert(v);
        this.kdt.insert(v);
        this.hnsw.insert(v, getDistFn(this.currentMetric));
        return v.id;
    }

    remove(id) {
        if (!this.store.has(id)) return false;
        this.store.delete(id);
        this.bf.remove(id);
        this.hnsw.remove(id);
        this.rebuild(this.currentMetric); // Rebuild with active metric
        return true;
    }

    // FIX #3: Rebuild the graph automatically if the user changes the UI metric
    rebuild(newMetric) {
        console.log(`\n🔄 Rebuilding HNSW & KD-Tree for metric: ${newMetric}...`);
        this.currentMetric = newMetric;
        let dfn = getDistFn(newMetric);
        
        this.kdt = new KDTree(this.dims);
        this.hnsw = new HNSW(16, 200);
        
        let items = Array.from(this.store.values());
        for (let v of items) {
            this.kdt.insert(v);
            this.hnsw.insert(v, dfn);
        }
        console.log(`✅ Rebuild complete!`);
    }

    search(q, k, metric, algo) {
        // Check if graph needs rebuilding based on UI selection
        if (metric !== this.currentMetric) {
            this.rebuild(metric);
        }

        let searchMetric = metric;
        let dfn = getDistFn(searchMetric);

        // FIX #2: Prevent KD-Tree from breaking with Cosine Similarity
        if (algo === "kdtree" && metric === "cosine") {
            // KD-Trees require spatial boundaries. Since our vectors are normalized,
            // Euclidean math ranks items identically to Cosine, but allows the tree to prune correctly!
            searchMetric = "euclidean";
            dfn = getDistFn("euclidean");
        }

        let t0 = performance.now();
        let raw = [];
        
        if (algo === "bruteforce") raw = this.bf.knn(q, k, dfn);
        else if (algo === "kdtree") raw = this.kdt.knn(q, k, dfn);
        else raw = this.hnsw.knn(q, k, 50, dfn);
        let us = Math.round((performance.now() - t0) * 1000);
        
        let hits = raw.filter(r => this.store.has(r.id)).map(r => {
            let v = this.store.get(r.id);
            let finalDist = r.dist;
            
            // If we secretly used Euclidean for the KD-Tree, calculate the true Cosine distance for the UI
            if (algo === "kdtree" && metric === "cosine") {
                finalDist = Distances.cosine(q, v.emb);
            }
            
            return { id: v.id, metadata: v.metadata, category: v.category, distance: finalDist, embedding: v.emb };
        });

        // Re-sort just in case the math bridge slightly shifted identical values
        hits.sort((a, b) => a.distance - b.distance);

        return { hits: hits.slice(0, k), us, algo, metric };
    }

    benchmark(q, k, metric) {
        if (metric !== this.currentMetric) this.rebuild(metric);
        
        let dfn = getDistFn(metric);
        // Apply KD-Tree fix to benchmarks as well
        let kdDfn = metric === "cosine" ? getDistFn("euclidean") : dfn;

        const time = (fn) => {
            let t0 = performance.now();
            fn();
            return Math.round((performance.now() - t0) * 1000);
        };
        return {
            bfUs: time(() => this.bf.knn(q, k, dfn)),
            kdUs: time(() => this.kdt.knn(q, k, kdDfn)),
            hnswUs: time(() => this.hnsw.knn(q, k, 50, dfn)),
            itemCount: this.store.size
        };
    }

    all() { return Array.from(this.store.values()); }
    hnswInfo() { return this.hnsw.getInfo(); }
    size() { return this.store.size; }
}

class DocumentDB {
    constructor() {
        this.store = new Map();
        this.hnsw = new HNSW(16, 200);
        this.bf = new BruteForce();
        this.nextId = 1;
        this.dims = 0;
    }
    insert(title, text, emb) {
        if (this.dims === 0) this.dims = emb.length;
        let item = { id: this.nextId++, title, text, emb };
        this.store.set(item.id, item);
        let vi = { id: item.id, metadata: title, category: "doc", emb };
        this.hnsw.insert(vi, Distances.cosine);
        this.bf.insert(vi);
        return item.id;
    }
    search(q, k, max_dist = 0.7) {
        if (this.store.size === 0) return[];
        let raw = this.store.size < 10 ? this.bf.knn(q, k, Distances.cosine) : this.hnsw.knn(q, k, 50, Distances.cosine);
        return raw.filter(r => r.dist <= max_dist && this.store.has(r.id))
                  .map(r => ({ dist: r.dist, doc: this.store.get(r.id) }));
    }
    remove(id) {
        if (!this.store.has(id)) return false;
        this.store.delete(id); this.hnsw.remove(id); this.bf.remove(id);
        return true;
    }
    all() { return Array.from(this.store.values()); }
    size() { return this.store.size; }
}

// =====================================================================
//  HELPERS & OLLAMA
// =====================================================================
function chunkText(text, chunkWords = 250, overlapWords = 30) {
    let words = text.split(/\s+/).filter(w => w.length > 0);
    if (words.length === 0) return [];
    if (words.length <= chunkWords) return[text];
    
    let chunks =[];
    let step = chunkWords - overlapWords;
    for (let i = 0; i < words.length; i += step) {
        let end = Math.min(i + chunkWords, words.length);
        chunks.push(words.slice(i, end).join(' '));
        if (end === words.length) break;
    }
    return chunks;
}

const Ollama = {
    host: "http://127.0.0.1:11434",
    embedModel: "nomic-embed-text",
    genModel: "llama3.2",
    
    async isAvailable() {
        try {
            let res = await fetch(`${this.host}/api/tags`);
            return res.status === 200;
        } catch { return false; }
    },
    async embed(text) {
        try {
            let res = await fetch(`${this.host}/api/embeddings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: this.embedModel, prompt: text })
            });
            let data = await res.json();
            return data.embedding || [];
        } catch { return[]; }
    },
    async generate(prompt) {
        try {
            let res = await fetch(`${this.host}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: this.genModel, prompt, stream: false })
            });
            let data = await res.json();
            return data.response || "";
        } catch { return "ERROR: Ollama unavailable. Run: ollama serve"; }
    }
};

// =====================================================================
//  DEMO DATA
// =====================================================================
function loadDemo(db) {
    const dataSeed = {
        cs: [
            "Linked List: linear data structure", "Binary Search Tree: logarithmic search",
            "Dynamic Programming: subproblem optimization", "Graph BFS: breadth-first traversal",
            "Hash Table: constant time O(1) lookup", "React: component-based UI library",
            "Node.js: server-side javascript runtime", "Docker: containerization platform",
            "Kubernetes: container orchestration", "Git: distributed version control",
            "SQL: relational database query language", "NoSQL: document-based storage",
            "Encryption: securing data with keys", "Machine Learning: neural network training",
            "Compiler: translating source to machine code", "Operating System: kernel and shell",
            "Virtual Machine: hardware emulation", "TCP/IP: networking protocol suite",
            "Cybersecurity: firewalls and penetration testing", "Cloud Computing: AWS and Azure",
            "API: application programming interface", "Microservices: distributed architecture",
            "Python: high-level interpreted language", "C++: high-performance systems language",
            "Recursion: function calling itself", "Pointer: memory address reference"
        ],
        math: [
            "Calculus: derivatives and integrals", "Linear Algebra: matrices and vectors",
            "Probability: study of random variables", "Number Theory: primes and modular math",
            "Combinatorics: permutations and combinations", "Topology: continuous deformations",
            "Geometry: shapes and spatial properties", "Statistics: data analysis and inference",
            "Fractals: self-similar mathematical patterns", "Logarithms: inverse of exponentiation",
            "Trigonometry: sine cosine and triangles", "Complex Numbers: real and imaginary parts",
            "Differential Equations: rate of change modeling", "Graph Theory: nodes and edges",
            "Set Theory: collections of objects", "Logic: truth tables and propositions",
            "Fibonacci Sequence: golden ratio growth", "Prime Numbers: integers divisible by one",
            "Game Theory: strategic decision making", "Chaos Theory: sensitive initial conditions",
            "Vector Space: linear transformations", "Matrix Multiplication: dot products",
            "Optimization: finding local minima", "Cryptography: RSA and elliptic curves",
            "Fourier Transform: signal processing math", "Boolean Algebra: binary logic gates"
        ],
        food: [
            "Neapolitan Pizza: margherita with basil", "Sushi: sashimi and nigiri rolls",
            "Ramen: tonkotsu pork bone broth", "Tacos: street style carnitas",
            "Croissant: buttery french pastry", "Burger: smash patty with cheese",
            "Pasta: carbonara with guanciale", "Steak: ribeye medium rare",
            "Dim Sum: steamed shrimp dumplings", "Gelato: artisan italian ice cream",
            "Pho: vietnamese noodle soup", "Paella: spanish saffron rice",
            "Falafel: chickpeas and tahini wrap", "Curry: spicy indian butter chicken",
            "Donut: glazed with chocolate sprinkles", "Waffles: belgian with maple syrup",
            "Salad: caesar with parmesan reggiano", "Burrito: beans rice and guac",
            "Pancakes: fluffy buttermilk stack", "Bagel: cream cheese and lox",
            "Espresso: dark roast coffee shot", "Chocolate: dark cocoa bean bar",
            "Baklava: honey and pistachio layers", "Oysters: raw on the half shell",
            "Kimchi: fermented spicy cabbage", "Lobster: butter poached tail"
        ],
        sports: [
            "Basketball: dribbling and slam dunks", "Football: touchdowns and yardage",
            "Tennis: grand slam serves and volleys", "Chess: grandmaster strategy and tactics",
            "Swimming: Olympic freestyle and butterfly", "Golf: fairway drives and putting",
            "Soccer: world cup goals and headers", "Boxing: heavyweight knockouts",
            "Baseball: home runs and strikeouts", "Cricket: wickets and boundaries",
            "F1 Racing: high speed cornering", "Surfing: riding massive ocean waves",
            "Skiing: downhill slalom racing", "UFC: mixed martial arts grappling",
            "Rugby: scrums and tries", "Volleyball: spikes and sets",
            "Badminton: shuttlecock rallies", "Cycling: tour de france sprints",
            "Gymnastics: floor routines and vaults", "Archery: bullseye target arrows",
            "Marathon: long distance endurance", "Skateboarding: kickflips and halfpipes",
            "Climbing: bouldering and belaying", "Fencing: epee and foil parries",
            "Bowling: strikes and spares", "Hockey: power plays and slap shots"
        ]
    };

    Object.entries(dataSeed).forEach(([cat, titles], catIdx) => {
        titles.forEach((title, titleIdx) => {
            // Initialize 16D vector with a low noise baseline
            let emb = Array.from({ length: 16 }, () => Math.random() * 0.1);

            // Create Clusters:
            // CS (dims 0-3), Math (dims 4-7), Food (dims 8-11), Sports (dims 12-15)
            const sectorStart = catIdx * 4;
            for (let i = sectorStart; i < sectorStart + 4; i++) {
                // Give a strong signal to the category's sector
                emb[i] = 0.7 + (Math.random() * 0.3);
            }

            // Add unique "fingerprint" based on the title string
            for (let i = 0; i < title.length; i++) {
                emb[i % 16] += (title.charCodeAt(i) % 10) / 100;
            }

            // Normalize slightly
            const magnitude = Math.sqrt(emb.reduce((s, v) => s + v * v, 0));
            emb = emb.map(v => v / magnitude);

            db.insert(title, cat, emb);
        });
    });

    console.log(`✅ Database Pre-loaded with ${db.size()} items across 4 categories.`);
}
// =====================================================================
//  HTTP SERVER (Express)
// =====================================================================
const app = express();
app.use(cors());
app.use(express.json());

// For processing memory multipart/form-data uploads
const upload = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 50 * 1024 * 1024 } // 50 MB limit
});

const db = new VectorDB(DIMS);
const docDB = new DocumentDB();
loadDemo(db);

// ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

app.get('/search', (req, res) => {
    let q = req.query.v ? req.query.v.split(',').map(Number) :[];
    if (q.length !== DIMS) return res.status(400).json({ error: `need ${DIMS}D vector` });
    
    let k = parseInt(req.query.k) || 5;
    let metric = req.query.metric || "cosine";
    let algo = req.query.algo || "hnsw";
    
    let out = db.search(q, k, metric, algo);
    res.json({ results: out.hits, latencyUs: out.us, algo: out.algo, metric: out.metric });
});

app.post('/insert', (req, res) => {
    let { metadata, category, embedding } = req.body;
    if (!metadata || !embedding || embedding.length !== DIMS) {
        return res.status(400).json({ error: "invalid body" });
    }
    let id = db.insert(metadata, category || "", embedding);
    res.json({ id });
});

app.delete('/delete/:id', (req, res) => {
    let id = parseInt(req.params.id);
    let ok = db.remove(id);
    res.json({ ok });
});

app.get('/items', (req, res) => res.json(db.all()));

app.get('/benchmark', (req, res) => {
    let q = req.query.v ? req.query.v.split(',').map(Number) :[];
    if (q.length !== DIMS) return res.status(400).json({ error: `need ${DIMS}D vector` });
    let k = parseInt(req.query.k) || 5;
    let metric = req.query.metric || "cosine";
    res.json(db.benchmark(q, k, metric));
});

app.get('/hnsw-info', (req, res) => res.json(db.hnswInfo()));

app.get('/stats', (req, res) => {
    res.json({
        count: db.size(),
        dims: DIMS,
        algorithms:["bruteforce", "kdtree", "hnsw"],
        metrics: ["euclidean", "cosine", "manhattan"]
    });
});

// ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

// POST /doc/insert (Plain text)
app.post('/doc/insert', async (req, res) => {
    let { title, text } = req.body;
    
    if (!title || !text) return res.status(400).json({ error: "need title and text" });

    let chunks = chunkText(text, 250, 30);
    let ids =[];
    
    for (let i = 0; i < chunks.length; i++) {
        let emb = await Ollama.embed(chunks[i]);
        if (emb.length === 0) return res.status(500).json({ error: "Ollama unavailable" });
        let cTitle = chunks.length > 1 ? `${title} [${i+1}/${chunks.length}]` : title;
        ids.push(docDB.insert(cTitle, chunks[i], emb));
    }
    res.json({ ids, chunks: chunks.length, dims: docDB.dims });
});
// POST /doc/upload (PDFs & Documents!)
app.post('/doc/upload', upload.single('file'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: "No file uploaded. Use field name 'file'." });

    let filename = req.file.originalname;
    let text = "";

    try {
        console.log(`\n[1/3] Receiving file: ${filename} (${(req.file.size / 1024 / 1024).toFixed(2)} MB)`);
        
        if (filename.toLowerCase().endsWith('.pdf')) {
            if (typeof pdfParse !== 'function') {
                return res.status(500).json({ error: "PDF library failed to load correctly." });
            }
            console.log(`[2/3] Extracting text from PDF...`);
            let pdfData = await pdfParse(req.file.buffer);
            text = pdfData.text;
        } else if (/\.(txt|md|csv)$/i.test(filename)) {
            text = req.file.buffer.toString('utf-8');
        } else {
            return res.status(400).json({ error: "Unsupported format. Use .pdf, .txt, .md, .csv" });
        }

        // Safety check: Did we actually find any text?
        if (!text || text.trim() === '') {
            console.log(`❌ Error: No readable text found. Is this a scanned PDF?`);
            return res.status(400).json({ error: "No readable text found in document. If this is a scanned PDF, it cannot be read without OCR." });
        }

        let chunks = chunkText(text, 250, 30);
        let ids = [];

        console.log(`[3/3] Text extracted! Creating ${chunks.length} embeddings... (This might take a minute depending on your PC speed)`);

        for (let i = 0; i < chunks.length; i++) {
            console.log(`      -> Processing chunk ${i + 1} of ${chunks.length}...`);
            let emb = await Ollama.embed(chunks[i]);
            
            if (emb.length === 0) {
                console.log(`❌ Error: Ollama failed on chunk ${i+1}`);
                return res.status(500).json({ error: `Ollama unavailable on chunk ${i+1}. Ensure it is running.` });
            }
            
            let chunkTitle = chunks.length > 1 ? `${filename}[${i+1}/${chunks.length}]` : filename;
            ids.push(docDB.insert(chunkTitle, chunks[i], emb));
        }

        console.log(`✅ Successfully processed and stored ${filename}!`);
        res.json({ ids, chunks: chunks.length, dims: docDB.dims, filename });

    } catch (error) {
        console.error("❌ Upload error:", error);
        res.status(500).json({ error: "Failed to process the document. " + error.message });
    }
});

app.delete('/doc/delete/:id', (req, res) => {
    let id = parseInt(req.params.id);
    let ok = docDB.remove(id);
    res.json({ ok });
});

app.get('/doc/list', (req, res) => {
    res.json(docDB.all().map(d => ({
        id: d.id, title: d.title, 
        preview: d.text.substring(0, 120) + (d.text.length > 120 ? "…" : ""),
        words: d.text.split(/\s+/).length
    })));
});

app.post('/doc/search', async (req, res) => {
    let { question, k = 3 } = req.body;
    if (!question) return res.status(400).json({ error: "need question" });

    let qEmb = await Ollama.embed(question);
    if (qEmb.length === 0) return res.status(500).json({ error: "Ollama unavailable" });

    let hits = docDB.search(qEmb, k);
    res.json({
        contexts: hits.map(h => ({ id: h.doc.id, title: h.doc.title, distance: h.dist }))
    });
});

app.post('/doc/ask', async (req, res) => {
    let { question, k = 3 } = req.body;
    if (!question) return res.status(400).json({ error: "need question" });

    let qEmb = await Ollama.embed(question);
    if (qEmb.length === 0) return res.status(500).json({ error: "Ollama unavailable" });

    let hits = docDB.search(qEmb, k);
    let ctx = hits.map((h, i) => `[${i+1}] ${h.doc.title}:\n${h.doc.text}\n\n`).join('');
    
    let prompt = `You are a helpful assistant. Answer the user's question directly. Use the provided context if it contains relevant information. If it doesn't, just use your own general knowledge. IMPORTANT: Do NOT mention the 'context' or say things like 'the context doesn't mention'. Just answer naturally.\n\nContext:\n${ctx}Question: ${question}\n\nAnswer:`;

    let answer = await Ollama.generate(prompt);
    
    res.json({
        answer, model: Ollama.genModel,
        contexts: hits.map(h => ({ id: h.doc.id, title: h.doc.title, text: h.doc.text, distance: h.dist })),
        docCount: docDB.size()
    });
});

app.get('/status', async (req, res) => {
    let up = await Ollama.isAvailable();
    res.json({
        ollamaAvailable: up, embedModel: Ollama.embedModel, genModel: Ollama.genModel,
        docCount: docDB.size(), docDims: docDB.dims, demoDims: DIMS, demoCount: db.size()
    });
});

// Serve frontend
app.use(express.static(path.join(__dirname, '/')));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});
app.post('/api/manual-insert', async (req, res) => {
    const { title, category } = req.body;
    try {
        let emb = await Ollama.embed(title);
        
        if (emb.length > 0 && emb.length !== DIMS) {
            // AI is online! Compress massive 1024D vector into 16D for our graph
            const compressed = new Array(DIMS).fill(0);
            const ratio = Math.floor(emb.length / DIMS);
            for(let i=0; i<DIMS; i++) {
                // Sum blocks of dimensions to preserve semantic meaning
                for(let j=0; j<ratio; j++) compressed[i] += Math.abs(emb[(i * ratio) + j]);
            }
            emb = compressed;
        } else if (emb.length === 0) {
            // AI is offline: Fallback to character hash
            emb = new Array(DIMS).fill(0.1);
            for(let i=0; i<title.length; i++) emb[i % DIMS] += (title.charCodeAt(i) % 50) / 100;
            for(let i=0; i<4; i++) emb[i] += 0.5; 
        }

        const id = db.insert(title, category || "manual", emb);
        res.json({ success: true, id });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});
// Start Server
app.listen(8080, async () => {
    console.log("=== Node.js VectorDB Engine ===");
    console.log("http://localhost:8080");
    console.log(`${db.size()} demo vectors | ${DIMS} dims | HNSW + KD-Tree + BruteForce`);
    
    let up = await Ollama.isAvailable();
    console.log(`Ollama: ${up ? "ONLINE" : "OFFLINE (install from ollama.com)"}`);
    if (up) console.log(`  embed model: ${Ollama.embedModel}  |  gen model: ${Ollama.genModel}`);
});
