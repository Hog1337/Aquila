#!/usr/bin/env python3
"""Falcon Dataset Cleaning Tool — exclude individual bad frames.
Uses V2 model (ReID-trained) for accurate similarity."""
import os, sys, csv, json, io, base64, time
from pathlib import Path
from PIL import Image, ImageDraw
from collections import defaultdict
import numpy as np
import torch

from flask import Flask, request, jsonify, render_template_string

BASE = Path("/home/limon/data/university/lct")
SPLITS = BASE / "reid" / "splits"
IMGS = BASE / "data" / "reid" / "images"
RESULTS = BASE / "reid" / "cleaning_results.csv"
CACHE = BASE / "reid" / "cleaning_emb_cache.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
tr = list(csv.DictReader(open(SPLITS / "train_sub.csv")))
val_q = list(csv.DictReader(open(SPLITS / "val_query.csv")))
val_g = list(csv.DictReader(open(SPLITS / "val_gallery.csv")))
all_rows = tr + val_q + val_g
for r in tr: r['_src'] = 'train'
for r in val_q: r['_src'] = 'val_q'
for r in val_g: r['_src'] = 'val_g'

by_veh = defaultdict(list)
for r in all_rows:
    by_veh[r['vehicle_id']].append(r)

all_veh = sorted(by_veh.keys(), key=int)
print(f"Loaded {len(all_rows)} rows, {len(all_veh)} vehicles", flush=True)

# Load existing excluded images
excluded = set()
if RESULTS.exists():
    with open(RESULTS) as f:
        for row in csv.DictReader(f):
            excluded.add((row['vehicle_id'], row['image_id'], row.get('src', '')))
print(f"Already excluded: {len(excluded)} frames", flush=True)

# ---- V2 Model ----
print("Loading V2 model for similarity...", flush=True)
sys.path.insert(0, str(BASE))
from reid.embed_v2 import V2Embedder
ckpt_path = str(BASE / "reid" / "baseline" / "JDNFV_SUPCON_PROTO_R32_best_ema_fp16.pt")
embedder = V2Embedder(ckpt_path, device=DEVICE)

# Embedding cache
emb_cache = {}  # image_id -> (embedding, timestamp)

def crop_img(r):
    x, y, w, h = int(r['x']), int(r['y']), int(r['w']), int(r['h'])
    return Image.open(IMGS / f"{r['image_id']}.jpg").convert("RGB").crop((x, y, x + w, y + h))

def get_embs(rows):
    """Get V2 embeddings for a list of rows. Returns list of (row, emb) tuples."""
    needed = []
    for r in rows:
        if r['image_id'] not in emb_cache:
            needed.append(r)
    
    if needed:
        imgs = [crop_img(r) for r in needed]
        embs_list = embedder.embed(imgs)
        for r, e in zip(needed, embs_list):
            emb_cache[r['image_id']] = (e, time.time())
    
    result = []
    for r in rows:
        if r['image_id'] in emb_cache:
            result.append((r, emb_cache[r['image_id']][0]))
    return result

def compute_similarities(rows):
    """Compute pairwise cosine similarities for a list of rows using V2 embeddings."""
    data = get_embs(rows)
    if len(data) < 2:
        return [r for r, _ in data], [], []
    
    embs = np.array([e for _, e in data])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim = embs @ embs.T
    
    scores = []
    for i in range(len(data)):
        others = np.concatenate([sim[i, :i], sim[i, i+1:]])
        is_out = float(others.mean()) < float(np.median(
            [sim[j].mean() for j in range(len(data)) if j != i])) - 0.05
        scores.append({
            'mean_sim': float(others.mean()),
            'min_sim': float(others.min()),
            'is_outlier': is_out,
        })
    
    return [r for r, _ in data], scores, sim.tolist()

# ---- Flask App ----
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Falcon — чистка кадров (V2 similarity)</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 15px; background: #0f0f1a; color: #e0e0e0; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px; }
  .header h1 { margin: 0; font-size: 20px; }
  .stats { font-size: 12px; color: #888; }
  
  .nav-bar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
  .nav-bar input[type=text] { 
    background: #1a1a2e; color: #fff; border: 1px solid #555; border-radius: 6px; 
    padding: 6px 10px; font-size: 13px; width: 100px; 
  }
  .nav-bar button { 
    background: #2a2a4e; color: #ddd; border: none; border-radius: 6px; 
    padding: 6px 14px; font-size: 13px; cursor: pointer; 
  }
  .nav-bar button:hover { background: #3a3a5e; }
  
  .veh-title { font-size: 22px; font-weight: bold; color: #ffd700; margin: 8px 0; }
  
  .grid { 
    display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
    gap: 14px; margin: 12px 0;
  }
  .card {
    border: 2px solid #333; border-radius: 10px; overflow: hidden;
    background: #16162a; padding: 8px; position: relative; cursor: pointer;
    transition: all 0.15s;
  }
  .card:hover { border-color: #666; }
  .card.excluded { opacity: 0.4; border-color: #661111; background: #111; }
  .card.excluded::after {
    content: '✕'; position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    font-size: 80px; color: #ff2222; opacity: 0.7; pointer-events: none;
  }
  .card.outlier { border-color: #ff6b35; }
  .card img { width: 100%; height: auto; display: block; border-radius: 4px; }
  .card .label { font-size: 12px; color: #888; text-align: center; margin-top: 3px; }
  .card .sim { font-size: 11px; text-align: center; padding: 2px 6px; border-radius: 4px; margin-top: 2px; }
  .sim-high { background: #1a3a1a; color: #4caf50; }
  .sim-mid { background: #3a3a1a; color: #ffeb3b; }
  .sim-low { background: #3a1a1a; color: #f44336; }
  
  .footer-bar {
    position: sticky; bottom: 0; padding: 12px; margin-top: 15px;
    background: #0f0f1a; display: flex; gap: 12px; align-items: center; justify-content: center;
    border-top: 1px solid #333;
  }
  .btn-confirm {
    padding: 12px 36px; font-size: 16px; border: none; border-radius: 8px;
    cursor: pointer; font-weight: bold; background: #4caf50; color: white;
  }
  .btn-confirm:hover { transform: scale(1.03); }
  .btn-confirm:disabled { background: #333; color: #666; cursor: not-allowed; transform: none; }
  .btn-nav { background: #2a2a4e; color: #ccc; border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer; }
  .btn-nav:hover { background: #3a3a5e; }
  .btn-reset { background: #3a1a1a; color: #f44336; border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer; }
  
  .sim-matrix { margin: 10px 0; text-align: center; font-size: 10px; overflow-x: auto; }
  .sim-matrix table { margin: 0 auto; border-collapse: collapse; }
  .sim-matrix td, .sim-matrix th { padding: 2px 6px; border: 1px solid #333; text-align: center; }
  .sim-matrix td.high { background: #1a4a1a; color: #4caf50; }
  .sim-matrix td.mid { background: #3a3a1a; color: #ffeb3b; }
  .sim-matrix td.low { background: #4a1a1a; color: #f44336; }
  
  .key-hint { text-align: center; color: #444; font-size: 11px; margin-top: 5px; }
</style>
</head>
<body>
  <div class="header">
    <h1>🧹 Чистка кадров Falcon <span style="font-size:12px;color:#666;">(V2 similarity)</span></h1>
    <div class="stats">
      Исключено: <b id="excluded-count">0</b> | Обработано: <b id="done-count">0</b> | Осталось ID: <b id="remain-count">0</b>
    </div>
  </div>
  
  <div class="nav-bar">
    <span style="font-size:13px;">ID:</span>
    <input type="text" id="veh-input" onkeydown="if(event.key==='Enter')goTo()">
    <button onclick="goTo()">Перейти</button>
    <button class="btn-nav" onclick="nav(-1)">◀</button>
    <span id="nav-pos" style="color:#888;font-size:12px;"></span>
    <button class="btn-nav" onclick="nav(1)">▶</button>
    <button class="btn-reset" onclick="resetAll()" style="margin-left:auto;">Сбросить всё</button>
  </div>
  
  <div id="main"><div style="text-align:center;color:#888;padding:40px;">Загрузка...</div></div>
  
  <div class="key-hint">
    Клик на фото — исключить/вернуть | <b>1-9</b> — toggle фото по номеру | <b>Enter</b> — подтвердить | <b>← →</b> — навигация
  </div>

<script>
let curVeh = null;
let vehList = [];
let curIdx = 0;
let excluded = {};
let hasChanges = false;

function stats() {
  fetch('/stats').then(r=>r.json()).then(d => {
    document.getElementById('excluded-count').textContent = d.excluded;
    document.getElementById('done-count').textContent = d.done;
    document.getElementById('remain-count').textContent = d.remain;
  });
}

function loadVeh(vid) {
  document.getElementById('main').innerHTML = '<div style="text-align:center;color:#888;padding:40px;">⏳ Загрузка эмбеддингов...</div>';
  fetch('/vehicle?vid=' + encodeURIComponent(vid)).then(r=>r.json()).then(d => {
    if (d.error) { document.getElementById('main').innerHTML = `<div style="text-align:center;color:#888;padding:40px;">${d.error}</div>`; return; }
    render(d);
    stats();
  });
}

function render(d) {
  curVeh = d.vehicle_id;
  document.getElementById('veh-input').value = d.vehicle_id;
  document.getElementById('nav-pos').textContent = `${curIdx+1}/${vehList.length}`;
  hasChanges = false;
  
  fetch('/excluded?vid=' + encodeURIComponent(d.vehicle_id)).then(r=>r.json()).then(ex => {
    excluded = {};
    for (const k of ex.keys) excluded[k] = true;
    
    let cards = d.images.map((img, i) => {
      const key = `${img.image_id}|${img.src}`;
      const isEx = excluded[key] ? 'excluded' : '';
      const isOut = img.is_outlier ? 'outlier' : '';
      let simClass = '', simText = '';
      if (img.mean_sim !== undefined) {
        simClass = img.mean_sim > 0.85 ? 'sim-high' : img.mean_sim > 0.7 ? 'sim-mid' : 'sim-low';
        simText = `<div class="sim ${simClass}">sim: ${img.mean_sim.toFixed(5)}</div>`;
      }
      return `<div class="card ${isEx} ${isOut}" onclick="toggle(${i})" id="card-${i}">
        <img src="data:image/jpeg;base64,${img.data}" alt="">
        <div class="label">${img.label}</div>
        ${simText}
      </div>`;
    }).join('');
    
    let matrix = '';
    if (d.sim_matrix && d.sim_matrix.length > 1) {
      matrix = '<div class="sim-matrix"><table><tr><th></th>';
      for (let j = 0; j < d.sim_matrix.length; j++) matrix += `<th>${j}</th>`;
      matrix += '</tr>';
      for (let i = 0; i < d.sim_matrix.length; i++) {
        matrix += `<tr><th>${i}</th>`;
        for (let j = 0; j < d.sim_matrix[i].length; j++) {
          let v = d.sim_matrix[i][j];
          matrix += `<td class="${v > 0.85 ? 'high' : v > 0.7 ? 'mid' : 'low'}">${v.toFixed(4)}</td>`;
        }
        matrix += '</tr>';
      }
      matrix += '</table></div>';
    }
    
    document.getElementById('main').innerHTML = `
      <div class="veh-title">🚗 ID ${d.vehicle_id} <span style="font-size:13px;color:#888;">(${d.images.length} фото)</span></div>
      <div class="grid">${cards}</div>
      ${matrix}
      <div class="footer-bar">
        <span id="change-indicator" style="font-size:12px;color:#888;"></span>
        <button class="btn-confirm" id="btn-confirm" onclick="saveExclusions()">✅ Подтвердить</button>
      </div>
    `;
  });
}

function toggle(idx) {
  const card = document.getElementById(`card-${idx}`);
  if (!card) return;
  card.classList.toggle('excluded');
  hasChanges = true;
  document.getElementById('change-indicator').textContent = '✏️ Есть изменения';
}

function saveExclusions() {
  const cards = document.querySelectorAll('.card');
  const keys = [];
  const actions = [];
  cards.forEach((card, idx) => {
    const isEx = card.classList.contains('excluded');
    const label = card.querySelector('.label').textContent;
    const parts = label.split(' ');
    const imageId = parts[0];
    const src = parts[2] ? parts[2].replace(/[()]/g,'') : 'train';
    keys.push(`${imageId}|${src}`);
    actions.push(isEx ? 'exclude' : 'include');
  });
  
  fetch('/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({vehicle_id: curVeh, keys: keys, actions: actions})
  }).then(r=>r.json()).then(d => {
    if (d.ok) {
      document.getElementById('change-indicator').textContent = '✅ Сохранено';
      hasChanges = false;
      stats();
      setTimeout(() => nav(1), 400);
    }
  });
}

function nav(delta) {
  if (hasChanges && !confirm('Есть несохранённые изменения. Перейти без сохранения?')) return;
  curIdx = Math.max(0, Math.min(vehList.length-1, curIdx + delta));
  loadVeh(vehList[curIdx]);
}

function goTo() {
  const vid = document.getElementById('veh-input').value.trim();
  if (!vid) return;
  const idx = vehList.indexOf(vid);
  if (idx >= 0) { curIdx = idx; loadVeh(vid); }
  else loadVeh(vid);
}

function resetAll() {
  if (!confirm('Сбросить ВСЕ исключения?')) return;
  fetch('/reset', {method:'POST'}).then(r=>r.json()).then(d => {
    if (d.ok) { loadVeh(curVeh); stats(); }
  });
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && document.activeElement?.tagName !== 'INPUT') {
    e.preventDefault(); saveExclusions();
  }
  if (e.key >= '1' && e.key <= '9') { toggle(parseInt(e.key)-1); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); nav(-1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); nav(1); }
});

fetch('/list').then(r=>r.json()).then(d => { vehList = d.vehicles; });
stats();
setTimeout(() => { if (vehList.length > 0) loadVeh(vehList[0]); }, 500);
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/stats')
def stats():
    exc = len(excluded)
    done_veh = len(set(v for v, _, _ in excluded))
    remain = max(0, len(all_veh) - done_veh)
    return jsonify(excluded=exc, done=done_veh, remain=remain)

@app.route('/list')
def list_():
    return jsonify(vehicles=all_veh)

@app.route('/vehicle')
def vehicle():
    vid = request.args.get('vid', '')
    if vid not in by_veh:
        return jsonify(error=f"ID {vid} не найден")
    
    rows = by_veh[vid]
    t0 = time.time()
    valid_rows, scores, sim_matrix = compute_similarities(rows)
    t = time.time() - t0
    
    images = []
    for i, r in enumerate(valid_rows):
        x, y, w, h = int(r['x']), int(r['y']), int(r['w']), int(r['h'])
        img_path = IMGS / f"{r['image_id']}.jpg"
        if not img_path.exists():
            continue
        
        pil = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(pil)
        draw.rectangle([x, y, x + w, y + h], outline='#00ff00', width=3)
        
        # Resize so longest side >= 1000px
        w_img, h_img = pil.size
        if max(w_img, h_img) < 1000:
            scale = 1000.0 / max(w_img, h_img)
            pil = pil.resize((int(w_img * scale), int(h_img * scale)), Image.LANCZOS)
        
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        img_data = {
            'data': b64,
            'label': f"{r['image_id']} cam:{r.get('camera_id','?')} ({r.get('_src','train')})",
            'image_id': r['image_id'],
            'src': r.get('_src', 'train'),
        }
        if scores:
            img_data.update({
                'mean_sim': scores[i]['mean_sim'],
                'min_sim': scores[i]['min_sim'],
                'is_outlier': scores[i]['is_outlier'],
            })
        images.append(img_data)
    
    return jsonify(
        vehicle_id=vid,
        images=images,
        sim_matrix=sim_matrix if sim_matrix else [],
        compute_time=t,
    )

@app.route('/excluded')
def get_excluded():
    vid = request.args.get('vid', '')
    keys = [f"{i}|{s}" for v, i, s in excluded if v == vid]
    return jsonify(keys=keys)

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    vid = data['vehicle_id']
    keys = data['keys']
    actions = data['actions']
    for k, a in zip(keys, actions):
        parts = k.split('|')
        if len(parts) != 2: continue
        img_id, src = parts
        key = (vid, img_id, src)
        if a == 'exclude':
            excluded.add(key)
        else:
            excluded.discard(key)
    with open(RESULTS, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['vehicle_id', 'image_id', 'src'])
        for v, i, s in sorted(excluded):
            w.writerow([v, i, s])
    return jsonify(ok=True)

@app.route('/reset', methods=['POST'])
def reset():
    excluded.clear()
    if RESULTS.exists():
        RESULTS.unlink()
    return jsonify(ok=True)

@app.route('/clear_cache')
def clear_cache():
    emb_cache.clear()
    return jsonify(ok=True, size=len(emb_cache))

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"🚀 Server on http://127.0.0.1:{port}", flush=True)
    app.run(host='127.0.0.1', port=port, debug=False)
