#!/usr/bin/env python3
"""Local web UI for ptpg_with_interior_hole.py.

    python packer_ui.py            # then open http://localhost:8765
    python packer_ui.py --port 9000

Needs only the solver file next to it (plus shapely, which the solver needs).
Enter the plot, the rooms, the adjacencies and the interior rooms, press Run.
"""
import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import ptpg_with_interior_hole as rp
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


# --------------------------------------------------------------------------
# validation + solving
# --------------------------------------------------------------------------

def validate(poly, rects, adj, interior):
    """Return (errors, warnings, info). Mirrors the checks the solver makes
    so the UI can say what is wrong BEFORE the search starts."""
    errs, warns, info = [], [], {}
    if len(poly) < 4:
        errs.append("The plot needs at least 4 corners.")
        return errs, warns, info
    P = Polygon(poly)
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if abs(x1 - x2) > 1e-9 and abs(y1 - y2) > 1e-9:
            errs.append(f"Edge {i + 1} ({x1:g},{y1:g})→({x2:g},{y2:g}) is slanted. "
                        "Plot edges must be horizontal or vertical.")
            break
    if not P.is_valid:
        errs.append("The plot outline crosses itself. List the corners in order "
                    "around the outline (like walking round it), not in zig-zag order.")
    elif P.area <= 1e-9:
        errs.append("The plot has zero area.")
    else:
        info["plot_area"] = P.area
    if not rects:
        errs.append("Add at least one room.")
    for i, (w, h) in rects.items():
        if w <= 0 or h <= 0:
            errs.append(f"Room {i} needs positive width and height.")
    for (a, b) in adj:
        if a == b:
            errs.append(f"Room {a} cannot be adjacent to itself.")
    if rects:
        total = sum(w * h for w, h in rects.values())
        info["rooms_area"] = total
        if "plot_area" in info:
            if total > info["plot_area"] + 1e-9:
                errs.append(f"Rooms need {total:g} area but the plot only has "
                            f"{info['plot_area']:g}.")
            elif total < info["plot_area"] - 1e-9:
                info["hole_area"] = info["plot_area"] - total
        if len(rects) > 1:
            ok, msg = rp.check_no_isolated(rects, set(adj), set(interior))
            if not ok:
                errs.append(msg)
        if interior and len(interior) == len(rects):
            errs.append("At least one room must touch the plot boundary.")
    return errs, warns, info


def solve(data):
    poly = [tuple(map(float, p)) for p in data["polygon"]]
    rects = {int(k): (float(v[0]), float(v[1])) for k, v in data["rects"].items()}
    adj = {tuple(sorted((int(a), int(b)))) for a, b in data["adjacency"]}
    interior = {int(i) for i in data.get("interior", [])}
    tl = float(data.get("time_limit", 30))
    kw = {k: bool(v) for k, v in (data.get("options") or {}).items()}

    errs, warns, info = validate(poly, rects, adj, interior)
    if errs:
        return {"status": "invalid", "errors": errs}

    t0 = time.time()
    found, sol, nodes = rp.pack_rectangles_frontier(
        poly, rects, adj, interior_ids=interior, time_limit=tl, **kw)
    dt = time.time() - t0
    out = {"status": {True: "solved", False: "none", None: "timeout"}[found],
           "nodes": nodes, "seconds": round(dt, 2), "info": info}
    if found:
        out["placements"] = {str(i): list(map(float, p)) for i, p in sol.items()}
        ov, outside, holes = rp.verify_placements(poly, sol)
        out["verify"] = {"overlaps": ov, "outside": outside, "holes": holes}
        left = Polygon(poly).difference(unary_union(
            [box(x, y, x + w, y + h) for x, y, w, h in sol.values()]))
        geoms = list(left.geoms) if left.geom_type == "MultiPolygon" else [left]
        out["hole"] = [list(g.exterior.coords) for g in geoms if g.area > 1e-9]
    return out


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html")
        elif self.path == "/examples":
            ex = {}
            for k, e in rp.EXAMPLES.items():
                ex[str(k)] = {
                    "polygon": [list(p) for p in e["polygon"]],
                    "rects": {str(i): list(v) for i, v in e["rects"].items()},
                    "adjacency": [list(a) for a in (e.get("required_adjacency") or [])],
                    "interior": list(e.get("interior", ())),
                }
            self._send(200, json.dumps(ex))
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n))
            if self.path == "/validate":
                poly = [tuple(map(float, p)) for p in data["polygon"]]
                rects = {int(k): (float(v[0]), float(v[1])) for k, v in data["rects"].items()}
                adj = {tuple(sorted((int(a), int(b)))) for a, b in data["adjacency"]}
                e, w, i = validate(poly, rects, adj, {int(x) for x in data.get("interior", [])})
                self._send(200, json.dumps({"errors": e, "warnings": w, "info": i}))
            elif self.path == "/solve":
                self._send(200, json.dumps(solve(data)))
            else:
                self._send(404, "{}")
        except Exception as ex:  # surface any problem in the page
            self._send(200, json.dumps({"status": "invalid", "errors": [f"{type(ex).__name__}: {ex}"]}))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Room packer</title>
<style>
:root{--bg:#f6f5f1;--card:#fff;--ink:#1d2327;--mut:#6b7279;--line:#dedbd2;--acc:#1f6f6b;--acc2:#e9f3f2;--bad:#b3261e;--badbg:#fbeceb;--ok:#1b6e3a;--okbg:#e7f4ec;--warn:#8a5a00;--warnbg:#fdf3dc;--int:#f4c06a;--room:#cfe3f4}
@media (prefers-color-scheme:dark){:root{--bg:#171a1c;--card:#1f2326;--ink:#e8e6e1;--mut:#9aa1a6;--line:#343a3e;--acc:#5fb8b2;--acc2:#1d3231;--bad:#f2a29c;--badbg:#3a2220;--ok:#8fd3a8;--okbg:#1c3025;--warn:#e8c06a;--warnbg:#352c14;--int:#8a6a22;--room:#2c4a63}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:18px 24px 6px}h1{margin:0;font-size:20px}header p{margin:2px 0 0;color:var(--mut)}
main{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:18px;padding:14px 24px 40px;align-items:start}
@media(max-width:980px){main{grid-template-columns:1fr}}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:14px}
h2{margin:0 0 8px;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);display:flex;justify-content:space-between;align-items:center}
label{display:inline-flex;align-items:center;gap:6px}
input[type=number],input[type=text],select,textarea{background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:5px 7px;font:inherit;width:70px}
textarea{width:100%;font-family:ui-monospace,monospace;font-size:13px}
input[type=text]{width:100%}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px;border-radius:8px;cursor:pointer}
button:hover{border-color:var(--acc)}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:600;padding:9px 20px}
button.primary:disabled{opacity:.5;cursor:wait}
button.ghost{border:none;color:var(--acc);padding:2px 6px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:6px 0}
.tabs{display:flex;gap:4px;margin-bottom:8px}.tabs button{border-radius:8px 8px 0 0;border-bottom:none}.tabs button.on{background:var(--acc2);border-color:var(--acc);color:var(--acc);font-weight:600}
table{border-collapse:collapse;width:100%}th{font-size:12px;color:var(--mut);font-weight:600;text-align:left;padding:2px 4px}td{padding:2px 4px}
td input[type=number]{width:62px}
.x{color:var(--mut);cursor:pointer;border:none;background:none;padding:0 6px}
.matrix{overflow:auto;max-height:300px}.matrix table{width:auto}.matrix td,.matrix th{padding:1px 2px;text-align:center;font-size:12px}
.matrix td.dead{background:var(--bg)}.matrix input{margin:0;width:15px;height:15px}
.msg{padding:8px 10px;border-radius:8px;margin:6px 0;font-size:14px}
.bad{background:var(--badbg);color:var(--bad)}.ok{background:var(--okbg);color:var(--ok)}.warn{background:var(--warnbg);color:var(--warn)}
.mut{color:var(--mut);font-size:13px}.stats{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0 10px}.stats b{font-size:18px;display:block}
svg{width:100%;height:auto;background:var(--bg);border-radius:8px;border:1px solid var(--line)}
details summary{cursor:pointer;color:var(--mut);font-size:13px}.opts label{display:flex;margin:4px 0;font-size:13px}
.sticky{position:sticky;top:12px}
</style></head><body>
<header><h1>Room packer</h1><p>Enter a plot, rooms, and which rooms must touch. Press Run.</p></header>
<main>
<div>
 <section><h2>1 · Plot <span class="mut" id="plotArea"></span></h2>
  <div class="tabs"><button id="tabL" class="on">L-shape</button><button id="tabC">Custom corners</button></div>
  <div id="lbox">
   <div class="row"><label>Width <input type="number" id="W" value="10" min="1" step="any"></label>
   <label>Height <input type="number" id="H" value="10" min="1" step="any"></label></div>
   <div class="row"><label>Notch width <input type="number" id="nw" value="0" min="0" step="any"></label>
   <label>Notch height <input type="number" id="nh" value="0" min="0" step="any"></label>
   <label>Notch at <select id="nc" style="width:auto"><option value="tr">top-right</option><option value="tl">top-left</option><option value="br">bottom-right</option><option value="bl">bottom-left</option></select></label></div>
   <div class="mut">Notch 0 gives a plain rectangle.</div>
  </div>
  <div id="cbox" style="display:none">
   <textarea id="corners" rows="4" placeholder="one corner per line:&#10;0,0&#10;10,0&#10;10,10&#10;0,10"></textarea>
   <div class="mut">Go around the outline in order (either direction). Edges must be horizontal or vertical.</div>
  </div>
  <svg id="plotsvg" viewBox="0 0 10 10" style="margin-top:8px;max-height:170px"></svg>
 </section>

 <section><h2>2 · Rooms <span><span class="mut" id="roomArea"></span> <button class="ghost" id="addRoom">+ add room</button></span></h2>
  <div style="overflow-x:auto"><table id="rooms"><thead></thead><tbody></tbody></table></div>
  <div class="row"><b>Add a room:</b> <label>width <input type="number" id="qw" value="3" min="0" step="any" style="width:60px"></label> <label>height <input type="number" id="qh" value="3" min="0" step="any" style="width:60px"></label> <label>copies <input type="number" id="qn" value="1" min="1" style="width:50px"></label> <button id="qadd">Add</button></div>
  <div class="mut">Tick a box if the room in that row and the room in that column must share a wall (ticks are mirrored). Unticked pairs must NOT touch. Rooms may be rotated. "Interior" = touches the plot boundary nowhere.</div>
 </section>

 <section><h2>Run</h2>
  <div class="row"><label>Time limit <input type="number" id="tl" value="30" min="1"> s</label>
   <select id="exSel" style="width:auto"><option value="">Load example…</option></select>
   <button id="save">Save</button><button id="load">Load</button><input type="file" id="file" accept=".json" style="display:none"></div>
  <div id="problems"></div>
  <div class="row"><button class="primary" id="run">Run</button><span class="mut" id="runStat"></span></div>
 </section>
</div>
<div class="sticky">
 <section><h2>Result</h2><div id="result" class="mut">Nothing run yet. The layout will appear here.</div></section>
</div>
</main>
<script>
const $=id=>document.getElementById(id);
let rooms=[{w:5,h:5,i:false},{w:5,h:5,i:false},{w:5,h:5,i:false},{w:5,h:5,i:false}];
let adj=new Set(["0-1","0-3","1-2","2-3"]);
let mode="L", timer=null;
const key=(a,b)=>a<b?a+"-"+b:b+"-"+a;

function polygon(){
 if(mode==="L"){
  const W=+$("W").value,H=+$("H").value,nw=+$("nw").value,nh=+$("nh").value,c=$("nc").value;
  if(!(nw>0&&nh>0)) return [[0,0],[W,0],[W,H],[0,H]];
  const P={tr:[[0,0],[W,0],[W,H-nh],[W-nw,H-nh],[W-nw,H],[0,H]],
           tl:[[0,0],[W,0],[W,H],[nw,H],[nw,H-nh],[0,H-nh]],
           br:[[0,0],[W-nw,0],[W-nw,nh],[W,nh],[W,H],[0,H]],
           bl:[[nw,0],[W,0],[W,H],[0,H],[0,nh],[nw,nh]]};
  return P[c];
 }
 return $("corners").value.split(/\n|;/).map(s=>s.trim()).filter(Boolean).map(s=>s.split(/[ ,]+/).map(Number));
}
function area(p){let a=0;for(let i=0;i<p.length;i++){const[x1,y1]=p[i],[x2,y2]=p[(i+1)%p.length];a+=x1*y2-x2*y1}return Math.abs(a)/2}
function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}

function drawPlot(){
 const p=polygon();const svg=$("plotsvg");
 if(p.length<3||p.some(q=>q.length<2||q.some(isNaN))){svg.innerHTML="";return}
 const xs=p.map(q=>q[0]),ys=p.map(q=>q[1]);const mx=Math.min(...xs),Mx=Math.max(...xs),my=Math.min(...ys),My=Math.max(...ys);
 const pad=Math.max(Mx-mx,My-my)*.06+.1;
 svg.setAttribute("viewBox",`${mx-pad} ${-(My+pad)} ${Mx-mx+2*pad} ${My-my+2*pad}`);
 svg.innerHTML=`<polygon points="${p.map(q=>q[0]+","+(-q[1])).join(" ")}" fill="var(--acc2)" stroke="var(--acc)" stroke-width="${pad/6}"/>`+
  p.map((q,i)=>`<text x="${q[0]}" y="${-q[1]}" font-size="${pad*.8}" fill="var(--mut)" text-anchor="middle" dy="-.3em">${i+1}</text>`).join("");
 $("plotArea").textContent="area "+(+area(p).toFixed(4));
}
function renderRooms(){
 const n=rooms.length;
 $("rooms").querySelector("thead").innerHTML="<tr><th></th><th>#</th><th>Width</th><th>Height</th><th title='must not touch the plot boundary, not even at a point'>Interior</th><th style='border-left:2px solid var(--line)'></th>"+rooms.map((r,j)=>`<th class="cj" data-j="${j}" style="text-align:center">${j}<br><small class="mut">${r.w}×${r.h}</small></th>`).join("")+"</tr>";
 const tb=$("rooms").querySelector("tbody");tb.innerHTML="";
 rooms.forEach((r,i)=>{const tr=document.createElement("tr");
  let ticks="";for(let j=0;j<n;j++)ticks+=j===i?`<td style="background:var(--bg)"></td>`:`<td style="text-align:center"><input type="checkbox" data-a="${i}" data-b="${j}" ${adj.has(key(i,j))?"checked":""}></td>`;
  tr.innerHTML=`<td><button class="x" title="remove room ${i}">✕</button></td><td><b>${i}</b></td><td><input type="number" min="0" step="any" value="${r.w}"></td><td><input type="number" min="0" step="any" value="${r.h}"></td><td><input type="checkbox" class="int" ${r.i?"checked":""}></td><td style="border-left:2px solid var(--line)"></td>${ticks}`;
  const [w,h]=tr.querySelectorAll("input[type=number]"),c=tr.querySelector("input.int");
  w.oninput=()=>{r.w=+w.value;changed();heads()};h.oninput=()=>{r.h=+h.value;changed();heads()};c.onchange=()=>{r.i=c.checked;changed()};
  tr.querySelector(".x").onclick=()=>{removeRoom(i)};
  tr.querySelectorAll("input[data-a]").forEach(t=>t.onchange=()=>{const k=key(+t.dataset.a,+t.dataset.b);t.checked?adj.add(k):adj.delete(k);syncTicks();changed()});
  tb.appendChild(tr)});
 const t=rooms.reduce((s,r)=>s+r.w*r.h,0);$("roomArea").textContent=n+" rooms · area "+(+t.toFixed(4));
 refreshTouches();
}
function heads(){document.querySelectorAll("th.cj").forEach(h=>{h.querySelector("small").textContent=rooms[+h.dataset.j].w+"×"+rooms[+h.dataset.j].h})}
function syncTicks(){document.querySelectorAll("input[data-a]").forEach(t=>{t.checked=adj.has(key(+t.dataset.a,+t.dataset.b))});refreshTouches()}
function removeRoom(k){
 rooms.splice(k,1);const n=new Set();
 adj.forEach(e=>{let[a,b]=e.split("-").map(Number);if(a===k||b===k)return;if(a>k)a--;if(b>k)b--;n.add(key(a,b))});adj=n;renderRooms();changed()}
function refreshTouches(){}
function payload(){
 const rs={};rooms.forEach((r,i)=>rs[i]=[r.w,r.h]);
 return {polygon:polygon(),rects:rs,adjacency:[...adj].map(e=>e.split("-").map(Number)),
  interior:rooms.map((r,i)=>r.i?i:-1).filter(i=>i>=0),time_limit:+$("tl").value,
  options:{}}
}
function changed(){
 const t=rooms.reduce((s,r)=>s+r.w*r.h,0);$("roomArea").textContent=rooms.length+" rooms · area "+(+t.toFixed(4));
 drawPlot();clearTimeout(timer);timer=setTimeout(check,250)}
async function check(){
 let r;try{r=await (await fetch("/validate",{method:"POST",body:JSON.stringify(payload())})).json()}catch(e){return}
 let h="";(r.errors||[]).forEach(e=>h+=`<div class="msg bad">${esc(e)}</div>`);
 if(!(r.errors||[]).length){const i=r.info||{};
  h+=`<div class="msg ok">Input looks valid.${i.hole_area?` Rooms leave an interior hole of area ${+i.hole_area.toFixed(4)}.`:" The rooms fill the plot exactly."}</div>`}
 $("problems").innerHTML=h;$("run").dataset.bad=(r.errors||[]).length?1:"";
}
function drawResult(res,poly){
 const xs=poly.map(q=>q[0]),ys=poly.map(q=>q[1]);const mx=Math.min(...xs),Mx=Math.max(...xs),my=Math.min(...ys),My=Math.max(...ys);
 const pad=Math.max(Mx-mx,My-my)*.04;const s=Math.max(Mx-mx,My-my)/22;
 let g=`<svg viewBox="${mx-pad} ${-(My+pad)} ${Mx-mx+2*pad} ${My-my+2*pad}">`;
 g+=`<polygon points="${poly.map(q=>q[0]+","+(-q[1])).join(" ")}" fill="none" stroke="var(--ink)" stroke-width="${s/4}"/>`;
 (res.hole||[]).forEach(h=>g+=`<polygon points="${h.map(q=>q[0]+","+(-q[1])).join(" ")}" fill="url(#hatch)" stroke="var(--mut)" stroke-dasharray="${s/2}" stroke-width="${s/8}"/>`);
 const inter=new Set(payload().interior);
 for(const[i,[x,y,w,h]]of Object.entries(res.placements)){
  g+=`<rect x="${x}" y="${-(y+h)}" width="${w}" height="${h}" fill="${inter.has(+i)?"var(--int)":"var(--room)"}" stroke="var(--ink)" stroke-width="${s/8}"/>`+
     `<text x="${x+w/2}" y="${-(y+h/2)}" font-size="${Math.min(s*1.3,Math.min(w,h)*.6)}" text-anchor="middle" dominant-baseline="central" fill="var(--ink)" font-weight="600">${i}</text>`}
 return g.replace("<svg ",`<svg `)+`<defs><pattern id="hatch" width="${s}" height="${s}" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="${s}" stroke="var(--mut)" stroke-width="${s/10}"/></pattern></defs></svg>`;
}
$("run").onclick=async()=>{
 const b=$("run");if(b.dataset.bad){$("result").innerHTML='<div class="msg bad">Fix the problems listed under “Run” first.</div>';return}
 b.disabled=true;$("runStat").textContent="searching… (up to "+$("tl").value+" s)";const p=payload();
 let res;try{res=await (await fetch("/solve",{method:"POST",body:JSON.stringify(p)})).json()}catch(e){res={status:"invalid",errors:[String(e)]}}
 b.disabled=false;$("runStat").textContent="";const R=$("result");
 if(res.status==="invalid"){R.innerHTML=res.errors.map(e=>`<div class="msg bad">${esc(e)}</div>`).join("");return}
 const stats=`<div class="stats"><div><b>${res.seconds}s</b>time</div><div><b>${res.nodes}</b>search nodes</div></div>`;
 if(res.status==="solved"){const v=res.verify;
  R.innerHTML=`<div class="msg ok"><b>Layout found.</b> Checked: ${v.overlaps} overlaps, ${v.outside} outside the plot, ${v.holes} extra holes.</div>`+stats+drawResult(res,p.polygon)+
   `<div class="mut" style="margin-top:6px">Blue = ordinary room, orange = interior room, hatched = hole.</div>`}
 else if(res.status==="none")
  R.innerHTML=`<div class="msg warn"><b>No layout found.</b> The search finished without one. This is very likely infeasible, but a “no” is not a proof: the corner rules and pruning are heuristics.</div>`+stats
 else R.innerHTML=`<div class="msg warn"><b>Timed out</b> after ${res.seconds}s. Undecided. Try a longer time limit.</div>`+stats;
};
// plot controls
["W","H","nw","nh","nc","corners"].forEach(id=>$(id).addEventListener("input",changed));
$("tabL").onclick=()=>{mode="L";$("tabL").className="on";$("tabC").className="";$("lbox").style.display="";$("cbox").style.display="none";changed()};
$("tabC").onclick=()=>{mode="C";if(!$("corners").value)$("corners").value=polygon().map(q=>q.join(",")).join("\n");$("tabC").className="on";$("tabL").className="";$("cbox").style.display="";$("lbox").style.display="none";changed()};
$("addRoom").onclick=()=>{rooms.push({w:3,h:3,i:false});renderRooms();changed()};
$("qadd").onclick=()=>{for(let k=0;k<+$("qn").value;k++)rooms.push({w:+$("qw").value,h:+$("qh").value,i:false});renderRooms();changed()};
function setState(d){
 mode="C";$("tabC").onclick();$("corners").value=d.polygon.map(q=>q.join(",")).join("\n");
 const n=Math.max(...Object.keys(d.rects).map(Number))+1;
 rooms=[];for(let i=0;i<n;i++){const r=d.rects[i]||[1,1];rooms.push({w:r[0],h:r[1],i:(d.interior||[]).includes(i)})}
 adj=new Set(d.adjacency.map(([a,b])=>key(a,b)));renderRooms();changed()}
$("save").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(payload(),null,1)],{type:"application/json"}));a.download="packer_example.json";a.click()};
$("load").onclick=()=>$("file").click();
$("file").onchange=async e=>{const f=e.target.files[0];if(f)setState(JSON.parse(await f.text()))};
fetch("/examples").then(r=>r.json()).then(ex=>{for(const k in ex){const o=document.createElement("option");o.value=k;o.textContent=`Example ${k} (${Object.keys(ex[k].rects).length} rooms)`;$("exSel").appendChild(o)}
 $("exSel").onchange=()=>{if($("exSel").value)setState(ex[$("exSel").value])}});
renderRooms();changed();
</script></body></html>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}"
    print(f"Room packer UI running at {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass