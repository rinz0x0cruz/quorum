"""Self-contained offline HTML dashboard.

Builds a single ``dashboard.html`` (inline CSS/JS, zero external requests) from
the stored sessions + benchmark rows: a strategy comparison table and a browser
of recent deliberations with each round's score and the final answer. Mirrors the
sibling tools' render language (inline JSON via ``__DATA__`` replacement).
"""
from __future__ import annotations

import html
import json
import os
from typing import Any

from . import bench as bench_mod
from . import throttle as throttle_mod
from .store import Store, now_iso


def build(cfg: dict, store: Store) -> str:
    sessions = store.recent_sessions(25)
    rows = store.bench_rows()
    strategies = sorted({r["strategy"] for r in rows})
    n_tasks = len({r["task_id"] for r in rows}) or 1
    summary = bench_mod.aggregate(
        [{"strategy": r["strategy"], "task_id": r["task_id"], "score": r["score"],
          "rounds": r["rounds"], "tokens": r["tokens_in"] + r["tokens_out"],
          "cost_usd": r["cost_usd"], "seconds": r["seconds"]} for r in rows],
        strategies, n_tasks) if rows else []

    payload: dict[str, Any] = {
        "generated": now_iso(),
        "sessions": [_slim(s) for s in sessions],
        "bench": summary,
        "throttle": _throttle(store),
    }
    path = cfg["output"]["dashboard_path"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_render(payload))
    return path


def _throttle(store: Store) -> Any:
    """A throttle summary for the dashboard panel, or None when no telemetry yet."""
    rows = store.api_calls_recent(5000) if hasattr(store, "api_calls_recent") else []
    if not rows:
        return None
    summary = throttle_mod.summarize(rows)
    summary["free_rpm"] = throttle_mod.FREE_RPM
    return summary


def _slim(s: dict[str, Any]) -> dict[str, Any]:
    """Keep the transcript but cap very long strings so the file stays light."""
    def clip(t: str, n: int = 4000) -> str:
        return t if len(t or "") <= n else t[:n] + "\u2026"
    rounds = []
    for r in s.get("rounds", []):
        rounds.append({
            "index": r["index"],
            "score": (r.get("verdict") or {}).get("score"),
            "best": (r.get("verdict") or {}).get("best_label"),
            "rationale": (r.get("verdict") or {}).get("rationale", ""),
            "turns": [{"member": t["member"], "kind": t["kind"], "model": t["model"],
                       "content": clip(t["content"])} for t in r.get("turns", [])],
        })
    return {
        "id": s["id"], "created": s["created"], "task": s["task"], "strategy": s["strategy"],
        "final": clip(s.get("final", ""), 8000), "final_score": s.get("final_score", 0),
        "stop_reason": s.get("stop_reason", ""), "status": s.get("status", "ok"),
        "cost_usd": s.get("cost_usd", 0), "tokens": s.get("tokens_in", 0) + s.get("tokens_out", 0),
        "prompt": clip(s.get("prompt", ""), 1200), "rounds": rounds,
    }


def _render(payload: dict) -> str:
    data = json.dumps(payload).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", data).replace("__GENERATED__", html.escape(now_iso()))


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>quorum</title>
<style>
:root{--bg:#0b0b0d;--card:#141418;--bd:#24242b;--fg:#ededf2;--dim:#a0a0ab;--acc:#7c6cff;
--ok:#22c55e;--warn:#f59e0b;--crit:#f43f5e;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:28px 20px 80px}
header{display:flex;align-items:baseline;gap:12px;border-bottom:1px solid var(--bd);padding-bottom:14px;margin-bottom:22px}
h1{font-size:22px;margin:0;letter-spacing:.5px}h1 b{color:var(--acc)}
.sub{color:var(--dim);font-size:12px}
h2{font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:26px 0 10px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--bd);border-radius:10px;overflow:hidden}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--bd);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{color:var(--dim);font-weight:600;font-size:12px}
tr:last-child td{border-bottom:0}.win{color:var(--ok);font-weight:700}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;margin:12px 0;overflow:hidden}
.chead{display:flex;gap:10px;align-items:center;padding:13px 16px;cursor:pointer}
.chead:hover{background:#17171c}.pill{font-size:11px;padding:2px 9px;border-radius:999px;background:#20202a;color:var(--dim)}
.score{margin-left:auto;font-weight:700;font-variant-numeric:tabular-nums}
.task{font-weight:600}.body{display:none;padding:0 16px 16px;border-top:1px solid var(--bd)}
.open .body{display:block}.round{margin:14px 0;padding-left:12px;border-left:2px solid var(--bd)}
.rhead{color:var(--dim);font-size:12px;margin-bottom:6px}.turn{margin:6px 0;font-size:13px}
.who{color:var(--acc);font-weight:600}.kind{color:var(--dim);font-size:11px}
.final{white-space:pre-wrap;background:#101015;border:1px solid var(--bd);border-radius:8px;padding:12px;margin-top:10px}
.bar{height:6px;border-radius:4px;background:linear-gradient(90deg,var(--acc),#4f46e5)}
.muted{color:var(--dim)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
</style></head><body><div class="wrap">
<header><h1><b>quorum</b> deliberations</h1><span class="sub" id="gen"></span></header>
<div id="benchWrap"><h2>strategy comparison</h2><div id="bench"></div></div>
<div id="throttleWrap" style="display:none"><h2>api throttle</h2><div id="throttle"></div></div>
<h2>recent sessions</h2><div id="sessions"></div>
<p class="muted" style="margin-top:40px">Generated offline. Your data stays on this machine.</p>
</div>
<script>
const D = __DATA__;
document.getElementById('gen').textContent = 'generated ' + D.generated;

function el(t, c, txt){const e=document.createElement(t); if(c)e.className=c; if(txt!=null)e.textContent=txt; return e;}

function renderBench(){
  const host=document.getElementById('bench');
  if(!D.bench.length){document.getElementById('benchWrap').style.display='none';return;}
  const cols=[['strategy','strategy'],['mean_score','score'],['win_rate','win%'],
    ['mean_rounds','rounds'],['mean_tokens','tokens'],['mean_cost_usd','cost$'],['mean_seconds','sec']];
  const tbl=el('table'), thead=el('tr');
  cols.forEach(c=>thead.appendChild(el('th',null,c[1]))); tbl.appendChild(thead);
  D.bench.forEach((r,i)=>{const tr=el('tr');
    cols.forEach(c=>{let v=r[c[0]]; if(typeof v==='number')v=(c[0]==='mean_cost_usd')?v.toFixed(4):(Number.isInteger(v)?v:v.toFixed(1));
      const td=el('td',null,String(v)); if(i===0&&c[0]==='strategy')td.className='win'; tr.appendChild(td);});
    tbl.appendChild(tr);});
  host.appendChild(tbl);
}

function scoreColor(s){return s>=85?'var(--ok)':s>=70?'var(--warn)':'var(--crit)';}

function renderThrottle(){
  const t=D.throttle; if(!t||!t.total){return;}
  document.getElementById('throttleWrap').style.display='';
  const host=document.getElementById('throttle');
  const peak=Math.max(0,...Object.values(t.peak_rpm||{}));
  const sum=el('div','muted mono'); sum.style.margin='0 0 8px';
  sum.textContent=`${t.total} attempts · ${t.throttled} × 429 · peak ${peak}/min (free limit ${t.free_rpm})`;
  host.appendChild(sum);
  const barWrap=el('div'); barWrap.style.cssText='height:6px;background:#101015;border:1px solid var(--bd);border-radius:4px;margin:0 0 12px;overflow:hidden';
  const bar=el('div','bar'); bar.style.width=Math.min(100,peak/(t.free_rpm||20)*100)+'%';
  if(peak>=t.free_rpm)bar.style.background='var(--crit)';
  barWrap.appendChild(bar); host.appendChild(barWrap);
  const cols=[['model','model'],['total','reqs'],['ok','ok'],['throttled','429'],['rate_429','429%'],['avg_latency_ms','lat ms']];
  const tbl=el('table'), thead=el('tr'); cols.forEach(c=>thead.appendChild(el('th',null,c[1]))); tbl.appendChild(thead);
  Object.entries(t.by_model).sort((a,b)=>b[1].total-a[1].total).forEach(([model,s])=>{
    const tr=el('tr');
    const cells=[model,s.total,s.ok,s.throttled,(s.rate_429*100).toFixed(0)+'%',s.avg_latency_ms];
    cells.forEach((v,i)=>{const td=el('td',null,String(v)); if(i===3&&s.throttled>0)td.style.color='var(--crit)'; tr.appendChild(td);});
    tbl.appendChild(tr);});
  host.appendChild(tbl);
}

function renderSessions(){
  const host=document.getElementById('sessions');
  if(!D.sessions.length){host.appendChild(el('p','muted','No deliberations yet. Run: quorum run "..."'));return;}
  D.sessions.forEach(s=>{
    const card=el('div','card');
    const head=el('div','chead');
    head.appendChild(el('span','pill',s.strategy));
    head.appendChild(el('span','task',s.task));
    const sc=el('span','score',(s.final_score||0).toFixed(0)); sc.style.color=scoreColor(s.final_score||0);
    head.appendChild(sc);
    head.onclick=()=>card.classList.toggle('open');
    card.appendChild(head);
    const body=el('div','body');
    const meta=el('div','muted mono'); meta.style.fontSize='11px'; meta.style.margin='10px 0';
    meta.textContent=`${s.id} · ${s.created} · ${s.tokens} tokens · $${(s.cost_usd||0).toFixed(4)} · stop: ${s.stop_reason} · ${s.status}`;
    body.appendChild(meta);
    (s.rounds||[]).forEach(r=>{
      const rd=el('div','round');
      const label=r.index===0?'promptsmith':('round '+r.index);
      const rh=el('div','rhead', r.score!=null?`${label} — score ${r.score} (best ${r.best||'?'})`:label);
      rd.appendChild(rh);
      (r.turns||[]).forEach(t=>{const tn=el('div','turn');
        tn.appendChild(el('span','who',t.member+' '));
        tn.appendChild(el('span','kind','('+t.kind+') '));
        tn.appendChild(document.createTextNode(t.content));
        rd.appendChild(tn);});
      if(r.rationale){const jr=el('div','muted',' judge: '+r.rationale); jr.style.fontSize='12px'; rd.appendChild(jr);}
      body.appendChild(rd);
    });
    const fin=el('div','final',s.final||'(none)'); body.appendChild(el('div','rhead','final answer')); body.appendChild(fin);
    card.appendChild(body);
    host.appendChild(card);
  });
}
renderBench(); renderThrottle(); renderSessions();
</script></body></html>
"""
