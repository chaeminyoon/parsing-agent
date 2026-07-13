"""골든셋 라벨링 도구 — 원본 PDF와 파싱 결과를 나란히 놓고 클릭으로 라벨링한다.

사용:
    uv run python golden/labeling_tool.py "/Volumes/T7/1. 논문정리/golden-labeling"
    → http://127.0.0.1:8500 접속

워크스페이스 구조(준비 스크립트가 만든 그대로):
    originals/<문서>.pdf   parsed/<문서>/<문서>.md   labels/<문서>.json

stdlib만 사용한다. 저장 버튼이 labels/<문서>.json의 ratings[0]을 갱신한다.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

PORT = 8500

_PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>골든셋 라벨링</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Apple SD Gothic Neo", sans-serif; display: flex; height: 100vh; }
  #sidebar { width: 230px; background: #1e293b; color: #e2e8f0; padding: 12px; overflow-y: auto; flex-shrink: 0; }
  #sidebar h1 { font-size: 15px; margin: 4px 0 12px; }
  #sidebar .doc { padding: 7px 9px; border-radius: 6px; cursor: pointer; font-size: 12.5px; margin-bottom: 3px; }
  #sidebar .doc:hover { background: #334155; }
  #sidebar .doc.active { background: #3b82f6; color: white; }
  #sidebar .doc.done::after { content: " ✓"; color: #4ade80; }
  #sidebar .doc.missing { opacity: 0.45; }
  #progress { font-size: 12px; color: #94a3b8; margin-bottom: 10px; }
  #main { flex: 1; display: flex; min-width: 0; }
  #pdf { flex: 1; border: none; min-width: 0; }
  #right { flex: 1; display: flex; flex-direction: column; min-width: 0; border-left: 2px solid #cbd5e1; }
  #md { flex: 1; overflow-y: auto; padding: 16px 20px; font-size: 13.5px; line-height: 1.55; }
  #md table { border-collapse: collapse; margin: 10px 0; }
  #md th, #md td { border: 1px solid #94a3b8; padding: 3px 8px; font-size: 12.5px; }
  #md th { background: #f1f5f9; }
  #md h1 { font-size: 19px; border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; }
  #md h2 { font-size: 16px; } #md h3 { font-size: 14px; }
  #md .marker { color: #94a3b8; font-size: 11px; border-top: 1px dashed #cbd5e1; margin-top: 8px; }
  #form { border-top: 2px solid #cbd5e1; padding: 10px 16px; background: #f8fafc; flex-shrink: 0; }
  .row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .row label.name { width: 150px; font-size: 12.5px; font-weight: 600; }
  .row .hint { font-size: 11px; color: #64748b; margin-left: 6px; }
  .score { display: flex; gap: 4px; }
  .score button { width: 30px; height: 26px; border: 1px solid #cbd5e1; background: white; border-radius: 5px; cursor: pointer; font-size: 13px; }
  .score button.sel { background: #3b82f6; color: white; border-color: #3b82f6; }
  #notes { width: 100%; height: 44px; font-size: 12.5px; padding: 6px; border: 1px solid #cbd5e1; border-radius: 6px; resize: vertical; }
  #save { margin-top: 6px; padding: 8px 22px; background: #16a34a; color: white; border: none; border-radius: 7px; font-size: 14px; font-weight: 700; cursor: pointer; }
  #save:disabled { background: #94a3b8; cursor: not-allowed; }
  #saved-msg { margin-left: 10px; color: #16a34a; font-size: 13px; font-weight: 600; }
  #halluc { transform: scale(1.15); }
</style></head>
<body>
<div id="sidebar">
  <h1>골든셋 라벨링</h1>
  <div id="progress"></div>
  <div id="docs"></div>
</div>
<div id="main">
  <embed id="pdf" type="application/pdf" src="">
  <div id="right">
    <div id="md"></div>
    <div id="form">
      <div class="row"><label class="name">종합 품질</label><div class="score" data-f="overall_quality"></div><span class="hint">5=RAG에 그대로 · 3=부분수정 · 1=사용불가</span></div>
      <div class="row"><label class="name">본문 보존</label><div class="score" data-f="text_coverage"></div><span class="hint">5=누락없음 · 3=문단누락 · 1=절반이상 누락</span></div>
      <div class="row"><label class="name">표 정확성</label><div class="score" data-f="table_correctness"></div><span class="hint">5=정확(표 없으면 5) · 3=구조 일부 깨짐 · 1=대부분 소실</span></div>
      <div class="row"><label class="name">구조 보존</label><div class="score" data-f="structure_preservation"></div><span class="hint">5=헤딩·순서 일치 · 3=절반 유실 · 1=식별불가</span></div>
      <div class="row"><label class="name">환각 있음</label><input type="checkbox" id="halluc"><span class="hint">원본에 없는 내용이 생성됐으면 체크</span></div>
      <textarea id="notes" placeholder="3점 이하 항목은 근거 필수 (예: p6 표 병합셀 깨짐)"></textarea>
      <button id="save">저장</button><span id="saved-msg"></span>
    </div>
  </div>
</div>
<script>
let docs = [], current = null, scores = {};
const FIELDS = ["overall_quality","text_coverage","table_correctness","structure_preservation"];

function mdToHtml(md) {
  const esc = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  const inline = s => esc(s).replace(/\\*\\*([^*]+)\\*\\*/g,"<b>$1</b>");
  const lines = md.split("\\n"); let out = [], table = [];
  const flushTable = () => {
    if (!table.length) return;
    let html = "<table>";
    table.forEach((cells, i) => {
      if (cells.every(c => /^[-: ]+$/.test(c))) return;
      const tag = i === 0 ? "th" : "td";
      html += "<tr>" + cells.map(c => `<${tag}>${inline(c)}</${tag}>`).join("") + "</tr>";
    });
    out.push(html + "</table>"); table = [];
  };
  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith("|") && t.endsWith("|")) { table.push(t.slice(1,-1).split("|").map(c=>c.trim())); continue; }
    flushTable();
    let m;
    if ((m = t.match(/^(#{1,6})\\s+(.*)/))) out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`);
    else if ((m = t.match(/^<!--\\s*(.*?)\\s*-->$/))) out.push(`<div class="marker">◾ ${esc(m[1])}</div>`);
    else if (t.startsWith("- ")) out.push(`<div>• ${inline(t.slice(2))}</div>`);
    else if (t === "") out.push("<div style='height:8px'></div>");
    else out.push(`<div>${inline(t)}</div>`);
  }
  flushTable();
  return out.join("");
}

function renderScores() {
  document.querySelectorAll(".score").forEach(el => {
    const f = el.dataset.f; el.innerHTML = "";
    for (let v = 1; v <= 5; v++) {
      const b = document.createElement("button");
      b.textContent = v;
      if (scores[f] === v) b.classList.add("sel");
      b.onclick = () => { scores[f] = v; renderScores(); checkSave(); };
      el.appendChild(b);
    }
  });
}
function checkSave() {
  document.getElementById("save").disabled = !FIELDS.every(f => scores[f] >= 1);
}
async function loadDocs() {
  docs = await (await fetch("/api/docs")).json();
  const box = document.getElementById("docs"); box.innerHTML = "";
  let done = 0;
  docs.forEach(d => {
    if (d.labeled) done++;
    const el = document.createElement("div");
    el.className = "doc" + (d.labeled ? " done" : "") + (d.has_md ? "" : " missing") + (current === d.name ? " active" : "");
    el.textContent = d.name + (d.has_md ? "" : " (파싱 대기)");
    el.onclick = () => openDoc(d.name);
    box.appendChild(el);
  });
  document.getElementById("progress").textContent = `완료 ${done} / ${docs.length}`;
}
async function openDoc(name) {
  current = name;
  document.getElementById("pdf").src = `/original/${encodeURIComponent(name)}.pdf`;
  const md = await (await fetch(`/api/md/${encodeURIComponent(name)}`)).text();
  document.getElementById("md").innerHTML = mdToHtml(md);
  const label = await (await fetch(`/api/label/${encodeURIComponent(name)}`)).json();
  const r = label.ratings[0];
  scores = {}; FIELDS.forEach(f => { if (r[f] >= 1) scores[f] = r[f]; });
  document.getElementById("halluc").checked = !!r.hallucination;
  document.getElementById("notes").value = r.notes || "";
  document.getElementById("saved-msg").textContent = "";
  renderScores(); checkSave(); loadDocs();
}
document.getElementById("save").onclick = async () => {
  const body = { ...scores, hallucination: document.getElementById("halluc").checked,
                 notes: document.getElementById("notes").value };
  const res = await fetch(`/api/save/${encodeURIComponent(current)}`, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
  document.getElementById("saved-msg").textContent = res.ok ? "저장됨 ✓" : "저장 실패";
  loadDocs();
};
loadDocs().then(() => { const first = docs.find(d => d.has_md && !d.labeled) || docs[0]; if (first) openDoc(first.name); });
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    workspace: Path

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _doc_names(self) -> list[str]:
        return sorted(
            p.stem for p in (self.workspace / "originals").glob("*.pdf") if not p.name.startswith("._")
        )

    def _label_path(self, name: str) -> Path:
        return self.workspace / "labels" / f"{name}.json"

    def do_GET(self) -> None:  # noqa: N802 - http.server 규약
        path = unquote(self.path)
        if path == "/":
            self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/docs":
            docs = []
            for name in self._doc_names():
                md = self.workspace / "parsed" / name / f"{name}.md"
                labeled = False
                label_path = self._label_path(name)
                if label_path.exists():
                    rating = json.loads(label_path.read_text(encoding="utf-8"))["ratings"][0]
                    labeled = all(rating.get(f, -1) >= 1 for f in (
                        "overall_quality", "text_coverage", "table_correctness", "structure_preservation"))
                docs.append({"name": name, "has_md": md.exists(), "labeled": labeled})
            self._send(200, json.dumps(docs, ensure_ascii=False).encode(), "application/json")
        elif path.startswith("/original/"):
            pdf = self.workspace / "originals" / path.removeprefix("/original/")
            if pdf.exists() and pdf.suffix == ".pdf":
                self._send(200, pdf.read_bytes(), "application/pdf")
            else:
                self._send(404, b"not found", "text/plain")
        elif path.startswith("/api/md/"):
            name = path.removeprefix("/api/md/")
            md = self.workspace / "parsed" / name / f"{name}.md"
            body = md.read_text(encoding="utf-8") if md.exists() else "(아직 파싱 결과가 없습니다)"
            self._send(200, body.encode(), "text/plain; charset=utf-8")
        elif path.startswith("/api/label/"):
            label_path = self._label_path(path.removeprefix("/api/label/"))
            if label_path.exists():
                self._send(200, label_path.read_bytes(), "application/json")
            else:
                self._send(404, b"{}", "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802 - http.server 규약
        path = unquote(self.path)
        if not path.startswith("/api/save/"):
            self._send(404, b"not found", "text/plain")
            return
        name = path.removeprefix("/api/save/")
        label_path = self._label_path(name)
        if not label_path.exists():
            self._send(404, b"label template missing", "text/plain")
            return
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        label = json.loads(label_path.read_text(encoding="utf-8"))
        rating = label["ratings"][0]
        for field in ("overall_quality", "text_coverage", "table_correctness", "structure_preservation"):
            if isinstance(payload.get(field), int):
                rating[field] = payload[field]
        rating["hallucination"] = bool(payload.get("hallucination", False))
        rating["notes"] = str(payload.get("notes", ""))
        rating["labeled_at"] = date.today().isoformat()
        label_path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
        self._send(200, b"ok", "text/plain")

    def log_message(self, *args) -> None:  # 조용히
        pass


def main() -> int:
    # 인자가 없으면 이 스크립트가 놓인 폴더를 워크스페이스로 쓴다
    # (워크스페이스에 복사해 두면 `python3 labeling_tool.py`만으로 실행).
    workspace = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    for required in ("originals", "labels"):
        if not (workspace / required).is_dir():
            print(f"워크스페이스가 아닙니다: {workspace} ({required}/ 없음)")
            return 1
    Handler.workspace = workspace
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"라벨링 도구: http://127.0.0.1:{PORT}  (중지: Ctrl+C)")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
