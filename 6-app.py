#!/usr/bin/env python3
"""Local interactive demo for the PU lacquer scratch-hardness model.

Run:  python3 app.py [port]   then open http://127.0.0.1:8000
Pure stdlib server + the saved GP model. No data leaves the machine.
Optional deep link: http://127.0.0.1:8000/?vernetzung=0.9&hdi_ipdi=0.1 presets sliders.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
FEATS = ["vernetzung", "hdi_ipdi", "mattierungsmittel", "pigmentpaste"]
LABELS = {
    "vernetzung": "Vernetzung (crosslinking)",
    "hdi_ipdi": "HDI-IPDI ratio",
    "mattierungsmittel": "Mattierungsmittel (matting agent)",
    "pigmentpaste": "Pigmentpaste",
}
try:
    gp = joblib.load(HERE / "model.joblib")
except Exception as exc:  # version mismatch between sklearn builds
    sys.exit("Could not load model.joblib ({}). Install the pinned versions with pip install -r requirements.txt, "
             "or retrain with python3 alchemy_lacquer_model.py".format(type(exc).__name__))
card = json.load(open(HERE / "model_card.json"))
INFL = card["inflation_factor"]


def predict_block(x):
    mean, std = gp.predict(np.array([x], dtype=float), return_std=True)
    hw = 1.96 * float(std[0]) * INFL
    lines = [f"Predicted Kratzhaerte: {mean[0]:.3f}",
             f"Calibrated 95% interval: [{mean[0] - hw:.3f}, {mean[0] + hw:.3f}] (half-width {hw:.3f})",
             f"Best measured in training data: {card['best_measured']:.2f}", "",
             "Source-scale values (authors' unnorm mapping, unit labels not documented):"]
    for f, v in zip(FEATS, x):
        pv = card["denorm"][f]["slope"] * float(v) + card["denorm"][f]["intercept"]
        lines.append(f"  {f}: {pv:.3f}")
    return "\n".join(lines)


INITIAL = predict_block([0.5, 0.5, 0.5, 0.5])

PAGE = r"""<!doctype html><meta charset=utf-8><title>Lacquer model demo</title>
<style>
 body{font-family:sans-serif;max-width:760px;margin:2em auto;color:#222}
 label{display:block;margin-top:1em;font-weight:600}
 input[type=range]{width:100%}
 #out{margin-top:1.5em;padding:1em;background:#f4f6fb;border-radius:8px;white-space:pre-wrap}
 small{color:#666}
</style>
<h2>PU lacquer scratch-hardness model (local demo)</h2>
<small>Trained on 42 measured PURformance formulations (Zenodo 10.5281/zenodo.13742097, CC-BY-4.0).
Method demonstration only - predictions need lab validation. Sliders are normalized recipe
coordinates in [0,1]; source-scale values are shown using the authors' own mapping (unit labels not documented).</small>
SLIDERS
<div id=out>%INITIAL%</div>
<script>
const FEATS = %FEATS%;
function fmt(x){return (+x).toFixed(3)}
async function go(){
  for (const f of FEATS) document.getElementById(f+'_v').textContent = document.getElementById(f).value;
  const q = new URLSearchParams();
  for (const f of FEATS) q.set(f, document.getElementById(f).value);
  const r = await fetch('/predict?'+q.toString());
  const j = await r.json();
  if (j.error) { document.getElementById('out').textContent = 'error: '+j.error; return; }
  const lines = [
    'Predicted Kratzhaerte: '+fmt(j.mean),
    'Calibrated 95% interval: ['+fmt(j.lo)+', '+fmt(j.hi)+'] (half-width '+fmt(j.hw)+')',
    'Best measured in training data: '+fmt(j.best),
    '',
    "Source-scale values (authors' unnorm mapping, unit labels not documented):"
  ];
  for (const f of FEATS) lines.push('  '+f+': '+fmt(j.source_scale[f]));
  if (j.warn) lines.push('', 'WARNING: '+j.warn);
  document.getElementById('out').textContent = lines.join('\n');
}
// optional deep link: ?vernetzung=0.9&hdi_ipdi=0.1 presets the sliders
const init = new URLSearchParams(location.search);
let preset = false;
for (const f of FEATS) {
  const v = parseFloat(init.get(f));
  if (!isNaN(v)) { document.getElementById(f).value = Math.min(1, Math.max(0, v)); preset = true; }
}
for (const f of FEATS) document.getElementById(f).addEventListener('input', go);
if (preset) {
  go();  // URL preset: fetch the matching prediction from the server
} else {
  // server-rendered block already shows the center-point prediction; sync labels only
  for (const f of FEATS) document.getElementById(f+'_v').textContent = document.getElementById(f).value;
}
</script>"""

sliders = "".join(
    f'<label>{LABELS[f]} <span id="{f}_v"></span>'
    f'<input type=range id="{f}" min=0 max=1 step=0.005 value=0.5></label>'
    for f in FEATS)
PAGE = (PAGE.replace("%INITIAL%", INITIAL)
        .replace("SLIDERS", sliders)
        .replace("%FEATS%", json.dumps(FEATS)))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/predict":
            q = parse_qs(u.query)
            try:
                x = [float(q[f][0]) for f in FEATS]
            except Exception:
                self._send(b'{"error":"bad input"}', "application/json")
                return
            mean, std = gp.predict(np.array([x], dtype=float), return_std=True)
            hw = 1.96 * float(std[0]) * INFL
            warn = ""
            for f, v in zip(FEATS, x):
                if not (0 <= v <= 1):
                    warn += f"{f} outside tested range [0,1]; extrapolation. "
            source_scale = {f: card["denorm"][f]["slope"] * float(v) + card["denorm"][f]["intercept"]
                            for f, v in zip(FEATS, x)}
            out = {"mean": float(mean[0]), "lo": float(mean[0]) - hw,
                   "hi": float(mean[0]) + hw, "hw": hw,
                   "best": card["best_measured"], "source_scale": source_scale,
                   "warn": warn}
            self._send(json.dumps(out).encode(), "application/json")
        else:
            self._send(PAGE.encode(), "text/html")

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Open http://127.0.0.1:{port}  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
