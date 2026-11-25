#!/usr/bin/env python3
"""Smoke tests for the Alchemy lacquer model package. Run: python3 test_model.py"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

HERE = Path(__file__).resolve().parent
ok = 0


def check(name, cond):
    global ok
    assert cond, f"FAIL: {name}"
    ok += 1
    print(f"pass: {name}")


# data integrity
df = pd.read_csv(HERE / "data" / "purformance_labeled.csv")
check("42 labeled rows", len(df) == 42)
check("no missing targets", df["kratzhaerte"].notna().all())
check("features in [0,1]",
      ((df[["vernetzung", "hdi_ipdi", "mattierungsmittel", "pigmentpaste"]] >= 0)
       & (df[["vernetzung", "hdi_ipdi", "mattierungsmittel", "pigmentpaste"]] <= 1)).all().all())
check("duplicate recipe rows 17/18 present", (df["row_id"] == 17).any() and (df["row_id"] == 18).any())

# model + card
card = json.load(open(HERE / "model_card.json"))
gp = joblib.load(HERE / "model.joblib")
check("inflation factor >= 1", card["inflation_factor"] >= 1.0)
x = np.array([[0.72, 0.0, 1.0, 0.66]])
m, s = gp.predict(x, return_std=True)
check("prediction in plausible range (5..30)", 5 < m[0] < 30)
check("positive std", s[0] > 0)

# denorm mapping sanity: pigmentpaste 0 -> 4.997, 1 -> 24.981
d = card["denorm"]["pigmentpaste"]
check("denorm pigmentpaste endpoints",
      abs(d["intercept"] - 4.997254) < 1e-4 and abs(d["slope"] + d["intercept"] - 24.981165) < 1e-3)

# results consistency: GP beats mean baseline RMSE on final test B
res = json.load(open(HERE / "results.json"))
tb = res["splitB_final_test_train27_test15"]["models"]
sel = res["model_selected_on_splitA"]
check("model was selected on split A, not by hand", sel == min(
    res["splitA_validation_train10_test17"]["models"],
    key=lambda k: res["splitA_validation_train10_test17"]["models"][k]["rmse"]))
check("selected model beats mean baseline RMSE on final test B",
      tb[sel]["rmse"] < tb["mean_baseline"]["rmse"])
check("calibrated coverage reported on B", "coverage_95_calibrated" in tb[sel])

# recommendations file
recs = pd.read_csv(HERE / "next_experiments.csv")
check("5 recommendations", len(recs) == 5)
check("source-scale columns present", "vernetzung_source_scale" in recs.columns)
check("diverse batch: entries 2-5 at least 0.25 L2 from rank 1",
      all(np.linalg.norm(recs.iloc[i][["vernetzung", "hdi_ipdi", "mattierungsmittel", "pigmentpaste"]].values
                         - recs.iloc[0][["vernetzung", "hdi_ipdi", "mattierungsmittel", "pigmentpaste"]].values) >= 0.25 - 1e-6
          for i in range(1, 5)))

# predict CLI smoke test
out = subprocess.run([sys.executable, str(HERE / "predict.py"), "0.7208", "0.0036", "0.9973", "0.6596"],
                     capture_output=True, text=True)
check("predict.py runs", out.returncode == 0)
check("predict.py prints interval", "95% interval" in out.stdout)
out2 = subprocess.run([sys.executable, str(HERE / "predict.py"), "--source-scale", "1.0946", "0.0036", "4.9667", "18.18"],
                      capture_output=True, text=True)
check("predict.py --source-scale runs", out2.returncode == 0 and "predicted Kratzhaerte" in out2.stdout)
m1 = [l for l in out.stdout.splitlines() if l.startswith("predicted")][0]
m2 = [l for l in out2.stdout.splitlines() if l.startswith("predicted")][0]
check("normalized and source-scale modes agree", m1.split(":")[1].strip()[:5] == m2.split(":")[1].strip()[:5])

# live app integration test: same request shape the page JS sends
import time, urllib.request
PORT = 8123
srv = subprocess.Popen([sys.executable, str(HERE / "app.py"), str(PORT)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    page = None
    for _ in range(30):
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=2).read().decode()
            break
        except Exception:
            time.sleep(0.3)
    check("app serves page", page is not None and "scratch-hardness model" in page)
    check("page contains server-rendered prediction", "Predicted Kratzhaerte: 15.480" in page)
    q = "vernetzung=0.9&hdi_ipdi=0.1&mattierungsmittel=0.8&pigmentpaste=0.2"
    j = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/predict?{q}", timeout=5).read())
    check("predict endpoint accepts named keys", "mean" in j and "source_scale" in j)
    check("predict endpoint responds to slider change (not center value)",
          abs(j["mean"] - 15.480) > 0.01)
    check("interval ordering", j["lo"] < j["mean"] < j["hi"])
finally:
    srv.terminate()
    srv.wait(timeout=5)

print(f"\nAll {ok} checks passed.")

