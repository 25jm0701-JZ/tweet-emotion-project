# -*- coding: utf-8 -*-
"""从 hf-mirror 并行分段下载 cardiffnlp/twitter-roberta-base-emotion 到本地目录。"""
import os
import sys
import threading
import time
import urllib.request

REPO = "cardiffnlp/twitter-roberta-base-emotion"
BASE = f"https://hf-mirror.com/{REPO}/resolve/main/"
FILES = ["config.json", "merges.txt", "vocab.json", "special_tokens_map.json",
         "pytorch_model.bin"]
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_model")
N_CONN = 16
CHUNK = 8 * 1024 * 1024


def get_size(name):
    """返回 (总大小, 若不是 206 则返回整段内容)。"""
    req = urllib.request.Request(BASE + name,
                                 headers={"Range": "bytes=0-0",
                                          "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
        body = r.read()
    if "/" in cr:
        return int(cr.rsplit("/", 1)[1]), None
    # 服务器不支持 Range（小文件常见），直接把整段返回
    return len(body), body


def fetch_range(name, start, end, path, idx, prog):
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                BASE + name, headers={"Range": f"bytes={start}-{end}",
                                      "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} != {end - start + 1}")
            with open(path, "r+b") as f:
                f.seek(start)
                f.write(data)
            prog[idx] = len(data)
            return
        except Exception as e:
            print(f"  retry {name} {start}-{end}: {e}", flush=True)
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"failed {name} {start}-{end}")


def download(name):
    size, whole = get_size(name)
    path = os.path.join(DEST, name)
    if whole is not None:
        with open(path, "wb") as f:
            f.write(whole)
        print(f"fetched {name} ({size/1e6:.2f} MB)", flush=True)
        return
    if os.path.exists(path) and os.path.getsize(path) == size:
        print(f"skip {name} ({size/1e6:.1f} MB)", flush=True)
        return
    print(f"download {name}: {size/1e6:.1f} MB", flush=True)
    with open(path, "wb") as f:
        f.truncate(size)
    spans = [(s, min(s + CHUNK - 1, size - 1)) for s in range(0, size, CHUNK)]
    prog = [0] * len(spans)
    t0 = time.time()
    for i in range(0, len(spans), N_CONN):
        group = list(range(i, min(i + N_CONN, len(spans))))
        ts = [threading.Thread(target=fetch_range, args=(name, *spans[j], path, j, prog))
              for j in group]
        [t.start() for t in ts]
        [t.join() for t in ts]
        done = sum(prog)
        dt = time.time() - t0
        print(f"  {name}: {done/1e6:.0f}/{size/1e6:.0f} MB "
              f"({done/max(dt,1e-9)/1e6:.2f} MB/s)", flush=True)
    assert os.path.getsize(path) == size, "size mismatch"
    print(f"done {name} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    for f in FILES:
        download(f)
    print("ALL DONE", DEST)
