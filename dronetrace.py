#!/usr/bin/env python3
"""
DroneTrace  —  DJI telemetry HUD overlay generator for any video editor.

Reads the telemetry DJI embeds inside each MP4 (a subtitle stream, no sidecar
.SRT needed) and renders a transparent overlay clip (ProRes 4444 with alpha):
  - a moving satellite mini-map with the GPS track + live dot
  - a SPEED readout
  - an ALTITUDE readout (true above-ground-level, via terrain model)

Drop the resulting *.overlay.mov on a track ABOVE your clip in any NLE. It
carries its own alpha, so it just floats over the footage — no keyframing,
no data entry.

Usage:
  dronetrace CLIP.MP4                 # overlay clip (panel-sized, ProRes 4444)
  dronetrace CLIP.MP4 --still 8       # one preview PNG at t=8s
  dronetrace CLIP.MP4 --fast          # half-res, much faster
  dronetrace FOLDER --batch           # every MP4 in a folder (shared home point)
  dronetrace CLIP.MP4 --alt asl       # above-sea-level instead of above-ground

Dependencies: ffmpeg/ffprobe on PATH, Pillow, numpy. Internet is used for map
tiles and the terrain-elevation lookup (both cached); offline it degrades
gracefully (plain map panel, relative altitude).

License: MIT.
"""

import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageChops

TILE = 256
TILE_SOURCES = {
    "sat": "https://server.arcgisonline.com/ArcGIS/rest/services/"
           "World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}
UA = "drone-overlay/1.0 (personal video tool)"


# ----------------------------------------------------------------------------- fonts
def _find_font():
    candidates = [
        "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


FONT_PATH = _find_font()


def font(size, bold=False):
    if FONT_PATH is None:
        return ImageFont.load_default()
    try:
        # .ttc with bold face index where available; otherwise plain
        if FONT_PATH.endswith(".ttc"):
            return ImageFont.truetype(FONT_PATH, size, index=1 if bold else 0)
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.truetype(FONT_PATH, size)


# ----------------------------------------------------------------------------- ffprobe
def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=0", path],
        capture_output=True, text=True, check=True).stdout
    d = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    num, den = d["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return int(d["width"]), int(d["height"]), fps, float(d["duration"])


# ----------------------------------------------------------------------------- telemetry
SRT_GPS = re.compile(r"GPS\s*\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)")
SRT_H   = re.compile(r"(?<![.\w])H\s+([-\d.]+)\s*m")          # height
SRT_HS  = re.compile(r"H\.S\s+([-\d.]+)\s*m/s")               # horizontal speed
SRT_VS  = re.compile(r"V\.S\s+([-\d.]+)\s*m/s")               # vertical speed
SRT_D   = re.compile(r"(?<![.\w])D\s+([-\d.]+)\s*m")          # distance from home
SRT_TS  = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->")


def extract_telemetry(path):
    """Return list of samples: dicts with t, lon, lat, alt, hs, vs, dist."""
    srt = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:s:0", "-f", "srt", "-"],
        capture_output=True, text=True)
    if srt.returncode != 0 or not srt.stdout.strip():
        raise RuntimeError("No telemetry subtitle stream found in this MP4.")
    blocks = re.split(r"\n\s*\n", srt.stdout.strip())
    samples = []
    for b in blocks:
        mts = SRT_TS.search(b)
        gps = SRT_GPS.search(b)
        if not (mts and gps):
            continue
        hh, mm, ss, ms = map(int, mts.groups())
        t = hh * 3600 + mm * 60 + ss + ms / 1000.0
        s = {"t": t,
             "lon": float(gps.group(1)),
             "lat": float(gps.group(2)),
             "alt": 0.0, "hs": 0.0, "vs": 0.0, "dist": 0.0}
        for key, rx in (("alt", SRT_H), ("hs", SRT_HS), ("vs", SRT_VS), ("dist", SRT_D)):
            m = rx.search(b)
            if m:
                s[key] = float(m.group(1))
        samples.append(s)
    if not samples:
        raise RuntimeError("Telemetry stream present but no GPS rows parsed.")
    return samples


def smooth_track(samples, win=9):
    """DJI logs GPS to only 4 decimals (~10 m steps), so the raw path is a
    staircase. Smooth lon/lat with a centered moving average (reflected edges)
    to recover a flowing flight path. Speed/altitude come from their own
    fields and are left untouched."""
    if win < 3 or len(samples) < 3:
        return
    lon = np.array([s["lon"] for s in samples], dtype=float)
    lat = np.array([s["lat"] for s in samples], dtype=float)
    k = win // 2
    kern = np.ones(win) / win
    for arr, key in ((lon, "lon"), (lat, "lat")):
        padded = np.pad(arr, k, mode="reflect")
        sm = np.convolve(padded, kern, mode="valid")
        for i, s in enumerate(samples):
            s[key] = float(sm[i])


# --------------------------------------------------------------------- elevation (DEM)
ELEV_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          ".elevation_cache.json")
_elev = None


def _ekey(lat, lon):
    return f"{lat:.5f},{lon:.5f}"


def elevations(coords):
    """coords: list of (lat, lon). Returns ground elevation (m ASL) for each,
    via opentopodata SRTM-30m, cached to disk so batch runs don't re-query."""
    global _elev
    if _elev is None:
        try:
            with open(ELEV_CACHE) as f:
                _elev = json.load(f)
        except Exception:
            _elev = {}
    todo = []
    seen = set()
    for lat, lon in coords:
        k = _ekey(lat, lon)
        if k not in _elev and k not in seen:
            seen.add(k)
            todo.append((lat, lon))
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        locs = "|".join(f"{lat:.5f},{lon:.5f}" for lat, lon in batch)
        url = f"https://api.opentopodata.org/v1/srtm30m?locations={locs}"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        d = json.loads(urllib.request.urlopen(req, timeout=30).read())
        for (lat, lon), r in zip(batch, d["results"]):
            _elev[_ekey(lat, lon)] = r["elevation"]
        if i + 100 < len(todo):
            time.sleep(1.0)  # opentopodata: max 1 req/sec
    if todo:
        try:
            with open(ELEV_CACHE, "w") as f:
                json.dump(_elev, f)
        except Exception:
            pass
    return [_elev[_ekey(lat, lon)] for lat, lon in coords]


def trilaterate_home(samples):
    """Recover the home/takeoff point from the (GPS, distance-from-home) pairs
    DJI logs. The home is the point that lies distance D_i from each GPS_i.
    DJI's D is the HORIZONTAL ground distance. Returns (lat, lon, fit_m) or
    None if the track doesn't move enough to locate it."""
    pts = [(s["lon"], s["lat"], s["dist"]) for s in samples if s["dist"] > 0.5]
    if len(pts) < 8:
        return None
    lon0, lat0 = pts[0][0], pts[0][1]
    mlon = 111320.0 * math.cos(math.radians(lat0))
    mlat = 111320.0
    xs = np.array([(lo - lon0) * mlon for lo, la, d in pts])
    ys = np.array([(la - lat0) * mlat for lo, la, d in pts])
    ds = np.array([d for lo, la, d in pts])
    if max(xs.max() - xs.min(), ys.max() - ys.min()) < 25.0:
        return None  # essentially a hover — geometry too poor
    x0, y0, d0 = xs[0], ys[0], ds[0]
    A = np.column_stack([2 * (xs - x0), 2 * (ys - y0)])
    b = (xs * xs - x0 * x0) + (ys * ys - y0 * y0) - (ds * ds - d0 * d0)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    X, Y = sol
    fit = float(np.abs(np.hypot(xs - X, ys - Y) - ds).mean())
    if fit > 25.0:
        return None
    return (lat0 + Y / mlat, lon0 + X / mlon, fit)


def apply_altitude(samples, mode, home=None):
    """Replace each sample's 'alt' (relative height H) with the chosen mode:
       rel = height above takeoff (unchanged)
       asl = metres above sea level  = home_ground + H
       agl = metres above ground now = home_ground + H - terrain_below
    'home' may be a (lat, lon) the caller already resolved (e.g. pooled over a
    whole session). Returns the label to show; falls back to 'rel' on failure."""
    if mode == "rel":
        return "ALT"
    try:
        if home is None:
            tri = trilaterate_home(samples)
            if tri:
                home = (tri[0], tri[1])
                src = f"trilaterated, fit ~{tri[2]:.0f} m"
            else:
                home = (samples[0]["lat"], samples[0]["lon"])
                src = "clip start (couldn't locate home — hover/short clip)"
        else:
            src = "session home"
        home_elev = elevations([home])[0]
        if mode == "agl":
            terr = elevations([(s["lat"], s["lon"]) for s in samples])
            for s, te in zip(samples, terr):
                s["alt"] = (home_elev + s["alt"]) - te
            label = "AGL"
        else:
            for s in samples:
                s["alt"] = home_elev + s["alt"]
            label = "ASL"
        print(f"  altitude: {label} (home ground ~{home_elev:.0f} m ASL, {src})")
        return label
    except Exception as e:
        sys.stderr.write(f"  [alt] DEM lookup failed ({e}); using relative height\n")
        return "ALT"


def session_home(files):
    """Pool the telemetry of every clip in a folder and trilaterate one shared
    home point — robust even when individual clips are hovers."""
    pooled = []
    for p in files:
        try:
            pooled.extend(extract_telemetry(p))
        except Exception:
            pass
    tri = trilaterate_home(pooled) if pooled else None
    return (tri[0], tri[1]) if tri else None


def lerp(a, b, f):
    return a + (b - a) * f


def sample_at(samples, t):
    """Linear interpolation of telemetry at time t (seconds)."""
    if t <= samples[0]["t"]:
        return samples[0]
    if t >= samples[-1]["t"]:
        return samples[-1]
    lo, hi = 0, len(samples) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if samples[mid]["t"] <= t:
            lo = mid
        else:
            hi = mid
    a, b = samples[lo], samples[hi]
    span = b["t"] - a["t"] or 1.0
    f = (t - a["t"]) / span
    return {k: lerp(a[k], b[k], f) for k in ("t", "lon", "lat", "alt", "hs", "vs", "dist")}


# ----------------------------------------------------------------------------- map
def latlon_to_world(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n * TILE
    latr = math.radians(lat)
    y = (1.0 - math.log(math.tan(latr) + 1.0 / math.cos(latr)) / math.pi) / 2.0 * n * TILE
    return x, y


def choose_zoom(samples, win_px, min_span_m=80.0):
    lons = [s["lon"] for s in samples]
    lats = [s["lat"] for s in samples]
    clat = sum(lats) / len(lats)
    clon = sum(lons) / len(lons)
    # meters-per-degree at this latitude
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(clat))
    span_m_x = max((max(lons) - min(lons)) * m_per_deg_lon, min_span_m)
    span_m_y = max((max(lats) - min(lats)) * m_per_deg_lat, min_span_m)
    span_m = max(span_m_x, span_m_y) * 1.35  # padding
    # ground resolution (m/px) at zoom z, latitude clat:  156543.03 * cos(lat) / 2^z
    # want span_m to fit in win_px:  span_m / (win_px) = m/px
    target_mpp = span_m / win_px
    z = math.log2(156543.03392 * math.cos(math.radians(clat)) / target_mpp)
    z = int(max(2, min(18, math.floor(z))))
    return z, clon, clat


def fetch_tile(z, x, y, cache, source):
    key = (z, x, y)
    if key in cache:
        return cache[key]
    url = TILE_SOURCES[source].format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        img = Image.open(io.BytesIO(r.read())).convert("RGBA")
    cache[key] = img
    return img


def build_map(samples, win_px, source="sat"):
    """Return (map_image win_px square, projector(lon,lat)->(px,py)) or (None, None)."""
    z, clon, clat = choose_zoom(samples, win_px)
    cx, cy = latlon_to_world(clon, clat, z)
    left = cx - win_px / 2.0
    top = cy - win_px / 2.0
    try:
        cache = {}
        tx0, tx1 = int(left // TILE), int((left + win_px) // TILE)
        ty0, ty1 = int(top // TILE), int((top + win_px) // TILE)
        stitched = Image.new("RGBA", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE))
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                tile = fetch_tile(z, tx % (2 ** z), ty % (2 ** z), cache, source)
                stitched.paste(tile, ((tx - tx0) * TILE, (ty - ty0) * TILE))
        crop_x = left - tx0 * TILE
        crop_y = top - ty0 * TILE
        mp = stitched.crop((int(crop_x), int(crop_y),
                            int(crop_x) + win_px, int(crop_y) + win_px))
        # satellite imagery (esp. forest/fields) is very dark — lift it so the
        # map actually reads as a map under the HUD
        mp = mp.convert("RGB")
        mp = ImageEnhance.Brightness(mp).enhance(1.65)
        mp = ImageEnhance.Contrast(mp).enhance(1.12)
        mp = ImageEnhance.Color(mp).enhance(1.22)
        mp = mp.convert("RGBA")

        def project(lon, lat):
            wx, wy = latlon_to_world(lon, lat, z)
            return wx - left, wy - top
        return mp, project
    except Exception as e:
        sys.stderr.write(f"  [map] tiles unavailable ({e}); using plain panel\n")
        return None, None


def plain_projector(samples, win_px):
    """Fallback: project the track onto a plain square panel (no tiles)."""
    lons = [s["lon"] for s in samples]
    lats = [s["lat"] for s in samples]
    clat = sum(lats) / len(lats)
    mlon = math.cos(math.radians(clat))
    xs = [s["lon"] * mlon for s in samples]
    ys = [s["lat"] for s in samples]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = (maxx - minx) or 1e-6
    spany = (maxy - miny) or 1e-6
    span = max(spanx, spany) * 1.3
    cxw = (minx + maxx) / 2
    cyw = (miny + maxy) / 2
    pad = win_px * 0.12
    usable = win_px - 2 * pad

    def project(lon, lat):
        px = pad + ((lon * mlon - cxw) / span + 0.5) * usable
        py = pad + (0.5 - (lat - cyw) / span) * usable  # y down
        return px, py
    return project


# ----------------------------------------------------------------------------- drawing
COL_BG     = (10, 14, 22, 175)
COL_STROKE = (255, 255, 255, 28)
COL_TRACK  = (120, 200, 255, 110)
COL_TRAIL  = (90, 220, 255, 230)
COL_DOT    = (60, 235, 255, 255)
COL_LABEL  = (180, 195, 215, 235)
COL_VALUE  = (255, 255, 255, 255)
COL_UNIT   = (150, 170, 195, 230)
COL_ACCENT = (60, 235, 255, 255)


def rounded(draw, box, r, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


class Layout:
    """Geometry for the HUD, scaled to the overlay resolution."""
    def __init__(self, W, H):
        self.W, self.H = W, H
        s = H / 2160.0
        self.s = s
        self.margin = int(64 * s)
        self.map_px = int(560 * s)
        self.gap = int(26 * s)
        self.pad = int(26 * s)
        self.map_x = self.margin
        self.map_y = H - self.margin - self.map_px
        # card column to the right of the map
        self.card_w = int(360 * s)
        self.card_h = int((self.map_px - self.gap) / 2)
        self.card_x = self.map_x + self.map_px + self.gap
        self.card1_y = self.map_y
        self.card2_y = self.map_y + self.card_h + self.gap
        # dynamic bbox covers map + cards (with a little slack)
        self.bbox = (self.map_x - 4,
                     self.map_y - 4,
                     self.card_x + self.card_w + 4,
                     self.map_y + self.map_px + 4)
        # fonts
        self.f_label = font(int(34 * s), bold=True)
        self.f_value = font(int(96 * s), bold=True)
        self.f_unit  = font(int(34 * s), bold=True)
        self.f_tag   = font(int(26 * s), bold=True)


# ----------------------------------------------------------------------------- spline
def _catmull(p0, p1, p2, p3, u):
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


def build_dense_track(samples, subdiv=12):
    """Catmull-Rom spline through the (already lightly smoothed) GPS samples,
    so the drawn path is a flowing curve instead of straight 1 Hz segments.
    Returns a dense list of (t, lon, lat)."""
    n = len(samples)
    if n < 3 or subdiv < 2:
        return [(s["t"], s["lon"], s["lat"]) for s in samples]
    lon = [s["lon"] for s in samples]
    lat = [s["lat"] for s in samples]
    tt = [s["t"] for s in samples]
    dense = []
    for i in range(n - 1):
        i0, i1, i2, i3 = max(0, i - 1), i, i + 1, min(n - 1, i + 2)
        for k in range(subdiv):
            u = k / subdiv
            dense.append((tt[i1] + (tt[i2] - tt[i1]) * u,
                          _catmull(lon[i0], lon[i1], lon[i2], lon[i3], u),
                          _catmull(lat[i0], lat[i1], lat[i2], lat[i3], u)))
    dense.append((tt[-1], lon[-1], lat[-1]))
    return dense


def dense_pos_at(dense, t):
    """Position (lon, lat) on the dense spline at time t."""
    if t <= dense[0][0]:
        return dense[0][1], dense[0][2]
    if t >= dense[-1][0]:
        return dense[-1][1], dense[-1][2]
    lo, hi = 0, len(dense) - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if dense[m][0] <= t:
            lo = m
        else:
            hi = m
    a, b = dense[lo], dense[hi]
    f = (t - a[0]) / ((b[0] - a[0]) or 1.0)
    return a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f


def draw_static(W, H, lay, map_img, project, samples, dense, alt_label="ALT"):
    """Everything that never changes: panels, map tiles, full track, labels."""
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(base, "RGBA")
    s = lay.s

    # --- map panel
    mbox = (lay.map_x, lay.map_y, lay.map_x + lay.map_px, lay.map_y + lay.map_px)
    rad = int(22 * s)
    rounded(d, mbox, rad, fill=COL_BG)
    # map image clipped to rounded panel
    if map_img is not None:
        mask = Image.new("L", (lay.map_px, lay.map_px), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, lay.map_px - 1, lay.map_px - 1), radius=rad, fill=255)
        base.paste(map_img, (lay.map_x, lay.map_y), mask)
    rounded(d, mbox, rad, outline=COL_STROKE, width=max(1, int(2 * s)))

    # --- full GPS track on the map (clipped to panel)
    track_layer = Image.new("RGBA", (lay.map_px, lay.map_px), (0, 0, 0, 0))
    td = ImageDraw.Draw(track_layer, "RGBA")
    pts = [project(lo, la) for (_, lo, la) in dense]
    if len(pts) >= 2:
        td.line(pts, fill=COL_TRACK, width=max(2, int(4 * s)), joint="curve")
        sx, sy = pts[0]
        r0 = int(7 * s)
        td.ellipse((sx - r0, sy - r0, sx + r0, sy + r0),
                   outline=(255, 255, 255, 180), width=max(1, int(2 * s)))
    mask = Image.new("L", (lay.map_px, lay.map_px), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, lay.map_px - 1, lay.map_px - 1), radius=rad, fill=255)
    # composite using the track's OWN alpha (clipped to the rounded panel) so
    # the transparent areas don't paint over the map tiles
    track_layer.putalpha(ImageChops.multiply(track_layer.getchannel("A"), mask))
    base.alpha_composite(track_layer, (lay.map_x, lay.map_y))

    # "GPS" tag on the map
    d.text((lay.map_x + int(20 * s), lay.map_y + int(16 * s)),
           "GPS TRACK", font=lay.f_tag, fill=COL_LABEL)

    # --- the two value cards (labels are static, values drawn per-frame)
    for (cy, label) in ((lay.card1_y, "SPEED"), (lay.card2_y, alt_label)):
        cbox = (lay.card_x, cy, lay.card_x + lay.card_w, cy + lay.card_h)
        rounded(d, cbox, rad, fill=COL_BG, outline=COL_STROKE, width=max(1, int(2 * s)))
        d.text((lay.card_x + lay.pad, cy + lay.pad), label,
               font=lay.f_label, fill=COL_LABEL)
    return base


def draw_dynamic(tile, lay, project, samples, dense, t, units):
    """Draw the moving dot + live numbers onto a copy of the static bbox tile."""
    d = ImageDraw.Draw(tile, "RGBA")
    s = lay.s
    ox, oy = lay.bbox[0], lay.bbox[1]          # tile origin in full-frame coords
    cur = sample_at(samples, t)

    # ---- moving trail + dot on the map (follow the dense spline)
    rad = int(22 * s)
    map_layer = Image.new("RGBA", (lay.map_px, lay.map_px), (0, 0, 0, 0))
    md = ImageDraw.Draw(map_layer, "RGBA")
    past = [project(lo, la) for (tm, lo, la) in dense if tm <= t]
    plon, plat = dense_pos_at(dense, t)
    px, py = project(plon, plat)
    past.append((px, py))
    if len(past) >= 2:
        md.line(past, fill=COL_TRAIL, width=max(2, int(5 * s)), joint="curve")
    # glow + dot
    for rr, col in ((int(16 * s), (60, 235, 255, 70)),
                    (int(11 * s), (60, 235, 255, 130))):
        md.ellipse((px - rr, py - rr, px + rr, py + rr), fill=col)
    r = int(7 * s)
    md.ellipse((px - r, py - r, px + r, py + r), fill=COL_DOT,
               outline=(255, 255, 255, 255), width=max(1, int(2 * s)))
    mask = Image.new("L", (lay.map_px, lay.map_px), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, lay.map_px - 1, lay.map_px - 1), radius=rad, fill=255)
    map_layer.putalpha(ImageChops.multiply(map_layer.getchannel("A"), mask))
    tile.alpha_composite(map_layer, (lay.map_x - ox, lay.map_y - oy))

    # ---- values
    if units == "ms":
        spd, su = cur["hs"], "m/s"
    elif units == "mph":
        spd, su = cur["hs"] * 2.23694, "mph"
    else:
        spd, su = cur["hs"] * 3.6, "km/h"
    alt = cur["alt"]

    def put_value(cy, value_str, unit_str):
        vx = lay.card_x - ox + lay.pad
        vy = cy - oy + lay.card_h - int(118 * s)
        d.text((vx, vy), value_str, font=lay.f_value, fill=COL_VALUE)
        w = d.textlength(value_str, font=lay.f_value)
        d.text((vx + w + int(12 * s), vy + int(58 * s)), unit_str,
               font=lay.f_unit, fill=COL_UNIT)

    put_value(lay.card1_y, f"{spd:.1f}", su)
    put_value(lay.card2_y, f"{alt:.1f}", "m")
    return tile


# ----------------------------------------------------------------------------- render
_gpu_cache = None


def gpu_available():
    """True if ffmpeg has the Apple VideoToolbox encoders we use."""
    global _gpu_cache
    if _gpu_cache is None:
        try:
            enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                 capture_output=True, text=True).stdout
            _gpu_cache = ("h264_videotoolbox" in enc
                          and "prores_videotoolbox" in enc)
        except Exception:
            _gpu_cache = False
    return _gpu_cache


def render(path, out, units="kmh", scale=1.0, still=None, tiles="sat",
           start=0.0, seconds=None, smooth=9, alt="agl", home=None, full=False,
           composite=False, gpu=False):
    # don't write the output over the source file
    if out and os.path.abspath(out) == os.path.abspath(path):
        raise RuntimeError("output path is the same as the input; choose a "
                           "different -o output path")
    W0, H0, fps, dur = probe(path)
    if seconds is not None:
        dur = min(seconds, dur - start)
    W = int(round(W0 * scale)) // 2 * 2
    H = int(round(H0 * scale)) // 2 * 2
    print(f"  {os.path.basename(path)}: {W0}x{H0} {fps:.3f}fps {dur:.1f}s "
          f"-> overlay {W}x{H}")
    samples = extract_telemetry(path)
    smooth_track(samples, smooth)
    alt_label = apply_altitude(samples, alt, home)
    print(f"  telemetry: {len(samples)} samples "
          f"({samples[0]['t']:.0f}-{samples[-1]['t']:.0f}s)")

    lay = Layout(W, H)
    dense = build_dense_track(samples)
    map_img, project = build_map(samples, lay.map_px, tiles)
    if project is None:
        project = plain_projector(samples, lay.map_px)
    base = draw_static(W, H, lay, map_img, project, samples, dense, alt_label)
    base_arr = np.array(base)
    x0, y0, x1, y1 = lay.bbox
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    if (x1 - x0) % 2:
        x1 -= 1
    if (y1 - y0) % 2:
        y1 -= 1
    base_tile = base.crop((x0, y0, x1, y1))

    # ---- still-preview mode: composite over a real frame and save a PNG
    if still is not None:
        tile = base_tile.copy()
        draw_dynamic(tile, lay, project, samples, dense, still, units)
        frame_png = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(still), "-i", path,
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True).stdout
        bg = Image.open(io.BytesIO(frame_png)).convert("RGBA").resize((W, H))
        ov = base.copy()
        ov.paste(tile, (x0, y0), tile)
        comp = Image.alpha_composite(bg, ov).convert("RGB")
        comp.save(out)
        print(f"  wrote preview {out}")
        return

    # size of the RGBA frames we stream: HUD panel (default) or whole frame (--full)
    ow, oh = (W, H) if full else (x1 - x0, y1 - y0)
    nframes = int(round(dur * fps))
    if composite:
        # composite the HUD with the footage -> finished H.264 file
        ovx, ovy = (0, 0) if full else (x0, y0)
        print(f"  compositing footage + HUD -> {out}")
        cmd = ["ffmpeg", "-y", "-v", "error"]
        if start:
            cmd += ["-ss", f"{start}"]
        if seconds is not None:
            cmd += ["-t", f"{seconds}"]
        cmd += ["-i", path,                                  # 0: source footage
                "-f", "rawvideo", "-pix_fmt", "rgba",
                "-s", f"{ow}x{oh}", "-r", f"{fps}", "-i", "-",   # 1: HUD frames (stdin)
                "-filter_complex",
                f"[0:v]scale={W}:{H}[bg];[bg][1:v]overlay={ovx}:{ovy}[v]",
                "-map", "[v]", "-map", "0:a?"]
        if gpu:
            mbps = max(8, round(W * H * fps * 0.20 / 1e6))   # ~50M @4k30, ~12M @1080p30
            cmd += ["-c:v", "h264_videotoolbox", "-b:v", f"{mbps}M", "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
        cmd += ["-c:a", "aac", "-movflags", "+faststart", out]
    else:
        if full:
            print(f"  overlay output: {ow}x{oh} (full frame, auto-aligns)")
        else:
            print(f"  overlay output: {ow}x{oh} (panel only)")
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{ow}x{oh}", "-r", f"{fps}",
               "-i", "-"]
        if gpu:
            cmd += ["-c:v", "prores_videotoolbox", "-profile:v", "4444",
                    "-pix_fmt", "ayuv64le"]
        else:
            cmd += ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
        cmd += ["-r", f"{fps}", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    out_arr = base_arr.copy() if full else None
    try:
        for f in range(nframes):
            t = start + f / fps
            tile = base_tile.copy()
            draw_dynamic(tile, lay, project, samples, dense, t, units)
            if full:
                out_arr[y0:y1, x0:x1] = np.array(tile)
                ff.stdin.write(out_arr.tobytes())
            else:
                ff.stdin.write(np.array(tile).tobytes())
            if f % 150 == 0:
                pct = 100.0 * f / max(1, nframes)
                print(f"\r  rendering {pct:5.1f}%  ({f}/{nframes})", end="", flush=True)
        print(f"\r  rendering 100.0%  ({nframes}/{nframes})        ")
    finally:
        ff.stdin.close()
        ff.wait()
    print(f"  wrote {out}")


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description="DJI telemetry HUD overlay for drone footage")
    ap.add_argument("input", help="MP4 file, or a folder with --batch")
    ap.add_argument("-o", "--output", help="output .mov (default: <clip>.overlay.mov)")
    ap.add_argument("--units", choices=["kmh", "mph", "ms"], default="kmh")
    ap.add_argument("--tiles", choices=["sat", "osm"], default="sat",
                    help="map background: satellite (default) or street map")
    ap.add_argument("--fast", action="store_true", help="render at half resolution")
    ap.add_argument("--scale", type=float, default=None, help="resolution scale, e.g. 0.5")
    ap.add_argument("--still", type=float, default=None,
                    help="render ONE preview PNG at time T (seconds) instead of video")
    ap.add_argument("--batch", action="store_true", help="process every MP4 in a folder")
    ap.add_argument("--start", type=float, default=0.0, help="start time (s) for a test render")
    ap.add_argument("--seconds", type=float, default=None, help="duration (s) to render")
    ap.add_argument("--smooth", type=int, default=9,
                    help="GPS smoothing window in samples/seconds (0 = off, default 9)")
    ap.add_argument("--alt", choices=["agl", "asl", "rel"], default="agl",
                    help="altitude shown: agl=above ground below (default), "
                         "asl=above sea level, rel=above takeoff point")
    ap.add_argument("--home", help="override home point as 'LAT,LON' "
                    "(otherwise trilaterated from the distance-from-home data)")
    ap.add_argument("--full", action="store_true",
                    help="render the overlay at full video resolution (auto-aligns, "
                         "no positioning). Default renders only the HUD panel.")
    ap.add_argument("--composite", action="store_true",
                    help="composite the HUD with the footage into a finished H.264 "
                         "video (<clip>.hud.mp4) instead of a transparent overlay")
    ap.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None,
                    help="use Apple VideoToolbox hardware encoders (much faster on "
                         "Apple Silicon). On by default when available; "
                         "use --no-gpu to force CPU encoding")
    args = ap.parse_args()

    scale = args.scale if args.scale else (0.5 if args.fast else 1.0)

    # GPU on by default when the hardware encoders are present; --no-gpu forces CPU
    use_gpu = gpu_available() if args.gpu is None else args.gpu

    home = None
    if args.home:
        lat, lon = (float(v) for v in args.home.split(","))
        home = (lat, lon)

    if args.batch or os.path.isdir(args.input):
        files = sorted(f for f in os.listdir(args.input)
                       if f.lower().endswith(".mp4"))
        paths = [os.path.join(args.input, n) for n in files]
        # one shared home for the whole session, pooled across all clips
        if home is None and args.alt in ("agl", "asl"):
            home = session_home(paths)
            if home:
                print(f"session home: {home[0]:.5f}, {home[1]:.5f} "
                      f"(pooled from {len(files)} clips)")
        ext = ".hud.mp4" if args.composite else ".overlay.mov"
        for p in paths:
            out = os.path.splitext(p)[0] + ext
            try:
                render(p, out, args.units, scale, tiles=args.tiles,
                       smooth=args.smooth, alt=args.alt, home=home,
                       full=args.full, composite=args.composite, gpu=use_gpu)
            except Exception as e:
                print(f"  SKIP {os.path.basename(p)}: {e}")
        return

    if args.still is not None:
        out = args.output or (os.path.splitext(args.input)[0] + f".preview.png")
        render(args.input, out, args.units, scale, still=args.still, tiles=args.tiles,
               smooth=args.smooth, alt=args.alt, home=home)
    else:
        ext = ".hud.mp4" if args.composite else ".overlay.mov"
        out = args.output or (os.path.splitext(args.input)[0] + ext)
        render(args.input, out, args.units, scale, tiles=args.tiles,
               start=args.start, seconds=args.seconds, smooth=args.smooth,
               alt=args.alt, home=home, full=args.full, composite=args.composite,
               gpu=use_gpu)


if __name__ == "__main__":
    main()
