#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# -----------------------------
# Config (can be overridden by env vars)
# -----------------------------
URLS_TXT = os.environ.get("YTDLP_URLS_TXT", "youtube_urls.txt")
OUTPUT_DIR = Path(os.environ.get("YTDLP_OUTPUT_DIR", "output_mp3"))
ARCHIVE_FILE = Path(os.environ.get("YTDLP_ARCHIVE_FILE", str(OUTPUT_DIR / "downloaded.txt")))

# Browser for cookies: "chrome", "chromium", "firefox", ...
COOKIES_FROM_BROWSER = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "chrome")

MIN_SLEEP_BETWEEN_VIDEOS = float(os.environ.get("YTDLP_MIN_SLEEP", "8"))
MAX_SLEEP_BETWEEN_VIDEOS = float(os.environ.get("YTDLP_MAX_SLEEP", "20"))

# Retry behavior
MAX_RETRIES_PER_URL = int(os.environ.get("YTDLP_MAX_RETRIES", "3"))
BASE_BACKOFF_SECONDS = float(os.environ.get("YTDLP_BASE_BACKOFF", "5"))

# Optional: limit speed to look less bot-like (e.g. "2M" or "500K"), empty disables
RATE_LIMIT = os.environ.get("YTDLP_RATE_LIMIT", "").strip()

# -----------------------------
# Helpers
# -----------------------------
def normalize_youtube_url(url: str) -> str:
    """Normalize youtu.be and youtube.com URLs to https://www.youtube.com/watch?v=VIDEO_ID"""
    url = (url or "").strip()
    if not url:
        return url

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()

    if "youtu.be" in host and parsed.path.strip("/"):
        video_id = parsed.path.strip("/")
        return f"https://www.youtube.com/watch?v={video_id}"

    if "youtube.com" in host:
        qs = parse_qs(parsed.query)
        video_ids = qs.get("v")
        if video_ids:
            return f"https://www.youtube.com/watch?v={video_ids[0]}"

    return url


def load_urls(path: str):
    p = Path(path)
    if not p.exists():
        print(f"❌ URL-Datei '{path}' nicht gefunden.")
        return []

    urls = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw and not raw.startswith("#"):
                urls.append(normalize_youtube_url(raw))
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u and u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def polite_sleep_between_videos(i: int, total: int):
    """Sleep between videos to reduce throttling/bot triggers."""
    if i >= total - 1:
        return
    t = random.uniform(MIN_SLEEP_BETWEEN_VIDEOS, MAX_SLEEP_BETWEEN_VIDEOS)
    print(f"\n⏳ Pause: {t:.1f} Sekunden...")
    time.sleep(t)


def build_ytdlp_cmd(url: str):
    """
    Build a robust yt-dlp command for:
    - cookies from browser
    - extract audio -> mp3
    - download archive (skip already downloaded)
    - safe filenames
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",

        # cookies (fixes "Sign in to confirm you're not a bot" in many cases)
        "--cookies-from-browser", COOKIES_FROM_BROWSER,

        # resume + skip already done
        "--download-archive", str(ARCHIVE_FILE),
        "--continue",
        "--no-overwrites",

        # stable, safe filenames
        "--restrict-filenames",
        "--windows-filenames",  # avoids weird characters
        "--paths", str(OUTPUT_DIR),

        # include id to avoid collisions
        "-o", "%(title)s [%(id)s].%(ext)s",

        # keep going if one URL fails
        "--ignore-errors",

        # cleaner progress in logs
        "--newline",
    ]

    if RATE_LIMIT:
        cmd += ["--limit-rate", RATE_LIMIT]

    cmd += [url]
    return cmd


def looks_like_bot_block(stderr_text: str) -> bool:
    s = (stderr_text or "").lower()
    return (
        "sign in to confirm you’re not a bot" in s
        or "sign in to confirm you're not a bot" in s
        or "confirm you’re not a bot" in s
        or "captcha" in s
        or "too many requests" in s
        or "http error 429" in s
    )


def run_with_retries(url: str) -> bool:
    """
    Runs yt-dlp with retries and exponential backoff.
    Returns True if success, False if ultimately failed.
    """
    cmd = build_ytdlp_cmd(url)
    for attempt in range(1, MAX_RETRIES_PER_URL + 1):
        print(f"\n▶ Lade: {url}")
        if attempt > 1:
            print(f"🔁 Retry {attempt}/{MAX_RETRIES_PER_URL}")

        try:
            # capture output so we can print helpful hints on failure
            proc = subprocess.run(
                cmd,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # print a little bit of stdout for transparency (optional)
            if proc.stdout.strip():
                print(proc.stdout.strip())
            print("  ✅ Fertig")
            return True

        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            stdout = (e.stdout or "").strip()

            if stdout:
                print(stdout)
            if stderr:
                print(stderr)

            # Helpful hints
            if looks_like_bot_block(stderr):
                print("\n🧩 Hinweis: Das sieht nach YouTube Bot/Rate-Limit aus.")
                print("   - Stelle sicher, dass du im Browser eingeloggt bist (YouTube) und ggf. eine Bestätigung erledigt hast.")
                print("   - Falls es weiter passiert: Deno installieren (JS runtime) und/oder Netzwerk wechseln (Hotspot).")
                print("   - Optional: setze YTDLP_RATE_LIMIT z.B. auf '2M'.")

            # Backoff
            if attempt < MAX_RETRIES_PER_URL:
                backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 2.0)
                print(f"⏳ Warte {backoff:.1f}s vor nächstem Versuch...")
                time.sleep(backoff)
            else:
                print("  ❌ endgültig fehlgeschlagen")
                return False


def print_quick_deps_hints():
    print("ℹ️  Setup-Tipps (falls Warnungen auftauchen):")
    print("   - secretstorage für bessere Cookie-Entschlüsselung:")
    print("       python3 -m pip install -U secretstorage")
    print("   - Deno (JS runtime) für stabilere YouTube-Extraktion:")
    print("       sudo snap install deno")
    print("       deno --version")
    print("")


def main():
    urls = load_urls(URLS_TXT)
    if not urls:
        print("Keine gültigen URLs gefunden.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_FILE.parent.mkdir(parents=True, exist_ok=True)

    print_quick_deps_hints()

    total = len(urls)
    print(f"✅ {total} URL(s) gefunden.")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"🍪 Cookies aus Browser: {COOKIES_FROM_BROWSER}")
    if RATE_LIMIT:
        print(f"🐢 Rate limit: {RATE_LIMIT}")
    print(f"🗃️  Archive: {ARCHIVE_FILE}")
    print("")

    ok = 0
    fail = 0
    failed_urls = []

    for i, url in enumerate(urls):
        print("============================")
        print(f"Video {i+1}/{total}")
        print("============================")

        success = run_with_retries(url)
        if success:
            ok += 1
        else:
            fail += 1
            failed_urls.append(url)

        polite_sleep_between_videos(i, total)

    print("\n============================")
    print("🏁 Fertig")
    print("============================")
    print(f"✅ Erfolgreich: {ok}")
    print(f"❌ Fehlgeschlagen: {fail}")

    if failed_urls:
        failed_file = OUTPUT_DIR / "failed_urls.txt"
        failed_file.write_text("\n".join(failed_urls) + "\n", encoding="utf-8")
        print(f"\n📝 Fehlgeschlagene URLs gespeichert in: {failed_file}")


if __name__ == "__main__":
    # Nur für Inhalte nutzen, die du rechtlich herunterladen darfst.
    main()
