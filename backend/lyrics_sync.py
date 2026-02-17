from __future__ import annotations

from typing import List, Dict, Any

from rapidfuzz import fuzz


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("’", "'")
    return " ".join(s.split())


def _tokenize(s: str) -> list[str]:
    s = normalize_text(s)
    out: list[str] = []
    cur: list[str] = []
    for ch in s:
        if ch.isalnum() or ch == "'":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def align_lyrics_to_words(
    lyric_lines: List[str],
    words: List[Dict[str, Any]],
    search_window: int = 60,
) -> List[Dict[str, Any]]:
    """Greedy alignment (fallback)."""
    word_tokens = [normalize_text(w.get("word", "")).strip() for w in words]
    out: list[dict] = []
    i = 0

    for line in lyric_lines:
        target_tokens = _tokenize(line)
        if not target_tokens:
            continue

        target = " ".join(target_tokens)
        best = None  # (adj_score, sidx, eidx, raw)

        for sidx in range(i, min(len(words), i + search_window)):
            acc_tokens = []
            for eidx in range(sidx, min(len(words), sidx + 50)):
                tok = word_tokens[eidx]
                if tok:
                    acc_tokens.append(tok)
                if not acc_tokens:
                    continue

                acc = " ".join(acc_tokens)
                raw = fuzz.partial_ratio(target, acc)

                jump = max(0, sidx - i)
                adj = raw - (jump * 0.15)

                if best is None or adj > best[0]:
                    best = (adj, sidx, eidx, raw)
                if raw >= 98:
                    break

        if best is None:
            t = float(words[i].get("start", 0.0)) if i < len(words) else 0.0
            out.append({"time": t, "text": line, "score": 0})
            continue

        _adj, sidx, eidx, raw = best
        t = float(words[sidx].get("start", 0.0)) if sidx < len(words) else 0.0
        out.append({"time": t, "text": line, "score": int(raw)})
        i = max(i, eidx + 1)

    last_t = 0.0
    for item in out:
        if item["time"] < last_t:
            item["time"] = last_t
        last_t = item["time"]

    return out


def align_lyrics_to_words_dp(
    lyric_lines: List[str],
    words: List[Dict[str, Any]],
    *,
    k_candidates: int = 6,
    max_start_lookahead: int = 220,
    min_words_per_line: int = 2,
) -> List[Dict[str, Any]]:
    """Global DP/Viterbi alignment using word timestamps."""

    if not lyric_lines or not words:
        return []

    word_tokens = [normalize_text(w.get("word", "")).strip() for w in words]

    candidates: list[list[tuple[int, int]]] = []  # per line: [(sidx, raw_score)]

    n = len(lyric_lines)
    first_t = float(words[0].get("start", 0.0))
    last_t = float(words[-1].get("start", first_t))
    span = max(1e-6, last_t - first_t)

    target_times: list[float] = []

    for idx, line in enumerate(lyric_lines):
        toks = _tokenize(line)
        if not toks:
            candidates.append([])
            target_times.append(first_t + (idx / max(1, n - 1)) * span)
            continue

        target = " ".join(toks)
        expected = max(min_words_per_line, len(toks))

        # approximate expected time for this line
        target_t = first_t + (idx / max(1, n - 1)) * span
        target_times.append(target_t)

        # binary search center index
        center = 0
        lo, hi = 0, len(words) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            mt = float(words[mid].get("start", 0.0))
            if mt < target_t:
                lo = mid + 1
                center = mid
            else:
                hi = mid - 1

        window = max_start_lookahead * 4
        start = max(0, center - window)
        end = min(len(words), center + window)

        scored: list[tuple[int, int]] = []
        stride = 2
        for sidx in range(start, end, stride):
            acc_tokens = []
            for eidx in range(sidx, min(len(words), sidx + expected + 25)):
                tok = word_tokens[eidx]
                if tok:
                    acc_tokens.append(tok)
                if len(acc_tokens) < min_words_per_line:
                    continue
                acc = " ".join(acc_tokens)
                raw = fuzz.partial_ratio(target, acc)
                if raw >= 55:
                    scored.append((sidx, raw))
                if raw >= 98:
                    break

        scored.sort(key=lambda x: x[1], reverse=True)
        top: list[tuple[int, int]] = []
        seen = set()
        for sidx, raw in scored:
            bucket = sidx // 6
            if bucket in seen:
                continue
            seen.add(bucket)
            top.append((sidx, raw))
            if len(top) >= k_candidates:
                break

        candidates.append(top)

    # DP
    import math

    dp: list[dict[int, tuple[float, int | None, int]]] = []

    for i_line in range(n):
        dp.append({})
        line_cands = candidates[i_line]
        if not line_cands:
            line_cands = [(0, 0)]
            candidates[i_line] = line_cands

        for ci, (sidx, raw) in enumerate(line_cands):
            emission = (100 - raw)

            # strong prior: keep each line near its expected timeline position
            try:
                cand_t = float(words[sidx].get("start", 0.0))
            except Exception:
                cand_t = 0.0
            prior = abs(cand_t - target_times[i_line]) * 0.9  # weight tuned empirically
            emission = emission + prior

            if i_line == 0:
                dp[i_line][ci] = (emission + (sidx * 0.02), None, sidx)
                continue

            best_cost = math.inf
            best_prev: int | None = None
            for pj, (psidx, _praw) in enumerate(candidates[i_line - 1] or [(0, 0)]):
                prev_state = dp[i_line - 1].get(pj)
                if not prev_state:
                    continue
                prev_cost, _pp, _ = prev_state

                if sidx <= psidx:
                    continue

                jump = sidx - psidx
                trans = (jump * 0.02)
                if jump > max_start_lookahead:
                    trans += (jump - max_start_lookahead) * 0.06

                cost = prev_cost + emission + trans
                if cost < best_cost:
                    best_cost = cost
                    best_prev = pj

            if best_prev is None:
                prev_best = min(dp[i_line - 1].items(), key=lambda kv: kv[1][0])[0]
                best_prev = prev_best
                best_cost = best_cost + 500

            dp[i_line][ci] = (best_cost, best_prev, sidx)

    # backtrack
    last_states = dp[-1]
    best_last = min(last_states.items(), key=lambda kv: kv[1][0])[0]

    chosen_sidxs = [0] * n
    cur = best_last
    for i_line in range(n - 1, -1, -1):
        _cost, prev, sidx = dp[i_line][cur]
        chosen_sidxs[i_line] = sidx
        if prev is None:
            break
        cur = prev

    for i_line in range(1, n):
        if chosen_sidxs[i_line] <= chosen_sidxs[i_line - 1]:
            chosen_sidxs[i_line] = min(len(words) - 1, chosen_sidxs[i_line - 1] + 1)

    out: list[dict] = []
    last_time = 0.0
    for i_line, sidx in enumerate(chosen_sidxs):
        t = float(words[sidx].get("start", 0.0)) if 0 <= sidx < len(words) else 0.0
        if t < last_time:
            t = last_time
        last_time = t

        raw_score = 0
        for cs, raw in candidates[i_line]:
            if cs == sidx:
                raw_score = raw
                break

        out.append({"time": t, "text": lyric_lines[i_line], "score": int(raw_score)})

    return out


def align_lyrics_to_words_anchored(
    lyric_lines: List[str],
    words: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Anchor-based alignment for repeated sections.

    Uses a few strong anchors spread over the song, then runs DP within the
    ranges between anchors. This reduces "matched the wrong chorus" errors.
    """

    if not lyric_lines or not words:
        return []

    word_tokens = [normalize_text(w.get("word", "")).strip() for w in words]

    def best_start_for_line(line: str) -> tuple[int, int]:
        toks = _tokenize(line)
        if not toks:
            return (0, 0)
        target = " ".join(toks)
        expected = max(2, len(toks))
        best_score = 0
        best_sidx = 0
        stride = 3
        for sidx in range(0, len(words), stride):
            acc_tokens = []
            for eidx in range(sidx, min(len(words), sidx + expected + 25)):
                tok = word_tokens[eidx]
                if tok:
                    acc_tokens.append(tok)
                if len(acc_tokens) < 2:
                    continue
                acc = " ".join(acc_tokens)
                raw = fuzz.partial_ratio(target, acc)
                if raw > best_score:
                    best_score = raw
                    best_sidx = sidx
                if raw >= 99:
                    break
        return best_sidx, best_score

    n = len(lyric_lines)

    # Pick more anchors (spread out) + add some long/unique-looking lines.
    spread = max(1, n // 16)
    anchor_indices = set(range(0, n, spread))
    anchor_indices.add(0)
    anchor_indices.add(n - 1)

    # add longest lines as anchors (helps with repeated chorus ambiguity)
    ranked = sorted(
        range(n),
        key=lambda i: len(_tokenize(lyric_lines[i])),
        reverse=True,
    )
    for i in ranked[:12]:
        anchor_indices.add(i)

    anchor_indices = sorted(anchor_indices)

    anchors: list[tuple[int, int, int]] = []  # (line_idx, word_idx, score)
    for li in anchor_indices:
        sidx, score = best_start_for_line(lyric_lines[li])
        if score >= 85:
            anchors.append((li, sidx, score))

    # force monotonic anchors
    anchors.sort(key=lambda x: x[0])
    filtered: list[tuple[int, int, int]] = []
    last_w = -1
    for li, wi, sc in anchors:
        if wi > last_w:
            filtered.append((li, wi, sc))
            last_w = wi
    anchors = filtered

    if len(anchors) < 2:
        return align_lyrics_to_words_dp(lyric_lines, words)

    def run_dp(line_a: int, line_b: int, word_a: int, word_b: int):
        seg_lines = lyric_lines[line_a:line_b]
        seg_words = words[word_a:word_b]
        if not seg_lines or not seg_words:
            return []
        return align_lyrics_to_words_dp(seg_lines, seg_words)

    out: list[dict] = []

    # before first anchor
    li0, wi0, _ = anchors[0]
    if li0 > 0:
        out.extend(run_dp(0, li0, 0, min(len(words), wi0 + 1200)))

    # between anchors
    for (lia, wia, _), (lib, wib, _) in zip(anchors, anchors[1:]):
        if lib <= lia:
            continue
        ws = max(0, wia - 800)
        we = min(len(words), wib + 800)
        out.extend(run_dp(lia, lib, ws, we))

    # after last anchor
    lil, wil, _ = anchors[-1]
    if lil < n:
        out.extend(run_dp(lil, n, max(0, wil - 800), len(words)))

    # preserve lyric order + enforce monotonic time
    # (out from segments already monotonic inside; we just stitch carefully)
    stitched: list[dict] = []
    last_t = 0.0
    for item in out:
        t = float(item.get("time", 0.0))
        if t < last_t:
            t = last_t
            item["time"] = t
        last_t = t
        stitched.append(item)

    # Keep original order (not time order) to avoid shuffled repeats
    # If we lost/duplicated lines due to segmentation, fall back.
    if len(stitched) < max(1, int(0.6 * len(lyric_lines))):
        return align_lyrics_to_words_dp(lyric_lines, words)

    return stitched


def align_lyrics_to_segments(
    lyric_lines: List[str],
    segments: List[Dict[str, Any]],
    search_window: int = 10,
) -> List[Dict[str, Any]]:
    """Fallback alignment using segment text only."""

    seg_texts = [normalize_text(s.get("text", "")) for s in segments]
    out = []
    i = 0

    for line in lyric_lines:
        target = normalize_text(line)
        if not target:
            continue

        best = None  # (score, start_idx, end_idx)
        for sidx in range(i, min(len(segments), i + search_window)):
            acc = ""
            for eidx in range(sidx, min(len(segments), sidx + 5)):
                acc = (acc + " " + seg_texts[eidx]).strip()
                if not acc:
                    continue
                score = fuzz.partial_ratio(target, acc)
                if best is None or score > best[0]:
                    best = (score, sidx, eidx)
                if score >= 95:
                    break

        if best is None:
            t = float(segments[i]["start"]) if i < len(segments) else 0.0
            out.append({"time": t, "text": line, "score": 0})
            continue

        score, sidx, eidx = best
        t = float(segments[sidx]["start"]) if sidx < len(segments) else 0.0
        out.append({"time": t, "text": line, "score": int(score)})
        i = max(i, eidx + 1)

    last_t = 0.0
    for item in out:
        if item["time"] < last_t:
            item["time"] = last_t
        last_t = item["time"]

    return out
