#!/usr/bin/env python3
"""
Recursive cleaner for .txt files.

Rules implemented:
- Remove everything before the first line that starts with the word "Valid" (case-insensitive).
- If there is a line containing "Graphics Available" (case-insensitive), remove the immediately preceding non-empty line (the name) and everything after it.

Usage:
    python clean_texts.py --root path/to/folder [--backup] [--dry-run] [--verbose]

The script is safe by default when used with --dry-run; use --backup to keep .bak copies of modified files.
"""
import argparse
import os
import re
import shutil
import datetime


def process_file(path, dry_run=True, backup=True, verbose=False):
    """Process one file. Returns (changed: bool, reason:str)."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            orig = f.read()
    except Exception as e:
        return False, f'read-error: {e}'

    lines = orig.splitlines(True)  # keep line endings

    # Find first line starting with 'valid' (case-insensitive)
    valid_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^\s*valid\b', line, re.I):
            valid_idx = i
            break

    if valid_idx is None:
        return False, 'no-valid-line'

    new_lines = lines[valid_idx:]

    # Find 'graphics available' line (case-insensitive)
    graphics_idx = None
    for i, line in enumerate(new_lines):
        # match 'graphics available', 'graphics are available', etc. (case-insensitive)
        if re.search(r'graphics\b(?:\s+are)?\s+available', line, re.I):
            graphics_idx = i
            break

    if graphics_idx is not None:
        # find previous non-empty line before graphics_idx
        prev_nonempty = None
        for j in range(graphics_idx - 1, -1, -1):
            if new_lines[j].strip():
                prev_nonempty = j
                break
        if prev_nonempty is not None:
            # keep everything before that name line
            new_lines = new_lines[:prev_nonempty]
        else:
            # nothing meaningful before graphics; remove everything
            new_lines = []

    new_content = ''.join(new_lines).rstrip() + '\n' if new_lines else ''

    if new_content == orig:
        return False, 'unchanged'

    if dry_run:
        if verbose:
            print(f'[dry-run] would modify: {path}')
        return True, 'would-modify'

    # create backup
    if backup:
        ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        bak_path = path + f'.bak.{ts}'
        try:
            shutil.copy2(path, bak_path)
        except Exception as e:
            return False, f'backup-failed: {e}'

    # write new content
    try:
        with open(path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(new_content)
    except Exception as e:
        return False, f'write-error: {e}'

    if verbose:
        print(f'[modified] {path}')
    return True, 'modified'


def walk_and_process(root, dry_run=True, backup=True, verbose=False):
    processed = 0
    changed = 0
    skipped = 0
    reasons = {}

    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith('.txt'):
                continue
            path = os.path.join(dirpath, fn)
            processed += 1
            ok, reason = process_file(path, dry_run=dry_run, backup=backup, verbose=verbose)
            if ok:
                changed += 1
            else:
                skipped += 1
            reasons.setdefault(reason, 0)
            reasons[reason] += 1

    return {'processed': processed, 'changed': changed, 'skipped': skipped, 'reasons': reasons}


def prune_folder_keep_slots(root, dry_run=True, backup=True, verbose=False, morning_hours=None, evening_hours=None):
    """Prune files in each directory: if >2 .txt files, keep latest morning and latest evening and delete others.

    morning_hours/evening_hours: lists of 2-digit hour strings (e.g. ['08'] and ['19','20']).
    Selection of 'latest' prefers a numeric suffix in filename (e.g. '08-11' -> 11), otherwise uses mtime.
    """
    # Default: morning = 00-12, evening = 13-23 (inclusive)
    if morning_hours is None:
        morning_hours = [f"{h:02d}" for h in range(0, 12)]
    if evening_hours is None:
        evening_hours = [f"{h:02d}" for h in range(12, 24)]

    total_dirs = 0
    total_files = 0
    total_deleted = 0

    for dirpath, dirnames, filenames in os.walk(root):
        txts = [fn for fn in filenames if fn.lower().endswith('.txt')]
        if len(txts) <= 2:
            continue
        total_dirs += 1
        total_files += len(txts)

        # build candidate metadata
        meta = []
        for fn in txts:
            path = os.path.join(dirpath, fn)
            base = os.path.splitext(fn)[0]
            m = re.match(r'^(\d{2})\D*(\d+)', base)
            if m:
                hour = m.group(1)
                suffix = int(m.group(2))
            else:
                # fallback: try first two digits as hour
                m2 = re.match(r'^(\d{2})', base)
                hour = m2.group(1) if m2 else ''
                suffix = None
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                mtime = 0
            meta.append({'fn': fn, 'path': path, 'hour': hour, 'suffix': suffix, 'mtime': mtime})

        # find latest morning and evening
        keep = set()

        def pick_latest(cands):
            if not cands:
                return None
            # prefer suffix if available
            with_suffix = [c for c in cands if c['suffix'] is not None]
            if with_suffix:
                best = max(with_suffix, key=lambda x: x['suffix'])
            else:
                best = max(cands, key=lambda x: x['mtime'])
            return best

        morning_cands = [c for c in meta if c['hour'] in morning_hours]
        evening_cands = [c for c in meta if c['hour'] in evening_hours]

        m_best = pick_latest(morning_cands)
        e_best = pick_latest(evening_cands)
        if m_best:
            keep.add(m_best['fn'])
        if e_best:
            keep.add(e_best['fn'])

        # If neither morning nor evening found, keep the most recent file overall
        if not keep:
            overall = max(meta, key=lambda x: (x['suffix'] if x['suffix'] is not None else -1, x['mtime']))
            keep.add(overall['fn'])

        # Delete others
        for c in meta:
            if c['fn'] in keep:
                if verbose:
                    print(f'[keep] {c["path"]}')
                continue
            # delete
            total_deleted += 1
            if dry_run:
                print(f'[dry-run] would delete: {c["path"]}')
            else:
                # backup if requested
                if backup:
                    try:
                        ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                        bak = c['path'] + f'.bak.{ts}'
                        shutil.copy2(c['path'], bak)
                        if verbose:
                            print(f'[backup] {c["path"]} -> {bak}')
                    except Exception as e:
                        print(f'[error] backup failed for {c["path"]}: {e}')
                try:
                    os.remove(c['path'])
                    if verbose:
                        print(f'[deleted] {c["path"]}')
                except Exception as e:
                    print(f'[error] delete failed for {c["path"]}: {e}')

    return {'dirs_examined': total_dirs, 'files_examined': total_files, 'files_deleted': total_deleted}


def main():
    p = argparse.ArgumentParser(description='Clean .txt files recursively (Valid/Graphics trimming).')
    p.add_argument('--root', '-r', required=True, help='Root directory to search for .txt files')
    p.add_argument('--dry-run', action='store_true', help='Do not write changes; just report')
    p.add_argument('--prune', action='store_true', help='Prune directories keeping only latest morning/evening files when >2 files present')
    p.add_argument('--morning-hours', help="Comma-separated hours or ranges to treat as 'morning' (default '00-12'). Examples: '00-12', '08,09', '07-09,11'.")
    p.add_argument('--evening-hours', help="Comma-separated hours or ranges to treat as 'evening' (default '13-23'). Examples: '13-23', '19,20', '18-20'.")
    p.add_argument('--no-backup', dest='backup', action='store_false', help="Don't create .bak before writing")
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = p.parse_args()

    def parse_hours(spec):
        """Parse a comma-separated hour list or ranges into a sorted list of 2-digit hour strings.

        Examples:
            '00-12' -> ['00','01',...,'12']
            '08,09,10' -> ['08','09','10']
            '20-02' -> ['20','21','22','23','00','01','02'] (wrap-around supported)
        Returns None when spec is falsy.
        """
        if not spec:
            return None
        out = set()
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-', 1)
                try:
                    a_i = int(a)
                    b_i = int(b)
                except ValueError:
                    continue
                # clamp to 0..24 (allow 24 for convenience); wrap-around if a > b
                a_i = max(0, min(24, a_i))
                b_i = max(0, min(24, b_i))
                if a_i <= b_i:
                    rng = range(a_i, b_i + 1)
                else:
                    rng = list(range(a_i, 25)) + list(range(0, b_i + 1))
                for h in rng:
                    out.add(f"{h:02d}")
            else:
                try:
                    h_i = int(part)
                except ValueError:
                    continue
                h_i = max(0, min(24, h_i))
                out.add(f"{h_i:02d}")
        return sorted(out)

    if not os.path.isdir(args.root):
        print(f'Error: root is not a directory: {args.root}')
        raise SystemExit(2)

    result = walk_and_process(args.root, dry_run=args.dry_run, backup=args.backup, verbose=args.verbose)

    print('\nSummary:')
    print(f"  processed: {result['processed']}")
    print(f"  changed (or would change in dry-run): {result['changed']}")
    print(f"  skipped: {result['skipped']}")
    print('  reasons:')
    for k, v in sorted(result['reasons'].items(), key=lambda x: -x[1]):
        print(f'    {k}: {v}')

    if args.prune:
        mh = parse_hours(args.morning_hours)
        eh = parse_hours(args.evening_hours)
        prune_res = prune_folder_keep_slots(args.root, dry_run=args.dry_run, backup=args.backup, verbose=args.verbose, morning_hours=mh, evening_hours=eh)
        print('\nPrune summary:')
        print(f"  dirs examined: {prune_res['dirs_examined']}")
        print(f"  files examined: {prune_res['files_examined']}")
        print(f"  files deleted (or would delete in dry-run): {prune_res['files_deleted']}")


if __name__ == '__main__':
    main()
