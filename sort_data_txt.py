"""
Match .nc files (in nested folders) to text-information folders named YYYY-MM-DD.

Usage examples (PowerShell):
  # Dry-run: only show matches
  python sort_data_txt.py --nc-root C:\path\to\nc_root --text-root C:\path\to\text_root --dry-run

  # Copy matched .nc files into their text folder
  python sort_data_txt.py --nc-root C:\nc --text-root C:\texts --copy --output mapping.csv

  # Move matched .nc files (destructive)
  python sort_data_txt.py --nc-root C:\nc --text-root C:\texts --move

The script will try several strategies to extract the date for each .nc file:
 - Look for an 8-digit YYYYMMDD in the filename
 - Look for YYYY, MM, DD components in the file path
 - Look for YYYY and MM in path and YYYYMMDD in filename

The default behavior (no --move/--copy) is to produce a CSV mapping and print a summary.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from datetime import datetime
from typing import Iterable, List, Optional, Tuple


DATE_RE = re.compile(r"(\d{8})")  # YYYYMMDD


def find_nc_files(root: str) -> Iterable[str]:
	for dirpath, _, filenames in os.walk(root):
		for fn in filenames:
			if fn.lower().endswith('.nc'):
				yield os.path.join(dirpath, fn)


def date_from_filename(fname: str) -> Optional[datetime]:
	m = DATE_RE.search(fname)
	if m:
		s = m.group(1)
		try:
			return datetime.strptime(s, "%Y%m%d")
		except ValueError:
			return None
	return None


def date_from_path_components(path: str) -> Optional[datetime]:
	# Look for consecutive components that look like year(4)/month(2)/day(2)
	comps = [c for c in path.replace('\\', '/').split('/') if c]
	for i in range(len(comps) - 2):
		y, m, d = comps[i:i+3]
		if re.fullmatch(r"\d{4}", y) and re.fullmatch(r"\d{1,2}", m) and re.fullmatch(r"\d{1,2}", d):
			try:
				return datetime(int(y), int(m), int(d))
			except ValueError:
				continue
	# fallback: try find year and month (no day)
	for i in range(len(comps) - 1):
		y, m = comps[i:i+2]
		if re.fullmatch(r"\d{4}", y) and re.fullmatch(r"\d{1,2}", m):
			# day unknown
			try:
				return datetime(int(y), int(m), 1)
			except ValueError:
				continue
	return None


def date_for_nc(nc_path: str) -> Optional[datetime]:
	# Prefer filename-based full date
	fn = os.path.basename(nc_path)
	d = date_from_filename(fn)
	if d:
		return d
	# Next try path components
	d = date_from_path_components(nc_path)
	return d


def format_date_folder(dt: datetime) -> str:
	return dt.strftime('%Y-%m-%d')


def ensure_output_dir(path: str) -> None:
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
	p = argparse.ArgumentParser(description='Match .nc files to text folders named YYYY-MM-DD')
	p.add_argument('--nc-root', required=True, help='Root directory that contains nested .nc files')
	p.add_argument('--text-root', required=True, help='Root directory that contains text folders named YYYY-MM-DD')
	p.add_argument('--output', default='mapping.csv', help='Output CSV path (nc_path, matched_text_folder_or_empty)')
	p.add_argument('--move', action='store_true', help='Move matched .nc files into their text folder')
	p.add_argument('--copy', action='store_true', help='Copy matched .nc files into their text folder')
	p.add_argument('--dry-run', action='store_true', help='Do not perform move/copy, only report')
	p.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
	args = p.parse_args(argv)

	if args.move and args.copy:
		p.error('Choose either --move or --copy, not both')

	nc_root = os.path.abspath(args.nc_root)
	text_root = os.path.abspath(args.text_root)

	rows: List[Tuple[str, str]] = []
	matched = 0
	moved = 0
	copied = 0
	unmatched_list: List[str] = []

	for nc in find_nc_files(nc_root):
		dt = date_for_nc(nc)
		matched_folder = ''
		if dt:
			# If date has day=1 from fallback, still format as YYYY-MM-DD (day 01)
			candidate = format_date_folder(dt)
			candidate_path = os.path.join(text_root, candidate)
			if os.path.isdir(candidate_path):
				matched_folder = candidate_path
		if not matched_folder:
			# No match found
			rows.append((nc, ''))
			unmatched_list.append(nc)
			if args.verbose:
				print(f'UNMATCHED: {nc}')
			continue

		matched += 1
		rows.append((nc, matched_folder))

		if args.dry_run:
			if args.verbose:
				print(f'[DRY] Would match: {nc} -> {matched_folder}')
			continue

		# perform action
		dest = os.path.join(matched_folder, os.path.basename(nc))
		# Avoid overwriting a file with same name: add suffix if exists
		if os.path.exists(dest):
			base, ext = os.path.splitext(dest)
			i = 1
			new_dest = f"{base}_{i}{ext}"
			while os.path.exists(new_dest):
				i += 1
				new_dest = f"{base}_{i}{ext}"
			dest = new_dest

		try:
			if args.move:
				shutil.move(nc, dest)
				moved += 1
				if args.verbose:
					print(f'MOVED: {nc} -> {dest}')
			elif args.copy:
				shutil.copy2(nc, dest)
				copied += 1
				if args.verbose:
					print(f'COPIED: {nc} -> {dest}')
		except Exception as exc:
			print(f'ERROR handling {nc} -> {dest}: {exc}')

	# write CSV output
	try:
		ensure_output_dir(args.output)
		with open(args.output, 'w', newline='', encoding='utf-8') as f:
			w = csv.writer(f)
			w.writerow(['nc_path', 'matched_text_folder'])
			for r in rows:
				w.writerow(r)
	except Exception as exc:
		print(f'ERROR writing output {args.output}: {exc}')
		return 2

	# Summary
	print('\nSummary:')
	print(f'  total .nc files scanned: {len(rows) + len(unmatched_list)}')
	print(f'  matched: {matched}')
	print(f'  unmatched: {len(unmatched_list)}')
	print(f'  mapping saved to: {os.path.abspath(args.output)}')
	if args.move:
		print(f'  moved: {moved}')
	if args.copy:
		print(f'  copied: {copied}')

	if unmatched_list:
		print('\nUnmatched examples (up to 10):')
		for u in unmatched_list[:10]:
			print('  ' + u)

	return 0


if __name__ == '__main__':
	raise SystemExit(main())
