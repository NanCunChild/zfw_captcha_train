# label_tool.py
"""Manual labeling tool for captcha images that failed automatic recognition.

Scans the data directory for ``Unknown_*`` directories, displays each image
in a tkinter window, and lets the user type the correct 4-digit label. The
image is then moved to the appropriate label directory with sequential naming.

Usage:
    python src/label_tool.py
    python src/label_tool.py --data-dir /path/to/data/captcha_get
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def find_unknown_dirs(data_dir: str) -> list[str]:
    """Find all Unknown_* directories sorted alphabetically."""
    pattern = os.path.join(data_dir, 'Unknown_*')
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    dirs.sort()
    return dirs


def get_images_in_dir(directory: str) -> list[str]:
    """Get all image files in a directory, sorted."""
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(directory, ext)))
    images.sort()
    return images


def next_filename(target_dir: str) -> str:
    """Determine the next sequential filename (e.g. 0001.png, 0002.png)."""
    os.makedirs(target_dir, exist_ok=True)
    existing = glob.glob(os.path.join(target_dir, '*.png'))
    max_num = 0
    for f in existing:
        name = os.path.splitext(os.path.basename(f))[0]
        if name.isdigit():
            max_num = max(max_num, int(name))
    return os.path.join(target_dir, f'{max_num + 1:04d}.png')


def extract_guess(dirname: str) -> str:
    """Extract the auto-tool's guess from 'Unknown_XXXX' directory name."""
    match = re.search(r'Unknown_(.+)$', os.path.basename(dirname))
    return match.group(1) if match else ''


class LabelApp:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.unknown_dirs = find_unknown_dirs(data_dir)

        if not self.unknown_dirs:
            print('No Unknown_* directories found. Nothing to label.')
            sys.exit(0)

        self.all_tasks: list[tuple[str, str, str]] = []
        for d in self.unknown_dirs:
            guess = extract_guess(d)
            for img_path in get_images_in_dir(d):
                self.all_tasks.append((img_path, d, guess))

        if not self.all_tasks:
            print('No images found in Unknown_* directories.')
            sys.exit(0)

        self.current_idx = 0
        self.stats = {'labeled': 0, 'skipped': 0}

        self.root = tk.Tk()
        self.root.title('Captcha Label Tool')
        self.root.resizable(False, False)

        self._build_ui()
        self._load_current()

        self.root.protocol('WM_DELETE_WINDOW', self._on_quit)
        self.root.mainloop()

    def _build_ui(self):
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack()

        self.progress_label = tk.Label(main_frame, text='', font=('monospace', 11))
        self.progress_label.pack(pady=(0, 10))

        self.image_label = tk.Label(main_frame, borderwidth=2, relief='solid')
        self.image_label.pack(pady=10)

        info_frame = tk.Frame(main_frame)
        info_frame.pack(pady=5)

        tk.Label(info_frame, text='Auto guess:', font=('monospace', 10)).grid(row=0, column=0, sticky='e')
        self.guess_label = tk.Label(info_frame, text='', font=('monospace', 12, 'bold'), fg='red')
        self.guess_label.grid(row=0, column=1, sticky='w', padx=(5, 0))

        tk.Label(info_frame, text='Source:', font=('monospace', 10)).grid(row=1, column=0, sticky='e')
        self.source_label = tk.Label(info_frame, text='', font=('monospace', 9), fg='gray')
        self.source_label.grid(row=1, column=1, sticky='w', padx=(5, 0))

        entry_frame = tk.Frame(main_frame)
        entry_frame.pack(pady=15)

        tk.Label(entry_frame, text='Correct label:', font=('monospace', 11)).pack(side='left')
        self.entry = tk.Entry(entry_frame, font=('monospace', 14, 'bold'), width=6, justify='center')
        self.entry.pack(side='left', padx=(10, 0))
        self.entry.bind('<Return>', lambda e: self._submit())

        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(pady=10)

        self.submit_btn = tk.Button(btn_frame, text='Submit (Enter)', command=self._submit,
                                    width=14, font=('monospace', 10))
        self.submit_btn.pack(side='left', padx=5)

        self.skip_btn = tk.Button(btn_frame, text='Skip (Esc)', command=self._skip,
                                  width=14, font=('monospace', 10))
        self.skip_btn.pack(side='left', padx=5)

        self.quit_btn = tk.Button(btn_frame, text='Quit (Q)', command=self._on_quit,
                                  width=14, font=('monospace', 10))
        self.quit_btn.pack(side='left', padx=5)

        self.root.bind('<Escape>', lambda e: self._skip())
        self.root.bind('q', lambda e: self._on_quit())
        self.root.bind('Q', lambda e: self._on_quit())

    def _load_current(self):
        if self.current_idx >= len(self.all_tasks):
            self._finish()
            return

        img_path, src_dir, guess = self.all_tasks[self.current_idx]

        self.progress_label.config(
            text=f'[{self.current_idx + 1}/{len(self.all_tasks)}]  '
                 f'Labeled: {self.stats["labeled"]}  Skipped: {self.stats["skipped"]}'
        )

        img = Image.open(img_path).convert('RGB')
        display_scale = 3
        display_size = (img.width * display_scale, img.height * display_scale)
        img_display = img.resize(display_size, Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(img_display)
        self.image_label.config(image=self.tk_image)

        self.guess_label.config(text=guess)
        self.source_label.config(text=os.path.basename(img_path))

        self.entry.delete(0, tk.END)
        self.entry.focus_set()

    def _submit(self):
        label = self.entry.get().strip()

        if not re.match(r'^\d{4}$', label):
            messagebox.showwarning('Invalid', 'Please enter exactly 4 digits (0-9).')
            self.entry.focus_set()
            return

        img_path, src_dir, guess = self.all_tasks[self.current_idx]
        target_dir = os.path.join(self.data_dir, label)
        dest_path = next_filename(target_dir)

        shutil.move(img_path, dest_path)
        self.stats['labeled'] += 1

        remaining = get_images_in_dir(src_dir)
        if not remaining:
            os.rmdir(src_dir)

        self.current_idx += 1
        self._load_current()

    def _skip(self):
        self.stats['skipped'] += 1
        self.current_idx += 1
        self._load_current()

    def _finish(self):
        messagebox.showinfo(
            'Done',
            f'All images processed.\n\n'
            f'Labeled: {self.stats["labeled"]}\n'
            f'Skipped: {self.stats["skipped"]}'
        )
        self.root.destroy()

    def _on_quit(self):
        if messagebox.askyesno('Quit', f'Quit now?\n\nLabeled: {self.stats["labeled"]}\nSkipped: {self.stats["skipped"]}'):
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='Manual labeling tool for Unknown_* captcha images')
    parser.add_argument('--data-dir', default=config.DATA_DIR,
                        help='Path to data directory containing label folders and Unknown_* folders')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f'Error: Data directory not found: {args.data_dir}')
        print('Please specify the correct path with --data-dir')
        sys.exit(1)

    print(f'Scanning: {args.data_dir}')
    LabelApp(args.data_dir)


if __name__ == '__main__':
    main()
