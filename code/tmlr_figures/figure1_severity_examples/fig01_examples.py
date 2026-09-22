"""Fig 1 example-image panels, rebuilt with paper-consistent typography.

Images are the exact rasters embedded in the original paper panel PDFs
(extracted with pdfimages); the pred labels are re-set below each image
in the same serif style as the other figures, at a readable size.

Run: python fig01_examples.py <raster-dir> <orig-pdf-dir> <out-dir>
"""
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pypdf
from PIL import Image

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "text.usetex": False,
    "font.size": 10,
}


IMG_FRAC = 0.571  # image 15% smaller again per the author (0.672*0.85)
FIGSIZE = (1.45, 1.19)


def fit_fontsize(label, start=9.5, floor=6.0):
    """Largest fontsize <= start at which label fits the panel width."""
    fig = plt.figure(figsize=FIGSIZE)
    fs = start
    t = fig.text(0.5, 0.096, label, ha="center", va="center", fontsize=fs)
    fig.canvas.draw()
    while (t.get_window_extent().width
           > 0.97 * fig.get_window_extent().width) and fs > floor:
        fs -= 0.25
        t.set_fontsize(fs)
        fig.canvas.draw()
    plt.close(fig)
    return fs


def main(raster_dir, pdf_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC)
    pdfs = sorted(glob.glob(str(Path(pdf_dir) / "*.pdf")))
    labels = {Path(p).stem: pypdf.PdfReader(p).pages[0].extract_text().strip()
              for p in pdfs}
    # pred: prefix dropped per the author; fit to the values line
    labels = {k: (v.split("pred:", 1)[1].strip() if "pred:" in v else v)
              for k, v in labels.items()}
    fs = min(fit_fontsize(lab) for lab in labels.values())
    print("uniform fontsize:", fs)
    for stem, label in labels.items():
        img = Image.open(Path(raster_dir) / f"{stem}-000.png")
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_axes([(1 - IMG_FRAC) / 2, 0.245, IMG_FRAC, 0.72])
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(0.5, 0.096, label, ha="center", va="center", fontsize=fs)
        fig.savefig(out / f"{stem}.pdf")
        plt.close(fig)
        print("Saved", stem)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
