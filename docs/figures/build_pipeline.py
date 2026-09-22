"""Generate pipeline.pdf + pipeline.pptx for the CILN-Bench figure.

Design rules:
  - Same-width boxes across rows that share a sub-stage.
  - Sub-stage headers sit just above their row, horizontally centered on the row.
  - Sans-serif typeface throughout (matches the GitHub diagram).
  - Two stacked phase panels: Phase 1 (pale blue) + Phase 2 (pale yellow).
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl

# ===== Palette =====
PASTEL_BLUE   = '#C8D9E8'
PASTEL_GREEN  = '#CCE5C9'
PASTEL_PURPLE = '#DDD0E5'
PHASE1_BG     = '#E7EEF9'
PHASE2_BG     = '#FBF7EA'
PHASE1_TITLE  = '#2D4B7E'
PHASE2_TITLE  = '#7A6E1E'
DARK_GREY     = '#333333'
GREY          = '#7A7A7A'
WHITE         = '#FFFFFF'

W, H = 11.0, 13.0

# ===== Helpers =====
def box(ax, x_center, y_center, w, h, fill, edge=DARK_GREY,
        radius=0.10, lw=1.0, zorder=2):
    p = FancyBboxPatch((x_center - w/2, y_center - h/2), w, h,
                       boxstyle=f'round,pad=0.03,rounding_size={radius}',
                       linewidth=lw, edgecolor=edge,
                       facecolor=fill, zorder=zorder)
    ax.add_patch(p)


def arrow(ax, x1, y1, x2, y2, lw=1.3, scale=14):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=scale,
                        color=DARK_GREY, linewidth=lw, zorder=3)
    ax.add_patch(a)


# ===== Renderer (matplotlib for PDF) =====
def render_pdf(out_path):
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype']  = 42
    mpl.rcParams['font.family']  = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    mpl.rcParams['mathtext.fontset'] = 'dejavusans'

    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')

    # ============== Phase 1 panel ==============
    # Panel spans y = 8.7 ... 12.8
    box(ax, W/2, (9.7 + 12.8)/2, W - 0.6, 12.8 - 9.7,
        PHASE1_BG, edge=PHASE1_TITLE, radius=0.18, lw=1.3, zorder=1)

    ax.text(W/2, 12.45, 'Phase 1 — Setup (one-time per dataset)',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=PHASE1_TITLE, zorder=3)

    # Left column: VT + BT (both same width 2.5)
    LEFT_X = 2.25
    BOX_W = 2.5
    BOX_H = 0.55

    box(ax, LEFT_X, 11.55, BOX_W, BOX_H, PASTEL_BLUE)
    ax.text(LEFT_X, 11.55, 'VoterTrain', ha='center', va='center',
            fontsize=12, color=DARK_GREY, zorder=3)

    box(ax, LEFT_X, 10.85, BOX_W, BOX_H, PASTEL_GREEN)
    ax.text(LEFT_X, 10.85, 'BenchmarkTrain', ha='center', va='center',
            fontsize=12, color=DARK_GREY, zorder=3)

    # "split D into halves" caption
    ax.text(LEFT_X, 10.30, 'split D into halves', ha='center', va='center',
            fontsize=10, style='italic', color=GREY, zorder=3)

    # Right: Train voter pool box
    VOTER_X = 6.7
    box(ax, VOTER_X, 11.20, 3.4, 1.10, PASTEL_PURPLE, radius=0.12, lw=1.2)
    ax.text(VOTER_X, 11.45, 'Train voter pool V', ha='center', va='center',
            fontsize=13, fontweight='bold', color=DARK_GREY, zorder=3)
    ax.text(VOTER_X, 11.00, r'$f_1 \ldots f_M$ (pretrained, frozen)',
            ha='center', va='center', fontsize=11, style='italic',
            color=DARK_GREY, zorder=3)

    # Arrow VT -> Voter pool
    arrow(ax, LEFT_X + BOX_W/2, 11.55, VOTER_X - 1.7, 11.30, scale=16)

    # Right-side caption
    ax.text(9.6, 11.30, 'Voters trained on\nVoterTrain only',
            ha='center', va='center', fontsize=10.5, color=DARK_GREY, zorder=3)

    # ============== Inter-phase arrow ==============
    arrow(ax, 1.6, 9.65, 1.6, 7.90, scale=18, lw=1.5)
    ax.text(2.95, 8.27, 'for each (corruption, severity)',
            ha='left', va='center', fontsize=11, style='italic',
            color=DARK_GREY, zorder=3)

    # ============== Phase 2 panel ==============
    # Panel spans y = 1.0 ... 7.85
    box(ax, W/2, (1.0 + 7.85)/2, W - 0.6, 7.85 - 1.0,
        PHASE2_BG, edge=PHASE2_TITLE, radius=0.18, lw=1.3, zorder=1)

    ax.text(W/2, 7.45, r'Phase 2 — Generate one benchmark setting $B_\tau$',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=PHASE2_TITLE, zorder=3)

    # ---- Sub-stage 1: Apply corruption ----
    # Row vertical center: 6.30. Header centered above at 6.95.
    ROW1_Y = 6.30
    HEADER1_Y = 6.95
    ax.text(W/2, HEADER1_Y, 'Apply corruption', ha='center', va='center',
            fontsize=13, fontweight='bold', color=DARK_GREY, zorder=3)

    SUB_W = 2.6           # all boxes same width on this row
    SUB_H = 0.85          # tall enough for 2-line center C(x_i,τ) box
    GAP   = 0.65

    # Place 3 same-width boxes centered horizontally
    LEFT  = W/2 - SUB_W*1.5 - GAP    # left box center x
    MID   = W/2
    RIGHT = W/2 + SUB_W*1.5 + GAP    # right box center x
    # Wait — 3 boxes with width SUB_W and gap GAP between them:
    # total width = 3*SUB_W + 2*GAP. center is at W/2.
    # so leftmost center is at W/2 - SUB_W - GAP, rightmost at W/2 + SUB_W + GAP
    LEFT  = W/2 - SUB_W - GAP
    MID   = W/2
    RIGHT = W/2 + SUB_W + GAP

    box(ax, LEFT, ROW1_Y, SUB_W, SUB_H, PASTEL_BLUE)
    ax.text(LEFT, ROW1_Y, r'CleanInputs $x_i$' + '\n(BenchmarkTrain)',
            ha='center', va='center', fontsize=11, color=DARK_GREY, zorder=3)

    box(ax, MID, ROW1_Y, SUB_W, SUB_H, WHITE)
    ax.text(MID, ROW1_Y + 0.13, r'$C(x_i,\tau)$',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=DARK_GREY, zorder=3)
    ax.text(MID, ROW1_Y - 0.18, r'$\tau = (c, \ell)$',
            ha='center', va='center', fontsize=11, color=DARK_GREY, zorder=3)

    box(ax, RIGHT, ROW1_Y, SUB_W, SUB_H, PASTEL_GREEN)
    ax.text(RIGHT, ROW1_Y, r'CorruptedInputs $\tilde{x}_i$',
            ha='center', va='center', fontsize=11, color=DARK_GREY, zorder=3)

    # Arrows between row 1 boxes
    arrow(ax, LEFT + SUB_W/2 + 0.02, ROW1_Y, MID - SUB_W/2 - 0.02, ROW1_Y)
    arrow(ax, MID + SUB_W/2 + 0.02, ROW1_Y, RIGHT - SUB_W/2 - 0.02, ROW1_Y)

    # Sub-stage 1 caption
    ax.text(W/2, ROW1_Y - 0.80,
            'each image logged with RNG seed, parameters, and output sha256',
            ha='center', va='center', fontsize=10, style='italic',
            color=GREY, zorder=3)

    # ---- Sub-stage 2: Label with voter pool ----
    # Row center y=3.95, voter-pool box height 1.25 -> top edge at 4.575.
    # Match the "Apply corruption" header-to-box gap of ~0.225 inches.
    ROW2_Y = 3.95
    HEADER2_Y = 4.80
    ax.text(W/2, HEADER2_Y, 'Label with voter pool', ha='center', va='center',
            fontsize=13, fontweight='bold', color=DARK_GREY, zorder=3)

    # We'll lay 3 columns of same-width boxes (left, center, right):
    #   Left:  CorruptedInputs (1 box)
    #   Mid:   Voter pool V (1 taller box)
    #   Right: two stacked boxes (Aggregated, Per-voter)
    R2_BOX_W = 2.6
    R2_LEFT  = W/2 - R2_BOX_W - GAP
    R2_MID   = W/2
    R2_RIGHT = W/2 + R2_BOX_W + GAP

    # Left box
    box(ax, R2_LEFT, ROW2_Y, R2_BOX_W, 0.75, PASTEL_GREEN)
    ax.text(R2_LEFT, ROW2_Y, r'CorruptedInputs $\tilde{x}_i$',
            ha='center', va='center', fontsize=11, color=DARK_GREY, zorder=3)

    # Center voter-pool box (taller, since the right column has 2 boxes ~1.2 tall total)
    box(ax, R2_MID, ROW2_Y, R2_BOX_W, 1.25, PASTEL_PURPLE, radius=0.12, lw=1.2)
    ax.text(R2_MID, ROW2_Y + 0.22, 'Voter pool V', ha='center', va='center',
            fontsize=13, fontweight='bold', color=DARK_GREY, zorder=3)
    ax.text(R2_MID, ROW2_Y - 0.22, r'$f_1 \ldots f_M$', ha='center', va='center',
            fontsize=11, style='italic', color=DARK_GREY, zorder=3)

    # Right column: 2 stacked boxes, each smaller height
    box(ax, R2_RIGHT, ROW2_Y, R2_BOX_W, 0.85, PASTEL_GREEN)
    ax.text(R2_RIGHT, ROW2_Y + 0.18, r'Per-voter $\{p_i^{(m)}\}$',
            ha='center', va='center', fontsize=11, color=DARK_GREY, zorder=3)
    ax.text(R2_RIGHT, ROW2_Y - 0.20, r'vote shares $v_i$',
            ha='center', va='center', fontsize=11, color=DARK_GREY, zorder=3)

    # Arrows
    arrow(ax, R2_LEFT + R2_BOX_W/2 + 0.02, ROW2_Y, R2_MID - R2_BOX_W/2 - 0.02, ROW2_Y)
    arrow(ax, R2_MID + R2_BOX_W/2 + 0.02, ROW2_Y,
              R2_RIGHT - R2_BOX_W/2 - 0.02, ROW2_Y)

    # ---- Sub-stage 3: Release benchmark Bτ ----
    HEADER3_Y = 2.85
    ax.text(W/2, HEADER3_Y, r'Release benchmark $B_\tau$',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=DARK_GREY, zorder=3)

    ax.text(W/2, 2.25,
            r'$B_\tau = \{\, x_i,\ \tilde{x}_i,\ y_i,\ \tilde{y}_i,\ v_i,\ \{p_i^{(m)}\},\ \tau = (c, \ell)\, \}$',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=DARK_GREY, zorder=3)

    ax.text(W/2, 1.65,
            r'clean input $\cdot$ corrupted input $\cdot$ ground-truth label $\cdot$ sampled training label $\cdot$ '
            r'vote shares $\cdot$ per-voter dists $\cdot$ corruption metadata',
            ha='center', va='center', fontsize=9, style='italic',
            color=GREY, zorder=3)

    # ============== Bottom summary ==============
    ax.text(W/2, 0.45,
            r'optional CILN-C filter: keep only images all voters classify '
            r'correctly on the clean input',
            ha='center', va='center', fontsize=9.5, style='italic',
            color=GREY, zorder=3)

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f'Saved {out_path}')


# ===== python-pptx renderer (same layout, editable PPTX) =====
def render_pptx(out_path):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

    def hx(h):
        h = h.lstrip('#')
        return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def to_top(y, h_box):
        # mpl: y is the *center* in inches-from-bottom. pptx wants top-left in EMU.
        return Inches(H - y - h_box/2)

    def set_text(shape, text, size=11, bold=False, italic=False, color=DARK_GREY):
        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.05)
        tf.margin_top = tf.margin_bottom = Inches(0.03)
        tf.clear()
        for i, line in enumerate(text.split('\n')):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = line
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = hx(color)

    def add_text(slide, x_center, y_center, width, height, text,
                 size=11, bold=False, italic=False, color=DARK_GREY):
        tb = slide.shapes.add_textbox(
            Inches(x_center - width/2), to_top(y_center, height),
            Inches(width), Inches(height))
        set_text(tb, text, size=size, bold=bold, italic=italic, color=color)

    def add_box(slide, x_center, y_center, w, h, fill, text,
                size=11, bold=False, italic=False, edge=DARK_GREY):
        s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x_center - w/2), to_top(y_center, h),
            Inches(w), Inches(h))
        s.fill.solid()
        s.fill.fore_color.rgb = hx(fill)
        s.line.color.rgb = hx(edge)
        s.line.width = Pt(1.0)
        set_text(s, text, size=size, bold=bold, italic=italic)

    prs = Presentation()
    prs.slide_width = Inches(W); prs.slide_height = Inches(H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # ----- Phase 1 panel -----
    p1 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.3), to_top((8.7 + 12.8)/2, 12.8 - 8.7),
        Inches(W - 0.6), Inches(12.8 - 8.7))
    p1.fill.solid(); p1.fill.fore_color.rgb = hx(PHASE1_BG)
    p1.line.color.rgb = hx(PHASE1_TITLE); p1.line.width = Pt(1.5)
    set_text(p1, '', size=11)

    add_text(slide, W/2, 12.45, W - 0.6, 0.45,
             'Phase 1 — Setup (one-time per dataset)',
             size=16, bold=True, color=PHASE1_TITLE)

    add_box(slide, 2.25, 11.55, 2.5, 0.55, PASTEL_BLUE, 'VoterTrain', size=12)
    add_box(slide, 2.25, 10.85, 2.5, 0.55, PASTEL_GREEN, 'BenchmarkTrain', size=12)
    add_text(slide, 2.25, 10.30, 2.5, 0.35, 'split D into halves',
             size=10, italic=True, color=GREY)
    add_box(slide, 6.7, 11.20, 3.4, 1.10, PASTEL_PURPLE,
            'Train voter pool V\nf1 ... fM (pretrained, frozen)',
            size=12, bold=True)
    add_text(slide, 9.6, 11.30, 2.0, 0.8,
             'Voters trained on\nVoterTrain only',
             size=10, color=DARK_GREY)

    add_text(slide, 4.0, 8.27, 4.5, 0.35,
             'for each (corruption, severity)',
             size=11, italic=True, color=DARK_GREY)

    # ----- Phase 2 panel -----
    p2 = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.3), to_top((1.0 + 7.85)/2, 7.85 - 1.0),
        Inches(W - 0.6), Inches(7.85 - 1.0))
    p2.fill.solid(); p2.fill.fore_color.rgb = hx(PHASE2_BG)
    p2.line.color.rgb = hx(PHASE2_TITLE); p2.line.width = Pt(1.5)
    set_text(p2, '', size=11)

    add_text(slide, W/2, 7.45, W - 0.6, 0.45,
             'Phase 2 — Generate one benchmark setting B_tau',
             size=16, bold=True, color=PHASE2_TITLE)

    # Sub-stage 1 (Apply corruption)
    add_text(slide, W/2, 6.95, W - 0.6, 0.35, 'Apply corruption',
             size=13, bold=True)
    GAP = 0.65; SUB_W = 2.6
    LEFT = W/2 - SUB_W - GAP; MID = W/2; RIGHT = W/2 + SUB_W + GAP
    add_box(slide, LEFT, 6.30, SUB_W, 0.85, PASTEL_BLUE,
            'CleanInputs x_i\n(BenchmarkTrain)', size=11)
    add_box(slide, MID, 6.30, SUB_W, 0.85, WHITE,
            'C(x_i, tau)\ntau = (c, l)', size=11, bold=True)
    add_box(slide, RIGHT, 6.30, SUB_W, 0.85, PASTEL_GREEN,
            'CorruptedInputs tilde_x_i', size=11)
    add_text(slide, W/2, 5.50, W - 0.6, 0.30,
             'each image logged with RNG seed, parameters, and output sha256',
             size=10, italic=True, color=GREY)

    # Sub-stage 2 (Label with voter pool)
    add_text(slide, W/2, 4.80, W - 0.6, 0.35, 'Label with voter pool',
             size=13, bold=True)
    R2_BOX_W = 2.6
    R2_LEFT = W/2 - R2_BOX_W - GAP
    R2_MID = W/2
    R2_RIGHT = W/2 + R2_BOX_W + GAP
    add_box(slide, R2_LEFT, 3.95, R2_BOX_W, 0.75, PASTEL_GREEN,
            'CorruptedInputs tilde_x_i', size=11)
    add_box(slide, R2_MID, 3.95, R2_BOX_W, 1.25, PASTEL_PURPLE,
            'Voter pool V\nf1 ... fM', size=12, bold=True)
    add_box(slide, R2_RIGHT, 4.30, R2_BOX_W, 0.55, PASTEL_GREEN,
            'Aggregated bar_p_i', size=11)
    add_box(slide, R2_RIGHT, 3.60, R2_BOX_W, 0.55, PASTEL_GREEN,
            'Per-voter {p_i^(m)}', size=11)

    # Sub-stage 3 (Release benchmark)
    add_text(slide, W/2, 2.85, W - 0.6, 0.35,
             'Release benchmark B_tau', size=13, bold=True)
    add_text(slide, W/2, 2.25, W - 0.6, 0.4,
             'B_tau = { x_i, tilde_x_i, y_i, tilde_y_i, v_i, {p_i^(m)}, tau = (c, l) }',
             size=14, bold=True)
    add_text(slide, W/2, 1.65, W - 0.6, 0.30,
             'clean input . corrupted input . ground-truth label . sampled training label . '
             'aggregated voter dist . per-voter dists . corruption metadata',
             size=10, italic=True, color=GREY)

    # Bottom summary
    add_text(slide, W/2, 0.45, W - 0.6, 0.30,
             'optional CILN-C filter: keep only images all voters classify '
             'correctly on the clean input',
             size=10, italic=True, color=GREY)

    prs.save(str(out_path))
    print(f'Saved {out_path}')


if __name__ == '__main__':
    HERE = Path(__file__).parent
    render_pdf(HERE / 'pipeline_thesis.pdf')
    render_pptx(HERE / 'pipeline_thesis.pptx')
