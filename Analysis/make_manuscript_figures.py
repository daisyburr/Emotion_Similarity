"""Assemble the manuscript figures from the panels the consolidated notebook saves (Science format: 300 dpi, bold 10 pt panel letters).

Run after executing Emotion_Similarity_Consolidated.ipynb. Fig. 1 and Fig. 2 are two panels side by side (A, B); the other figures
are copied from the notebook's output so that every submitted figure sits in Analysis/figs/manuscript_figures.
"""
import os
import shutil

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
OUT = os.path.join(FIGS, "manuscript_figures")
FONT = "/System/Library/Fonts/Helvetica.ttc"
DPI = 300
LETTER_PT = 10
BAND = 52  # pixels above the panels, holding the panel letters


def two_panel(left, right, name):
    a, b = (Image.open(os.path.join(FIGS, p)).convert("RGB") for p in (left, right))
    canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height) + BAND), "white")
    canvas.paste(a, (0, BAND))
    canvas.paste(b, (a.width, BAND))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, round(LETTER_PT / 72 * DPI), index=1)  # Helvetica Bold
    draw.text((8, 4), "A", fill="black", font=font)
    draw.text((a.width + 8, 4), "B", fill="black", font=font)
    canvas.save(os.path.join(OUT, name), dpi=(DPI, DPI))


os.makedirs(OUT, exist_ok=True)
two_panel("IAPS_MTURK_sim_app_valence_barplot_correlation.png", "IAPS_MTURK_PCA_90_Var_App.png", "Fig1.png")
two_panel("CK_App__valence_barplot_correlation.png", "CK_Emo__valence_barplot_correlation.png", "Fig2.png")
for source, name in (("trans/emo_transformation_nestedCVR2_barplot_valence.png", "Fig3.png"),
                     ("image_neural_similarity/brain_map_pos_vs_neg.png", "Fig4.png"),
                     ("trans/trans_plot_by_valence.png", "FigS1.png"),
                     ("trans/emo_transformation_R2_barplot_valence.png", "FigS2.png")):
    shutil.copy(os.path.join(FIGS, source), os.path.join(OUT, name))
