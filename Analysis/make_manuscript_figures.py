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


def brain_map(name):
    """Fig. 4: the brain map with the color bar labeled (nilearn clips the label at the image edge)."""
    img = Image.open(os.path.join(FIGS, "image_neural_similarity", "brain_map_pos_vs_neg.png")).convert("RGB")
    margin = 90
    canvas = Image.new("RGB", (img.width + margin, img.height), "white")
    canvas.paste(img, (0, 0))
    font = ImageFont.truetype(FONT, round(7 / 72 * DPI))
    text = "Negative minus positive similarity (Fisher z)"
    box = font.getbbox(text)
    label = Image.new("RGB", (box[2] + 4, box[3] + 4), "white")
    ImageDraw.Draw(label).text((2, 0), text, fill="black", font=font)
    label = label.rotate(90, expand=True)
    canvas.paste(label, (img.width + 10, (img.height - label.height) // 2))
    width = round(18.4 / 2.54 * DPI)  # keep the printed width at 18.4 cm
    canvas.resize((width, round(canvas.height * width / canvas.width)), Image.LANCZOS).save(os.path.join(OUT, name), dpi=(DPI, DPI))


brain_map("Fig4.png")
for source, name in (("trans/emo_transformation_nestedCVR2_barplot_valence.png", "Fig3.png"),
                     ("trans/trans_plot_by_valence.png", "FigS1.png"),
                     ("trans/emo_transformation_R2_barplot_valence.png", "FigS2.png")):
    shutil.copy(os.path.join(FIGS, source), os.path.join(OUT, name))
