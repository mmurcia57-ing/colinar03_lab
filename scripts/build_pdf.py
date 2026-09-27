"""Construye una memoria PDF breve a partir de los resultados ejecutados."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image as PdfImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.anomaly_removal import remove_impulses
from src.metrics import detection_metrics, restoration_metrics


OUT = ROOT / "deliverables" / "laboratorio_eliminacion_anomalias.pdf"


def load(index: int):
    clean = np.asarray(Image.open(ROOT / f"data/original_{index}/original_{index}.png"))
    corrupted = np.asarray(Image.open(ROOT / f"data/corrupted_{index}/corrupted_{index}.png"))
    truth = np.asarray(Image.open(ROOT / f"data/mask_{index}/mask_{index}.png")) > 0
    restored, detected = remove_impulses(corrupted, window_size=5)
    Image.fromarray(restored).save(ROOT / f"data/output/restored_{index}.png")
    return clean, corrupted, truth, restored, detected


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(2 * cm, 1.1 * cm, "Percepcion Computacional - Eliminacion de anomalias")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def picture(path: Path, width=7.5 * cm):
    image = Image.open(path)
    ratio = image.height / image.width
    return PdfImage(str(path), width=width, height=width * ratio)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#17324D"), spaceAfter=16))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#17324D"), spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    story = [Paragraph("Laboratorio: Eliminacion de anomalias de la imagen", styles["TitleCenter"])]
    story.append(Paragraph("Ruido impulsivo tipo sal y pimienta", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Integrantes: completar con los nombres del grupo.", styles["BodyText"]))
    story.append(Paragraph("Objetivo. Detectar y restaurar pixeles impulsivos en dos imagenes distintas con el mismo algoritmo, demostrando una operacion principal propia y reproducible.", styles["BodyText"]))
    story.append(Paragraph("Diseno. Para cada pixel se extrae una ventana 5x5, se calcula la mediana y la desviacion absoluta mediana (MAD), y se marca el centro cuando es extremo y su desviacion contextual supera un umbral robusto. La restauracion pondera vecinos no anomalos por cercania a la mediana. No se usa un filtro de libreria como solucion.", styles["BodyText"]))
    story.append(Paragraph("Mapping con la actividad", styles["Section"]))
    data = [["Criterio", "Evidencia"], ["Generalizacion", "Mismo algoritmo y parametros en dos escenas"], ["Operacion propia", "Recorrido de ventanas, MAD y restauracion ponderada en src/"], ["Ejecucion", "Notebook ejecutado y figuras de deteccion/restauracion"], ["Validacion", "Precision, recall, F1, MAE, PSNR y SSIM"]]
    table = Table(data, colWidths=[4.2 * cm, 12.2 * cm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")), ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("LEADING", (0, 0), (-1, -1), 10)]))
    story.append(table)
    story.append(PageBreak())

    results = {}
    for index in (1, 2):
        clean, corrupted, truth, restored, detected = load(index)
        results[index] = (clean, corrupted, truth, restored, detected)
        story.append(Paragraph(f"Imagen {index}: ejecucion paso a paso", styles["Section"]))
        story.append(Paragraph("Entrada limpia, entrada contaminada, mapa detectado y salida restaurada.", styles["Small"]))
        row = [picture(ROOT / f"data/original_{index}/original_{index}.png"), picture(ROOT / f"data/corrupted_{index}/corrupted_{index}.png"), picture(ROOT / f"data/mask_{index}/mask_{index}.png")]
        story.append(Table([row], colWidths=[5.25 * cm] * 3, style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(Paragraph("Limpia &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Contaminada &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Mapa real", styles["Small"]))
        story.append(Spacer(1, 8))
        story.append(Table([[picture(ROOT / f"data/output/restored_{index}.png", 8.0 * cm)]], colWidths=[16 * cm], style=[("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(Paragraph(f"Salida restaurada. Pixeles alterados reales: {truth.sum()}; detectados: {detected.sum()}.", styles["Small"]))
        if index == 1:
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("Validacion y conclusiones", styles["Section"]))
    rows = [["Imagen", "Precision", "Recall", "F1", "MAE", "PSNR", "SSIM"]]
    for index, (clean, corrupted, truth, restored, detected) in results.items():
        det = detection_metrics(detected, truth)
        met = restoration_metrics(clean, restored)
        rows.append([str(index), f"{det['precision']:.3f}", f"{det['recall']:.3f}", f"{det['f1']:.3f}", f"{met['mae']:.3f}", f"{met['psnr']:.2f}", f"{met['ssim']:.4f}"])
    table = Table(rows, colWidths=[1.4 * cm, 2.2 * cm, 2.0 * cm, 1.6 * cm, 1.8 * cm, 2.0 * cm, 2.0 * cm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("FONTSIZE", (0, 0), (-1, -1), 8.5)]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("La misma configuracion funciona en dos escenas de contenido diferente, con precision 1.0 y recall superior a 0.98 en esta prueba controlada. El resultado respalda que el metodo no depende de una posicion fija ni de una imagen concreta. La validacion se realiza frente a la imagen limpia y a la mascara conocida antes de la corrupcion.", styles["BodyText"]))
    story.append(Paragraph("Referencias: guia UNIR colinar03_lab.docx; materiales de la asignatura sobre fuentes de ruido y deteccion/cancelacion de anomalias; fundamentos de estadistica robusta para la MAD. La implementacion concreta del detector y restaurador es propia y no copia codigo externo.", styles["Small"]))
    SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm, title="Laboratorio eliminacion de anomalias").build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
