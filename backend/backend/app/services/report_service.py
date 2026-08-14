"""
Report Generation Service (Updated)
Generates JSON and PDF reports using segmentation results.
"""
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any
from loguru import logger
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import inch
from io import BytesIO
from app.utils.chart_utils import generate_pie_chart
from app.utils.file_utils import get_report_path


class ReportService:
    def generate_json(self, scan_id: str, fruits: List[Dict], grade_counts: Dict,
                      shelf_life: str, market: Dict) -> Dict[str, Any]:
        return {
            "scan_id": scan_id,
            "generated_at": datetime.utcnow().isoformat(),
            "total_fruits": len(fruits),
            "grades": grade_counts,
            "shelf_life": shelf_life,
            "market": market,
            "fruits": fruits,
        }

    def generate_final_report(self, scan_id: str, fruits: List[Dict]) -> Dict[str, Any]:
        total = len(fruits)
        good = sum(1 for f in fruits if f.get("grade", "").lower() == "good")
        better = sum(1 for f in fruits if f.get("grade", "").lower() == "better")
        reject = sum(1 for f in fruits if f.get("grade", "").lower() == "reject")

        shelf_lives = []
        for f in fruits:
            days = f.get("shelf_life_days")
            if days is None:
                match = re.search(r'\d+', str(f.get("shelf_life", "")))
                if match:
                    days = int(match.group())
            if days is not None:
                shelf_lives.append(days)

        avg_days = round(sum(shelf_lives) / len(shelf_lives)) if shelf_lives else 0
        avg_shelf_life_str = f"{avg_days} days"

        fruit_distribution = {}
        for f in fruits:
            t = f.get("fruit_type", "Unknown")
            if t:
                t_proper = t.strip().capitalize()
                fruit_distribution[t_proper] = fruit_distribution.get(t_proper, 0) + 1

        return {
            "scan_id": scan_id,
            "total_fruits": total,
            "good": good,
            "better": better,
            "reject": reject,
            "average_shelf_life": avg_shelf_life_str,
            "fruit_distribution": fruit_distribution,
        }

    def generate_pdf(self, scan_id: str, image_path: str, fruits: List[Dict],
                     grade_counts: Dict, shelf_life: str, market: Dict,
                     annotated_path: str = None) -> str:
        """
        PDF report generator.
        Updated to use segmentation-based annotated images.
        """
        pdf_path = get_report_path(scan_id)
        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4,
            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40,
        )
        styles = getSampleStyleSheet()
        story = []

        if annotated_path and os.path.exists(annotated_path):
            try:
                import cv2
                img_orig = cv2.imread(annotated_path)
                if img_orig is not None:
                    h, w = img_orig.shape[:2]
                    img_width = 360
                    img_height = int(img_width * (h / w))

                    story.append(Paragraph("Segmented Fruit Analysis", styles["Heading2"]))
                    story.append(Spacer(1, 6))
                    story.append(RLImage(annotated_path, width=img_width, height=img_height))
                    story.append(Spacer(1, 16))
            except Exception as e:
                logger.warning(f"Failed to embed annotated image in PDF: {e}")

        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            fontSize=22, textColor=colors.HexColor("#2E7D32"),
        )
        story.append(Paragraph("HarvestLenz — Fruit Quality Report", title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Scan ID: {scan_id}", styles["Normal"]))
        story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}", styles["Normal"]))
        story.append(Spacer(1, 16))

        fruit_type = fruits[0].get("fruit_type", "N/A") if fruits else "N/A"
        summary_data = [
            ["Metric", "Value"],
            ["Fruit Type", fruit_type.capitalize()],
            ["Total Fruits Detected", str(len(fruits))],
            ["Good", str(grade_counts.get("good", 0) + grade_counts.get("Good", 0))],
            ["Better", str(grade_counts.get("better", 0) + grade_counts.get("Better", 0))],
            ["Reject", str(grade_counts.get("reject", 0) + grade_counts.get("Reject", 0))],
            ["Average Shelf Life", shelf_life],
            ["Best Market", market.get("best_market", market.get("recommended_market", "N/A"))],
            ["Expected Price", market.get("expected_price", market.get("estimated_price", "N/A"))],
        ]
        tbl = Table(summary_data, colWidths=[3 * inch, 3 * inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F8E9")]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 20))

        try:
            grades_lower = {
                "good": grade_counts.get("good", 0) + grade_counts.get("Good", 0),
                "better": grade_counts.get("better", 0) + grade_counts.get("Better", 0),
                "reject": grade_counts.get("reject", 0) + grade_counts.get("Reject", 0),
            }
            chart_bytes = generate_pie_chart(grades_lower)
            chart_img = RLImage(BytesIO(chart_bytes), width=3 * inch, height=3 * inch)
            story.append(chart_img)
            story.append(Spacer(1, 12))
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")

        story.append(Paragraph("Fruit-Level Details", styles["Heading2"]))
        fruit_data = [["#", "Type", "Grade", "Shelf Life", "Market"]]
        for i, f in enumerate(fruits[:50], 1):
            fruit_data.append([
                str(i),
                f.get("fruit_type", "N/A").capitalize(),
                f.get("grade", "N/A"),
                f.get("shelf_life", "N/A"),
                f.get("market_recommendation", "N/A"),
            ])
        ft = Table(fruit_data, colWidths=[0.4*inch, 1.2*inch, 1*inch, 1.2*inch, 2.2*inch])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#388E3C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FBE7")]),
        ]))
        story.append(ft)

        doc.build(story)
        logger.info(f"PDF report saved: {pdf_path}")
        return pdf_path


report_service = ReportService()
