#!/usr/bin/env python
"""Build and validate the benchmark workbook with the approved openpyxl fallback.

The canonical benchmark data remains JSON/JSONL/CSV.  This reporter creates a
human-readable Excel view without changing any scientific measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


LOG = logging.getLogger("benchmark_excel")
MAX_EXCEL_ROWS = 1_048_576
MAX_CELL_TEXT = 32_700

DARK = "374151"
DARKER = "1F2937"
MID = "6B7280"
LIGHT = "E5E7EB"
LIGHTER = "F9FAFB"
ACCENT = "0F766E"
GOOD = "DCFCE7"
WARN = "FEF3C7"
BAD = "FEE2E2"
WHITE = "FFFFFF"
TEXT = "111827"

THIN_GRAY = Side(style="thin", color="D1D5DB")
SUBTLE_BORDER = Border(bottom=Side(style="thin", color=LIGHT))

SUMMARY_COLUMNS = [
    "direction",
    "query_variant",
    "search_mode",
    "model_id",
    "total_queries",
    "ok_queries",
    "rejected_queries",
    "error_queries",
    "coverage",
    "success_30m_queries",
    "success_30m",
    "success_30m_ci95_low",
    "success_30m_ci95_high",
    "success_30m_failure_rate",
    "auc_30m",
    "mean_error_under_30m",
    "median_error_under_30m",
    "mean_error_m",
    "median_error_m",
    "median_error_ci95_low",
    "median_error_ci95_high",
    "p90_error_m",
    "p95_error_m",
    "success_5m",
    "success_10m",
    "success_10m_ci95_low",
    "success_10m_ci95_high",
    "success_25m",
    "success_50m",
    "mean_top1_score",
    "mean_search_seconds",
    "total_search_seconds",
]

QUERY_COLUMNS = [
    "direction",
    "query_variant",
    "query_id",
    "block_id",
    "center_easting_m",
    "center_northing_m",
    "source_row",
    "source_col",
    "query_std",
    "query_entropy",
    "dark_fraction",
    "raw_tile_file",
]

SUMMARY_FORMATS = {
    "coverage": "0.0%",
    "success_30m": "0.0%",
    "success_30m_ci95_low": "0.0%",
    "success_30m_ci95_high": "0.0%",
    "success_30m_failure_rate": "0.0%",
    "auc_30m": "0.0%",
    "mean_error_under_30m": "0.00",
    "median_error_under_30m": "0.00",
    "mean_error_m": "0.00",
    "median_error_m": "0.00",
    "median_error_ci95_low": "0.00",
    "median_error_ci95_high": "0.00",
    "p90_error_m": "0.00",
    "p95_error_m": "0.00",
    "success_5m": "0.0%",
    "success_10m": "0.0%",
    "success_10m_ci95_low": "0.0%",
    "success_10m_ci95_high": "0.0%",
    "success_25m": "0.0%",
    "success_50m": "0.0%",
    "mean_top1_score": "0.0000",
    "mean_search_seconds": "0.000",
    "total_search_seconds": "0.0",
}

RESULT_FORMATS = {
    "query_center_easting_m": "0.000",
    "query_center_northing_m": "0.000",
    "expected_center_easting_m": "0.000",
    "expected_center_northing_m": "0.000",
    "predicted_center_easting_m": "0.000",
    "predicted_center_northing_m": "0.000",
    "error_m": "0.000",
    "error_px": "0.000",
    "top1_score": "0.0000",
    "top2_score": "0.0000",
    "peak_margin": "0.0000",
    "psr": "0.00",
    "query_std": "0.00",
    "query_entropy": "0.00",
    "query_dark_fraction": "0.0%",
    "success_30m": "0",
    "search_seconds": "0.000",
    "query_inference_seconds": "0.000",
    "map_build_seconds": "0.0",
}

QUERY_FORMATS = {
    "center_easting_m": "0.000",
    "center_northing_m": "0.000",
    "query_std": "0.00",
    "query_entropy": "0.00",
    "dark_fraction": "0.0%",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sonuçlarını XLSX raporuna dönüştürür.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def read_json(path: Path, fallback: Any) -> Any:
    if not path.is_file():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} bir JSON nesnesi değil.")
            rows.append(value)
    return rows


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def excel_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value)
    if len(text) > MAX_CELL_TEXT:
        text = text[: MAX_CELL_TEXT - 25] + " … [kısaltıldı]"
    return text


def ordered_columns(rows: Sequence[dict[str, Any]], fallback: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    columns: list[str] = []
    for name in fallback:
        if name not in seen:
            seen.add(name)
            columns.append(name)
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                columns.append(name)
    return columns


def find_query_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest_paths = sorted(run_dir.rglob("query_manifest.json")) + sorted(
        run_dir.rglob("query_variant_manifest.json")
    )
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path, {})
        try:
            direction = manifest_path.relative_to(run_dir).parts[0]
        except IndexError:
            direction = "unknown"
        query_variant = str(manifest.get("query_variant", "clean"))
        for query in manifest.get("queries", []):
            if isinstance(query, dict):
                row = {"direction": direction, "query_variant": query_variant, **query}
                if isinstance(row.get("augmentation"), dict):
                    row["augmentation"] = json.dumps(
                        row["augmentation"], ensure_ascii=False, sort_keys=True
                    )
                rows.append(row)
    return rows


def safe_table_name(prefix: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", prefix)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"T_{cleaned}"
    return f"{cleaned}_{index:02d}"


def configure_sheet(
    ws, *, freeze: str = "A2", fit_width: int = 1, orientation: str = "portrait"
) -> None:
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = freeze
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = fit_width
    ws.page_setup.fitToHeight = 0
    ws.page_setup.orientation = orientation
    ws.print_title_rows = "1:1"
    ws.sheet_view.zoomScale = 90


def style_header(row: Iterable[Any]) -> None:
    for cell in row:
        cell.fill = PatternFill("solid", fgColor=DARK)
        cell.font = Font(name="Aptos", size=10, bold=True, color=WHITE)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=Side(style="medium", color=MID))


def write_table(
    ws,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    table_name: str,
    number_formats: dict[str, str] | None = None,
) -> None:
    ws.append(list(columns))
    for row in rows:
        ws.append([excel_value(row.get(column)) for column in columns])
    # Excel desktop rejects a table whose reference contains only its header.
    # Keep a single genuinely blank data row for empty datasets.
    if not rows:
        ws.append([None] * len(columns))
    style_header(ws[1])
    ws.row_dimensions[1].height = 32
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.font = Font(name="Aptos", size=9, color=TEXT)
            cell.alignment = Alignment(vertical="center")
            cell.border = SUBTLE_BORDER
    if number_formats:
        for index, column in enumerate(columns, start=1):
            number_format = number_formats.get(column)
            if number_format:
                for cell in ws.iter_cols(
                    min_col=index, max_col=index, min_row=2, max_row=max(2, ws.max_row)
                ):
                    for item in cell:
                        item.number_format = number_format
                        item.alignment = Alignment(horizontal="right", vertical="center")
    end_column = get_column_letter(len(columns))
    table = Table(displayName=table_name, ref=f"A1:{end_column}{max(1, ws.max_row)}")
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(table)
    column_count = len(columns)
    configure_sheet(
        ws,
        fit_width=3 if column_count > 30 else 2 if column_count > 15 else 1,
        orientation="landscape" if column_count > 6 else "portrait",
    )


def set_widths(
    ws,
    columns: Sequence[str],
    *,
    overrides: dict[str, float] | None = None,
    sample_rows: int = 250,
) -> None:
    overrides = overrides or {}
    for column_index, name in enumerate(columns, start=1):
        if name in overrides:
            width = overrides[name]
        else:
            values = [str(name)]
            for row_index in range(2, min(ws.max_row, sample_rows + 1) + 1):
                value = ws.cell(row=row_index, column=column_index).value
                if value is not None:
                    values.append(str(value))
            width = min(24.0, max(10.0, max(len(value) for value in values) + 2.0))
        ws.column_dimensions[get_column_letter(column_index)].width = width


def add_error_conditional_formatting(ws, columns: Sequence[str]) -> None:
    if ws.max_row < 2:
        return
    if "error_m" in columns:
        index = columns.index("error_m") + 1
        letter = get_column_letter(index)
        ws.conditional_formatting.add(
            f"{letter}2:{letter}{ws.max_row}",
            ColorScaleRule(
                start_type="min",
                start_color="DCFCE7",
                mid_type="percentile",
                mid_value=50,
                mid_color="FEF3C7",
                end_type="max",
                end_color="FECACA",
            ),
        )


def write_dashboard(
    wb: Workbook,
    config: dict[str, Any],
    summary_rows: Sequence[dict[str, Any]],
    result_chunks: Sequence[tuple[str, Sequence[dict[str, Any]], Sequence[str]]],
) -> None:
    ws = wb["Özet"]
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A3"
    ws.sheet_view.zoomScale = 90
    ws.merge_cells("A1:H2")
    ws["A1"] = "Jeoreferanslı Ortak-Temsil Benchmark Sonuçları"
    ws["A1"].fill = PatternFill("solid", fgColor=DARKER)
    ws["A1"].font = Font(name="Aptos Display", size=18, bold=True, color=WHITE)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 28
    ws.row_dimensions[2].height = 18

    labels = ["Koşu", "Toplam kayıt", "Model/yön kombinasyonu", "Başarılı kayıt", "Hata/reddetme"]
    for row_index, label in enumerate(labels, start=4):
        ws.cell(row=row_index, column=1, value=label)
        ws.cell(row=row_index, column=1).fill = PatternFill("solid", fgColor=LIGHT)
        ws.cell(row=row_index, column=1).font = Font(name="Aptos", bold=True, color=DARK)
        ws.cell(row=row_index, column=2).fill = PatternFill("solid", fgColor=LIGHTER)
        ws.cell(row=row_index, column=2).font = Font(name="Aptos", bold=True, color=TEXT)
        for column in (1, 2):
            ws.cell(row=row_index, column=column).border = Border(bottom=THIN_GRAY)
            ws.cell(row=row_index, column=column).alignment = Alignment(vertical="center")

    ws["B4"] = config.get("run_id") or "benchmark"
    count_formulas: list[str] = []
    ok_formulas: list[str] = []
    for sheet_name, rows, columns in result_chunks:
        last_row = max(2, len(rows) + 1)
        id_index = 1
        status_index = columns.index("status") + 1 if "status" in columns else None
        id_letter = get_column_letter(id_index)
        count_formulas.append(f"COUNTA('{sheet_name}'!{id_letter}2:{id_letter}{last_row})")
        if status_index is not None:
            status_letter = get_column_letter(status_index)
            ok_formulas.append(
                f'COUNTIF(\'{sheet_name}\'!{status_letter}2:{status_letter}{last_row},"ok")'
            )
    ws["B5"] = "=" + "+".join(count_formulas or ["0"])
    summary_last_row = max(2, len(summary_rows) + 1)
    ws["B6"] = f"=COUNTA('Model Özeti'!A2:A{summary_last_row})"
    ws["B7"] = "=" + "+".join(ok_formulas or ["0"])
    ws["B8"] = "=B5-B7"
    for cell in ("B5", "B6", "B7", "B8"):
        ws[cell].number_format = "#,##0"

    ws.merge_cells("A10:H11")
    ws["A10"] = (
        "Ana değerlendirme: aynı model hem sorgu parçasına hem arama haritasına "
        "uygulanır. Ana başarı ölçütü tüm sorgular üzerinden 30 m içinde konumlamadır; "
        "clean ve hard_v1 koşulları ayrı raporlanır; tüm-hata ortalaması yalnız tanısaldır."
    )
    ws["A10"].fill = PatternFill("solid", fgColor="F3F4F6")
    ws["A10"].font = Font(name="Aptos", italic=True, color="4B5563")
    ws["A10"].alignment = Alignment(wrap_text=True, vertical="center")
    ws["A10"].border = Border(left=THIN_GRAY, right=THIN_GRAY, top=THIN_GRAY, bottom=THIN_GRAY)
    ws.row_dimensions[10].height = 26
    ws.row_dimensions[11].height = 18

    ranking_headers = [
        "Model ve arama modu",
        "30 m başarı",
        "30 m içi medyan hata",
        "AUC@30m",
    ]
    for column, header in enumerate(ranking_headers, start=1):
        ws.cell(row=14, column=column, value=header)
    style_header(ws[14][:4])
    ws.row_dimensions[14].height = 32

    summary_positions = sorted(
        [
            (index + 2, row)
            for index, row in enumerate(summary_rows)
            if row.get("success_30m") is not None
        ],
        key=lambda item: (
            -float(item[1].get("success_30m") or 0.0),
            -float(item[1].get("auc_30m") or 0.0),
            (
                float(item[1]["median_error_under_30m"])
                if item[1].get("median_error_under_30m") is not None
                else float("inf")
            ),
        ),
    )[:15]
    summary_column_letters = {
        name: get_column_letter(index + 1) for index, name in enumerate(SUMMARY_COLUMNS)
    }
    for offset, (source_row, _) in enumerate(summary_positions, start=15):
        model_letter = summary_column_letters["model_id"]
        variant_letter = summary_column_letters["query_variant"]
        mode_letter = summary_column_letters["search_mode"]
        ws.cell(
            row=offset,
            column=1,
            value=(
                f"=IF(LEN('Model Özeti'!{model_letter}{source_row})>24,"
                f"LEFT('Model Özeti'!{model_letter}{source_row},24)&\"...\","
                f"'Model Özeti'!{model_letter}{source_row})&\" [\"&"
                f"'Model Özeti'!{variant_letter}{source_row}&\" | \"&"
                f"'Model Özeti'!{mode_letter}{source_row}&\"]\""
            ),
        )
        ws.cell(
            row=offset,
            column=2,
            value=f"='Model Özeti'!{summary_column_letters['success_30m']}{source_row}",
        )
        ws.cell(
            row=offset,
            column=3,
            value=(
                f"='Model Özeti'!"
                f"{summary_column_letters['median_error_under_30m']}{source_row}"
            ),
        )
        ws.cell(
            row=offset,
            column=4,
            value=f"='Model Özeti'!{summary_column_letters['auc_30m']}{source_row}",
        )
        for column in range(1, 5):
            ws.cell(row=offset, column=column).border = SUBTLE_BORDER
            ws.cell(row=offset, column=column).font = Font(name="Aptos", size=9, color=TEXT)
    for row in range(15 + len(summary_positions), 30):
        for column in range(1, 5):
            ws.cell(row=row, column=column).border = SUBTLE_BORDER
    for row in range(15, 30):
        ws.cell(row=row, column=2).number_format = "0.0%"
        ws.cell(row=row, column=3).number_format = "0.00"
        ws.cell(row=row, column=4).number_format = "0.0%"

    if summary_positions:
        chart = BarChart()
        chart.type = "bar"
        chart.style = 10
        chart.title = "30 m konumlama başarısı - en iyi modeller"
        chart.height = 8.5
        chart.width = 17.0
        chart.varyColors = False
        data = Reference(ws, min_col=2, min_row=14, max_row=14 + len(summary_positions))
        categories = Reference(ws, min_col=1, min_row=15, max_row=14 + len(summary_positions))
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(categories)
        chart.legend = None
        chart.y_axis.scaling.orientation = "maxMin"
        chart.y_axis.numFmt = "0%"
        if chart.series:
            chart.series[0].graphicalProperties.solidFill = ACCENT
            chart.series[0].graphicalProperties.line.solidFill = ACCENT
        ws.add_chart(chart, "F14")

    widths = {"A": 42, "B": 18, "C": 16, "D": 16, "E": 3}
    for column, width in widths.items():
        ws.column_dimensions[column].width = width
    for column in range(6, 14):
        ws.column_dimensions[get_column_letter(column)].width = 12
    ws.print_area = "A1:M31"
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1


def build_workbook(run_dir: Path, output_path: Path) -> dict[str, Any]:
    LOG.info("Kaynak dosyalar okunuyor: %s", run_dir)
    config = read_json(run_dir / "run_config.json", {})
    summary_rows = read_json(run_dir / "summary.json", [])
    result_rows = read_jsonl(run_dir / "results.jsonl")
    error_rows = read_jsonl(run_dir / "model_errors.jsonl")
    query_rows = find_query_rows(run_dir)
    if not isinstance(summary_rows, list):
        raise ValueError("summary.json bir liste içermelidir.")
    LOG.info(
        "Kayıt sayıları | sonuç=%d | özet=%d | sorgu=%d | model_hatası=%d",
        len(result_rows),
        len(summary_rows),
        len(query_rows),
        len(error_rows),
    )

    wb = Workbook()
    wb.remove(wb.active)
    wb.create_sheet("Özet")
    summary_ws = wb.create_sheet("Model Özeti")
    raw_ws = wb.create_sheet("Ham Sonuçlar")
    query_ws = wb.create_sheet("Sorgu Manifesti")
    config_ws = wb.create_sheet("Yapılandırma")
    error_ws = wb.create_sheet("Hatalar")
    try:
        wb.calculation.fullCalcOnLoad = True
        wb.calculation.forceFullCalc = True
        wb.calculation.calcMode = "auto"
    except AttributeError:
        pass

    summary_columns = ordered_columns(summary_rows, SUMMARY_COLUMNS)
    LOG.info("Model Özeti sayfası yazılıyor (%d satır).", len(summary_rows))
    write_table(
        summary_ws,
        summary_rows,
        summary_columns,
        table_name="ModelSummary_01",
        number_formats=SUMMARY_FORMATS,
    )
    set_widths(
        summary_ws,
        summary_columns,
        overrides={"direction": 38, "query_variant": 16, "search_mode": 18, "model_id": 48},
    )
    if summary_ws.max_row >= 2 and "coverage" in summary_columns:
        coverage_letter = get_column_letter(summary_columns.index("coverage") + 1)
        summary_ws.conditional_formatting.add(
            f"{coverage_letter}2:{coverage_letter}{summary_ws.max_row}",
            CellIsRule(operator="lessThan", formula=["1"], fill=PatternFill("solid", fgColor=BAD)),
        )
    if summary_ws.max_row >= 2 and "success_30m" in summary_columns:
        success_letter = get_column_letter(summary_columns.index("success_30m") + 1)
        summary_ws.conditional_formatting.add(
            f"{success_letter}2:{success_letter}{summary_ws.max_row}",
            ColorScaleRule(
                start_type="min",
                start_color="FECACA",
                mid_type="percentile",
                mid_value=50,
                mid_color="FEF3C7",
                end_type="max",
                end_color="BBF7D0",
            ),
        )
    add_error_conditional_formatting(summary_ws, summary_columns)

    result_columns = ordered_columns(result_rows, ["run_id", "status"])
    per_sheet = MAX_EXCEL_ROWS - 1
    raw_chunks: list[tuple[str, Sequence[dict[str, Any]], Sequence[str]]] = []
    chunks = [result_rows[index : index + per_sheet] for index in range(0, len(result_rows), per_sheet)]
    if not chunks:
        chunks = [[]]
    for chunk_index, chunk in enumerate(chunks, start=1):
        if chunk_index == 1:
            ws = raw_ws
            sheet_name = "Ham Sonuçlar"
        else:
            sheet_name = f"Ham Sonuçlar {chunk_index}"
            ws = wb.create_sheet(sheet_name)
        LOG.info("%s sayfası yazılıyor (%d satır).", sheet_name, len(chunk))
        write_table(
            ws,
            chunk,
            result_columns,
            table_name=safe_table_name("RawResults", chunk_index),
            number_formats=RESULT_FORMATS,
        )
        set_widths(
            ws,
            result_columns,
            overrides={
                "model_id": 48,
                "model_file": 56,
                "source_query_raster": 48,
                "source_map_raster": 48,
                "reason": 38,
            },
        )
        add_error_conditional_formatting(ws, result_columns)
        raw_chunks.append((sheet_name, chunk, result_columns))

    query_columns = ordered_columns(query_rows, QUERY_COLUMNS)
    LOG.info("Sorgu Manifesti sayfası yazılıyor (%d satır).", len(query_rows))
    write_table(
        query_ws,
        query_rows,
        query_columns,
        table_name="QueryManifest_01",
        number_formats=QUERY_FORMATS,
    )
    set_widths(
        query_ws,
        query_columns,
        overrides={"direction": 38, "raw_tile_file": 58},
    )

    config_rows = [
        {"Parametre": key, "Değer": excel_value(value)} for key, value in sorted(config.items())
    ]
    LOG.info("Yapılandırma sayfası yazılıyor (%d parametre).", len(config_rows))
    write_table(
        config_ws,
        config_rows,
        ["Parametre", "Değer"],
        table_name="RunConfig_01",
    )
    set_widths(config_ws, ["Parametre", "Değer"], overrides={"Parametre": 34, "Değer": 90})
    for cell in config_ws["B"]:
        cell.alignment = Alignment(vertical="top", wrap_text=True)

    error_columns = ordered_columns(
        error_rows, ["created_at_utc", "direction", "model_id", "error_type", "error"]
    )
    LOG.info("Hatalar sayfası yazılıyor (%d satır).", len(error_rows))
    write_table(
        error_ws,
        error_rows,
        error_columns,
        table_name="ModelErrors_01",
    )
    set_widths(
        error_ws,
        error_columns,
        overrides={"model_file": 58, "error": 62, "traceback": 70},
    )
    for name in ("error", "traceback"):
        if name in error_columns:
            index = error_columns.index(name) + 1
            for row in range(2, error_ws.max_row + 1):
                error_ws.cell(row=row, column=index).alignment = Alignment(vertical="top", wrap_text=True)

    LOG.info("Özet panosu ve formül bağlı grafik hazırlanıyor.")
    write_dashboard(wb, config, summary_rows, raw_chunks)
    wb.active = wb.sheetnames.index("Özet")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    LOG.info("Çalışma kitabı geçici dosyaya yazılıyor: %s", temporary)
    wb.save(temporary)
    os.replace(temporary, output_path)
    LOG.info("Çalışma kitabı kaydedildi: %s (%.2f MiB)", output_path, output_path.stat().st_size / 2**20)
    return validate_workbook(output_path, run_dir)


def validate_workbook(output_path: Path, run_dir: Path) -> dict[str, Any]:
    LOG.info("XLSX ZIP bütünlüğü ve sayfa yapısı doğrulanıyor.")
    with zipfile.ZipFile(output_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"XLSX arşiv üyesi bozuk: {bad_member}")

    wb = load_workbook(output_path, data_only=False, read_only=False)
    required = ["Özet", "Model Özeti", "Ham Sonuçlar", "Sorgu Manifesti", "Yapılandırma", "Hatalar"]
    missing = [name for name in required if name not in wb.sheetnames]
    if missing:
        raise RuntimeError(f"Eksik çalışma sayfaları: {missing}")

    formula_count = 0
    formula_error_literals: list[str] = []
    sheets: list[dict[str, Any]] = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                value = cell.value
                if isinstance(value, str) and value.startswith("="):
                    formula_count += 1
                elif isinstance(value, str) and value in {
                    "#REF!",
                    "#DIV/0!",
                    "#VALUE!",
                    "#NAME?",
                    "#N/A",
                }:
                    formula_error_literals.append(f"{ws.title}!{cell.coordinate}:{value}")
        sheets.append(
            {
                "name": ws.title,
                "rows": ws.max_row,
                "columns": ws.max_column,
                "tables": sorted(ws.tables.keys()),
                "charts": len(ws._charts),
            }
        )
    if formula_error_literals:
        raise RuntimeError(f"Formül hata sabitleri bulundu: {formula_error_literals[:10]}")

    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    report = {
        "validated_at_utc": utc_now_iso(),
        "engine": "openpyxl",
        "openpyxl_version": __import__("openpyxl").__version__,
        "workbook": str(output_path.resolve()),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256,
        "sheet_count": len(wb.sheetnames),
        "sheets": sheets,
        "formula_count": formula_count,
        "formula_error_literals": formula_error_literals,
        "zip_integrity": "ok",
    }
    validation_path = run_dir / "excel_validation.json"
    validation_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info(
        "XLSX doğrulandı | sayfa=%d | formül=%d | grafik=%d | sha256=%s",
        len(wb.sheetnames),
        formula_count,
        sum(item["charts"] for item in sheets),
        sha256[:16],
    )
    LOG.info("Doğrulama kaydı: %s", validation_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | EXCEL | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    output_path = (args.output or (run_dir / "benchmark_results.xlsx")).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Koşu klasörü bulunamadı: {run_dir}")
    build_workbook(run_dir, output_path)
    print(f"Workbook ready: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
