#!/usr/bin/env node
// Build the benchmark workbook from canonical JSON/JSONL outputs.
// Spreadsheet authoring intentionally uses @oai/artifact-tool only.

import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (!key.startsWith("--")) continue;
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) {
      result[key.slice(2)] = true;
    } else {
      result[key.slice(2)] = value;
      index += 1;
    }
  }
  return result;
}


async function readJson(filePath, fallback) {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return fallback;
    throw error;
  }
}


async function readJsonl(filePath) {
  try {
    const text = await fs.readFile(filePath, "utf8");
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
}


async function findFiles(root, targetName) {
  const found = [];
  async function walk(directory) {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) await walk(full);
      else if (entry.name === targetName) found.push(full);
    }
  }
  await walk(root);
  return found.sort();
}


function matrixFromRows(rows, columns) {
  return [columns, ...rows.map((row) => columns.map((column) => row?.[column] ?? null))];
}


function applyHeader(range) {
  range.format = {
    fill: "#374151",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#6B7280" },
  };
  range.format.rowHeight = 30;
}


function styleFlatSheet(sheet, usedRange, freezeRows = 1) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(freezeRows);
  usedRange.format.font = { name: "Aptos", size: 10, color: "#111827" };
  usedRange.format.verticalAlignment = "center";
  usedRange.format.borders = {
    insideHorizontal: { style: "thin", color: "#E5E7EB" },
    bottom: { style: "thin", color: "#D1D5DB" },
  };
}


function addTable(sheet, matrix, startRow, startCol, tableName) {
  const rowCount = matrix.length;
  const colCount = matrix[0].length;
  const range = sheet.getRangeByIndexes(startRow, startCol, rowCount, colCount);
  range.values = matrix;
  const table = sheet.tables.add(range, true, tableName);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
  applyHeader(range.getRow(0));
  return { range, table };
}


function numberFormatColumns(sheet, columns, formats, firstDataRow, lastDataRow) {
  for (let index = 0; index < columns.length; index += 1) {
    const format = formats[columns[index]];
    if (!format || lastDataRow < firstDataRow) continue;
    sheet
      .getRangeByIndexes(firstDataRow, index, lastDataRow - firstDataRow + 1, 1)
      .format.numberFormat = format;
  }
}


function safeTableName(prefix, index) {
  return `${prefix}${String(index + 1).padStart(2, "0")}`;
}


const args = parseArgs(process.argv.slice(2));
if (!args["run-dir"]) throw new Error("--run-dir zorunludur");
const runDir = path.resolve(args["run-dir"]);
const outputPath = path.resolve(args.output ?? path.join(runDir, "benchmark_results.xlsx"));
const previewDir = path.join(runDir, "excel_previews");

const config = await readJson(path.join(runDir, "run_config.json"), {});
const summaryRows = await readJson(path.join(runDir, "summary.json"), []);
const resultRows = await readJsonl(path.join(runDir, "results.jsonl"));
const errorRows = await readJsonl(path.join(runDir, "model_errors.jsonl"));
const manifestFiles = await findFiles(runDir, "query_manifest.json");
const queryRows = [];
for (const manifestPath of manifestFiles) {
  const manifest = await readJson(manifestPath, {});
  const direction = path.basename(path.dirname(path.dirname(manifestPath)));
  for (const query of manifest.queries ?? []) queryRows.push({ direction, ...query });
}

const workbook = Workbook.create();
const dashboard = workbook.worksheets.add("Özet");
const summarySheet = workbook.worksheets.add("Model Özeti");
const rawSheet = workbook.worksheets.add("Ham Sonuçlar");
const querySheet = workbook.worksheets.add("Sorgu Manifesti");
const configSheet = workbook.worksheets.add("Yapılandırma");
const errorSheet = workbook.worksheets.add("Hatalar");

// Dashboard
dashboard.showGridLines = false;
dashboard.getRange("A1:H2").merge();
dashboard.getRange("A1").values = [["Jeoreferanslı Ortak-Temsil Benchmark Sonuçları"]];
dashboard.getRange("A1:H2").format = {
  fill: "#1F2937",
  font: { name: "Aptos Display", size: 18, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
dashboard.getRange("A4:B8").values = [
  ["Koşu", config.run_id ?? path.basename(runDir)],
  ["Toplam kayıt", resultRows.length],
  ["Model/yön kombinasyonu", summaryRows.length],
  ["Başarılı kayıt", resultRows.filter((row) => row.status === "ok").length],
  ["Hata/reddetme", resultRows.filter((row) => row.status !== "ok").length],
];
dashboard.getRange("A4:A8").format = {
  fill: "#E5E7EB",
  font: { bold: true, color: "#374151" },
};
dashboard.getRange("B4:B8").format = {
  fill: "#F9FAFB",
  font: { bold: true, color: "#111827" },
  numberFormat: "#,##0",
};
dashboard.getRange("A10:H11").merge();
dashboard.getRange("A10").values = [[
  "Ana değerlendirme: aynı model hem sorgu parçasına hem arama haritasına uygulanır. Ana başarı ölçütü tüm sorgular üzerinden 30 m içinde konumlamadır; tüm-hata ortalaması yalnız tanısaldır.",
]];
dashboard.getRange("A10:H11").format = {
  fill: "#F3F4F6",
  font: { italic: true, color: "#4B5563" },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: "#D1D5DB" },
};

const ranked = [...summaryRows]
  .filter((row) => row.success_30m !== null && row.success_30m !== undefined)
  .sort((a, b) =>
    Number(b.success_30m) - Number(a.success_30m)
    || Number(b.auc_30m ?? 0) - Number(a.auc_30m ?? 0)
    || Number(a.median_error_under_30m ?? Number.POSITIVE_INFINITY)
      - Number(b.median_error_under_30m ?? Number.POSITIVE_INFINITY)
  )
  .slice(0, 15);
dashboard.getRange("A14:D14").values = [["Model", "30 m başarı", "30 m içi medyan hata", "AUC@30m"]];
dashboard.getRange("A15:D29").values = Array.from({ length: 15 }, (_, index) => {
  const row = ranked[index];
  return row
    ? [
        `${row.model_id} [${row.search_mode ?? "global"}]`,
        row.success_30m,
        row.median_error_under_30m,
        row.auc_30m,
      ]
    : [null, null, null, null];
});
applyHeader(dashboard.getRange("A14:D14"));
dashboard.getRange("B15:B29").format.numberFormat = "0.0%";
dashboard.getRange("C15:C29").format.numberFormat = "0.00";
dashboard.getRange("D15:D29").format.numberFormat = "0.0%";
dashboard.getRange("A14:D29").format.borders = {
  insideHorizontal: { style: "thin", color: "#E5E7EB" },
  bottom: { style: "thin", color: "#D1D5DB" },
};
if (ranked.length > 0) {
  const chartEnd = 14 + ranked.length;
  const chart = dashboard.charts.add("bar", dashboard.getRange(`A14:B${chartEnd}`));
  chart.title = "30 m konumlama başarısı - en iyi modeller";
  chart.hasLegend = false;
  chart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
  chart.yAxis = { numberFormatCode: "0.0%" };
  chart.setPosition("F14", "M31");
}
dashboard.freezePanes.freezeRows(2);
dashboard.getRange("A1:M31").format.font = { name: "Aptos", size: 10, color: "#111827" };
dashboard.getRange("A1:A31").format.columnWidth = 42;
dashboard.getRange("B1:D31").format.columnWidth = 17;
dashboard.getRange("E1:E31").format.columnWidth = 3;
dashboard.getRange("E1:M31").format.columnWidth = 12;

// Model summary sheet
const summaryColumns = [
  "direction", "search_mode", "model_id", "total_queries", "ok_queries", "rejected_queries",
  "error_queries", "coverage", "success_30m_queries", "success_30m",
  "success_30m_ci95_low", "success_30m_ci95_high", "success_30m_failure_rate",
  "auc_30m", "mean_error_under_30m", "median_error_under_30m",
  "mean_error_m", "median_error_m",
  "median_error_ci95_low", "median_error_ci95_high", "p90_error_m", "p95_error_m",
  "success_5m", "success_10m", "success_10m_ci95_low", "success_10m_ci95_high",
  "success_25m", "success_50m",
  "mean_top1_score", "mean_search_seconds", "total_search_seconds",
];
const summaryMatrix = matrixFromRows(summaryRows, summaryColumns);
const summaryBlock = addTable(summarySheet, summaryMatrix, 0, 0, "ModelSummaryTable");
styleFlatSheet(summarySheet, summaryBlock.range);
numberFormatColumns(summarySheet, summaryColumns, {
  coverage: "0.0%", success_30m: "0.0%", success_30m_ci95_low: "0.0%",
  success_30m_ci95_high: "0.0%", success_30m_failure_rate: "0.0%", auc_30m: "0.0%",
  mean_error_under_30m: "0.00", median_error_under_30m: "0.00",
  mean_error_m: "0.00", median_error_m: "0.00",
  median_error_ci95_low: "0.00", median_error_ci95_high: "0.00",
  success_10m_ci95_low: "0.0%", success_10m_ci95_high: "0.0%",
  p90_error_m: "0.00", p95_error_m: "0.00", success_5m: "0.0%",
  success_10m: "0.0%", success_25m: "0.0%", success_50m: "0.0%",
  mean_top1_score: "0.0000", mean_search_seconds: "0.000",
  total_search_seconds: "0.0",
}, 1, summaryRows.length);
summarySheet.getRangeByIndexes(0, 0, summaryRows.length + 1, 1).format.columnWidth = 36;
summarySheet.getRangeByIndexes(0, 1, summaryRows.length + 1, 1).format.columnWidth = 18;
summarySheet.getRangeByIndexes(0, 2, summaryRows.length + 1, 1).format.columnWidth = 48;
summarySheet.getRangeByIndexes(0, 3, summaryRows.length + 1, summaryColumns.length - 3).format.columnWidth = 16;

// Raw results sheet
const resultColumns = resultRows.length > 0 ? Object.keys(resultRows[0]) : ["run_id", "status"];
const maxExcelRows = 1_048_575;
const rawChunks = [];
for (let offset = 0; offset < resultRows.length || (resultRows.length === 0 && offset === 0); offset += maxExcelRows) {
  rawChunks.push(resultRows.slice(offset, offset + maxExcelRows));
  if (resultRows.length === 0) break;
}
const firstRawMatrix = matrixFromRows(rawChunks[0], resultColumns);
const rawBlock = addTable(rawSheet, firstRawMatrix, 0, 0, "RawResultsTable01");
styleFlatSheet(rawSheet, rawBlock.range);
numberFormatColumns(rawSheet, resultColumns, {
  query_center_easting_m: "0.000", query_center_northing_m: "0.000",
  expected_center_easting_m: "0.000", expected_center_northing_m: "0.000",
  predicted_center_easting_m: "0.000", predicted_center_northing_m: "0.000",
  error_m: "0.000", error_px: "0.000", top1_score: "0.0000",
  top2_score: "0.0000", peak_margin: "0.0000", psr: "0.00",
  query_std: "0.00", query_entropy: "0.00", query_dark_fraction: "0.0%",
  success_30m: "0",
  search_seconds: "0.000", query_inference_seconds: "0.000", map_build_seconds: "0.0",
}, 1, rawChunks[0].length);
rawSheet.getRangeByIndexes(0, 0, Math.max(1, firstRawMatrix.length), resultColumns.length).format.columnWidth = 15;
for (const name of ["model_id", "model_file", "source_query_raster", "source_map_raster", "reason"]) {
  const index = resultColumns.indexOf(name);
  if (index >= 0) rawSheet.getRangeByIndexes(0, index, Math.max(1, firstRawMatrix.length), 1).format.columnWidth = name === "reason" ? 32 : 42;
}

for (let chunkIndex = 1; chunkIndex < rawChunks.length; chunkIndex += 1) {
  const sheet = workbook.worksheets.add(`Ham Sonuçlar ${chunkIndex + 1}`);
  const matrix = matrixFromRows(rawChunks[chunkIndex], resultColumns);
  const block = addTable(sheet, matrix, 0, 0, safeTableName("RawResultsTable", chunkIndex));
  styleFlatSheet(sheet, block.range);
  numberFormatColumns(sheet, resultColumns, {
    error_m: "0.000", error_px: "0.000", top1_score: "0.0000",
    top2_score: "0.0000", peak_margin: "0.0000", psr: "0.00",
  }, 1, rawChunks[chunkIndex].length);
}

// Query manifest
const queryColumns = [
  "direction", "query_id", "block_id", "center_easting_m", "center_northing_m",
  "source_row", "source_col", "query_std", "query_entropy", "dark_fraction", "raw_tile_file",
];
const queryMatrix = matrixFromRows(queryRows, queryColumns);
const queryBlock = addTable(querySheet, queryMatrix, 0, 0, "QueryManifestTable");
styleFlatSheet(querySheet, queryBlock.range);
numberFormatColumns(querySheet, queryColumns, {
  center_easting_m: "0.000", center_northing_m: "0.000", query_std: "0.00",
  query_entropy: "0.00", dark_fraction: "0.0%",
}, 1, queryRows.length);
querySheet.getRangeByIndexes(0, 0, queryRows.length + 1, 1).format.columnWidth = 36;
querySheet.getRangeByIndexes(0, 1, queryRows.length + 1, 2).format.columnWidth = 16;
querySheet.getRangeByIndexes(0, 3, queryRows.length + 1, 7).format.columnWidth = 16;
querySheet.getRangeByIndexes(0, 10, queryRows.length + 1, 1).format.columnWidth = 54;

// Configuration
const configRows = Object.entries(config).map(([key, value]) => [
  key,
  typeof value === "object" && value !== null ? JSON.stringify(value) : value,
]);
configSheet.getRangeByIndexes(0, 0, configRows.length + 1, 2).values = [["Parametre", "Değer"], ...configRows];
applyHeader(configSheet.getRange("A1:B1"));
styleFlatSheet(configSheet, configSheet.getRangeByIndexes(0, 0, configRows.length + 1, 2));
configSheet.getRangeByIndexes(0, 0, configRows.length + 1, 1).format.columnWidth = 34;
configSheet.getRangeByIndexes(0, 1, configRows.length + 1, 1).format.columnWidth = 90;
configSheet.getRangeByIndexes(0, 1, configRows.length + 1, 1).format.wrapText = true;

// Errors
const errorColumns = errorRows.length > 0
  ? Object.keys(errorRows[0])
  : ["created_at_utc", "direction", "model_id", "error_type", "error"];
const errorMatrix = matrixFromRows(errorRows, errorColumns);
const errorBlock = addTable(errorSheet, errorMatrix, 0, 0, "ModelErrorsTable");
styleFlatSheet(errorSheet, errorBlock.range);
errorSheet.getRangeByIndexes(0, 0, Math.max(1, errorMatrix.length), errorColumns.length).format.columnWidth = 24;
for (const name of ["model_file", "error", "traceback"]) {
  const index = errorColumns.indexOf(name);
  if (index >= 0) {
    errorSheet.getRangeByIndexes(0, index, Math.max(1, errorMatrix.length), 1).format.columnWidth = 60;
    errorSheet.getRangeByIndexes(0, index, Math.max(1, errorMatrix.length), 1).format.wrapText = true;
  }
}

await fs.mkdir(previewDir, { recursive: true });
const sheets = workbook.worksheets.items;
for (const sheet of sheets) {
  const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
  const safeName = sheet.name.replace(/[^a-zA-Z0-9_-]+/g, "_");
  await fs.writeFile(path.join(previewDir, `${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const keyInspect = await workbook.inspect({
  kind: "table",
  range: "Özet!A1:M31",
  include: "values,formulas",
  tableMaxRows: 31,
  tableMaxCols: 13,
  maxChars: 12000,
});
await fs.writeFile(path.join(runDir, "excel_inspect.ndjson"), keyInspect.ndjson, "utf8");
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(runDir, "excel_formula_errors.ndjson"), formulaErrors.ndjson, "utf8");
if (/"matchCount"\s*:\s*[1-9]/.test(formulaErrors.ndjson)) {
  throw new Error(`Excel formül hata taraması başarısız: ${formulaErrors.ndjson}`);
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
process.stdout.write(`Workbook ready: ${outputPath}\n`);
