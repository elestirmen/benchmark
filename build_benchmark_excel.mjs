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


function modelLabel(identity, modelId) {
  if (modelId === "RAW_BASELINE") return "RAW_BASELINE";
  if (!identity) return modelId;
  const relativePath = String(identity.relative_path ?? "");
  const parts = relativePath.split(/[\\/]/).filter(Boolean);
  const parent = parts.length > 1 ? parts[parts.length - 2] : "";
  const rank = parent.match(/^(\d{2})_/);
  const lineage = rank ? `Lineage ${rank[1]}` : (parent || "Model");
  const checkpoint = identity.checkpoint_number;
  const epoch = checkpoint !== undefined && checkpoint !== null
    ? `epoch_${String(Number(checkpoint)).padStart(5, "0")}`
    : (parts.length > 0 ? parts[parts.length - 1].replace(/\.[^.]+$/, "") : modelId);
  return `${lineage} | ${epoch}`;
}

function enrichSummaryRows(rows, catalog) {
  const models = Array.isArray(catalog?.models) ? catalog.models : [];
  const byId = new Map(models.filter((item) => item?.model_id).map((item) => [String(item.model_id), item]));
  const sequenceById = new Map(
    models
      .filter((item) => item?.model_id)
      .map((item, index) => [String(item.model_id), index + 1]),
  );
  return rows.map((source) => {
    const row = { ...source };
    const modelId = String(row.model_id ?? "");
    const identity = byId.get(modelId);
    row.model_sequence = modelId === "RAW_BASELINE" ? 0 : (sequenceById.get(modelId) ?? null);
    row.model_label = modelLabel(identity, modelId);
    row.model_epoch = identity?.checkpoint_number ?? null;
    row.model_relative_path = identity?.relative_path ?? null;
    row.model_sha256 = identity?.sha256 ?? null;
    return row;
  });
}

function displayDirection(value) {
  if (value == null) return null;
  const text = String(value);
  const lowered = text.toLowerCase();
  const gmapsIndex = lowered.indexOf("gmaps");
  const bingIndex = lowered.indexOf("bingmap");
  if (gmapsIndex >= 0 && bingIndex >= 0) return gmapsIndex < bingIndex ? "Google → Bing" : "Bing → Google";
  return text.replaceAll("__TO__", " → ");
}

function compactDirectionLabel(value) {
  const text = String(value ?? "").toLowerCase();
  const gmapsIndex = text.includes("gmaps") ? text.indexOf("gmaps") : text.indexOf("google");
  const bingIndex = text.includes("bingmap") ? text.indexOf("bingmap") : text.indexOf("bing");
  if (gmapsIndex >= 0 && bingIndex >= 0) return gmapsIndex < bingIndex ? "G→B" : "B→G";
  return "Yön";
}

function displayVariant(value) {
  return { clean: "Temiz", hard_v1: "Hard" }[String(value)] ?? value;
}

function displaySearchMode(value) {
  if (value == null) return null;
  const text = String(value);
  if (text === "global") return "Global";
  const match = text.match(/^roi[_-]?(\d+)(?:m)?$/i);
  return match ? `ROI ${Number(match[1])} m` : text;
}

function prepareModelSummaryRows(rows) {
  return rows.map((source) => ({
    ...source,
    direction: displayDirection(source.direction),
    query_variant: displayVariant(source.query_variant),
    search_mode: displaySearchMode(source.search_mode),
  })).sort((a, b) => {
    const aSequence = a.model_sequence ?? Number.POSITIVE_INFINITY;
    const bSequence = b.model_sequence ?? Number.POSITIVE_INFINITY;
    return aSequence - bSequence;
  });
}

function compactModelLabel(row) {
  const label = String(row.model_label ?? row.model_id ?? "Model");
  if (label === "RAW_BASELINE") return "RAW";
  const match = label.match(/Lineage\s+(\d+)/);
  const epoch = row.model_epoch;
  if (match && epoch !== null && epoch !== undefined) {
    return "L" + String(Number(match[1])).padStart(2, "0")
      + " | E" + String(Number(epoch)).padStart(3, "0");
  }
  return epoch !== null && epoch !== undefined
    ? label.slice(0, 14) + " | E" + String(Number(epoch)).padStart(3, "0")
    : label.slice(0, 24);
}


function buildModelCatalogRows(catalog, summaryRows) {
  const models = Array.isArray(catalog?.models) ? catalog.models : [];
  const ids = [...models.map((item) => item?.model_id).filter(Boolean), ...summaryRows.map((row) => row.model_id).filter(Boolean)];
  const uniqueIds = [...new Set(ids.map((value) => String(value)))];
  const byId = new Map(models.filter((item) => item?.model_id).map((item) => [String(item.model_id), item]));
  return uniqueIds.map((modelId) => {
    const identity = byId.get(modelId);
    return {
      model_id: modelId,
      model_label: modelLabel(identity, modelId),
      model_epoch: identity?.checkpoint_number ?? null,
      model_relative_path: identity?.relative_path ?? null,
      model_sha256: identity?.sha256 ?? null,
    };
  });
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
const rawSummaryRows = await readJson(path.join(runDir, "summary.json"), []);
const modelCatalog = await readJson(path.join(runDir, "model_catalog.json"), {});
const summaryRows = enrichSummaryRows(rawSummaryRows, modelCatalog);
const summaryDisplayRows = prepareModelSummaryRows(summaryRows);
const catalogRows = buildModelCatalogRows(modelCatalog, summaryRows);
const resultRows = await readJsonl(path.join(runDir, "results.jsonl"));
const errorRows = await readJsonl(path.join(runDir, "model_errors.jsonl"));
const manifestFiles = [
  ...(await findFiles(runDir, "query_manifest.json")),
  ...(await findFiles(runDir, "query_variant_manifest.json")),
].sort();
const queryRows = [];
for (const manifestPath of manifestFiles) {
  const manifest = await readJson(manifestPath, {});
  const direction = path.relative(runDir, manifestPath).split(path.sep)[0] ?? "unknown";
  const queryVariant = manifest.query_variant ?? "clean";
  for (const query of manifest.queries ?? []) {
    queryRows.push({ direction, query_variant: queryVariant, ...query });
  }
}

const workbook = Workbook.create();
const dashboard = workbook.worksheets.add("Özet");
const summarySheet = workbook.worksheets.add("Model Özeti");
const catalogSheet = workbook.worksheets.add("Model Kataloğu");
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
  "Ana değerlendirme: aynı model hem sorgu parçasına hem arama haritasına uygulanır. clean ve hard_v1 koşulları ayrı raporlanır; ana başarı ölçütü tüm sorgular üzerinden 25 m içinde konumlamadır. Aşağıdaki sıralama yalnız global aramayı gösterir.",
]];
dashboard.getRange("A10:H11").format = {
  fill: "#F3F4F6",
  font: { italic: true, color: "#4B5563" },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: "#D1D5DB" },
};

const ranked = [...summaryDisplayRows]
  .filter((row) =>
    row.search_mode === "Global"
    && row.success_25m !== null
    && row.success_25m !== undefined
  )
  .sort((a, b) =>
    Number(b.success_25m) - Number(a.success_25m)
    || Number(b.auc_25m ?? 0) - Number(a.auc_25m ?? 0)
    || Number(a.median_error_under_25m ?? Number.POSITIVE_INFINITY)
      - Number(b.median_error_under_25m ?? Number.POSITIVE_INFINITY)
  )
  .slice(0, 10);

dashboard.getRange("A14:G14").values = [[
  "Sıra", "Model", "Senaryo", "Yön", "Başarı ≤25 m", "AUC@25 m", "Medyan hata (m)",
]];
dashboard.getRange("A15:G24").values = Array.from({ length: 10 }, (_, index) => {
  const row = ranked[index];
  return row
    ? [
        index + 1,
        compactModelLabel(row),
        row.query_variant,
        row.direction,
        row.success_25m,
        row.auc_25m,
        row.median_error_under_25m,
      ]
    : [null, null, null, null, null, null, null];
});
applyHeader(dashboard.getRange("A14:G14"));
dashboard.getRange("E15:F24").format.numberFormat = "0.0%";
dashboard.getRange("G15:G24").format.numberFormat = "0.00";
dashboard.getRange("A14:G24").format.borders = {
  insideHorizontal: { style: "thin", color: "#E5E7EB" },
  bottom: { style: "thin", color: "#D1D5DB" },
};

const chartRows = ranked.slice(0, 5);
dashboard.getRange("I14:J14").values = [["Model | Senaryo", "Başarı (%)"]];
if (chartRows.length > 0) {
  dashboard.getRange(`I15:I${14 + chartRows.length}`).values = chartRows.map((row) => [
    `${compactModelLabel(row)} | ${row.query_variant} | ${compactDirectionLabel(row.direction)}`,
  ]);
  dashboard.getRange(`J15:J${14 + chartRows.length}`).formulas = chartRows.map((_, index) => {
    const row = 15 + index;
    return [`=E${row}*100`];
  });
}
applyHeader(dashboard.getRange("I14:J14"));
dashboard.getRange("J15:J19").format.numberFormat = "0.0";
dashboard.getRange("I14:J19").format.borders = {
  insideHorizontal: { style: "thin", color: "#E5E7EB" },
  bottom: { style: "thin", color: "#D1D5DB" },
};
if (chartRows.length > 0) {
  const chartEnd = 14 + chartRows.length;
  const chart = dashboard.charts.add("bar", dashboard.getRange("I14:J" + chartEnd));
  chart.title = "En iyi 5 model | 25 m başarı (%)";
  chart.hasLegend = false;
  chart.xAxis = { numberFormatCode: "0", min: 0, max: 100, textStyle: { fontSize: 9 } };
  chart.yAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
  chart.setPosition("L14", "R29");
}
dashboard.freezePanes.freezeRows(2);
// Compact human-facing model summary.
const summaryColumns = [
  "model_sequence", "direction", "query_variant", "search_mode", "model_label", "model_id", "model_epoch",
  "total_queries", "ok_queries", "coverage", "success_25m", "auc_25m",
  "median_error_under_25m", "p90_error_m", "mean_search_seconds",
];
const summaryHeaders = [
  "Sıra No", "Yön", "Senaryo", "Arama", "Model", "Model ID", "Epoch", "N", "Başarılı N",
  "Kapsam", "Başarı ≤25 m", "AUC@25 m", "Medyan hata ≤25 m (m)",
  "P90 hata (m)", "Ort. arama (sn)",
];
const summaryMatrix = [summaryHeaders, ...summaryDisplayRows.map((row) => summaryColumns.map((column) => row?.[column] ?? null))];
const summaryBlock = addTable(summarySheet, summaryMatrix, 0, 0, "ModelSummaryTable");
styleFlatSheet(summarySheet, summaryBlock.range);
numberFormatColumns(summarySheet, summaryColumns, {
  coverage: "0.0%", success_25m: "0.0%", auc_25m: "0.0%",
  median_error_under_25m: "0.00", p90_error_m: "0.00", mean_search_seconds: "0.000",
  model_sequence: "0", model_epoch: "0", total_queries: "#,##0", ok_queries: "#,##0",
}, 1, summaryDisplayRows.length);
summarySheet.getRangeByIndexes(0, 0, summaryDisplayRows.length + 1, 1).format.columnWidth = 10;
summarySheet.getRangeByIndexes(0, 1, summaryDisplayRows.length + 1, 1).format.columnWidth = 18;
summarySheet.getRangeByIndexes(0, 2, summaryDisplayRows.length + 1, 2).format.columnWidth = 14;
summarySheet.getRangeByIndexes(0, 4, summaryDisplayRows.length + 1, 1).format.columnWidth = 28;
summarySheet.getRangeByIndexes(0, 5, summaryDisplayRows.length + 1, 1).format.columnWidth = 48;
summarySheet.getRangeByIndexes(0, 6, summaryDisplayRows.length + 1, 4).format.columnWidth = 11;
summarySheet.getRangeByIndexes(0, 10, summaryDisplayRows.length + 1, 2).format.columnWidth = 14;
summarySheet.getRangeByIndexes(0, 12, summaryDisplayRows.length + 1, 1).format.columnWidth = 20;
summarySheet.getRangeByIndexes(0, 13, summaryDisplayRows.length + 1, 2).format.columnWidth = 16;

// Technical model identity and audit fields.
const catalogColumns = ["model_id", "model_label", "model_epoch", "model_relative_path", "model_sha256"];
const catalogHeaders = ["Model ID (kanonik)", "Model", "Epoch", "Göreli yol", "SHA256"];
const catalogMatrix = [catalogHeaders, ...catalogRows.map((row) => catalogColumns.map((column) => row?.[column] ?? null))];
const catalogBlock = addTable(catalogSheet, catalogMatrix, 0, 0, "ModelCatalogTable");
styleFlatSheet(catalogSheet, catalogBlock.range);
numberFormatColumns(catalogSheet, catalogColumns, { model_epoch: "0" }, 1, catalogRows.length);
catalogSheet.getRangeByIndexes(0, 0, catalogRows.length + 1, 1).format.columnWidth = 48;
catalogSheet.getRangeByIndexes(0, 1, catalogRows.length + 1, 1).format.columnWidth = 28;
catalogSheet.getRangeByIndexes(0, 2, catalogRows.length + 1, 1).format.columnWidth = 10;
catalogSheet.getRangeByIndexes(0, 3, catalogRows.length + 1, 1).format.columnWidth = 60;
catalogSheet.getRangeByIndexes(0, 4, catalogRows.length + 1, 1).format.columnWidth = 68;

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
  success_25m: "0",
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
  "direction", "query_variant", "query_id", "block_id", "center_easting_m", "center_northing_m",
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
querySheet.getRangeByIndexes(0, 1, queryRows.length + 1, 3).format.columnWidth = 16;
querySheet.getRangeByIndexes(0, 4, queryRows.length + 1, 7).format.columnWidth = 16;
querySheet.getRangeByIndexes(0, 11, queryRows.length + 1, 1).format.columnWidth = 54;

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
