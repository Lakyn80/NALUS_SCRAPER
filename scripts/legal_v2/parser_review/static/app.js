const classes = [
  "metadata", "heading", "numbered_paragraph_start", "numbered_paragraph_continuation",
  "prose_start", "prose_continuation", "citation_continuation", "list_or_table",
  "signature", "instruction", "layout_noise", "unresolved"
];
const boundaryDecisions = ["split", "merge", "preserve_parser", "unresolved"];
let documents = [];
let current = null;

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function postJson(url, body) {
  const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function options(values, selected, labeler = value => value) {
  return values.map(v => `<option value="${v}" ${v === selected ? "selected" : ""}>${escapeHtml(labeler(v))}</option>`).join("");
}

async function load() {
  const docs = await getJson("/api/documents");
  documents = docs.documents;
  document.getElementById("documents").innerHTML = documents.map(d =>
    `<li data-id="${d.document_id}">
      <span>${String(d.review_number).padStart(2, "0")} ${d.court}</span>
      <span class="status-badge ${parserStatusClass(d.parser_validation_status)}">${escapeHtml(d.parser_validation_label)}</span>
      <br><span class="pill">${d.source_id}</span>
      <small>Manual review: ${d.manual_line_reviewed}/${d.manual_line_total} lines · ${d.manual_boundary_reviewed}/${d.manual_boundary_total} boundaries</small>
    </li>`
  ).join("");
  for (const li of document.querySelectorAll("li[data-id]")) li.onclick = () => selectDoc(li.dataset.id);
  await refreshProgress();
  if (documents.length) await selectDoc(documents[0].document_id);
}

async function refreshProgress() {
  const progress = await getJson("/api/progress");
  document.getElementById("progress").innerHTML = `
    <section class="sidebar-progress">
      <h2>Parser validation</h2>
      <p>${progress.parser_validation.line_validated}/${progress.parser_validation.line_total} lines · ${progress.parser_validation.boundary_validated}/${progress.parser_validation.boundary_total} boundaries</p>
      <p>${progress.parser_validation.block_validated}/${progress.parser_validation.block_total} blocks · conflicts ${progress.parser_validation.conflicts}</p>
      <h2>Manual review</h2>
      <p>${progress.line_reviewed}/${progress.line_total} lines · ${progress.boundary_reviewed}/${progress.boundary_total} boundaries</p>
      <p>not reviewed ${progress.manual_review.not_reviewed} · stale ${progress.manual_review.stale}</p>
    </section>`;
}

async function selectDoc(id) {
  current = documents.find(d => d.document_id === id);
  document.querySelectorAll("li").forEach(li => li.classList.toggle("active", li.dataset.id === id));
  document.getElementById("documentTitle").textContent = `${current.review_number}. ${current.case_number || current.source_id}`;
  document.getElementById("documentMeta").innerHTML = `
    <span class="status-badge ${parserStatusClass(current.parser_validation_status)}">${escapeHtml(current.parser_validation_label)}</span>
    <span>${escapeHtml(current.court)} · ${escapeHtml(current.decision_date || "")} · ${escapeHtml(current.document_id)}</span>
    <span>Parser v7: ${current.parser_line_validated}/${current.parser_line_total} lines · ${current.parser_boundary_validated}/${current.parser_boundary_total} boundaries · ${current.parser_block_validated}/${current.parser_block_total} blocks</span>
    <span>Manual review: ${current.manual_line_reviewed}/${current.manual_line_total} lines · ${current.manual_boundary_reviewed}/${current.manual_boundary_total} boundaries</span>`;
  await render();
}

async function render() {
  if (!current) return;
  const mode = document.getElementById("mode").value;
  if (mode === "assisted") return renderAssisted();
  if (mode === "parser-v7-changes") return renderParserV7Changes();
  if (mode === "parser-v6-changes") return renderParserV6Changes();
  if (mode === "problems") return renderProblems();
  if (mode === "progress") return renderProgressView();
  if (mode === "full-corpus-v7") return renderFullCorpusV7();
  if (mode === "full-corpus-v6") return renderFullCorpusV6();
  if (mode === "boundaries") return renderBoundaries();
  return renderLines();
}

async function renderLines() {
  const data = await getJson(`/api/lines?document_id=${encodeURIComponent(current.document_id)}`);
  document.getElementById("work").innerHTML = `<div class="line-list">` +
    data.lines.map(renderLineCard).join("") + "</div>";
}

function renderLineCard(line) {
  return `<article class="line-card">
    <header class="item-status-header">
      ${renderParserBadge(line)}
      ${renderManualBadge(line)}
    </header>
    <h3>Line ${line.raw_line_number}</h3>
    <dl>
      <dt>Parser v7 result</dt><dd>${escapeHtml(line.parser_proposed_line_class)}</dd>
      <dt>Reason</dt><dd>${escapeHtml(line.parser_validation_reason || line.parser_reason_code || "")}</dd>
    </dl>
    <p class="raw-line">${escapeHtml(line.raw_text)}</p>
    <details class="manual-controls">
      <summary>Manual review</summary>
      <div class="controls">
        <select id="class-${line.item_id}">${options(classes, line.manual_class || "unresolved")}</select>
        <input id="comment-${line.item_id}" placeholder="comment">
        <button onclick="saveLine('${line.item_id}', '${line.document_id}', '${line.source_checksum}')">Save line</button>
      </div>
    </details>
  </article>`;
}

async function renderBoundaries() {
  const data = await getJson(`/api/boundary-cards?document_id=${encodeURIComponent(current.document_id)}`);
  document.getElementById("work").innerHTML = `<div class="boundary-list">` +
    data.boundaries.map(renderBoundaryCard).join("") + "</div>";
}

function renderBoundaryCard(card) {
  const selectId = `boundary-${safeId(card.boundary_id)}`;
  const commentId = `comment-${safeId(card.boundary_id)}`;
  const statusId = `save-${safeId(card.boundary_id)}`;
  const parserDisplay = card.parser_boundary.display;
  return `<article class="boundary-card" aria-labelledby="title-${safeId(card.boundary_id)}">
    <header class="boundary-card-header">
      <div>
        <h3 id="title-${safeId(card.boundary_id)}">Boundary ${card.boundary_number}: Line ${card.before.line_number} -> Line ${card.after.line_number}</h3>
        <p>${renderParserBadge(card)} ${renderManualBadge(card)}</p>
      </div>
      <div class="decision-badges" aria-label="Boundary decisions">
        <span class="decision-badge parser">PARSER v7: ${parserDisplay}</span>
        <span class="decision-badge previous">PREVIOUS: ${card.previous_boundary.display}</span>
      </div>
    </header>

    <section class="context-panel" aria-label="Surrounding source context">
      <h4>Context</h4>
      ${card.context_before.map(line => renderContextLine(line, false)).join("")}
      ${renderContextLine(card.before, true)}
      <div class="boundary-marker">BOUNDARY BETWEEN L${card.before.line_number} AND L${card.after.line_number}</div>
      ${renderContextLine(card.after, true)}
      ${card.context_after.map(line => renderContextLine(line, false)).join("")}
    </section>

    <div class="boundary-lines">
      ${renderBoundaryLine("LINE BEFORE BOUNDARY", card.before)}
      ${renderBoundaryLine("LINE AFTER BOUNDARY", card.after)}
    </div>

    <section class="decision-panel">
      <div>
        <h4>Parser v7 result</h4>
        <p><strong>PARSER v7: ${parserDisplay}</strong></p>
        <p>${escapeHtml(boundaryExplanation(card.parser_boundary, card.before.line_number, card.after.line_number))}</p>
        <p>${escapeHtml(card.parser_block_context)}</p>
      </div>
      <div>
        <h4>Previous Annotation</h4>
        <p><strong>PREVIOUS: ${card.previous_boundary.display}</strong></p>
        <p>${escapeHtml(boundaryExplanation(card.previous_boundary, card.before.line_number, card.after.line_number))}</p>
        <p>${escapeHtml(card.conflict.text)}</p>
      </div>
    </section>

    <details class="manual-panel manual-controls">
      <summary>Manual review</summary>
      <label for="${selectId}">Decision</label>
      <select id="${selectId}" onchange="updateBoundaryPreview('${card.boundary_id}', '${parserDisplay}', ${card.before.line_number}, ${card.after.line_number})">
        ${options(boundaryDecisions, card.manual_decision.stored_value || "preserve_parser", value => boundaryChoiceLabel(value, parserDisplay, card.before.line_number, card.after.line_number))}
      </select>
      <p id="preview-${safeId(card.boundary_id)}" class="manual-preview">${escapeHtml(manualPreview(card.manual_decision.stored_value || "preserve_parser", parserDisplay, card.before.line_number, card.after.line_number))}</p>
      <label for="${commentId}">Comment</label>
      <input id="${commentId}" placeholder="Optional reviewer comment">
      <button onclick="saveBoundary('${card.boundary_id}', '${card.document_id}', '${card.source_checksum}', '${parserDisplay}', ${card.before.line_number}, ${card.after.line_number})">Save boundary</button>
      <p id="${statusId}" class="save-status" role="status" aria-live="polite"></p>
    </details>

    ${card.suspicious_reasons.length ? `<section class="suspicious"><h4>Suspicious Reasons</h4><p>${card.suspicious_reasons.map(escapeHtml).join(", ")}</p></section>` : ""}
  </article>`;
}

function renderBoundaryLine(label, line) {
  return `<section class="source-line-detail">
    <h4>${label}</h4>
    <dl>
      <dt>Line</dt><dd>${line.line_number}</dd>
      <dt>Page</dt><dd>${line.page ?? "n/a"}</dd>
      <dt>Parser block</dt><dd>${escapeHtml(line.parser_block_id || "n/a")}</dd>
      <dt>Parser class</dt><dd>${escapeHtml(line.parser_class || "n/a")}</dd>
      <dt>Previous class</dt><dd>${escapeHtml(line.previous_class || "n/a")}</dd>
    </dl>
    <p class="raw-line">${escapeHtml(line.raw_text || "")}</p>
  </section>`;
}

function renderContextLine(line, highlighted) {
  return `<div class="context-line ${highlighted ? "highlight" : ""}">
    <span class="context-number">L${line.line_number}</span>
    <span class="context-text">${escapeHtml(line.raw_text || "")}</span>
  </div>`;
}

function boundaryChoiceLabel(value, parserDisplay, beforeLine, afterLine) {
  if (value === "preserve_parser") return `Accept parser: ${parserDisplay}`;
  if (value === "split") return `Force SPLIT before line ${afterLine}`;
  if (value === "merge") return `Force MERGE with line ${beforeLine}`;
  return "Manual review: mark as conflict";
}

function manualPreview(value, parserDisplay, beforeLine, afterLine) {
  if (value === "preserve_parser") return `This will save: ${parserDisplay} between lines ${beforeLine} and ${afterLine}. preserve_parser -> ${parserDisplay}.`;
  if (value === "split") return `This will save: SPLIT before line ${afterLine}.`;
  if (value === "merge") return `This will save: MERGE between lines ${beforeLine} and ${afterLine}.`;
  return "This will save: manual conflict for this boundary.";
}

function boundaryExplanation(boundary, beforeLine, afterLine) {
  if (boundary.display === "SPLIT") return `SPLIT: line ${afterLine} starts a new block.`;
  if (boundary.display === "MERGE") return `MERGE: lines ${beforeLine} and ${afterLine} remain in the same block.`;
  return boundary.explanation || "No boundary annotation is available.";
}

function updateBoundaryPreview(itemId, parserDisplay, beforeLine, afterLine) {
  const select = document.getElementById(`boundary-${safeId(itemId)}`);
  const preview = document.getElementById(`preview-${safeId(itemId)}`);
  preview.textContent = manualPreview(select.value, parserDisplay, beforeLine, afterLine);
}

async function saveLine(itemId, documentId, checksum) {
  await postJson("/api/decision", {
    item_type: "line", item_id: itemId, document_id: documentId, source_checksum: checksum,
    decision_status: "overridden", manual_class: document.getElementById(`class-${itemId}`).value,
    reviewer_comment: document.getElementById(`comment-${itemId}`).value
  });
  await refreshProgress();
  await render();
}

async function saveBoundary(itemId, documentId, checksum, parserDisplay, beforeLine, afterLine) {
  const key = safeId(itemId);
  const status = document.getElementById(`save-${key}`);
  const select = document.getElementById(`boundary-${key}`);
  status.textContent = "";
  try {
    const result = await postJson("/api/decision", {
      item_type: "boundary", item_id: itemId, document_id: documentId, source_checksum: checksum,
      decision_status: select.value === "unresolved" ? "unresolved" : "overridden",
      manual_boundary_decision: select.value,
      reviewer_comment: document.getElementById(`comment-${key}`).value
    });
    status.textContent = `Saved successfully. Revision ${result.decision.revision_number}. ${manualPreview(select.value, parserDisplay, beforeLine, afterLine)}`;
    await refreshProgress();
  } catch (err) {
    status.textContent = `Save failed: ${err.message}`;
  }
}

async function renderAssisted() {
  const data = await getJson("/api/assisted/rules");
  const summary = await getJson("/api/assisted/summary");
  const batchesByRule = new Map(data.batches.map(batch => [batch.rule_id, batch]));
  document.getElementById("work").innerHTML = `<section class="assisted-summary">
    <h3>Assisted Review</h3>
    <p>Completed evidence documents: ${summary.summary.completed_evidence_documents.length}</p>
    <p>SAFE rules: ${summary.summary.safe_rules}; REVIEW rules: ${summary.summary.review_rules}; BLOCKED rules: ${summary.summary.blocked_rules}</p>
    <p>High Court gated: ${(summary.summary.high_court_gated || []).join(", ") || "none"}</p>
  </section>
  <div class="assisted-rules">` + data.rules.map(rule => renderAssistedRule(rule, batchesByRule.get(rule.rule_id))).join("") + "</div>";
}

async function renderParserV7Changes() {
  const data = await getJson(`/api/parser-v7/changes?document_id=${encodeURIComponent(current.document_id)}`);
  const total = data.class_count + data.boundary_count + data.block_count;
  document.getElementById("work").innerHTML = `<section class="change-summary">
    <h3>Changed by parser v7</h3>
    <p>Changed lines/classes: ${data.class_count}; changed boundaries: ${data.boundary_count}; changed blocks: ${data.block_count}</p>
  </section>
  <div class="change-queue">
    ${renderChangeSection("Changed Lines / Classes", data.changed_classes, (row) => renderClassChange(row, "v6", "v7"))}
    ${renderChangeSection("Changed Boundaries", data.changed_boundaries, (row) => renderBoundaryChange(row, "v6", "v7"))}
    ${renderChangeSection("Changed Blocks", data.changed_blocks, (row) => renderBlockChange(row, "v6", "v7"))}
    ${total === 0 ? "<p>No parser v7 changes for this document.</p>" : ""}
  </div>`;
}

async function renderParserV6Changes() {
  const data = await getJson(`/api/parser-v6/changes?document_id=${encodeURIComponent(current.document_id)}`);
  const total = data.class_count + data.boundary_count + data.block_count;
  document.getElementById("work").innerHTML = `<section class="change-summary">
    <h3>Changed by parser v6 (historical)</h3>
    <p>Changed lines/classes: ${data.class_count}; changed boundaries: ${data.boundary_count}; changed blocks: ${data.block_count}</p>
  </section>
  <div class="change-queue">
    ${renderChangeSection("Changed Lines / Classes", data.changed_classes, (row) => renderClassChange(row, "v5", "v6"))}
    ${renderChangeSection("Changed Boundaries", data.changed_boundaries, (row) => renderBoundaryChange(row, "v5", "v6"))}
    ${renderChangeSection("Changed Blocks", data.changed_blocks, (row) => renderBlockChange(row, "v5", "v6"))}
    ${total === 0 ? "<p>No parser v6 changes for this document.</p>" : ""}
  </div>`;
}

function renderChangeSection(title, rows, renderer) {
  return `<section class="change-section">
    <h3>${title}</h3>
    ${rows.length ? rows.map(renderer).join("") : "<p>No items in this queue.</p>"}
  </section>`;
}

function renderBoundaryChange(change, beforeKey = "v5", afterKey = "v6") {
  const before = change[`${beforeKey}_boundary`] || "";
  const after = change[`${afterKey}_boundary`] || "";
  return `<article class="change-card">
    <h3>Boundary L${change.before_line} -> L${change.after_line}</h3>
    <p>${renderParserBadge(change)} ${renderManualBadge(change)}</p>
    <p><strong>${escapeHtml(change.court || "")}</strong> · ${escapeHtml(change.document_id || "")} · ${escapeHtml(change.source_id || "")}</p>
    <p><strong>${beforeKey}:</strong> ${escapeHtml(before)} · <strong>${afterKey}:</strong> ${escapeHtml(after)}</p>
    <p><strong>Impact:</strong> ${escapeHtml(change.block_impact || "")} · <strong>Reason:</strong> ${escapeHtml(change.reason || "")}</p>
    <p class="raw-line"><strong>Before:</strong> ${escapeHtml(change.before_text || "")}</p>
    <p class="raw-line"><strong>After:</strong> ${escapeHtml(change.after_text || "")}</p>
  </article>`;
}

function renderClassChange(change, beforeKey = "v5", afterKey = "v6") {
  const before = change[`${beforeKey}_class`] || "";
  const after = change[`${afterKey}_class`] || "";
  return `<article class="change-card">
    <h3>Line ${change.line}</h3>
    <p>${renderParserBadge(change)} ${renderManualBadge(change)}</p>
    <p><strong>${escapeHtml(change.court || "")}</strong> · ${escapeHtml(change.document_id || "")} · ${escapeHtml(change.source_id || "")}</p>
    <p><strong>${beforeKey}:</strong> ${escapeHtml(before)} · <strong>${afterKey}:</strong> ${escapeHtml(after)}</p>
    <p><strong>Reason:</strong> ${escapeHtml(change.reason || "")}</p>
    <p class="raw-line">${escapeHtml(change.text || "")}</p>
  </article>`;
}

function renderBlockChange(change, beforeKey = "v5", afterKey = "v6") {
  const beforeRange = change[`${beforeKey}_range`];
  const afterRange = change[`${afterKey}_range`];
  const beforeClasses = change[`${beforeKey}_classes`] || [];
  const afterClasses = change[`${afterKey}_classes`] || [];
  return `<article class="change-card">
    <h3>Block ${change.block_index}</h3>
    <p>${renderParserBadge(change)}</p>
    <p><strong>${escapeHtml(change.court || "")}</strong> · ${escapeHtml(change.document_id || "")} · ${escapeHtml(change.source_id || "")}</p>
    <p><strong>${beforeKey} range:</strong> ${escapeHtml(formatRange(beforeRange))} · <strong>${afterKey} range:</strong> ${escapeHtml(formatRange(afterRange))}</p>
    <p><strong>${beforeKey} classes:</strong> ${escapeHtml(beforeClasses.join(", "))}</p>
    <p><strong>${afterKey} classes:</strong> ${escapeHtml(afterClasses.join(", "))}</p>
    <p><strong>Reason:</strong> ${escapeHtml(change.reason || "")}</p>
  </article>`;
}

async function renderProblems() {
  const data = await getJson(`/api/problems?document_id=${encodeURIComponent(current.document_id)}`);
  document.getElementById("work").innerHTML = `<section class="problems-view">
    <h3>Problems</h3>
    <p>Parser conflicts: ${data.parser_conflict_count}; manual conflicts or stale decisions: ${data.manual_conflict_count}</p>
    ${data.parser_conflicts.length === 0 && data.manual_conflicts.length === 0 ? "<p>No genuine conflicts for this document.</p>" : ""}
    ${data.parser_conflicts.map(renderProblem).join("")}
    ${data.manual_conflicts.map(renderProblem).join("")}
  </section>`;
}

function renderProblem(item) {
  const line = item.raw_line_number || item.previous_line_number || "";
  return `<article class="change-card">
    <h3>${escapeHtml(item.item_type || "item")} ${escapeHtml(line)}</h3>
    <p>${renderParserBadge(item)} ${renderManualBadge(item)}</p>
    <p class="raw-line">${escapeHtml(item.raw_text || item.before_text || "")}</p>
  </article>`;
}

async function renderProgressView() {
  const progress = await getJson("/api/progress");
  document.getElementById("work").innerHTML = `<section class="progress-view">
    <h3>Parser validation</h3>
    <dl>
      <dt>Validated lines</dt><dd>${progress.parser_validation.line_validated}/${progress.parser_validation.line_total}</dd>
      <dt>Validated boundaries</dt><dd>${progress.parser_validation.boundary_validated}/${progress.parser_validation.boundary_total}</dd>
      <dt>Validated blocks</dt><dd>${progress.parser_validation.block_validated}/${progress.parser_validation.block_total}</dd>
      <dt>Golden-covered items</dt><dd>${progress.parser_validation.golden_covered_items}</dd>
      <dt>Invariant-covered items</dt><dd>${progress.parser_validation.invariant_covered_items}</dd>
      <dt>Review recommended</dt><dd>${progress.parser_validation.review_recommended}</dd>
      <dt>Conflicts</dt><dd>${progress.parser_validation.conflicts}</dd>
    </dl>
    <h3>Manual review</h3>
    <dl>
      <dt>Manually reviewed lines</dt><dd>${progress.manual_review.reviewed_lines}/${progress.line_total}</dd>
      <dt>Manually reviewed boundaries</dt><dd>${progress.manual_review.reviewed_boundaries}/${progress.boundary_total}</dd>
      <dt>Accepted</dt><dd>${progress.manual_review.accepted}</dd>
      <dt>Overridden</dt><dd>${progress.manual_review.overridden}</dd>
      <dt>Stale</dt><dd>${progress.manual_review.stale}</dd>
      <dt>Not reviewed</dt><dd>${progress.manual_review.not_reviewed}</dd>
    </dl>
  </section>`;
}

function renderFullCorpusView(data, title, apiPrefix) {
  const renderDoc = (doc, golden) => `<article class="corpus-card ${golden ? "golden" : "remaining"}">
    <header>
      <h3>${String(doc.review_number).padStart(2, "0")} · ${escapeHtml(doc.court)} · ${escapeHtml(doc.case_number || doc.source_id)}</h3>
      <span class="status-badge ${parserStatusClass(doc.parser_validation_status)}">${escapeHtml(doc.display_parser_label)}</span>
    </header>
    <p><strong>Document ID:</strong> ${escapeHtml(doc.document_id)} · <strong>Source:</strong> ${escapeHtml(doc.source_id)}</p>
    <p><strong>Exact golden:</strong> ${doc.exact_golden_coverage ? "yes" : "no"}</p>
    <p><strong>Changed:</strong> lines ${doc.changed_line_count}, boundaries ${doc.changed_boundary_count}, blocks ${doc.changed_block_count}</p>
    <p><strong>Hierarchy:</strong> blocks ${doc.hierarchy_summary.block_count}, numbered ${doc.hierarchy_summary.numbered_paragraph_count}, lists/tables ${doc.hierarchy_summary.list_or_table_count}, headings ${doc.hierarchy_summary.heading_count}</p>
    <p><strong>Manual review:</strong> ${doc.manual_line_reviewed}/${doc.manual_line_total} lines · ${doc.manual_boundary_reviewed}/${doc.manual_boundary_total} boundaries</p>
    <p><strong>Potential review candidates:</strong> ${doc.potential_review_candidates.review_recommended ? "review recommended" : "no parser-change review queue"}</p>
    ${golden ? "" : `<button type="button" onclick="copyDocumentReviewById('${doc.document_id}', '${apiPrefix}')">Copy document review</button>`}
  </article>`;
  document.getElementById("work").innerHTML = `<section class="full-corpus-view">
    <h3>${title}</h3>
    <p>All 20 documents with golden / non-golden separation. Exact GOLDEN PASS is reserved for documents 05, 11 and 16. Targeted structural regressions display TARGETED REGRESSION PASS separately.</p>
    <p class="export-links">
      <a href="${data.exports.json_url}" ${data.exports.json_exists ? "" : 'aria-disabled="true"'}>Download complete JSON</a>
      ·
      <a href="${data.exports.markdown_url}" ${data.exports.markdown_exists ? "" : 'aria-disabled="true"'}>Download complete Markdown</a>
    </p>
    <h4>Golden documents</h4>
    <div class="corpus-grid">${data.golden_documents.map(doc => renderDoc(doc, true)).join("")}</div>
    <h4>Remaining non-golden documents</h4>
    <div class="corpus-grid">${data.remaining_documents.map(doc => renderDoc(doc, false)).join("")}</div>
  </section>`;
}

async function renderFullCorpusV7() {
  const data = await getJson("/api/full-corpus-v7");
  renderFullCorpusView(data, "Full corpus v7 review", "full-corpus-v7");
}

async function renderFullCorpusV6() {
  const data = await getJson("/api/full-corpus-v6");
  renderFullCorpusView(data, "Full corpus v6 review (historical)", "full-corpus-v6");
}

async function copyDocumentReviewById(documentId, apiPrefix = "full-corpus-v7") {
  const data = await getJson(`/api/${apiPrefix}/document-markdown?document_id=${encodeURIComponent(documentId)}`);
  await navigator.clipboard.writeText(data.markdown);
  const meta = document.getElementById("documentMeta");
  if (meta) {
    meta.innerHTML += `<span class="pill">Copied complete Markdown for document ${data.review_number}</span>`;
  }
}

async function copyDocumentReview() {
  if (!current) return;
  await copyDocumentReviewById(current.document_id);
}

function renderParserBadge(item) {
  return `<span class="status-badge ${parserStatusClass(item.parser_validation_status)}">Parser validation: ${escapeHtml(item.parser_validation_label || "PARSER NOT VALIDATED")}</span>`;
}

function renderManualBadge(item) {
  return `<span class="status-badge manual">Manual review: ${escapeHtml((item.manual_review_label || "Manual review: not performed").replace(/^Manual review:\\s*/, ""))}</span>`;
}

function parserStatusClass(status) {
  if (status === "AUTO_VALIDATED_GOLDEN") return "success";
  if (status === "PARSER_VALIDATED") return "validated";
  if (status === "PARSER_CHANGED_NEEDS_REVIEW") return "review";
  if (status === "PARSER_CONFLICT") return "conflict";
  return "unknown";
}

function formatRange(range) {
  if (!range || !range.length) return "n/a";
  return `L${range[0]}-L${range[1]}`;
}

function renderAssistedRule(rule, batch) {
  const applyAllowed = batch && batch.apply_allowed;
  const confirmation = batch ? batch.confirmation : "";
  return `<article class="assisted-rule">
    <header class="assisted-rule-header">
      <div>
        <h3>${escapeHtml(rule.rule_id)}</h3>
        <p>${escapeHtml(rule.court)} · ${escapeHtml(rule.confidence)} · ${escapeHtml(rule.rule_type)} · ${escapeHtml(rule.item_type)}</p>
      </div>
      <span class="decision-badge ${rule.confidence === "SAFE" ? "parser" : "previous"}">${escapeHtml(rule.confidence)}</span>
    </header>
    <p><strong>Proposed:</strong> ${escapeHtml(rule.target_value)}</p>
    <p><strong>Evidence:</strong> ${rule.source_document_ids.map(escapeHtml).join(", ")}</p>
    <p>${escapeHtml(rule.rationale)}</p>
    <p>Matching pending items: ${batch ? batch.occurrence_count : 0}; excluded: ${batch ? batch.excluded_count : 0}</p>
    <div class="assisted-actions">
      <button onclick="previewRule('${rule.rule_id}')">Preview occurrences</button>
      ${applyAllowed ? `<button onclick="applyAssistedRule('${rule.rule_id}', '${confirmation}')">Apply entire safe batch</button>` : ""}
      <input id="revert-${safeId(rule.rule_id)}" placeholder="batch id to revert">
      <button onclick="revertAssistedBatch('${safeId(rule.rule_id)}')">Revert applied batch</button>
    </div>
    <div id="occ-${safeId(rule.rule_id)}" class="occurrences"></div>
  </article>`;
}

async function previewRule(ruleId) {
  const data = await getJson(`/api/assisted/rules/${encodeURIComponent(ruleId)}/occurrences`);
  const target = document.getElementById(`occ-${safeId(ruleId)}`);
  target.innerHTML = data.occurrences.map(item => `<section class="occurrence ${item.excluded ? "excluded" : ""}">
    <h4>${escapeHtml(item.document_id)} · ${escapeHtml(item.item_type)} · ${escapeHtml(item.item_id)}</h4>
    <p><strong>Proposed:</strong> ${escapeHtml(item.proposed_manual_class || item.proposed_boundary_decision || "")}</p>
    <p><strong>Parser:</strong> ${escapeHtml(item.parser_proposal || "")}; <strong>Previous:</strong> ${escapeHtml(item.previous_annotation || "")}</p>
    <p class="raw-line">${escapeHtml(item.raw_text || "")}</p>
    <p>${item.excluded ? `Excluded: ${escapeHtml(item.excluded_reason || "")}` : "Included in safe batch"}</p>
  </section>`).join("");
}

async function applyAssistedRule(ruleId, expectedConfirmation) {
  const confirmation = prompt(`Type exact confirmation to apply this batch:\n${expectedConfirmation}`);
  if (confirmation !== expectedConfirmation) {
    alert("Confirmation rejected. No decisions were written.");
    return;
  }
  const result = await postJson("/api/assisted/apply", {rule_id: ruleId, confirmation});
  alert(`Applied batch ${result.result.batch_id}: ${result.result.applied_count} decisions.`);
  await renderAssisted();
  await refreshProgress();
}

async function revertAssistedBatch(inputKey) {
  const batchId = document.getElementById(`revert-${inputKey}`).value;
  const confirmation = prompt(`Type exact confirmation to revert this batch:\nREVERT ${batchId}`);
  if (confirmation !== `REVERT ${batchId}`) {
    alert("Confirmation rejected. No decisions were changed.");
    return;
  }
  const result = await postJson("/api/assisted/revert", {batch_id: batchId, confirmation});
  alert(`Reverted batch ${result.result.batch_id}: ${result.result.reverted_count} items.`);
  await renderAssisted();
  await refreshProgress();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
}

function safeId(value) {
  return String(value).replace(/[^A-Za-z0-9_-]/g, "_");
}

document.getElementById("mode").onchange = render;
document.getElementById("copyDocumentReview").onclick = () => {
  copyDocumentReview().catch(err => {
    document.getElementById("work").textContent = err.message;
  });
};
load().catch(err => { document.getElementById("work").textContent = err.message; });
