/* GeoStress AI - Frontend Logic v2.0 (Industrial Grade) */

var currentSource = "demo";
var currentWell = "3P";
var feedbackRating = 3;

// ── Helpers ───────────────────────────────────────

var _loadingTimer = null;
var _loadingStart = 0;
var _progressSource = null;

function showLoading(text) {
    var el = document.getElementById("loading-text");
    el.textContent = text || "Processing";
    document.getElementById("loading-overlay").classList.remove("d-none");
    document.getElementById("loading-progress").style.display = "none";
    document.getElementById("loading-detail").textContent = "";
    _loadingStart = Date.now();
    if (_loadingTimer) clearInterval(_loadingTimer);
    _loadingTimer = setInterval(function() {
        var elapsed = Math.round((Date.now() - _loadingStart) / 1000);
        el.textContent = (text || "Processing") + " (" + elapsed + "s)";
    }, 1000);
}

function showLoadingWithProgress(text, taskId) {
    showLoading(text);
    // Show progress bar and connect to SSE
    document.getElementById("loading-progress").style.display = "flex";
    document.getElementById("loading-bar").style.width = "0%";
    if (_progressSource) { _progressSource.close(); _progressSource = null; }
    _progressSource = new EventSource("/api/progress/" + taskId);
    _progressSource.onmessage = function(e) {
        try {
            var d = JSON.parse(e.data);
            document.getElementById("loading-bar").style.width = d.pct + "%";
            document.getElementById("loading-text").textContent = d.step;
            if (d.detail) document.getElementById("loading-detail").textContent = d.detail;
            if (d.pct >= 100) {
                _progressSource.close();
                _progressSource = null;
            }
        } catch (err) {}
    };
    _progressSource.onerror = function() {
        if (_progressSource) { _progressSource.close(); _progressSource = null; }
    };
}

function hideLoading() {
    document.getElementById("loading-overlay").classList.add("d-none");
    if (_loadingTimer) { clearInterval(_loadingTimer); _loadingTimer = null; }
    if (_progressSource) { _progressSource.close(); _progressSource = null; }
}

function generateTaskId() {
    return "task_" + Date.now() + "_" + Math.random().toString(36).substr(2, 6);
}

function showToast(msg, title) {
    document.getElementById("toast-title").textContent = title || "GeoStress AI";
    document.getElementById("toast-body").textContent = msg;
    var toast = new bootstrap.Toast(document.getElementById("toast"), { delay: 4000 });
    toast.show();
}

var _lastResponseTime = null;

async function api(url, options) {
    var t0 = performance.now();
    var resp = await fetch(url, options || {});
    var elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    _lastResponseTime = elapsed;
    // Show response time in status bar
    var timeEl = document.getElementById("response-time");
    if (timeEl) timeEl.textContent = elapsed + "s";
    if (!resp.ok) {
        var text = await resp.text();
        throw new Error(text || resp.statusText);
    }
    return resp.json();
}

async function apiPost(url, data) {
    return api(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });
}

function setImg(id, src) {
    var el = document.getElementById(id);
    if (el && src) el.src = src;
}

function val(id, v) {
    var el = document.getElementById(id);
    if (el) el.textContent = v;
}

function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
}

function createCell(tag, text, styles) {
    var cell = document.createElement(tag);
    cell.textContent = text;
    if (styles) {
        for (var k in styles) cell.style[k] = styles[k];
    }
    return cell;
}

function fmt(v, decimals) {
    if (v == null || v !== v) return "N/A";
    return Number(v).toFixed(decimals == null ? 1 : decimals);
}

/**
 * Render a stakeholder brief card — plain-English decision summary.
 * @param {string} containerId - DOM element ID to render into
 * @param {object} brief - stakeholder_brief from API response
 * @param {string} collapseId - unique ID for the collapsible detail section
 */
function renderStakeholderBrief(containerId, brief, collapseId) {
    var el = document.getElementById(containerId);
    if (!el || !brief) return;
    el.classList.remove("d-none");

    // Determine border color from risk_level if present
    var borderClass = "border-primary";
    var iconClass = "bi-clipboard-check";
    if (brief.risk_level === "RED") { borderClass = "border-danger"; iconClass = "bi-exclamation-triangle-fill"; }
    else if (brief.risk_level === "AMBER") { borderClass = "border-warning"; iconClass = "bi-exclamation-circle"; }
    else if (brief.risk_level === "GREEN") { borderClass = "border-success"; iconClass = "bi-check-circle-fill"; }

    var html = '<div class="card ' + borderClass + ' shadow-sm">';
    html += '<div class="card-body py-3">';
    html += '<h6 class="card-title mb-1"><i class="bi ' + iconClass + '"></i> ' + (brief.headline || "Analysis Complete") + '</h6>';

    if (brief.confidence_sentence) {
        html += '<p class="text-muted small mb-2">' + brief.confidence_sentence + '</p>';
    }
    if (brief.verdict) {
        html += '<p class="small mb-2"><strong>Verdict:</strong> ' + brief.verdict + '</p>';
    }
    if (brief.what_it_means) {
        html += '<p class="small mb-2">' + brief.what_it_means + '</p>';
    }
    if (brief.limiting_class) {
        html += '<p class="small mb-2 text-warning"><i class="bi bi-exclamation-triangle"></i> ' + brief.limiting_class + '</p>';
    }
    if (brief.tradeoff_explained) {
        html += '<p class="small mb-2">' + brief.tradeoff_explained + '</p>';
    }
    if (brief.what_agreement_means) {
        html += '<p class="small mb-2"><strong>Model Consensus:</strong> ' + brief.what_agreement_means + '</p>';
    }

    // Collapsible detail section
    var hasDetail = brief.next_action || brief.suitable_for || brief.not_suitable_for ||
                    brief.model_to_use || brief.recommended_use || brief.action ||
                    brief.critically_stressed_plain || brief.feedback_note ||
                    brief.why_these_samples;
    if (hasDetail && collapseId) {
        html += '<a class="small text-decoration-none" data-bs-toggle="collapse" href="#' + collapseId + '" role="button">';
        html += '<i class="bi bi-chevron-down"></i> What does this mean for operations?</a>';
        html += '<div class="collapse mt-2" id="' + collapseId + '">';
        html += '<ul class="small mb-0">';
        if (brief.critically_stressed_plain) html += '<li><strong>Fracture Risk:</strong> ' + brief.critically_stressed_plain + '</li>';
        if (brief.next_action) html += '<li><strong>Next Action:</strong> ' + brief.next_action + '</li>';
        if (brief.model_to_use) html += '<li><strong>Recommended Model:</strong> ' + brief.model_to_use + '</li>';
        if (brief.recommended_use) html += '<li>' + brief.recommended_use + '</li>';
        if (brief.action) html += '<li>' + brief.action + '</li>';
        if (brief.suitable_for) html += '<li><strong>Suitable for:</strong> ' + brief.suitable_for.join(", ") + '</li>';
        if (brief.not_suitable_for) html += '<li><strong>Not suitable for:</strong> ' + brief.not_suitable_for.join(", ") + '</li>';
        if (brief.caution) html += '<li class="text-warning"><i class="bi bi-exclamation-circle"></i> ' + brief.caution + '</li>';
        if (brief.why_these_samples) html += '<li>' + brief.why_these_samples + '</li>';
        if (brief.what_to_look_for) html += '<li><strong>What to look for:</strong> ' + brief.what_to_look_for + '</li>';
        if (brief.what_happens_next) html += '<li>' + brief.what_happens_next + '</li>';
        if (brief.progress) html += '<li><strong>Progress:</strong> ' + brief.progress + '</li>';
        if (brief.feedback_note) html += '<li class="text-info"><i class="bi bi-chat-dots"></i> ' + brief.feedback_note + '</li>';
        html += '</ul></div>';
    }

    html += '</div></div>';
    el.innerHTML = html;
}

function getPorePresure() {
    var ppEl = document.getElementById("pp-input");
    var ppVal = ppEl ? ppEl.value : "";
    return ppVal === "" ? null : parseFloat(ppVal);
}

function getWell() {
    var el = document.getElementById("well-select");
    return el ? el.value : "3P";
}

function getDepth() {
    var el = document.getElementById("depth-input");
    return el ? (parseFloat(el.value) || 3000) : 3000;
}

function getRegime() {
    var el = document.getElementById("regime-select");
    return el ? el.value : "auto";
}

// ── Tab Switching ─────────────────────────────────

var tabNames = {
    executive: "Executive Summary",
    data: "Data Overview",
    viz: "Visualizations",
    inversion: "Stress Inversion",
    models: "Model Comparison",
    shap: "Why It Predicts",
    classify: "ML Classification",
    cluster: "Fracture Clustering",
    sensitivity: "Sensitivity Analysis",
    risk: "Risk Assessment",
    uncertainty: "Uncertainty Budget",
    wells: "Well Comparison",
    report: "Well Report",
    feedback: "Expert Feedback",
    scenarios: "Scenario Comparison",
    decision: "Decision Support",
    montecarlo: "Monte Carlo Uncertainty",
    validation: "Data Validation",
    research: "Research & Methods",
    audit: "Audit Trail",
    calibration: "Model Calibration",
    glossary: "Glossary & Guide",
    mlops: "Production MLOps"
};

document.querySelectorAll("[data-tab]").forEach(function(link) {
    link.addEventListener("click", function(e) {
        e.preventDefault();
        var tab = link.dataset.tab;

        document.querySelectorAll(".sidebar-nav .nav-link").forEach(function(l) {
            l.classList.remove("active");
        });
        link.classList.add("active");

        document.querySelectorAll(".tab-content").forEach(function(c) {
            c.classList.remove("active");
        });
        document.getElementById("tab-" + tab).classList.add("active");
        document.getElementById("page-title").textContent = tabNames[tab] || tab;
    });
});

function switchTab(tab) {
    document.querySelectorAll(".sidebar-nav .nav-link").forEach(function(l) {
        l.classList.remove("active");
        if (l.dataset.tab === tab) l.classList.add("active");
    });
    document.querySelectorAll(".tab-content").forEach(function(c) { c.classList.remove("active"); });
    var el = document.getElementById("tab-" + tab);
    if (el) el.classList.add("active");
    document.getElementById("page-title").textContent = tabNames[tab] || tab;
}

// ── Well selector sync ────────────────────────────

document.getElementById("well-select").addEventListener("change", function() {
    currentWell = this.value;
});

// ── Executive Summary ─────────────────────────────

async function runExecutiveSummary() {
    showLoading("Generating executive summary...");
    try {
        var r = await api("/api/analysis/executive-summary", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                source: currentSource,
                well: currentWell || "3P",
                depth: parseFloat(document.getElementById("depth-input").value) || 3000
            })
        });

        // Risk banner
        var banner = document.getElementById("exec-risk-banner");
        banner.classList.remove("d-none");
        var alertEl = document.getElementById("exec-risk-alert");
        var riskColors = {RED: "danger", AMBER: "warning", GREEN: "success"};
        alertEl.className = "alert alert-" + (riskColors[r.overall_risk] || "info") + " p-4";
        alertEl.innerHTML = '<h4><i class="bi bi-' +
            (r.overall_risk === "GREEN" ? "check-circle" : r.overall_risk === "RED" ? "exclamation-octagon" : "exclamation-triangle") +
            '"></i> ' + r.overall_risk + ' — ' + (r.well || "All Wells") + '</h4>' +
            '<p class="mb-0">' + r.overall_message + '</p>';

        // Sections
        var sectionsEl = document.getElementById("exec-sections");
        clearChildren(sectionsEl);
        (r.sections || []).forEach(function(sec) {
            var card = document.createElement("div");
            card.className = "card mb-3";
            var html = '<div class="card-body"><h5 class="card-title"><i class="bi bi-' + sec.icon + '"></i> ' + sec.title + '</h5>';
            html += '<p class="card-text">' + sec.text + '</p>';
            if (sec.risk) {
                var riskBg = sec.risk === "RED" ? "danger" : sec.risk === "AMBER" ? "warning" : "success";
                html += '<div class="alert alert-' + riskBg + ' py-2 mb-0 small">' +
                    '<strong>' + sec.risk + ':</strong> ' + sec.risk_text + '</div>';
            }
            html += '</div>';
            card.innerHTML = html;
            sectionsEl.appendChild(card);
        });

        // GO/NO-GO Decision Matrix
        if (r.decision_matrix) {
            var dm = r.decision_matrix;
            var dmEl = document.getElementById("exec-decision-matrix");
            if (dmEl) {
                var verdictColors = {"GO": "success", "CONDITIONAL GO": "warning", "NO-GO": "danger"};
                var vc = verdictColors[dm.verdict] || "secondary";
                var dmhtml = '<div class="card border-' + vc + ' mb-3">' +
                    '<div class="card-header bg-' + vc + ' bg-opacity-10 py-2">' +
                    '<i class="bi bi-shield-check me-1"></i> <strong>Operational Decision: </strong>' +
                    '<span class="badge bg-' + vc + ' fs-6">' + dm.verdict + '</span></div>' +
                    '<div class="card-body py-2">' +
                    '<p class="mb-2">' + dm.verdict_note + '</p>' +
                    '<table class="table table-sm table-bordered mb-0"><thead class="table-light">' +
                    '<tr><th>Factor</th><th>Status</th><th>Detail</th></tr></thead><tbody>';
                dm.factors.forEach(function(f) {
                    var fc = f.status === "GREEN" ? "success" : f.status === "AMBER" ? "warning" : "danger";
                    dmhtml += '<tr><td><strong>' + f.factor + '</strong></td>' +
                        '<td><span class="badge bg-' + fc + '">' + f.status + '</span></td>' +
                        '<td class="small">' + f.detail + '</td></tr>';
                });
                dmhtml += '</tbody></table></div></div>';
                dmEl.innerHTML = dmhtml;
                dmEl.classList.remove("d-none");
            }
        }

        showToast("Executive summary: " + r.overall_risk + " | Decision: " + (r.decision_matrix ? r.decision_matrix.verdict : "N/A"));

        // Also load calibration status
        loadCalibrationStatus();
    } catch (err) {
        showToast("Executive summary error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


async function loadCalibrationStatus() {
    try {
        var r = await api("/api/calibration/measurements?well=" + (currentWell || "3P"));
        var measurements = r.measurements || [];
        var calDiv = document.getElementById("exec-calibration");
        var calBody = document.getElementById("exec-cal-body");
        var calCard = document.getElementById("exec-cal-card");

        if (measurements.length === 0) {
            calDiv.classList.remove("d-none");
            calCard.className = "card border-warning";
            calBody.innerHTML =
                '<div class="d-flex align-items-center">' +
                '<i class="bi bi-exclamation-triangle text-warning fs-4 me-3"></i>' +
                '<div>' +
                '<strong>Not Calibrated</strong> — No field measurements recorded for well ' + (currentWell || "3P") + '.' +
                '<br><small class="text-muted">Add LOT, XLOT, or minifrac test results in the Calibration tab to validate model predictions against ground truth.</small>' +
                '</div>' +
                '<button class="btn btn-outline-warning btn-sm ms-auto" onclick="switchTab(\'calibration\')">Go to Calibration</button>' +
                '</div>';
            return;
        }

        // Run validation
        var v = await api("/api/calibration/validate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                well: currentWell || "3P",
                source: currentSource,
                depth_m: parseFloat(document.getElementById("depth-input").value) || 3000,
                pp_mpa: 30
            })
        });

        calDiv.classList.remove("d-none");
        var scoreColor = v.calibration_score >= 80 ? "success" :
                        v.calibration_score >= 60 ? "info" :
                        v.calibration_score >= 40 ? "warning" : "danger";
        calCard.className = "card border-" + scoreColor;

        calBody.innerHTML =
            '<div class="d-flex align-items-center">' +
            '<div class="me-4 text-center">' +
            '<div class="fs-2 fw-bold text-' + scoreColor + '">' + v.calibration_score + '</div>' +
            '<small class="text-muted">out of 100</small>' +
            '</div>' +
            '<div class="flex-grow-1">' +
            '<strong class="text-' + scoreColor + '">' + v.overall_rating + '</strong>' +
            ' — ' + v.n_measurements + ' field measurement(s), ' +
            'avg stress error: ' + v.avg_stress_error_pct + '%' +
            (v.avg_azimuth_error_deg ? ', azimuth error: ' + v.avg_azimuth_error_deg + '\u00B0' : '') +
            '<br><small class="text-muted">' + (v.recommendations[0] || '') + '</small>' +
            '</div>' +
            '<button class="btn btn-outline-' + scoreColor + ' btn-sm" onclick="switchTab(\'calibration\')">Details</button>' +
            '</div>';

    } catch (err) {
        // Silently ignore — calibration status is informational
    }
}


async function runSufficiencyCheck() {
    showLoading("Checking data sufficiency...");
    try {
        var r = await api("/api/data/sufficiency", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource})
        });

        document.getElementById("suff-results").classList.remove("d-none");

        // Overall
        var overallColors = {"FULLY READY": "success", "PARTIALLY READY": "warning", "MORE DATA NEEDED": "danger"};
        var overallEl = document.getElementById("suff-overall");
        overallEl.className = "alert alert-" + (overallColors[r.overall_readiness] || "info") + " mb-3";
        overallEl.innerHTML = '<strong>' + r.overall_readiness + '</strong> — ' + r.overall_message +
            '<br><small>' + r.n_samples + ' samples, ' + r.n_wells + ' wells, ' + r.n_classes + ' fracture types</small>';

        // Analysis table
        var tbody = document.getElementById("suff-table-body");
        clearChildren(tbody);
        var statusColors = {READY: "success", MARGINAL: "warning", INSUFFICIENT: "danger"};
        (r.analyses || []).forEach(function(a) {
            var tr = document.createElement("tr");
            tr.innerHTML = '<td>' + a.analysis + '</td>' +
                '<td><span class="badge bg-' + (statusColors[a.status] || "secondary") + '">' + a.status + '</span></td>' +
                '<td class="small">' + a.message + '</td>' +
                '<td>' + a.min_needed + '</td>';
            tbody.appendChild(tr);
        });

        // Recommendations
        var recsEl = document.getElementById("suff-recs");
        clearChildren(recsEl);
        if (r.recommendations && r.recommendations.length > 0) {
            var h6 = document.createElement("h6");
            h6.textContent = "Recommended Actions";
            recsEl.appendChild(h6);
            r.recommendations.forEach(function(rec) {
                var div = document.createElement("div");
                var prioColors = {HIGH: "danger", MEDIUM: "warning", LOW: "info"};
                div.className = "alert alert-" + (prioColors[rec.priority] || "info") + " py-2 mb-2 small";
                div.innerHTML = '<strong>' + rec.priority + ':</strong> ' + rec.action;
                recsEl.appendChild(div);
            });
        }

        showToast("Data sufficiency: " + r.overall_readiness);
    } catch (err) {
        showToast("Sufficiency check error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Safety Check ─────────────────────────────────

async function runSafetyCheck() {
    showLoading("Running prediction safety check...");
    try {
        var r = await api("/api/analysis/safety-check", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, well: currentWell || null})
        });

        document.getElementById("safety-results").classList.remove("d-none");

        var decColors = {"GO": "success", "GO WITH MONITORING": "info", "PROCEED WITH CAUTION": "warning", "NO-GO": "danger"};
        var decEl = document.getElementById("safety-decision");
        decEl.className = "alert alert-" + (decColors[r.decision] || "secondary") + " mb-3";
        decEl.innerHTML = '<h5 class="mb-1">' + r.decision + '</h5><p class="mb-0">' + r.decision_message + '</p>';

        var blockersEl = document.getElementById("safety-blockers");
        clearChildren(blockersEl);
        if (r.blockers && r.blockers.length > 0) {
            var h6 = document.createElement("h6");
            h6.className = "small fw-bold text-danger";
            h6.textContent = "Critical Blockers";
            blockersEl.appendChild(h6);
            r.blockers.forEach(function(b) {
                var div = document.createElement("div");
                div.className = "alert alert-danger py-2 mb-2 small";
                div.innerHTML = '<strong>' + b.type + ':</strong> ' + b.message;
                blockersEl.appendChild(div);
            });
        }

        var warnsEl = document.getElementById("safety-warnings");
        clearChildren(warnsEl);
        if (r.warnings && r.warnings.length > 0) {
            var h6w = document.createElement("h6");
            h6w.className = "small fw-bold text-warning mt-2";
            h6w.textContent = "Warnings (" + r.warnings.length + ")";
            warnsEl.appendChild(h6w);
            r.warnings.forEach(function(w) {
                var sevColors = {HIGH: "danger", MEDIUM: "warning", LOW: "info"};
                var div = document.createElement("div");
                div.className = "alert alert-" + (sevColors[w.severity] || "info") + " py-2 mb-2 small";
                div.innerHTML = '<strong>' + w.type + ':</strong> ' + w.message;
                warnsEl.appendChild(div);
            });
        }

        showToast("Safety: " + r.decision);
    } catch (err) {
        showToast("Safety check error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Field Consistency ────────────────────────────

async function runFieldConsistency() {
    showLoading("Checking field-scale consistency...");
    try {
        var r = await api("/api/analysis/field-consistency", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                source: currentSource,
                depth: parseFloat(document.getElementById("depth-input").value) || 3000
            })
        });

        if (r.error) { showToast(r.error, "Error"); return; }

        document.getElementById("field-results").classList.remove("d-none");

        var recColors = {SEPARATE: "warning", BOTH: "info", COMBINED: "success"};
        var recEl = document.getElementById("field-rec");
        recEl.className = "alert alert-" + (recColors[r.recommendation] || "info") + " mb-3";
        recEl.innerHTML = '<strong>Recommendation: ' + r.recommendation + '</strong> — ' + r.recommendation_message;

        // SHmax
        var shmaxColors = {CONSISTENT: "success", MODERATE: "warning", INCONSISTENT: "danger"};
        var shmaxEl = document.getElementById("field-shmax");
        shmaxEl.innerHTML = '<div class="alert alert-' + (shmaxColors[r.shmax_consistency] || "info") + ' py-2 small">' +
            '<strong>' + r.shmax_consistency + '</strong> (max diff: ' + r.shmax_max_difference + '°)<br>' +
            r.shmax_message + '</div>';

        // Types
        var typeColors = {SIMILAR: "success", PARTIAL: "warning", DIFFERENT: "danger"};
        var typeEl = document.getElementById("field-types");
        typeEl.innerHTML = '<div class="alert alert-' + (typeColors[r.type_similarity] || "info") + ' py-2 small">' +
            '<strong>' + r.type_similarity + '</strong><br>' + r.type_message + '</div>';

        // Per-well table
        var tbody = document.getElementById("field-table-body");
        clearChildren(tbody);
        var wr = r.well_results || {};
        Object.keys(wr).forEach(function(well) {
            var w = wr[well];
            var tr = document.createElement("tr");
            if (w.error) {
                tr.innerHTML = '<td>' + well + '</td><td colspan="5" class="text-danger">' + w.error + '</td>';
            } else {
                tr.innerHTML = '<td>' + well + '</td><td>' + w.shmax + '°</td><td>' + w.sigma1 + '</td>' +
                    '<td>' + w.sigma3 + '</td><td>' + w.misfit + '</td><td>' + w.n_fractures + '</td>';
            }
            tbody.appendChild(tr);
        });

        showToast("Field consistency: " + r.shmax_consistency + " SHmax, recommend " + r.recommendation);
    } catch (err) {
        showToast("Field consistency error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Evidence Chain ─────────────────────────────────

async function runEvidenceChain() {
    var taskId = generateTaskId();
    showLoadingWithProgress("Building evidence chain...", taskId);
    try {
        var r = await apiPost("/api/analysis/evidence-chain", {
            source: currentSource, well: getWell(), depth: getDepth(),
            task_id: taskId
        });

        document.getElementById("evidence-results").classList.remove("d-none");

        // Overall recommendation banner
        var recColors = {PROCEED: "success", REVIEW: "warning", CAUTION: "danger"};
        var sumEl = document.getElementById("evidence-summary");
        sumEl.className = "alert alert-" + (recColors[r.overall_recommendation] || "info") + " p-3";
        sumEl.innerHTML = '<h5 class="mb-1"><i class="bi bi-clipboard-check"></i> ' +
            r.overall_recommendation + '</h5>' +
            '<p class="mb-1">' + r.overall_message + '</p>' +
            '<small>' + r.high_confidence_count + ' high-confidence, ' +
            r.low_confidence_count + ' low-confidence out of ' + r.n_evidence_items + ' evidence items</small>';

        // Evidence items
        var itemsEl = document.getElementById("evidence-items");
        itemsEl.innerHTML = '';

        var confColors = {HIGH: "success", MODERATE: "warning", LOW: "danger", NONE: "secondary"};
        var confIcons = {HIGH: "check-circle", MODERATE: "exclamation-triangle", LOW: "x-circle", NONE: "question-circle"};

        (r.evidence || []).forEach(function(item) {
            var color = confColors[item.confidence] || "secondary";
            var icon = confIcons[item.confidence] || "info-circle";

            var card = document.createElement("div");
            card.className = "card mb-3 border-" + color;

            var header = '<div class="card-header d-flex justify-content-between align-items-center">' +
                '<span><i class="bi bi-' + icon + ' text-' + color + '"></i> ' +
                '<strong>' + item.category + '</strong></span>' +
                '<span class="badge bg-' + color + '">' + item.confidence + '</span></div>';

            var body = '<div class="card-body">';
            body += '<h6 class="card-title">' + item.conclusion + '</h6>';

            // Evidence bullets
            body += '<div class="mb-2"><strong class="small text-muted">Evidence:</strong><ul class="small mb-2">';
            item.evidence.forEach(function(e) {
                body += '<li>' + e + '</li>';
            });
            body += '</ul></div>';

            // Risk and action
            body += '<div class="row">' +
                '<div class="col-md-6"><div class="alert alert-danger py-1 px-2 mb-1 small">' +
                '<strong>Risk if wrong:</strong> ' + item.risk_if_wrong + '</div></div>' +
                '<div class="col-md-6"><div class="alert alert-info py-1 px-2 mb-1 small">' +
                '<strong>Action:</strong> ' + item.action + '</div></div></div>';

            body += '</div>';
            card.innerHTML = header + body;
            itemsEl.appendChild(card);
        });

        showToast("Evidence chain: " + r.overall_recommendation);
    } catch (err) {
        showToast("Evidence chain error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Guided Analysis Wizard ──────────────────────────

async function runAnomalyDetection() {
    showLoading("Scanning for data anomalies...");
    try {
        var r = await apiPost("/api/data/anomaly-detection", {
            source: currentSource, well: getWell()
        });
        var el = document.getElementById("anomaly-results");
        el.classList.remove("d-none");
        var body = document.getElementById("anomaly-body");

        var rec = r.recommendation || {};
        var verdictColors = {
            DATA_ERRORS_FOUND: "danger", MANY_WARNINGS: "warning",
            SOME_WARNINGS: "info", DATA_CLEAN: "success"
        };
        var html = '<div class="alert alert-' + (verdictColors[rec.verdict] || "secondary") + ' p-3">' +
            '<h5><i class="bi bi-shield-check"></i> ' + (rec.verdict || "").replace(/_/g, " ") + '</h5>' +
            '<p class="mb-1">' + (rec.message || "") + '</p></div>';

        // Summary cards
        html += '<div class="row g-2 mb-3">';
        html += '<div class="col-md-3"><div class="card text-center"><div class="card-body py-2">' +
            '<h3>' + r.total_samples + '</h3><small class="text-muted">Total</small></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center border-success"><div class="card-body py-2">' +
            '<h3 class="text-success">' + r.clean_count + '</h3><small class="text-muted">Clean</small></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center border-warning"><div class="card-body py-2">' +
            '<h3 class="text-warning">' + r.flagged_count + '</h3><small class="text-muted">Flagged (' + r.flagged_pct + '%)</small></div></div></div>';
        var errCount = (r.severity_counts || {}).ERROR || 0;
        html += '<div class="col-md-3"><div class="card text-center border-danger"><div class="card-body py-2">' +
            '<h3 class="text-danger">' + errCount + '</h3><small class="text-muted">Errors</small></div></div></div>';
        html += '</div>';

        // Flag type breakdown
        if (r.flag_types && Object.keys(r.flag_types).length > 0) {
            html += '<h6><i class="bi bi-tags"></i> Flag Types</h6>';
            html += '<div class="d-flex flex-wrap gap-2 mb-3">';
            Object.keys(r.flag_types).forEach(function(ft) {
                var badgeColor = ft.includes("ERROR") || ft.includes("IMPOSSIBLE") ? "danger" :
                    (ft.includes("OUTLIER") || ft.includes("DUPLICATE") ? "warning" : "info");
                html += '<span class="badge bg-' + badgeColor + '">' + ft.replace(/_/g, " ") +
                    ': ' + r.flag_types[ft] + '</span>';
            });
            html += '</div>';
        }

        // Flagged samples table (first 30)
        if (r.flagged_samples && r.flagged_samples.length > 0) {
            html += '<h6><i class="bi bi-table"></i> Flagged Samples (' + r.flagged_samples.length + ')</h6>';
            html += '<div class="table-responsive"><table class="table table-sm table-striped"><thead><tr>' +
                '<th>#</th><th>Depth</th><th>Az</th><th>Dip</th><th>Type</th><th>Severity</th><th>Issues</th></tr></thead><tbody>';
            r.flagged_samples.slice(0, 30).forEach(function(s) {
                var rowClass = s.max_severity === "ERROR" ? "table-danger" :
                    (s.max_severity === "WARNING" ? "table-warning" : "");
                var issues = s.flags.map(function(f) { return f.message; }).join("; ");
                html += '<tr class="' + rowClass + '"><td>' + s.index + '</td>' +
                    '<td>' + (s.depth || "-") + '</td><td>' + (s.azimuth || "-") + '</td>' +
                    '<td>' + (s.dip || "-") + '</td><td>' + (s.type || "-") + '</td>' +
                    '<td><span class="badge bg-' + (s.max_severity === "ERROR" ? "danger" : (s.max_severity === "WARNING" ? "warning" : "info")) +
                    '">' + s.max_severity + '</span></td><td><small>' + issues + '</small></td></tr>';
            });
            html += '</tbody></table></div>';
            if (r.flagged_samples.length > 30) {
                html += '<small class="text-muted">Showing 30 of ' + r.flagged_samples.length + ' flagged samples</small>';
            }
        }

        body.innerHTML = html;
        showToast("Anomalies: " + r.flagged_count + " flagged, " + errCount + " errors");
    } catch (err) {
        showToast("Anomaly detection error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function runBatchAnalysis() {
    showLoading("Running batch analysis for all wells...");
    try {
        var r = await apiPost("/api/analysis/batch", {
            source: currentSource, depth: getDepth()
        });
        var el = document.getElementById("batch-results");
        el.classList.remove("d-none");
        var body = document.getElementById("batch-body");

        var fs = r.field_summary || {};
        // Field summary banner
        var riskColor = fs.worst_risk === "RED" ? "danger" : (fs.worst_risk === "AMBER" ? "warning" : "success");
        var html = '<div class="alert alert-' + riskColor + ' p-3 mb-3">' +
            '<h5><i class="bi bi-globe"></i> Field Summary</h5>' +
            '<div class="row">' +
            '<div class="col-md-3"><strong>' + fs.n_wells + '</strong> wells analyzed</div>' +
            '<div class="col-md-3"><strong>' + fs.total_fractures + '</strong> total fractures</div>';
        if (fs.shmax_range) {
            html += '<div class="col-md-3">SHmax: <strong>' + fs.shmax_range[0] + '°–' + fs.shmax_range[1] + '°</strong>' +
                ' (spread: ' + fs.shmax_spread + '°)</div>';
        }
        html += '<div class="col-md-3">Worst risk: <span class="badge bg-' + riskColor + '">' + (fs.worst_risk || "N/A") + '</span></div>';
        html += '</div>';
        if (fs.shmax_consistent === false) {
            html += '<small class="text-danger"><i class="bi bi-exclamation-triangle"></i> SHmax is INCONSISTENT across wells — possible structural domain boundary</small>';
        } else if (fs.shmax_consistent === true) {
            html += '<small class="text-success"><i class="bi bi-check-circle"></i> SHmax is consistent across wells</small>';
        }
        html += '</div>';

        // Per-well cards
        html += '<div class="row g-3">';
        Object.keys(r.wells).forEach(function(wellName) {
            var w = r.wells[wellName];
            var wRisk = (w.risk && w.risk.risk_level) || "N/A";
            var wRiskColor = wRisk === "RED" ? "danger" : (wRisk === "AMBER" ? "warning" : "success");

            html += '<div class="col-md-6"><div class="card border-' + wRiskColor + '">';
            html += '<div class="card-header d-flex justify-content-between align-items-center">' +
                '<strong><i class="bi bi-droplet"></i> Well ' + wellName + '</strong>' +
                '<span class="badge bg-' + wRiskColor + '">' + wRisk + '</span></div>';
            html += '<div class="card-body">';
            html += '<small class="text-muted">' + w.n_fractures + ' fractures</small>';

            // Stress
            if (w.stress && !w.stress.error) {
                html += '<div class="mt-2"><strong>Stress:</strong> ' + w.stress.regime +
                    ' | SHmax: ' + w.stress.shmax + '° | σ1: ' + w.stress.sigma1 +
                    ' | σ3: ' + w.stress.sigma3 + ' | μ: ' + w.stress.mu + '</div>';
            }

            // Classification
            if (w.classification && !w.classification.error) {
                html += '<div><strong>ML:</strong> ' + (w.classification.accuracy * 100).toFixed(1) +
                    '% accuracy | F1: ' + (w.classification.f1 * 100).toFixed(1) +
                    '% | ' + w.classification.n_classes + ' classes</div>';
            }

            // Risk
            if (w.risk && !w.risk.error) {
                html += '<div><strong>Risk:</strong> ' + w.risk.pct_critically_stressed +
                    '% critically stressed (' + w.risk.n_critical + ' fractures)</div>';
            }

            html += '</div></div></div>';
        });
        html += '</div>';

        // Comparison chart
        if (r.comparison_chart) {
            html += '<div class="card mt-3"><div class="card-header"><i class="bi bi-bar-chart-line"></i> Field Comparison Chart</div>' +
                '<div class="card-body text-center"><img src="' + r.comparison_chart + '" class="img-fluid" alt="Batch comparison"></div></div>';
        }

        body.innerHTML = html;
        showToast("Batch: " + fs.n_wells + " wells analyzed, worst risk: " + fs.worst_risk);
    } catch (err) {
        showToast("Batch error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function runGuidedWizard() {
    var taskId = generateTaskId();
    showLoadingWithProgress("Running guided analysis wizard...", taskId);
    try {
        var r = await apiPost("/api/analysis/guided-wizard", {
            source: currentSource, well: getWell(), depth: getDepth(),
            task_id: taskId
        });

        var container = document.getElementById("wizard-results");
        container.classList.remove("d-none");

        // Overall status banner
        var statusColors = {PROCEED: "success", PROCEED_WITH_REVIEW: "info", CAUTION: "warning", HALT: "danger"};
        var statusIcons = {PROCEED: "check-circle-fill", PROCEED_WITH_REVIEW: "info-circle-fill",
                           CAUTION: "exclamation-triangle-fill", HALT: "x-octagon-fill"};
        var color = statusColors[r.overall_status] || "secondary";
        var icon = statusIcons[r.overall_status] || "question-circle";

        var html = '<div class="card border-' + color + '">';
        html += '<div class="card-header bg-' + color + ' text-white">' +
            '<i class="bi bi-' + icon + '"></i> Guided Analysis — ' + r.overall_status.replace(/_/g, " ") +
            ' <span class="float-end badge bg-light text-dark">' + r.well + '</span></div>';

        html += '<div class="card-body">';
        html += '<div class="alert alert-' + color + ' mb-3"><strong>' + r.overall_message + '</strong></div>';

        // Step progress bar
        html += '<div class="d-flex gap-1 mb-4">';
        (r.steps || []).forEach(function(step) {
            var sc = {PASS: "success", WARN: "warning", FAIL: "danger",
                      PROCEED: "success", PROCEED_WITH_REVIEW: "info",
                      CAUTION: "warning", HALT: "danger"};
            var c = sc[step.status] || "secondary";
            var w = Math.floor(100 / r.n_steps);
            html += '<div class="progress flex-fill" style="height:28px">' +
                '<div class="progress-bar bg-' + c + '" style="width:100%" title="Step ' + step.step + '">' +
                step.step + '. ' + step.title + '</div></div>';
        });
        html += '</div>';

        // Each step card
        (r.steps || []).forEach(function(step) {
            var sc = {PASS: "success", WARN: "warning", FAIL: "danger",
                      PROCEED: "success", PROCEED_WITH_REVIEW: "info",
                      CAUTION: "warning", HALT: "danger"};
            var si = {PASS: "check-circle", WARN: "exclamation-triangle", FAIL: "x-circle",
                      PROCEED: "check-circle", PROCEED_WITH_REVIEW: "info-circle",
                      CAUTION: "exclamation-triangle", HALT: "x-octagon"};
            var c = sc[step.status] || "secondary";
            var i = si[step.status] || "question-circle";

            html += '<div class="card mb-2 border-' + c + '">';
            html += '<div class="card-header py-2 d-flex justify-content-between align-items-center">' +
                '<span><i class="bi bi-' + i + ' text-' + c + '"></i> ' +
                '<strong>Step ' + step.step + ': ' + step.title + '</strong></span>' +
                '<span class="badge bg-' + c + '">' + step.status + '</span></div>';

            html += '<div class="card-body py-2">';
            html += '<p class="mb-1 small">' + step.summary + '</p>';

            // Details
            if (step.details && Object.keys(step.details).length > 0) {
                html += '<div class="small text-muted mb-1">';
                Object.entries(step.details).forEach(function(pair) {
                    var k = pair[0], v = pair[1];
                    if (k === "issues" && Array.isArray(v) && v.length > 0) {
                        html += '<div class="mt-1"><strong>Issues:</strong><ul class="mb-0">';
                        v.forEach(function(issue) { html += '<li>' + issue + '</li>'; });
                        html += '</ul></div>';
                    } else if (k === "key_findings" && Array.isArray(v) && v.length > 0) {
                        html += '<div class="mt-1"><strong>Key Findings:</strong><ul class="mb-0">';
                        v.forEach(function(f) { html += '<li class="text-dark fw-bold">' + f + '</li>'; });
                        html += '</ul></div>';
                    } else if (k === "top_confusion" && v) {
                        html += '<span class="me-3"><strong>' + k + ':</strong> ' +
                            v.true_class + ' → ' + v.predicted_as + ' (' + v.count + ')</span>';
                    } else if (k !== "error" && !Array.isArray(v) && typeof v !== "object") {
                        html += '<span class="me-3"><strong>' + k + ':</strong> ' + v + '</span>';
                    }
                });
                html += '</div>';
            }

            // Next action
            html += '<div class="alert alert-' + c + ' py-1 px-2 mb-0 mt-1 small">' +
                '<i class="bi bi-arrow-right-circle"></i> ' + step.next_action + '</div>';

            html += '</div></div>';
        });

        // Key findings summary
        if (r.key_findings && r.key_findings.length > 0) {
            html += '<div class="alert alert-primary mt-3"><h6><i class="bi bi-star-fill"></i> Key Findings for Stakeholders</h6><ul class="mb-0">';
            r.key_findings.forEach(function(f) {
                html += '<li>' + f + '</li>';
            });
            html += '</ul></div>';
        }

        // Counts summary
        html += '<div class="d-flex gap-3 mt-2 small">' +
            '<span class="badge bg-success">' + r.pass_count + ' PASS</span>' +
            '<span class="badge bg-warning text-dark">' + r.warn_count + ' WARN</span>' +
            '<span class="badge bg-danger">' + r.fail_count + ' FAIL</span></div>';

        html += '</div></div>';
        container.innerHTML = html;

        showToast("Wizard: " + r.overall_status.replace(/_/g, " "));
    } catch (err) {
        showToast("Wizard error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Full Analysis Pipeline ────────────────────────

var _pipelineRunning = false;

function closePipelinePanel() {
    document.getElementById("pipeline-panel").classList.add("d-none");
}

function _updatePipelineStep(steps, idx, status) {
    // status: "running", "done", "error", "pending"
    var colors = {running: "primary", done: "success", error: "danger", pending: "secondary"};
    var icons = {running: "arrow-right-circle", done: "check-circle-fill", error: "x-circle-fill", pending: "circle"};
    var stepsEl = document.getElementById("pipeline-steps");
    var badges = stepsEl.querySelectorAll(".badge");
    if (badges[idx]) {
        badges[idx].className = "badge bg-" + colors[status] + " py-1 px-2";
        badges[idx].innerHTML = '<i class="bi bi-' + icons[status] + '"></i> ' + steps[idx].label;
    }
}

async function runFullPipeline() {
    if (_pipelineRunning) { showToast("Pipeline already running", "Warning"); return; }
    _pipelineRunning = true;

    var well = getWell();
    var depth = getDepth();
    var regime = getRegime();
    var panel = document.getElementById("pipeline-panel");
    panel.classList.remove("d-none");
    panel.scrollIntoView({behavior: "smooth"});

    var steps = [
        {label: "Data Validation", endpoint: "/api/data/validate-constraints", tab: "validation",
         body: {source: currentSource, well: well}},
        {label: "Stress Inversion", endpoint: "/api/analysis/inversion", tab: "inversion",
         body: {source: currentSource, well: well, depth_m: depth, regime: regime}},
        {label: "ML Classification", endpoint: "/api/analysis/classify", tab: "classify",
         body: {source: currentSource, well: well, classifier: "gradient_boosting"}},
        {label: "Risk Assessment", endpoint: "/api/analysis/risk-matrix", tab: "risk",
         body: {source: currentSource, well: well, depth: depth}},
        {label: "Uncertainty Budget", endpoint: "/api/analysis/uncertainty-budget", tab: "uncertainty",
         body: {source: currentSource, well: well, depth: depth}},
        {label: "Executive Summary", endpoint: "/api/analysis/executive-summary", tab: "executive",
         body: {source: currentSource, well: well, depth: depth}}
    ];

    // Render step badges
    var stepsEl = document.getElementById("pipeline-steps");
    stepsEl.innerHTML = steps.map(function(s) {
        return '<span class="badge bg-secondary py-1 px-2"><i class="bi bi-circle"></i> ' + s.label + '</span>';
    }).join("");

    var bar = document.getElementById("pipeline-bar");
    var statusEl = document.getElementById("pipeline-status");
    var elapsedEl = document.getElementById("pipeline-elapsed");
    var summaryEl = document.getElementById("pipeline-summary");
    summaryEl.classList.add("d-none");
    bar.style.width = "0%";
    bar.className = "progress-bar bg-warning";

    var startTime = Date.now();
    var results = {};
    var errors = [];
    var timer = setInterval(function() {
        var sec = ((Date.now() - startTime) / 1000).toFixed(0);
        elapsedEl.textContent = sec + "s";
    }, 500);

    document.getElementById("btn-pipeline").disabled = true;

    for (var i = 0; i < steps.length; i++) {
        var step = steps[i];
        _updatePipelineStep(steps, i, "running");
        statusEl.textContent = "Step " + (i + 1) + "/" + steps.length + ": " + step.label + "...";
        bar.style.width = Math.round((i / steps.length) * 100) + "%";

        try {
            var r = await apiPost(step.endpoint, step.body);
            results[step.label] = r;
            _updatePipelineStep(steps, i, "done");
        } catch (err) {
            errors.push(step.label + ": " + err.message);
            _updatePipelineStep(steps, i, "error");
        }
    }

    bar.style.width = "100%";
    clearInterval(timer);
    var totalSec = ((Date.now() - startTime) / 1000).toFixed(1);
    elapsedEl.textContent = totalSec + "s total";

    // Summary
    var nOk = steps.length - errors.length;
    bar.className = errors.length === 0 ? "progress-bar bg-success" : "progress-bar bg-warning";
    statusEl.textContent = errors.length === 0
        ? "Pipeline complete — all " + nOk + " steps passed"
        : nOk + "/" + steps.length + " steps completed, " + errors.length + " failed";

    // Build summary card
    var html = '<div class="row g-2">';
    // Inversion result
    if (results["Stress Inversion"]) {
        var inv = results["Stress Inversion"];
        html += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">SHmax</div>' +
            '<div class="metric-value">' + (inv.shmax_azimuth_deg || "--") + '&deg;</div></div></div>';
        html += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Regime</div>' +
            '<div class="metric-value text-capitalize">' + (inv.regime || "--") + '</div></div></div>';
    }
    // Risk result
    if (results["Risk Assessment"]) {
        var risk = results["Risk Assessment"];
        var rc = {HIGH: "danger", MODERATE: "warning", LOW: "success"};
        html += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Risk</div>' +
            '<div class="metric-value text-' + (rc[risk.overall_risk] || "muted") + '">' +
            (risk.overall_risk || "--") + '</div></div></div>';
    }
    // Executive summary verdict
    if (results["Executive Summary"]) {
        var exec = results["Executive Summary"];
        html += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Decision</div>' +
            '<div class="metric-value">' + (exec.overall_risk || exec.go_no_go || "--") + '</div></div></div>';
    }
    html += '</div>';

    if (errors.length > 0) {
        html += '<div class="alert alert-danger py-2 mt-2 small"><strong>Issues:</strong><ul class="mb-0">';
        errors.forEach(function(e) { html += '<li>' + e + '</li>'; });
        html += '</ul></div>';
    }

    html += '<div class="mt-2"><button class="btn btn-sm btn-outline-primary" onclick="switchTab(\'executive\')">View Executive Summary</button> ';
    html += '<button class="btn btn-sm btn-outline-secondary" onclick="switchTab(\'inversion\')">View Inversion</button> ';
    html += '<button class="btn btn-sm btn-outline-secondary" onclick="switchTab(\'risk\')">View Risk Matrix</button></div>';

    summaryEl.innerHTML = html;
    summaryEl.classList.remove("d-none");

    document.getElementById("btn-pipeline").disabled = false;
    _pipelineRunning = false;
    showToast("Pipeline complete in " + totalSec + "s (" + nOk + "/" + steps.length + " OK)");
}


// ── Data Loading ──────────────────────────────────

async function loadSummary() {
    try {
        var data = await api("/api/data/summary?source=" + currentSource);
        val("total-fractures", data.total_fractures);
        val("total-wells", data.wells.length);
        val("total-types", data.fracture_types.length);
        val("fracture-count-badge", data.total_fractures + " fractures");

        var tbody = document.querySelector("#summary-table tbody");
        clearChildren(tbody);
        data.summary.forEach(function(row) {
            var tr = document.createElement("tr");
            var tdWell = document.createElement("td");
            var strong = document.createElement("strong");
            strong.textContent = row.well;
            tdWell.appendChild(strong);
            tr.appendChild(tdWell);
            tr.appendChild(createCell("td", row.fracture_type));
            tr.appendChild(createCell("td", row.count));
            tr.appendChild(createCell("td", fmt(row.depth_min)));
            tr.appendChild(createCell("td", fmt(row.depth_max)));
            tr.appendChild(createCell("td", fmt(row.azimuth_mean)));
            tr.appendChild(createCell("td", fmt(row.dip_mean)));
            tbody.appendChild(tr);
        });

        var wellData = await api("/api/data/wells?source=" + currentSource);
        var sel = document.getElementById("well-select");
        clearChildren(sel);
        wellData.wells.forEach(function(w) {
            var opt = document.createElement("option");
            opt.value = w.name;
            opt.textContent = w.name + " (" + w.count + " fractures)";
            sel.appendChild(opt);
        });
        currentWell = wellData.wells[0] ? wellData.wells[0].name : "3P";
        sel.value = currentWell;

        var wellInfo = wellData.wells.find(function(w) { return w.name === currentWell; });
        if (wellInfo) {
            document.getElementById("depth-input").value = Math.round(wellInfo.avg_depth);
        }

        // Load data quality assessment
        loadDataQuality();
    } catch (err) {
        showToast("Error loading data: " + err.message, "Error");
    }
}

async function loadDataQuality() {
    try {
        var q = await api("/api/data/quality?source=" + currentSource);
        var banner = document.getElementById("quality-banner");
        banner.classList.remove("d-none");

        // Score and grade
        val("quality-score", q.score + "/100");
        var gradeBadge = document.getElementById("quality-grade-badge");
        gradeBadge.textContent = "Grade: " + q.grade;
        var gradeColors = { A: "bg-success", B: "bg-primary", C: "bg-warning text-dark", D: "bg-warning text-dark", F: "bg-danger" };
        gradeBadge.className = "badge fs-6 " + (gradeColors[q.grade] || "bg-secondary");

        // Progress bar
        var bar = document.getElementById("quality-bar");
        bar.style.width = q.score + "%";
        bar.className = "progress-bar " + (q.score >= 80 ? "bg-success" : q.score >= 60 ? "bg-warning" : "bg-danger");

        // Issues
        var issuesDiv = document.getElementById("quality-issues");
        clearChildren(issuesDiv);
        if (q.issues && q.issues.length > 0) {
            var issueAlert = document.createElement("div");
            issueAlert.className = "alert alert-danger py-2 mb-2 small";
            var issueTitle = document.createElement("strong");
            issueTitle.textContent = "Critical Issues: ";
            issueAlert.appendChild(issueTitle);
            q.issues.forEach(function(issue, i) {
                if (i > 0) issueAlert.appendChild(document.createTextNode(" | "));
                issueAlert.appendChild(document.createTextNode(issue));
            });
            issuesDiv.appendChild(issueAlert);
        }

        // Warnings
        var warningsDiv = document.getElementById("quality-warnings");
        clearChildren(warningsDiv);
        if (q.warnings && q.warnings.length > 0) {
            var warnAlert = document.createElement("div");
            warnAlert.className = "alert alert-warning py-2 mb-2 small";
            var warnTitle = document.createElement("strong");
            warnTitle.textContent = "Warnings: ";
            warnAlert.appendChild(warnTitle);
            var warnList = document.createElement("ul");
            warnList.className = "mb-0 mt-1";
            q.warnings.forEach(function(w) {
                var li = document.createElement("li");
                li.textContent = w;
                warnList.appendChild(li);
            });
            warnAlert.appendChild(warnList);
            warningsDiv.appendChild(warnAlert);
        }

        // Recommendations
        var recsDiv = document.getElementById("quality-recommendations");
        clearChildren(recsDiv);
        if (q.recommendations && q.recommendations.length > 0) {
            q.recommendations.forEach(function(rec) {
                var recDiv = document.createElement("div");
                recDiv.className = "small text-muted";
                var icon = document.createElement("i");
                icon.className = "bi bi-arrow-right-circle me-1";
                recDiv.appendChild(icon);
                recDiv.appendChild(document.createTextNode(rec));
                recsDiv.appendChild(recDiv);
            });
        }
    } catch (err) {
        // Non-critical - don't show error toast
    }
}

// ── File Upload ───────────────────────────────────

document.getElementById("file-upload").addEventListener("change", async function() {
    var file = this.files[0];
    if (!file) return;

    showLoading("Uploading " + file.name + "...");
    try {
        var formData = new FormData();
        formData.append("file", file);
        var result = await api("/api/data/upload", { method: "POST", body: formData });

        currentSource = "uploaded";
        var badge = document.getElementById("data-source-badge");
        badge.textContent = "Uploaded: " + result.filename;
        badge.classList.add("uploaded");
        val("data-source-label", result.filename);
        document.getElementById("btn-use-demo").classList.remove("d-none");

        // Show upload validation summary
        var warnings = [];
        var oodMsg = "";
        if (result.ood_check && result.ood_check.ood_detected) {
            warnings.push("Distribution shift: " + result.ood_check.severity + " — " + result.ood_check.message);
        }
        if (result.domain_warnings) {
            result.domain_warnings.forEach(function(w) { warnings.push(w); });
        }
        if (result.quality && result.quality.issues) {
            result.quality.issues.forEach(function(i) { warnings.push(i); });
        }

        // Build validation summary
        var valHtml = '<div class="card border-info mb-3"><div class="card-header bg-info text-white">' +
            '<i class="bi bi-clipboard-check"></i> Upload Validation — ' + result.filename + '</div><div class="card-body">';

        // Quality badge
        if (result.quality) {
            var qColor = result.quality.score >= 80 ? "success" : result.quality.score >= 60 ? "warning" : "danger";
            valHtml += '<div class="d-flex gap-3 mb-2"><span class="badge bg-' + qColor + ' fs-6">' +
                'Quality: ' + result.quality.grade + ' (' + result.quality.score + '/100)</span>';
        }
        if (result.sufficiency) {
            var sColor = result.sufficiency.overall === "FULLY READY" ? "success" :
                result.sufficiency.overall === "PARTIALLY READY" ? "warning" : "danger";
            valHtml += '<span class="badge bg-' + sColor + ' fs-6">' +
                result.sufficiency.ready_count + '/' + result.sufficiency.total_count + ' analyses ready</span></div>';
        }

        // Preview stats
        if (result.preview) {
            var p = result.preview;
            valHtml += '<div class="row g-2 mb-2">';
            if (p.depth_range) {
                valHtml += '<div class="col-md-4"><small class="text-muted">Depth:</small> ' +
                    p.depth_range[0] + ' — ' + p.depth_range[1] + ' m</div>';
            }
            if (p.azimuth_range) {
                valHtml += '<div class="col-md-4"><small class="text-muted">Azimuth:</small> ' +
                    p.azimuth_range[0] + '° — ' + p.azimuth_range[1] + '°</div>';
            }
            if (p.dip_range) {
                valHtml += '<div class="col-md-4"><small class="text-muted">Dip:</small> ' +
                    p.dip_range[0] + '° — ' + p.dip_range[1] + '°</div>';
            }
            valHtml += '</div>';
            if (p.type_distribution && Object.keys(p.type_distribution).length > 0) {
                valHtml += '<div class="mb-2"><small class="text-muted">Types:</small> ';
                Object.entries(p.type_distribution).forEach(function(e) {
                    valHtml += '<span class="badge bg-secondary me-1">' + e[0] + ': ' + e[1] + '</span>';
                });
                valHtml += '</div>';
            }
        }

        // Report Card (GO/CAUTION/NO-GO per analysis)
        if (result.report_card && result.report_card.length > 0) {
            valHtml += '<div class="mt-2 mb-2"><strong class="small">Analysis Readiness:</strong>';
            valHtml += '<div class="d-flex flex-wrap gap-2 mt-1">';
            result.report_card.forEach(function(rc) {
                var rcColor = rc.status === "GO" ? "success" : rc.status === "CAUTION" ? "warning" : "danger";
                var rcIcon = rc.status === "GO" ? "bi-check-circle" : rc.status === "CAUTION" ? "bi-exclamation-triangle" : "bi-x-circle";
                valHtml += '<div class="border rounded px-2 py-1 border-' + rcColor + '">' +
                    '<i class="bi ' + rcIcon + ' text-' + rcColor + '"></i> ' +
                    '<small><strong>' + rc.analysis + '</strong>: ' +
                    '<span class="badge bg-' + rcColor + '">' + rc.status + '</span> ' +
                    '<span class="text-muted">' + rc.reason + '</span></small></div>';
            });
            valHtml += '</div></div>';
        }

        // Validity pre-filter results
        if (result.validity) {
            var vl = result.validity;
            if (vl.suspicious_count > 0) {
                valHtml += '<div class="alert alert-danger py-2 mb-1 mt-2 small">' +
                    '<i class="bi bi-exclamation-octagon"></i> <strong>' + vl.suspicious_count +
                    ' suspicious measurements</strong> detected by validity pre-filter. Review before analysis.</div>';
            } else if (vl.borderline_count > 0) {
                valHtml += '<div class="alert alert-info py-1 mb-1 mt-2 small">' +
                    '<i class="bi bi-info-circle"></i> ' + vl.borderline_count +
                    ' borderline measurements (filter accuracy: ' + (vl.filter_accuracy * 100).toFixed(1) + '%)</div>';
            }
        }

        // Warnings
        if (warnings.length > 0) {
            valHtml += '<div class="alert alert-warning py-2 mb-0 mt-2 small"><strong>Warnings:</strong><ul class="mb-0">';
            warnings.forEach(function(w) { valHtml += '<li>' + w + '</li>'; });
            valHtml += '</ul></div>';
        }
        valHtml += '</div></div>';

        // Show in the data tab
        var dataTab = document.getElementById("tab-data");
        var existingVal = document.getElementById("upload-validation");
        if (existingVal) existingVal.remove();
        var valDiv = document.createElement("div");
        valDiv.id = "upload-validation";
        valDiv.innerHTML = valHtml;
        dataTab.insertBefore(valDiv, dataTab.firstChild);

        showToast("Loaded " + result.rows + " fractures from " + result.filename +
            (warnings.length > 0 ? " (" + warnings.length + " warnings)" : " — all checks passed"));
        await loadSummary();
    } catch (err) {
        showToast("Upload error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
});

function switchToDemo() {
    currentSource = "demo";
    var badge = document.getElementById("data-source-badge");
    badge.textContent = "Demo Data";
    badge.classList.remove("uploaded");
    val("data-source-label", "Demo");
    document.getElementById("btn-use-demo").classList.add("d-none");
    document.getElementById("file-upload").value = "";
    loadSummary();
}

// ── Visualizations ────────────────────────────────

async function loadAllViz() {
    showLoading("Generating visualizations...");
    try {
        var well = currentWell;
        var src = currentSource;

        var results = await Promise.all([
            api("/api/viz/rose?well=" + well + "&source=" + src),
            api("/api/viz/stereonet?well=" + well + "&source=" + src),
            api("/api/viz/depth-profile?source=" + src)
        ]);

        setImg("rose-img", results[0].image);
        setImg("stereonet-img", results[1].image);
        if (results[2].image) setImg("depth-img", results[2].image);
        showToast("Visualizations generated for Well " + well);
    } catch (err) {
        showToast("Visualization error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Inversion (Enhanced with pore pressure + interpretation) ──

async function runInversion() {
    var selectedRegime = document.getElementById("regime-select").value;
    var loadingMsg = selectedRegime === "auto"
        ? "Auto-detecting best stress regime (running all 3 inversions)..."
        : "Running stress inversion (with pore pressure correction)...";
    showLoading(loadingMsg);
    try {
        var geoInput = document.getElementById("geothermal-input");
        var geoGradient = geoInput ? parseFloat(geoInput.value) / 1000.0 : 0.030;  // Convert °C/km to °C/m
        var body = {
            well: currentWell,
            regime: selectedRegime,
            depth_m: parseFloat(document.getElementById("depth-input").value),
            cohesion: 0,
            source: currentSource,
            pore_pressure: getPorePresure(),
            geothermal_gradient: geoGradient
        };

        var r = await api("/api/analysis/inversion", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });

        document.getElementById("inversion-results").classList.remove("d-none");
        // Render stakeholder brief (plain-English decision summary)
        if (r.stakeholder_brief) {
            renderStakeholderBrief("inv-brief", r.stakeholder_brief, "inv-brief-detail");
        }
        val("inv-sigma1", r.sigma1.toFixed(1));
        val("inv-sigma2", r.sigma2.toFixed(1));
        val("inv-sigma3", r.sigma3.toFixed(1));
        val("inv-R", r.R.toFixed(3));
        val("inv-shmax", r.shmax_azimuth_deg.toFixed(1) + "\u00b0");
        val("inv-pp", (r.pore_pressure_mpa || 0).toFixed(1) + " MPa");

        // Data quality badge
        if (r.data_quality) {
            var dq = r.data_quality;
            var qc = dq.confidence_level === "HIGH" ? "success" : dq.confidence_level === "MODERATE" ? "warning" : "danger";
            var qBadge = document.getElementById("inv-quality-badge");
            if (qBadge) {
                qBadge.className = "badge bg-" + qc + " ms-2";
                qBadge.textContent = "Data: " + dq.grade + " (" + dq.score + "/100)";
                qBadge.title = dq.confidence_level + " confidence" +
                    (dq.issues.length > 0 ? " | Issues: " + dq.issues.join("; ") : "") +
                    (dq.warnings.length > 0 ? " | Warnings: " + dq.warnings.join("; ") : "");
            }
        }

        // Uncertainty confidence intervals
        if (r.uncertainty) {
            var uncEl = document.getElementById("inv-uncertainty");
            if (uncEl) {
                var u = r.uncertainty;
                var qColor = u.quality === "WELL_CONSTRAINED" ? "success" :
                             u.quality === "MODERATELY_CONSTRAINED" ? "warning" : "danger";
                var uhtml = '<div class="card border-' + qColor + ' mb-3">' +
                    '<div class="card-header py-2 bg-' + qColor + ' bg-opacity-10">' +
                    '<i class="bi bi-bar-chart me-1"></i> <strong>Uncertainty (90% CI)</strong> ' +
                    '<span class="badge bg-' + qColor + '">' + (u.quality || "").replace(/_/g, " ") + '</span></div>' +
                    '<div class="card-body py-2"><div class="row text-center">';
                if (u.shmax_ci_90 && u.shmax_ci_90.length === 2) {
                    uhtml += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">SHmax</div>' +
                        '<div class="metric-value">' + r.shmax_azimuth_deg.toFixed(1) + '°</div>' +
                        '<div class="small text-muted">' + u.shmax_ci_90[0].toFixed(1) + '° – ' + u.shmax_ci_90[1].toFixed(1) + '°</div></div></div>';
                }
                if (u.sigma1_ci_90 && u.sigma1_ci_90.length === 2) {
                    uhtml += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">σ1</div>' +
                        '<div class="metric-value">' + r.sigma1 + ' MPa</div>' +
                        '<div class="small text-muted">' + u.sigma1_ci_90[0].toFixed(1) + ' – ' + u.sigma1_ci_90[1].toFixed(1) + '</div></div></div>';
                }
                if (u.mu_ci_90 && u.mu_ci_90.length === 2) {
                    uhtml += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Friction (μ)</div>' +
                        '<div class="metric-value">' + r.mu + '</div>' +
                        '<div class="small text-muted">' + u.mu_ci_90[0].toFixed(3) + ' – ' + u.mu_ci_90[1].toFixed(3) + '</div></div></div>';
                }
                if (r.critically_stressed_range) {
                    var cs = r.critically_stressed_range;
                    uhtml += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">CS%</div>' +
                        '<div class="metric-value">' + cs.best_estimate + '%</div>' +
                        '<div class="small text-muted">' + cs.high_friction + '% – ' + cs.low_friction + '%</div></div></div>';
                }
                uhtml += '</div></div></div>';
                uncEl.innerHTML = uhtml;
                uncEl.classList.remove("d-none");
            }
        }

        // WSM Quality Ranking (international standard)
        if (r.uncertainty && r.uncertainty.wsm_quality_rank) {
            var wsmEl = document.getElementById("inv-wsm-quality");
            if (wsmEl) {
                var wsm = r.uncertainty;
                var wsmColor = wsm.wsm_quality_rank <= "B" ? "success" :
                               wsm.wsm_quality_rank === "C" ? "warning" :
                               wsm.wsm_quality_rank === "D" ? "danger" : "dark";
                wsmEl.innerHTML = '<div class="card border-' + wsmColor + ' mb-3">' +
                    '<div class="card-header py-2 bg-' + wsmColor + ' bg-opacity-10">' +
                    '<i class="bi bi-globe me-1"></i> <strong>WSM Quality Rank</strong> ' +
                    '<span class="badge bg-' + wsmColor + ' fs-6">Grade ' + wsm.wsm_quality_rank + '</span></div>' +
                    '<div class="card-body py-2">' +
                    '<p class="mb-1">' + wsm.wsm_quality_detail + '</p>' +
                    '<p class="small text-muted mb-0">World Stress Map 2025 standard (Heidbach et al.). ' +
                    'SHmax uncertainty: ±' + (wsm.shmax_std_deg || "?") + '°</p></div></div>';
                wsmEl.classList.remove("d-none");
            }
        }

        // Calibration Warning (stress magnitude underdetermination)
        if (r.calibration_warning) {
            var calWEl = document.getElementById("inv-calibration-warning");
            if (calWEl) {
                calWEl.innerHTML = '<div class="alert alert-warning mb-3 py-2">' +
                    '<i class="bi bi-exclamation-triangle me-1"></i> ' +
                    '<strong>Calibration Notice:</strong> ' + r.calibration_warning.message + '</div>';
                calWEl.classList.remove("d-none");
            }
        }

        // Multi-criteria CS% comparison
        if (r.multi_criteria_cs) {
            var mcEl = document.getElementById("inv-multi-criteria");
            if (mcEl) {
                var mc = r.multi_criteria_cs;
                mcEl.innerHTML = '<div class="card border-secondary mb-3">' +
                    '<div class="card-header py-2 bg-secondary bg-opacity-10">' +
                    '<i class="bi bi-layers me-1"></i> <strong>Multi-Criteria CS%</strong></div>' +
                    '<div class="card-body py-2"><div class="row text-center">' +
                    '<div class="col-md-4"><div class="metric-card"><div class="metric-label">Mohr-Coulomb</div>' +
                    '<div class="metric-value">' + mc.mohr_coulomb_pct + '%</div></div></div>' +
                    '<div class="col-md-4"><div class="metric-card"><div class="metric-label">Mogi-Coulomb</div>' +
                    '<div class="metric-value">' + mc.mogi_coulomb_pct + '%</div>' +
                    '<div class="small text-muted">Accounts for σ2</div></div></div>' +
                    '<div class="col-md-4"><div class="metric-card"><div class="metric-label">Drucker-Prager</div>' +
                    '<div class="metric-value">' + mc.drucker_prager_pct + '%</div>' +
                    '<div class="small text-muted">Smooth yield</div></div></div>' +
                    '</div><p class="small text-muted mt-2 mb-0">' + mc.note + '</p></div></div>';
                mcEl.classList.remove("d-none");
            }
        }

        // Mud Weight Window
        if (r.mud_weight_window) {
            var mwEl = document.getElementById("inv-mud-weight");
            if (mwEl) {
                var mw = r.mud_weight_window;
                var sw = mw.safe_window || {};
                var mwColor = mw.status === "SAFE" ? "success" :
                              mw.status === "NARROW" ? "warning" : "danger";
                mwEl.innerHTML = '<div class="card border-' + mwColor + ' mb-3">' +
                    '<div class="card-header py-2 bg-' + mwColor + ' bg-opacity-10">' +
                    '<i class="bi bi-droplet me-1"></i> <strong>Mud Weight Window</strong> ' +
                    '<span class="badge bg-' + mwColor + '">' + mw.status + '</span></div>' +
                    '<div class="card-body py-2"><div class="row text-center">' +
                    '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Pore Pressure</div>' +
                    '<div class="metric-value">' + (mw.pore_pressure ? mw.pore_pressure.ppg : "?") + ' ppg</div>' +
                    '<div class="small text-muted">' + (mw.pore_pressure ? mw.pore_pressure.sg : "?") + ' sg</div></div></div>' +
                    '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Lower Bound</div>' +
                    '<div class="metric-value">' + (sw.lower_ppg || "?") + ' ppg</div>' +
                    '<div class="small text-muted">Kick/collapse</div></div></div>' +
                    '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Upper Bound</div>' +
                    '<div class="metric-value">' + (sw.upper_ppg || "?") + ' ppg</div>' +
                    '<div class="small text-muted">Frac gradient</div></div></div>' +
                    '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Margin</div>' +
                    '<div class="metric-value">' + (sw.margin_ppg || "?") + ' ppg</div>' +
                    '<div class="small text-muted">' + (sw.margin_sg ? sw.margin_sg.toFixed(3) : "?") + ' sg</div></div></div>' +
                    '</div></div></div>';
                mwEl.classList.remove("d-none");
            }
        }

        // Show stress profile section (user clicks Generate to load)
        var spEl = document.getElementById("inv-stress-profile");
        if (spEl) spEl.classList.remove("d-none");

        // Risk categories
        if (r.risk_categories) {
            val("inv-high-risk", r.risk_categories.high);
            val("inv-mod-risk", r.risk_categories.moderate);
            val("inv-low-risk", r.risk_categories.low);
        }

        // Display interpretation for stakeholders
        if (r.interpretation) {
            renderInterpretation(r.interpretation);
        }

        setImg("mohr-img", r.mohr_circle_img);
        setImg("slip-img", r.slip_tendency_img);
        setImg("dilation-img", r.dilation_tendency_img);
        setImg("dashboard-img", r.dashboard_img);

        // Display recommendations
        if (r.recommendations && r.recommendations.length > 0) {
            var recEl = document.getElementById("inv-recommendations");
            if (recEl) {
                var rhtml = '<div class="card border-info mb-3"><div class="card-header bg-info text-white py-2">' +
                    '<i class="bi bi-lightbulb"></i> <strong>Actionable Recommendations</strong></div>' +
                    '<div class="card-body py-2"><ul class="list-group list-group-flush">';
                r.recommendations.forEach(function(rec) {
                    var pc = rec.priority === "HIGH" ? "danger" : rec.priority === "MODERATE" ? "warning" : "info";
                    rhtml += '<li class="list-group-item px-0 py-1"><span class="badge bg-' + pc + ' me-2">' +
                        rec.priority + '</span><strong>' + rec.category + ':</strong> ' + rec.text +
                        '<div class="text-muted small">' + rec.rationale + '</div></li>';
                });
                rhtml += '</ul></div></div>';
                recEl.innerHTML = rhtml;
                recEl.classList.remove("d-none");
            }
        }

        // Display temperature correction (2025 research)
        if (r.thermal_correction) {
            var thEl = document.getElementById("inv-thermal");
            if (thEl) {
                var tc = r.thermal_correction;
                var bgClass = tc.is_corrected ? "border-warning" : "border-success";
                var iconClass = tc.is_corrected ? "bi-thermometer-high text-warning" : "bi-thermometer-low text-success";
                var thtml = '<div class="card ' + bgClass + ' mb-0">' +
                    '<div class="card-header py-2"><i class="bi ' + iconClass + '"></i> ' +
                    '<strong>Thermal Correction</strong> — Formation: ' + tc.temperature_c + '°C' +
                    ' (gradient: ' + (tc.geothermal_gradient * 1000).toFixed(0) + ' °C/km)</div>' +
                    '<div class="card-body py-2">';
                if (tc.is_corrected) {
                    thtml += '<div class="alert alert-warning py-1 mb-2"><strong>Deep well thermal effect detected.</strong> ' +
                        tc.explanation + '</div>' +
                        '<div class="row text-center">' +
                        '<div class="col-4"><div class="metric-card"><div class="metric-label">μ original</div>' +
                        '<div class="metric-value">' + tc.mu_original.toFixed(3) + '</div></div></div>' +
                        '<div class="col-4"><div class="metric-card border-warning"><div class="metric-label">μ effective</div>' +
                        '<div class="metric-value text-warning">' + tc.mu_effective.toFixed(3) + '</div></div></div>' +
                        '<div class="col-4"><div class="metric-card"><div class="metric-label">New CS fractures</div>' +
                        '<div class="metric-value text-danger">+' + tc.new_critical_from_thermal + '</div>' +
                        '<div class="text-muted small">' + tc.cs_pct_original + '% → ' + tc.cs_pct_corrected + '% CS</div></div></div>' +
                        '</div>';
                } else {
                    thtml += '<span class="text-success"><i class="bi bi-check-circle"></i> ' + tc.explanation + '</span>';
                }
                thtml += '</div></div>';
                thEl.innerHTML = thtml;
                thEl.classList.remove("d-none");
            }
        }

        // Display auto-regime detection results if applicable
        renderAutoRegime(r.auto_regime);

        var regimeLabel = r.regime;
        if (r.auto_regime) {
            regimeLabel = r.auto_regime.best_regime + " (auto-detected, " + r.auto_regime.confidence + " confidence)";
        }
        showToast("Inversion complete: " + regimeLabel + ", SHmax=" + r.shmax_azimuth_deg + "\u00b0, Pp=" + (r.pore_pressure_mpa || 0).toFixed(1) + " MPa");
    } catch (err) {
        showToast("Inversion error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

function renderInterpretation(interp) {
    var section = document.getElementById("interpretation-section");
    var body = document.getElementById("interpretation-body");
    clearChildren(body);

    // Interpretations
    if (interp.interpretations) {
        interp.interpretations.forEach(function(item) {
            var div = document.createElement("div");
            div.className = "mb-3 p-3 bg-light rounded";

            var title = document.createElement("h6");
            title.className = "mb-1";
            title.textContent = item.title + ": " + item.value;
            div.appendChild(title);

            var explain = document.createElement("p");
            explain.className = "mb-1 small";
            explain.textContent = item.explanation;
            div.appendChild(explain);

            if (item.confidence) {
                var conf = document.createElement("small");
                conf.className = "text-muted";
                conf.textContent = "Basis: " + item.confidence;
                div.appendChild(conf);
            }

            body.appendChild(div);
        });
    }

    // Warnings
    if (interp.warnings && interp.warnings.length > 0) {
        var warnDiv = document.createElement("div");
        warnDiv.className = "alert alert-warning mt-3";
        var warnTitle = document.createElement("strong");
        warnTitle.textContent = "Warnings:";
        warnDiv.appendChild(warnTitle);
        var warnList = document.createElement("ul");
        warnList.className = "mb-0 mt-1";
        interp.warnings.forEach(function(w) {
            var li = document.createElement("li");
            li.textContent = w;
            warnList.appendChild(li);
        });
        warnDiv.appendChild(warnList);
        body.appendChild(warnDiv);
    }

    // Recommendations
    if (interp.recommendations && interp.recommendations.length > 0) {
        var recDiv = document.createElement("div");
        recDiv.className = "alert alert-success mt-3";
        var recTitle = document.createElement("strong");
        recTitle.textContent = "Recommendations:";
        recDiv.appendChild(recTitle);
        var recList = document.createElement("ol");
        recList.className = "mb-0 mt-1";
        interp.recommendations.forEach(function(r) {
            var li = document.createElement("li");
            li.textContent = r;
            recList.appendChild(li);
        });
        recDiv.appendChild(recList);
        body.appendChild(recDiv);
    }

    section.classList.remove("d-none");

    // Risk banner
    var banner = document.getElementById("risk-banner");
    if (interp.risk_level) {
        var colorMap = { HIGH: "alert-danger", MODERATE: "alert-warning", LOW: "alert-success" };
        banner.className = "alert mb-4 " + (colorMap[interp.risk_level] || "alert-info");
        val("risk-title", "Risk Level: " + interp.risk_level);
        val("risk-body", interp.risk_level === "HIGH"
            ? "Significant number of critically stressed fractures detected. Review recommendations below before proceeding with operations."
            : interp.risk_level === "MODERATE"
            ? "Moderate risk level. Some fractures may be reactivated during operations."
            : "Low risk. Most fractures are stable under current stress conditions.");
        banner.classList.remove("d-none");
    }
}

function renderAutoRegime(autoRegime) {
    var banner = document.getElementById("auto-regime-banner");
    if (!autoRegime) {
        banner.classList.add("d-none");
        return;
    }
    banner.classList.remove("d-none");

    // Confidence badge
    var badge = document.getElementById("auto-regime-confidence");
    var confColors = { HIGH: "bg-success", MODERATE: "bg-warning text-dark", LOW: "bg-danger" };
    badge.className = "badge ms-2 " + (confColors[autoRegime.confidence] || "bg-secondary");
    badge.textContent = autoRegime.confidence + " Confidence";

    // Summary
    document.getElementById("auto-regime-summary").textContent = autoRegime.stakeholder_summary;

    // Comparison table
    var tbody = document.querySelector("#auto-regime-table tbody");
    clearChildren(tbody);
    autoRegime.comparison.forEach(function(row) {
        var tr = document.createElement("tr");
        if (row.is_best) tr.className = "table-success";

        var tdName = document.createElement("td");
        var strong = document.createElement("strong");
        strong.textContent = row.regime.replace("_", "-");
        tdName.appendChild(strong);
        tr.appendChild(tdName);

        tr.appendChild(createCell("td", row.misfit.toFixed(1)));
        tr.appendChild(createCell("td", row.sigma1.toFixed(1)));
        tr.appendChild(createCell("td", row.sigma3.toFixed(1)));
        tr.appendChild(createCell("td", row.R.toFixed(3)));
        tr.appendChild(createCell("td", row.shmax_azimuth_deg.toFixed(1) + "\u00b0"));
        tr.appendChild(createCell("td", row.mu.toFixed(3)));

        var tdBest = document.createElement("td");
        if (row.is_best) {
            var bestBadge = document.createElement("span");
            bestBadge.className = "badge bg-success";
            bestBadge.textContent = "BEST FIT";
            tdBest.appendChild(bestBadge);
        }
        tr.appendChild(tdBest);
        tbody.appendChild(tr);
    });
}

function downloadCSV(csvContent, filename) {
    var blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
}

async function exportInversion() {
    showLoading("Exporting inversion results...");
    try {
        var r = await api("/api/export/inversion", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                well: currentWell,
                regime: document.getElementById("regime-select").value,
                depth_m: parseFloat(document.getElementById("depth-input").value),
                source: currentSource,
                pore_pressure: getPorePresure()
            })
        });
        downloadCSV(r.csv, r.filename);
        showToast("Exported " + r.rows + " fractures with tendencies to " + r.filename);
    } catch (err) {
        showToast("Export error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function exportData() {
    showLoading("Exporting fracture data...");
    try {
        var r = await api("/api/export/data", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                well: currentWell,
                source: currentSource
            })
        });
        downloadCSV(r.csv, r.filename);
        showToast("Exported " + r.rows + " fractures to " + r.filename);
    } catch (err) {
        showToast("Export error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function exportPdfReport() {
    var taskId = generateTaskId();
    showLoadingWithProgress("Generating PDF report...", taskId);
    try {
        var r = await apiPost("/api/export/pdf-report", {
            source: currentSource, well: getWell(), depth: getDepth(),
            task_id: taskId
        });
        // Convert base64 to blob and trigger download
        var byteChars = atob(r.pdf_base64);
        var byteArray = new Uint8Array(byteChars.length);
        for (var i = 0; i < byteChars.length; i++) {
            byteArray[i] = byteChars.charCodeAt(i);
        }
        var blob = new Blob([byteArray], { type: "application/pdf" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = r.filename || "GeoStress_Report.pdf";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast("PDF report downloaded (" + r.pages + " pages, " + r.size_kb + " KB)");
    } catch (err) {
        showToast("PDF export error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function exportFullJsonReport() {
    var taskId = generateTaskId();
    showLoadingWithProgress("Generating full JSON report...", taskId);
    try {
        var r = await apiPost("/api/export/full-report", {
            source: currentSource,
            well: getWell(),
            depth: getDepth(),
            regime: getRegime(),
            task_id: taskId
        });
        // Pretty-print and trigger download
        var jsonStr = JSON.stringify(r, null, 2);
        var blob = new Blob([jsonStr], { type: "application/json" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = "GeoStress_Report_" + (r.metadata && r.metadata.well || "all") + "_" +
                     new Date().toISOString().slice(0, 10) + ".json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Summary toast
        var sections = 0;
        if (r.stress_inversion && !r.stress_inversion.error) sections++;
        if (r.risk_assessment && !r.risk_assessment.error) sections++;
        if (r.classification && !r.classification.error) sections++;
        if (r.data_quality && !r.data_quality.error) sections++;
        if (r.uncertainty && !r.uncertainty.error) sections++;
        var sizeKb = Math.round(jsonStr.length / 1024);
        showToast("JSON report downloaded — " + sections + "/5 sections, " + sizeKb + " KB");
    } catch (err) {
        showToast("JSON export error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function runDecisionReadiness() {
    showLoading("Assessing decision readiness (6 quality signals)...");
    try {
        var r = await apiPost("/api/analysis/decision-readiness", {
            source: currentSource, well: getWell(), depth: getDepth()
        });
        var el = document.getElementById("readiness-results");
        el.classList.remove("d-none");
        var body = document.getElementById("readiness-body");

        var verdictColor = r.verdict === "NO_GO" ? "danger" : (r.verdict === "CAUTION" ? "warning" : "success");
        var verdictIcon = r.verdict === "NO_GO" ? "x-octagon" : (r.verdict === "CAUTION" ? "exclamation-triangle" : "check-circle");

        // Verdict banner
        var html = '<div class="alert alert-' + verdictColor + ' p-3 mb-3">' +
            '<h5><i class="bi bi-' + verdictIcon + '"></i> ' + r.verdict.replace(/_/g, " ") + '</h5>' +
            '<p class="mb-0">' + r.verdict_detail + '</p></div>';

        // Signal summary (traffic light)
        var ss = r.signal_summary || {};
        html += '<div class="d-flex gap-3 mb-3">';
        html += '<span class="badge bg-success fs-6">' + (ss.GREEN || 0) + ' GREEN</span>';
        html += '<span class="badge bg-warning text-dark fs-6">' + (ss.AMBER || 0) + ' AMBER</span>';
        html += '<span class="badge bg-danger fs-6">' + (ss.RED || 0) + ' RED</span>';
        html += '</div>';

        // Signal details table
        html += '<div class="table-responsive"><table class="table table-sm table-hover">';
        html += '<thead class="table-dark"><tr><th>Signal</th><th>Status</th><th>Detail</th><th>Action Required</th></tr></thead><tbody>';
        (r.signals || []).forEach(function(sig) {
            var gc = sig.grade === "RED" ? "danger" : (sig.grade === "AMBER" ? "warning" : "success");
            html += '<tr><td><strong>' + sig.signal + '</strong></td>' +
                '<td><span class="badge bg-' + gc + '">' + sig.grade + '</span></td>' +
                '<td>' + sig.detail + '</td>' +
                '<td><small>' + sig.action + '</small></td></tr>';
        });
        html += '</tbody></table></div>';

        html += '<small class="text-muted">Well: ' + r.well + ' | Computed in ' + r.computation_time_s + 's</small>';

        body.innerHTML = html;
        showToast("Decision: " + r.verdict.replace(/_/g, " ") + " (" + (ss.GREEN || 0) + " GREEN, " + (ss.RED || 0) + " RED)");
    } catch (err) {
        showToast("Decision readiness error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function runAllRegimes() {
    showLoading("Comparing all stress regimes...");
    try {
        var regimes = ["normal", "strike_slip", "thrust"];
        var results = [];

        for (var i = 0; i < regimes.length; i++) {
            var r = await api("/api/analysis/inversion", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    well: currentWell,
                    regime: regimes[i],
                    depth_m: parseFloat(document.getElementById("depth-input").value),
                    cohesion: 0,
                    source: currentSource,
                    pore_pressure: getPorePresure()
                })
            });
            results.push(r);
        }

        var tbody = document.querySelector("#regime-table tbody");
        clearChildren(tbody);
        results.forEach(function(r) {
            var tr = document.createElement("tr");
            var tdRegime = document.createElement("td");
            var strong = document.createElement("strong");
            strong.textContent = r.regime;
            tdRegime.appendChild(strong);
            tr.appendChild(tdRegime);
            tr.appendChild(createCell("td", r.sigma1.toFixed(1)));
            tr.appendChild(createCell("td", r.sigma2.toFixed(1)));
            tr.appendChild(createCell("td", r.sigma3.toFixed(1)));
            tr.appendChild(createCell("td", r.R.toFixed(3)));
            tr.appendChild(createCell("td", r.shmax_azimuth_deg.toFixed(1) + "\u00b0"));
            tr.appendChild(createCell("td", r.mu.toFixed(3)));
            tr.appendChild(createCell("td", r.critically_stressed_count + "/" + r.critically_stressed_total + " (" + r.critically_stressed_pct + "%)"));
            tbody.appendChild(tr);
        });

        document.getElementById("regime-comparison").classList.remove("d-none");
        showToast("Regime comparison complete (with pore pressure correction)");
    } catch (err) {
        showToast("Comparison error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Model Comparison (NEW) ────────────────────────

async function runModelComparison(fast) {
    showLoading(fast ? "Quick comparison (~30s)..." : "Full comparison with stacking ensemble (may take 2-4 minutes)...");
    try {
        var r = await api("/api/analysis/compare-models", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource, fast: !!fast })
        });

        document.getElementById("model-results").classList.remove("d-none");

        // Render stakeholder brief
        if (r.stakeholder_brief) {
            renderStakeholderBrief("mc-brief", r.stakeholder_brief, "mc-brief-detail");
        }

        // Ranking criterion notice
        if (r.ranking_criterion === "balanced_accuracy") {
            var notice = document.getElementById("mc-ranking-notice");
            if (notice) {
                notice.classList.remove("d-none");
                notice.innerHTML = '<i class="bi bi-info-circle me-1"></i>' +
                    '<strong>Ranked by balanced accuracy</strong> due to severe class imbalance. ' +
                    'This ensures the best model handles ALL fracture types, not just the majority ones.';
            }
        }

        // Summary metrics
        val("mc-best-model", r.best_model ? r.best_model.replace("_", " ") : "--");
        val("mc-best-acc", r.ranking && r.ranking[0] ? (r.ranking[0].accuracy * 100).toFixed(1) + "%" : "--");
        val("mc-agreement", (r.model_agreement_mean * 100).toFixed(1) + "%");
        val("mc-low-conf", r.low_confidence_count + " (" + r.low_confidence_pct + "%)");

        // Uncertainty explanation
        var alertEl = document.getElementById("mc-uncertainty-alert");
        if (r.low_confidence_pct > 10) {
            alertEl.classList.remove("d-none");
            var ucText = document.getElementById("mc-uncertainty-text");
            ucText.textContent = " When models disagree on " + r.low_confidence_pct +
                "% of fractures, those are uncertain classifications. " +
                "Consider expert review for these fractures, or provide additional data to improve accuracy.";
        } else {
            alertEl.classList.add("d-none");
        }

        // Ranking table
        var tbody = document.querySelector("#mc-ranking-table tbody");
        clearChildren(tbody);
        if (r.ranking) {
            r.ranking.forEach(function(row) {
                var tr = document.createElement("tr");
                if (row.rank === 1) tr.style.background = "#dcfce7";
                tr.appendChild(createCell("td", "#" + row.rank));
                var tdName = document.createElement("td");
                var nameStrong = document.createElement("strong");
                nameStrong.textContent = row.model.replace("_", " ");
                tdName.appendChild(nameStrong);
                tr.appendChild(tdName);
                tr.appendChild(createCell("td", (row.accuracy * 100).toFixed(1) + "%"));
                // Balanced accuracy (accounts for class imbalance)
                var balAcc = row.balanced_accuracy ? (row.balanced_accuracy * 100).toFixed(1) + "%" : "--";
                var balCell = createCell("td", balAcc);
                if (row.balanced_accuracy && row.accuracy - row.balanced_accuracy > 0.15) {
                    balCell.className = "text-danger fw-bold";
                    balCell.title = "Large gap between standard and balanced accuracy = model ignores minority classes";
                }
                tr.appendChild(balCell);
                tr.appendChild(createCell("td", (row.f1 * 100).toFixed(1) + "%"));
                tbody.appendChild(tr);
            });
        }

        // Per-model details
        var container = document.getElementById("mc-details-container");
        clearChildren(container);
        if (r.models) {
            Object.keys(r.models).forEach(function(modelName) {
                var m = r.models[modelName];
                var card = document.createElement("div");
                card.className = "card mb-3";

                var header = document.createElement("div");
                header.className = "card-header";
                header.textContent = modelName.replace("_", " ") +
                    " - Accuracy: " + (m.cv_accuracy_mean * 100).toFixed(1) +
                    "% \u00b1" + (m.cv_accuracy_std * 100).toFixed(1) + "%";
                card.appendChild(header);

                var body = document.createElement("div");
                body.className = "card-body";

                // Feature importances for this model
                if (m.feature_importances && Object.keys(m.feature_importances).length > 0) {
                    var fiTitle = document.createElement("h6");
                    fiTitle.textContent = "Top Features";
                    body.appendChild(fiTitle);

                    var sorted = Object.entries(m.feature_importances).sort(function(a, b) { return b[1] - a[1]; });
                    var topN = sorted.slice(0, 5);
                    var maxFI = topN[0][1] || 1;

                    topN.forEach(function(entry) {
                        var feat = entry[0], imp = entry[1];
                        var pct = (imp / maxFI * 100).toFixed(0);

                        var wrapper = document.createElement("div");
                        wrapper.className = "feat-bar-container";

                        var label = document.createElement("div");
                        label.className = "feat-bar-label";
                        label.textContent = feat;
                        wrapper.appendChild(label);

                        var bg = document.createElement("div");
                        bg.className = "feat-bar-bg";

                        var fill = document.createElement("div");
                        fill.className = "feat-bar-fill";
                        fill.style.width = pct + "%";
                        bg.appendChild(fill);

                        var valSpan = document.createElement("span");
                        valSpan.className = "feat-bar-value";
                        valSpan.textContent = imp.toFixed(3);
                        bg.appendChild(valSpan);

                        wrapper.appendChild(bg);
                        body.appendChild(wrapper);
                    });
                }

                // Per-class metrics table
                if (m.per_class_metrics && Object.keys(m.per_class_metrics).length > 0) {
                    var pcTitle = document.createElement("h6");
                    pcTitle.className = "mt-3";
                    pcTitle.textContent = "Per-Class Performance";
                    body.appendChild(pcTitle);

                    var table = document.createElement("table");
                    table.className = "table table-sm table-striped mb-0";
                    var thead = document.createElement("thead");
                    thead.innerHTML = "<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Samples</th></tr>";
                    table.appendChild(thead);
                    var tbody2 = document.createElement("tbody");

                    Object.entries(m.per_class_metrics).forEach(function(entry) {
                        var cls = entry[0], met = entry[1];
                        var tr = document.createElement("tr");
                        if (met.f1 === 0) tr.className = "table-danger";
                        else if (met.f1 < 0.5) tr.className = "table-warning";
                        tr.appendChild(createCell("td", cls));
                        tr.appendChild(createCell("td", (met.precision * 100).toFixed(1) + "%"));
                        tr.appendChild(createCell("td", (met.recall * 100).toFixed(1) + "%"));
                        tr.appendChild(createCell("td", (met.f1 * 100).toFixed(1) + "%"));
                        var countCell = createCell("td", met.support);
                        if (met.support < 30) {
                            countCell.className = "text-danger fw-bold";
                            countCell.title = "Under-represented class (<30 samples)";
                        }
                        tr.appendChild(countCell);
                        tbody2.appendChild(tr);
                    });

                    table.appendChild(tbody2);
                    body.appendChild(table);
                }

                // Overfit gap indicator
                if (m.overfit_gap !== undefined) {
                    var gapDiv = document.createElement("div");
                    gapDiv.className = "small mt-2";
                    var gapPct = (m.overfit_gap * 100).toFixed(1);
                    var gapColor = m.overfit_gap > 0.10 ? "text-danger" : m.overfit_gap > 0.05 ? "text-warning" : "text-success";
                    gapDiv.innerHTML = 'Train-Test Gap: <span class="' + gapColor + ' fw-bold">' + gapPct + '%</span>' +
                        (m.overfit_gap > 0.10 ? ' <i class="bi bi-exclamation-triangle text-danger"></i> Significant overfitting' :
                         m.overfit_gap > 0.05 ? ' <i class="bi bi-info-circle text-warning"></i> Mild overfitting' :
                         ' <i class="bi bi-check-circle text-success"></i> Good generalization');
                    body.appendChild(gapDiv);
                }

                card.appendChild(body);
                container.appendChild(card);
            });
        }

        // Conformal confidence section
        var confSection = document.getElementById("mc-conformal-section");
        if (r.conformal && r.conformal.available) {
            confSection.classList.remove("d-none");
            val("mc-conf-mean", (r.conformal.mean_confidence * 100).toFixed(1) + "%");
            val("mc-conf-high", r.conformal.high_confidence_pct + "%");
            val("mc-conf-uncertain", r.conformal.uncertain_count + " (" + r.conformal.uncertain_pct + "%)");
            val("mc-conf-min", (r.conformal.min_confidence * 100).toFixed(1) + "%");
        } else {
            confSection.classList.add("d-none");
        }

        // Generalization assessment
        var genSection = document.getElementById("mc-generalization");
        if (r.generalization) {
            genSection.classList.remove("d-none");
            var gen = r.generalization;
            val("mc-gen-gap", (gen.overfit_gap * 100).toFixed(1) + "%");
            val("mc-gen-stability", (gen.cv_stability * 100).toFixed(1) + "%");
            val("mc-gen-min-class", gen.min_class_count);

            var genWarnings = document.getElementById("mc-gen-warnings");
            clearChildren(genWarnings);
            if (gen.warnings && gen.warnings.length > 0) {
                gen.warnings.forEach(function(w) {
                    var div = document.createElement("div");
                    div.className = "alert alert-warning py-2 mb-2 small";
                    div.innerHTML = '<i class="bi bi-exclamation-triangle me-1"></i>' + w;
                    genWarnings.appendChild(div);
                });
            } else {
                var ok = document.createElement("div");
                ok.className = "alert alert-success py-2 mb-2 small";
                ok.innerHTML = '<i class="bi bi-check-circle me-1"></i>Model generalizes well. No significant overfitting or instability detected.';
                genWarnings.appendChild(ok);
            }
        } else {
            genSection.classList.add("d-none");
        }

        // Add overfit gap to ranking table
        if (r.ranking) {
            var rankingRows = document.querySelectorAll("#mc-ranking-table tbody tr");
            r.ranking.forEach(function(row, idx) {
                if (rankingRows[idx] && row.overfit_gap !== undefined) {
                    var gapCell = createCell("td", (row.overfit_gap * 100).toFixed(1) + "%");
                    if (row.overfit_gap > 0.10) gapCell.className = "text-danger";
                    else if (row.overfit_gap > 0.05) gapCell.className = "text-warning";
                    else gapCell.className = "text-success";
                    rankingRows[idx].appendChild(gapCell);
                }
            });
        }

        // SMOTE info
        var smoteEl = document.getElementById("mc-smote-info");
        if (smoteEl && r.smote) {
            smoteEl.classList.remove("d-none");
            var smoteHtml = '<i class="bi bi-';
            if (r.smote.applied) {
                smoteHtml += 'check-circle text-success"></i> <strong>SMOTE Active:</strong> ' +
                    'Synthetic samples generated for minority classes to improve balanced accuracy. ' +
                    r.smote.reason;
            } else {
                smoteHtml += 'info-circle text-muted"></i> SMOTE not applied: ' + r.smote.reason;
            }
            smoteEl.innerHTML = smoteHtml;
        }

        // Show comparison chart
        if (r.comparison_chart_img) {
            setImg("mc-chart-img", r.comparison_chart_img);
            document.getElementById("mc-chart-container").classList.remove("d-none");
        }

        showToast("Model comparison complete: " + (r.ranking ? r.ranking.length : 0) + " models evaluated" +
            (r.conformal && r.conformal.available ? " | Confidence: " + (r.conformal.mean_confidence * 100).toFixed(0) + "%" : ""));
    } catch (err) {
        showToast("Model comparison error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Classification (Enhanced) ─────────────────────

async function runClassification() {
    showLoading("Running ML classification with enhanced features...");
    try {
        var classifier = document.getElementById("classifier-select").value;
        var r = await api("/api/analysis/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ classifier: classifier, source: currentSource, enhanced: true })
        });

        document.getElementById("classify-results").classList.remove("d-none");
        // Render stakeholder brief
        if (r.stakeholder_brief) {
            renderStakeholderBrief("clf-brief", r.stakeholder_brief, "clf-brief-detail");
        }
        val("clf-accuracy", (r.cv_mean_accuracy * 100).toFixed(1) + "%");
        val("clf-std", "\u00b1" + (r.cv_std_accuracy * 100).toFixed(1) + "%");
        val("clf-f1", r.cv_f1_mean ? (r.cv_f1_mean * 100).toFixed(1) + "%" : "--");
        val("clf-type", classifier.replace("_", " "));

        // Confidence gate: color-code accuracy and add warning for low accuracy
        var accEl = document.getElementById("clf-accuracy");
        if (accEl) {
            if (r.cv_mean_accuracy >= 0.85) accEl.className = "metric-value text-success";
            else if (r.cv_mean_accuracy >= 0.70) accEl.className = "metric-value text-warning";
            else accEl.className = "metric-value text-danger";
        }

        // Feature importances
        var container = document.getElementById("feat-imp-container");
        clearChildren(container);
        var sorted = Object.entries(r.feature_importances).sort(function(a, b) { return b[1] - a[1]; });
        var maxVal = sorted[0] ? sorted[0][1] : 1;
        sorted.forEach(function(entry) {
            var feat = entry[0], imp = entry[1];
            var pct = (imp / maxVal * 100).toFixed(0);

            var wrapper = document.createElement("div");
            wrapper.className = "feat-bar-container";

            var label = document.createElement("div");
            label.className = "feat-bar-label";
            label.textContent = feat;
            wrapper.appendChild(label);

            var bg = document.createElement("div");
            bg.className = "feat-bar-bg";

            var fill = document.createElement("div");
            fill.className = "feat-bar-fill";
            fill.style.width = pct + "%";
            bg.appendChild(fill);

            var valSpan = document.createElement("span");
            valSpan.className = "feat-bar-value";
            valSpan.textContent = imp.toFixed(3);
            bg.appendChild(valSpan);

            wrapper.appendChild(bg);
            container.appendChild(wrapper);
        });

        // Confusion matrix
        var thead = document.querySelector("#confusion-table thead");
        var ctbody = document.querySelector("#confusion-table tbody");
        clearChildren(thead);
        clearChildren(ctbody);

        var headerRow = document.createElement("tr");
        headerRow.appendChild(createCell("th", ""));
        r.class_names.forEach(function(name) {
            headerRow.appendChild(createCell("th", name));
        });
        thead.appendChild(headerRow);

        r.confusion_matrix.forEach(function(row, i) {
            var tr = document.createElement("tr");
            var labelTd = document.createElement("td");
            var labelStrong = document.createElement("strong");
            labelStrong.textContent = r.class_names[i];
            labelTd.appendChild(labelStrong);
            tr.appendChild(labelTd);

            row.forEach(function(v, j) {
                var styles = { textAlign: "center" };
                if (i === j) styles.background = "#dcfce7";
                else if (v > 0) styles.background = "#fef2f2";
                tr.appendChild(createCell("td", v, styles));
            });
            ctbody.appendChild(tr);
        });

        // Spatial (depth-blocked) CV warning
        var spatEl = document.getElementById("clf-spatial-cv");
        if (spatEl && r.spatial_cv) {
            var sp = r.spatial_cv;
            var drop = ((r.cv_mean_accuracy - sp.spatial_cv_accuracy) * 100).toFixed(1);
            var spColor = parseFloat(drop) > 10 ? "danger" : parseFloat(drop) > 5 ? "warning" : "info";
            spatEl.innerHTML = '<div class="alert alert-' + spColor + ' py-2 mb-3">' +
                '<i class="bi bi-geo-alt me-1"></i> <strong>Spatial CV (depth-blocked):</strong> ' +
                (sp.spatial_cv_accuracy * 100).toFixed(1) + '% accuracy ' +
                '(±' + (sp.spatial_cv_std * 100).toFixed(1) + '%), ' +
                'F1: ' + (sp.spatial_cv_f1 * 100).toFixed(1) + '%. ' +
                '<span class="fw-bold">Accuracy drop: ' + drop + '%</span> vs random CV. ' +
                '<span class="text-muted">' + sp.note + '</span></div>';
            spatEl.classList.remove("d-none");
        } else if (spatEl) {
            spatEl.classList.add("d-none");
        }

        // Conformal Prediction bounds (ARMA 2025)
        var cpEl = document.getElementById("clf-conformal");
        if (cpEl && r.conformal_prediction) {
            var cp = r.conformal_prediction;
            var cpColor = cp.precision_ratio > 0.8 ? "success" : cp.precision_ratio > 0.5 ? "warning" : "danger";
            cpEl.innerHTML = '<div class="card border-' + cpColor + ' mb-3">' +
                '<div class="card-header py-2 bg-' + cpColor + ' bg-opacity-10">' +
                '<i class="bi bi-shield-check me-1"></i> <strong>Conformal Prediction</strong> ' +
                '<span class="badge bg-' + cpColor + '">Coverage: ' + (cp.empirical_coverage * 100).toFixed(0) + '%</span></div>' +
                '<div class="card-body py-2"><div class="row text-center">' +
                '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Coverage Target</div>' +
                '<div class="metric-value">' + (cp.coverage_target * 100).toFixed(0) + '%</div></div></div>' +
                '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Actual Coverage</div>' +
                '<div class="metric-value text-success">' + (cp.empirical_coverage * 100).toFixed(1) + '%</div></div></div>' +
                '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Avg Set Size</div>' +
                '<div class="metric-value">' + cp.avg_prediction_set_size + ' / ' + cp.n_classes + '</div></div></div>' +
                '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Precision</div>' +
                '<div class="metric-value">' + (cp.precision_ratio * 100).toFixed(1) + '%</div></div></div>' +
                '</div>' +
                '<p class="text-muted small mt-2 mb-0"><i class="bi bi-info-circle me-1"></i>' + cp.note + '</p>' +
                '</div></div>';
            cpEl.classList.remove("d-none");
        } else if (cpEl) {
            cpEl.classList.add("d-none");
        }

        showToast("Classification: " + (r.cv_mean_accuracy * 100).toFixed(1) + "% accuracy (" + classifier + ")");
    } catch (err) {
        showToast("Classification error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Deep Ensemble UQ (2025 Research) ──────────────

async function runDeepEnsemble() {
    showLoading("Training 5-model deep ensemble for uncertainty quantification...");
    try {
        var classifier = document.getElementById("classifier-select").value;
        var r = await api("/api/analysis/deep-ensemble", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ classifier: classifier, source: currentSource, n_ensemble: 5 })
        });

        var el = document.getElementById("ensemble-results");
        if (!el) return;

        var ep = r.epistemic_uncertainty;
        var al = r.aleatoric_uncertainty;
        var ag = r.agreement;
        var ai = r.actionable_insights;

        var html = '<div class="card border-info mb-0">' +
            '<div class="card-header bg-info text-white py-2">' +
            '<i class="bi bi-layers-fill"></i> <strong>Deep Ensemble Uncertainty</strong> — ' +
            r.n_ensemble + ' models × ' + r.classifier.replace("_", " ") +
            ' (accuracy: ' + (r.ensemble_accuracy * 100).toFixed(1) + '%)</div>' +
            '<div class="card-body py-2">' +
            '<div class="row g-2 mb-2">' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Epistemic (Reducible)</div>' +
            '<div class="metric-value text-primary">' + ep.high_pct + '%</div>' +
            '<div class="text-muted small">' + ep.high_count + ' high-uncertainty</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Aleatoric (Irreducible)</div>' +
            '<div class="metric-value text-secondary">' + al.high_pct + '%</div>' +
            '<div class="text-muted small">' + al.high_count + ' inherently noisy</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Full Agreement</div>' +
            '<div class="metric-value text-success">' + ag.full_agreement_pct + '%</div>' +
            '<div class="text-muted small">All ' + r.n_ensemble + ' models agree</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Low Agreement</div>' +
            '<div class="metric-value text-danger">' + ag.low_agreement_count + '</div>' +
            '<div class="text-muted small">&lt;60% model consensus</div></div></div>' +
            '</div>';

        // Actionable insights
        html += '<div class="alert alert-light py-2 mb-0"><strong>What to do:</strong><ul class="mb-0 small">';
        if (ai.needs_more_data_count > 0) {
            html += '<li><span class="badge bg-primary me-1">' + ai.needs_more_data_count + '</span> fractures need more training data — collecting labels here would most improve accuracy</li>';
        }
        if (ai.measurement_noise_count > 0) {
            html += '<li><span class="badge bg-secondary me-1">' + ai.measurement_noise_count + '</span> fractures have inherent measurement noise — more data won\'t help, consider manual expert review</li>';
        }
        if (ai.both_uncertain_count > 0) {
            html += '<li><span class="badge bg-warning text-dark me-1">' + ai.both_uncertain_count + '</span> fractures are uncertain in both dimensions — highest priority for expert review</li>';
        }
        html += '</ul></div></div></div>';

        el.innerHTML = html;
        el.classList.remove("d-none");

        showToast("Deep Ensemble: " + (r.ensemble_accuracy * 100).toFixed(1) + "% acc, " +
            ep.high_count + " high-uncertainty samples identified");
    } catch (err) {
        showToast("Deep Ensemble error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Transfer Learning (2025 Research) ──────────────

async function runTransferLearning() {
    showLoading("Evaluating well-to-well transfer learning (train→adapt→compare)...");
    try {
        var classifier = document.getElementById("classifier-select").value;
        // Determine source/target: train on current, test on the other
        var srcWell = currentWell;
        var tgtWell = srcWell === "3P" ? "6P" : "3P";

        var r = await api("/api/analysis/transfer-learning", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                source_well: srcWell,
                target_well: tgtWell,
                classifier: classifier,
                fine_tune_fraction: 0.2,
                source: currentSource
            })
        });

        var el = document.getElementById("transfer-results");
        if (!el) return;

        if (r.error) {
            el.innerHTML = '<div class="alert alert-warning">' + r.error + '</div>';
            el.classList.remove("d-none");
            return;
        }

        var qColor = r.transfer_quality === "GOOD" ? "success" :
                     r.transfer_quality === "MODERATE" ? "warning" : "danger";

        var html = '<div class="card border-' + qColor + ' mb-0">' +
            '<div class="card-header py-2"><i class="bi bi-arrow-left-right"></i> ' +
            '<strong>Transfer Learning</strong>: ' + r.source_well + ' → ' + r.target_well +
            ' <span class="badge bg-' + qColor + '">' + r.transfer_quality + '</span></div>' +
            '<div class="card-body py-2">' +
            '<div class="row g-2 mb-2">' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Source (' + r.source_well + ') CV</div>' +
            '<div class="metric-value">' + (r.source_cv_accuracy * 100).toFixed(1) + '%</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Zero-Shot → ' + r.target_well + '</div>' +
            '<div class="metric-value text-danger">' + (r.zero_shot_accuracy * 100).toFixed(1) + '%</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card border-' + qColor + '"><div class="metric-label">Fine-Tuned (' + r.fine_tune_n + ' samples)</div>' +
            '<div class="metric-value text-' + qColor + '">' + (r.fine_tuned_accuracy * 100).toFixed(1) + '%</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Target-Only CV</div>' +
            '<div class="metric-value">' + (r.target_only_accuracy * 100).toFixed(1) + '%</div></div></div>' +
            '</div>';

        html += '<div class="alert alert-' + qColor + ' py-1 mb-1 small">' + r.recommendation + '</div>';

        // Class overlap info
        if (r.source_only_classes && r.source_only_classes.length > 0) {
            html += '<div class="text-muted small">Classes only in ' + r.source_well + ': ' + r.source_only_classes.join(", ") + '</div>';
        }
        if (r.target_only_classes && r.target_only_classes.length > 0) {
            html += '<div class="text-muted small">Classes only in ' + r.target_well + ': ' + r.target_only_classes.join(", ") + '</div>';
        }

        html += '</div></div>';
        el.innerHTML = html;
        el.classList.remove("d-none");

        showToast("Transfer: " + r.source_well + "→" + r.target_well + " = " + r.transfer_quality +
            " (zero-shot " + (r.zero_shot_accuracy * 100).toFixed(0) + "%, fine-tuned " +
            (r.fine_tuned_accuracy * 100).toFixed(0) + "%)");
    } catch (err) {
        showToast("Transfer Learning error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Domain-Adapted Transfer ──────────────────────────
async function runTransferAdapted() {
    showLoading("Comparing 5 domain adaptation methods (MMD, pseudo-labeling, etc.)...");
    try {
        var srcWell = document.getElementById("ta-source-well").value;
        var tgtWell = document.getElementById("ta-target-well").value;
        var classifier = document.getElementById("classifier-select").value;

        var r = await api("/api/analysis/transfer-adapted", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                source_well: srcWell,
                target_well: tgtWell,
                classifier: classifier,
                source: currentSource
            })
        }, 120);

        var el = document.getElementById("ta-results");
        if (!el) return;

        if (r.error) {
            el.innerHTML = '<div class="alert alert-warning">' + r.error + '</div>';
            el.classList.remove("d-none");
            return;
        }

        // Brief
        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("ta-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        // Metrics
        document.getElementById("ta-best-method").textContent = (r.best_method || "").replace(/_/g, " ");
        document.getElementById("ta-best-acc").textContent = (r.best_accuracy * 100).toFixed(1) + "%";
        document.getElementById("ta-n-shifts").textContent = r.n_shifts;
        document.getElementById("ta-ft-size").textContent = r.n_finetune;

        // Plot
        if (r.plot) {
            document.getElementById("ta-plot-img").innerHTML = '<img src="data:image/png;base64,' + r.plot + '" class="img-fluid" style="max-height:400px">';
        }

        // Method table
        var tbody = document.getElementById("ta-table-body");
        tbody.innerHTML = "";
        var methods = r.results || {};
        Object.keys(methods).forEach(function(m) {
            var v = methods[m];
            var acc = v.accuracy || 0;
            var cls = acc >= 0.7 ? "success" : acc >= 0.4 ? "warning" : "danger";
            var details = "";
            if (v.rounds !== undefined) details = v.rounds + " rounds, " + (v.n_pseudo || 0) + " pseudo-labels";
            if (m === r.best_method) details += (details ? " — " : "") + "★ BEST";
            tbody.innerHTML += '<tr' + (m === r.best_method ? ' class="table-success"' : '') + '>' +
                '<td>' + m.replace(/_/g, " ") + '</td>' +
                '<td><span class="badge bg-' + cls + '">' + (acc * 100).toFixed(1) + '%</span></td>' +
                '<td>' + (v.f1 || 0).toFixed(3) + '</td>' +
                '<td class="text-muted small">' + details + '</td></tr>';
        });

        // Feature shifts
        var shifts = r.feature_shifts || [];
        if (shifts.length > 0) {
            var shiftBody = document.getElementById("ta-shifts-body");
            shiftBody.innerHTML = "";
            shifts.forEach(function(s) {
                var sc = s.severity === "HIGH" ? "danger" : "warning";
                shiftBody.innerHTML += '<tr><td>' + s.feature + '</td><td>' + s.cohens_d + '</td>' +
                    '<td><span class="badge bg-' + sc + '">' + s.severity + '</span></td></tr>';
            });
            document.getElementById("ta-shifts-section").classList.remove("d-none");
        }

        el.classList.remove("d-none");
        showToast("Transfer adapted: best=" + r.best_method + " (" + (r.best_accuracy * 100).toFixed(1) + "%)");
    } catch (err) {
        showToast("Transfer adapted error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Error Budget / Learning Curve ────────────────────
async function runErrorBudget() {
    showLoading("Computing learning curve and error budget...");
    try {
        var classifier = document.getElementById("classifier-select").value;

        var r = await api("/api/analysis/error-budget", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                classifier: classifier,
                source: currentSource
            })
        }, 120);

        var el = document.getElementById("eb-results");
        if (!el) return;

        // Brief
        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("eb-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '<br><em>' + sb.action + '</em></div>';

        // Metrics
        document.getElementById("eb-accuracy").textContent = (r.current_accuracy * 100).toFixed(1) + "%";
        document.getElementById("eb-gap").textContent = (r.train_test_gap * 100).toFixed(1) + "%";
        var diagColor = r.diagnosis === "IMPROVING" ? "text-success" : r.diagnosis === "PLATEAU" ? "text-warning" : "text-danger";
        document.getElementById("eb-diagnosis").className = "metric-value " + diagColor;
        document.getElementById("eb-diagnosis").textContent = r.diagnosis;
        document.getElementById("eb-samples-needed").textContent =
            r.samples_for_1pct_improvement > 0 ? "~" + r.samples_for_1pct_improvement : "N/A (plateau)";

        // Plot
        if (r.plot) {
            document.getElementById("eb-plot-img").innerHTML = '<img src="data:image/png;base64,' + r.plot + '" class="img-fluid" style="max-height:400px">';
        }

        // Curve table
        var tbody = document.getElementById("eb-table-body");
        tbody.innerHTML = "";
        (r.learning_curve || []).forEach(function(p) {
            tbody.innerHTML += '<tr><td>' + p.n_samples + '</td>' +
                '<td>' + (p.train_accuracy * 100).toFixed(1) + '%</td>' +
                '<td>' + (p.test_accuracy * 100).toFixed(1) + '%</td>' +
                '<td>±' + (p.test_std * 100).toFixed(1) + '%</td></tr>';
        });

        el.classList.remove("d-none");
        showToast("Error budget: " + r.diagnosis + " at " + (r.current_accuracy * 100).toFixed(1) + "%");
    } catch (err) {
        showToast("Error budget error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Stakeholder Decision Report ──────────────────────
async function runStakeholderReport() {
    showLoading("Generating comprehensive stakeholder decision report...");
    try {
        var classifier = document.getElementById("classifier-select").value;
        var r = await api("/api/report/stakeholder-decision", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, classifier: classifier, source: currentSource })
        }, 120);
        var el = document.getElementById("sdr-results");
        if (!el) return;

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("sdr-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '<br><em>' + sb.action + '</em></div>';

        var dColor = r.decision === "GO" ? "text-success" : r.decision.includes("CONDITIONAL") ? "text-warning" : "text-danger";
        document.getElementById("sdr-decision").className = "metric-value fs-5 " + dColor;
        document.getElementById("sdr-decision").textContent = r.decision;
        document.getElementById("sdr-score").textContent = r.score + "/100";
        document.getElementById("sdr-accuracy").textContent = (r.accuracy * 100).toFixed(1) + "%";
        var ei = r.economic_impact || {};
        document.getElementById("sdr-econ-risk").textContent = "$" + ((ei.total_economic_risk_usd || 0) / 1000).toFixed(0) + "K";
        document.getElementById("sdr-low-conf").textContent = (r.confidence_stats || {}).below_70pct || 0;
        document.getElementById("sdr-reviews").textContent = (r.feedback_summary || {}).total_reviews || 0;

        if (r.plot) {
            document.getElementById("sdr-plot-img").innerHTML = '<img src="data:image/png;base64,' + r.plot + '" class="img-fluid">';
        }

        // Class risks
        var crBody = document.getElementById("sdr-class-risk-body");
        crBody.innerHTML = "";
        (r.class_risks || []).forEach(function(cr) {
            var vc = cr.verdict === "HIGH" ? "danger" : cr.verdict === "MEDIUM" ? "warning" : "success";
            crBody.innerHTML += '<tr><td>' + cr.class + '</td><td>' + (cr.recall * 100).toFixed(1) + '%</td>' +
                '<td>' + cr.risk_score + '</td><td><span class="badge bg-' + vc + '">' + cr.verdict + '</span></td></tr>';
        });

        // Evidence
        var evBody = document.getElementById("sdr-evidence-body");
        evBody.innerHTML = "";
        (r.evidence || []).forEach(function(ev) {
            var sc = ev.severity === "POSITIVE" ? "success" : ev.severity === "HIGH" ? "danger" : ev.severity === "MEDIUM" ? "warning" : "secondary";
            evBody.innerHTML += '<tr><td>' + ev.factor + '</td><td>' + ev.impact + '</td>' +
                '<td><span class="badge bg-' + sc + '">' + ev.severity + '</span></td></tr>';
        });

        el.classList.remove("d-none");
        showToast("Decision: " + r.decision + " (score " + r.score + "/100)");
    } catch (err) {
        showToast("Stakeholder report error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Model Arena ──────────────────────────────────────
async function runModelArena() {
    showLoading("Running model arena — testing all classifiers (this may take a minute)...");
    try {
        var r = await api("/api/analysis/model-arena", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, source: currentSource })
        }, 180);
        var el = document.getElementById("arena-results");
        if (!el) return;

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("arena-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        document.getElementById("arena-best").textContent = (r.best_model || "").replace(/_/g, " ");
        var br = (r.results || {})[r.best_model] || {};
        document.getElementById("arena-best-acc").textContent = ((br.accuracy || 0) * 100).toFixed(1) + "%";
        document.getElementById("arena-n-models").textContent = r.n_models;
        document.getElementById("arena-composite").textContent = (r.best_composite || 0).toFixed(3);

        if (r.plot) {
            document.getElementById("arena-plot-img").innerHTML = '<img src="data:image/png;base64,' + r.plot + '" class="img-fluid">';
        }

        var tbody = document.getElementById("arena-table-body");
        tbody.innerHTML = "";
        (r.ranking || []).forEach(function(name) {
            var m = r.results[name];
            var isBest = name === r.best_model;
            tbody.innerHTML += '<tr' + (isBest ? ' class="table-success"' : '') + '>' +
                '<td>' + m.rank + '</td><td>' + name.replace(/_/g, " ") + (isBest ? ' ★' : '') + '</td>' +
                '<td>' + (m.accuracy * 100).toFixed(1) + '%</td>' +
                '<td>' + (m.f1 || 0).toFixed(3) + '</td>' +
                '<td>' + ((m.balanced_accuracy || 0) * 100).toFixed(1) + '%</td>' +
                '<td>' + (m.ece || 0).toFixed(3) + '</td>' +
                '<td>' + (m.speed_seconds || 0).toFixed(1) + 's</td>' +
                '<td><strong>' + (m.composite || 0).toFixed(3) + '</strong></td></tr>';
        });

        el.classList.remove("d-none");
        showToast("Arena: " + (r.best_model || "").replace(/_/g, " ") + " wins!");
    } catch (err) {
        showToast("Model arena error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Auto-Retrain from Feedback ───────────────────────
async function runAutoRetrain() {
    showLoading("Auto-retraining model from accumulated feedback...");
    try {
        var classifier = document.getElementById("classifier-select").value;
        var r = await api("/api/analysis/auto-retrain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, classifier: classifier, source: currentSource })
        }, 120);
        var el = document.getElementById("retrain-results");
        if (!el) return;

        if (r.status === "NO_FEEDBACK") {
            el.innerHTML = '<div class="alert alert-info"><i class="bi bi-info-circle"></i> ' + r.message + '</div>';
            el.classList.remove("d-none");
            return;
        }

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("retrain-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        var sc = r.status === "PROMOTED" ? "text-success" : "text-danger";
        document.getElementById("retrain-status").className = "metric-value " + sc;
        document.getElementById("retrain-status").textContent = r.status;
        document.getElementById("retrain-improvement").textContent = (r.improvement >= 0 ? "+" : "") + (r.improvement * 100).toFixed(1) + "%";
        var fu = r.feedback_used || {};
        document.getElementById("retrain-corrections").textContent = fu.corrections || 0;
        document.getElementById("retrain-failures").textContent = fu.failures || 0;

        if (r.plot) {
            document.getElementById("retrain-plot-img").innerHTML = '<img src="data:image/png;base64,' + r.plot + '" class="img-fluid" style="max-height:400px">';
        }

        el.classList.remove("d-none");
        showToast("Auto-retrain: " + r.status + " (" + (r.improvement >= 0 ? "+" : "") + (r.improvement * 100).toFixed(1) + "%)");
    } catch (err) {
        showToast("Auto-retrain error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Negative Outcome Learning ────────────────────────
async function runNegativeOutcomes() {
    showLoading("Analyzing failure patterns and learning from negative outcomes...");
    try {
        var classifier = document.getElementById("classifier-select").value;
        var r = await api("/api/analysis/negative-outcomes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, classifier: classifier, source: currentSource })
        }, 120);
        var el = document.getElementById("negout-results");
        if (!el) return;

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("negout-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '<br><em>' + sb.action + '</em></div>';

        document.getElementById("negout-errors").textContent = r.n_errors;
        document.getElementById("negout-biases").textContent = (r.systematic_biases || []).length;
        document.getElementById("negout-synthetic").textContent = r.n_synthetic_added;
        var impColor = r.improvement >= 0 ? "text-success" : "text-danger";
        document.getElementById("negout-improvement").className = "metric-value " + impColor;
        document.getElementById("negout-improvement").textContent = (r.improvement >= 0 ? "+" : "") + (r.improvement * 100).toFixed(1) + "%";

        if (r.plot) {
            document.getElementById("negout-plot-img").innerHTML = '<img src="data:image/png;base64,' + r.plot + '" class="img-fluid">';
        }

        var biases = r.systematic_biases || [];
        if (biases.length > 0) {
            var bBody = document.getElementById("negout-bias-body");
            bBody.innerHTML = "";
            biases.forEach(function(b) {
                bBody.innerHTML += '<tr><td>' + b.true_class + '</td><td>' + (b.error_rate * 100).toFixed(1) + '%</td>' +
                    '<td>' + b.confused_with + '</td><td>' + b.confusion_count + '</td></tr>';
            });
            document.getElementById("negout-biases-section").classList.remove("d-none");
        }

        el.classList.remove("d-none");
        showToast("Negative learning: " + biases.length + " biases, " + (r.improvement >= 0 ? "+" : "") + (r.improvement * 100).toFixed(1) + "%");
    } catch (err) {
        showToast("Negative outcome error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Data Validation ──────────────────────────────────
async function runDataValidation() {
    showLoading("Validating data quality...");
    try {
        var r = await api("/api/data/validate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, source: currentSource })
        });
        var el = document.getElementById("dv-results");
        if (!el) return;

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("dv-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.action + '</div>';

        var qColor = r.quality === "GOOD" ? "text-success" : r.quality === "ACCEPTABLE" ? "text-warning" : "text-danger";
        document.getElementById("dv-quality").className = "metric-value " + qColor;
        document.getElementById("dv-quality").textContent = r.quality;
        document.getElementById("dv-critical").textContent = r.n_critical;
        document.getElementById("dv-warnings").textContent = r.n_warnings;
        document.getElementById("dv-samples").textContent = r.n_samples;

        var tbody = document.getElementById("dv-issues-body");
        tbody.innerHTML = "";
        (r.issues || []).forEach(function(i) {
            var sc = i.severity === "CRITICAL" ? "danger" : i.severity === "WARNING" ? "warning" : "secondary";
            tbody.innerHTML += '<tr><td><span class="badge bg-' + sc + '">' + i.severity + '</span></td>' +
                '<td>' + i.field + '</td><td>' + i.detail + '</td></tr>';
        });
        (r.recommendations || []).forEach(function(rec) {
            tbody.innerHTML += '<tr class="table-info"><td><span class="badge bg-info">REC</span></td><td colspan="2">' + rec + '</td></tr>';
        });

        el.classList.remove("d-none");
        showToast("Data quality: " + r.quality + " (" + r.n_critical + " critical, " + r.n_warnings + " warnings)");
    } catch (err) {
        showToast("Validation error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Cache Warmup ─────────────────────────────────────
async function runCacheWarmup() {
    showLoading("Warming caches for faster response...");
    try {
        var r = await api("/api/system/warmup", { method: "POST" });
        showToast("Cache warming started: " + (r.targets || []).length + " targets. Responses will be faster.");
    } catch (err) {
        showToast("Warmup error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── RLHF Preference Model ───────────────────────────
async function runPreferenceModel() {
    showLoading("Building RLHF preference model from expert reviews...");
    try {
        var r = await api("/api/rlhf/preference-model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, source: currentSource })
        }, 120);
        var el = document.getElementById("pref-results");
        if (!el) return;

        if (r.status === "INSUFFICIENT_DATA") {
            el.innerHTML = '<div class="alert alert-info"><i class="bi bi-info-circle"></i> ' + r.message + '</div>';
            el.classList.remove("d-none");
            return;
        }

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("pref-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        document.getElementById("pref-reviews").textContent = r.n_reviews;
        document.getElementById("pref-accepted").textContent = r.accepted;
        document.getElementById("pref-rejected").textContent = r.rejected;
        var impColor = r.improvement >= 0 ? "text-success" : "text-danger";
        document.getElementById("pref-improvement").className = "metric-value " + impColor;
        document.getElementById("pref-improvement").textContent = (r.improvement >= 0 ? "+" : "") + (r.improvement * 100).toFixed(1) + "%";

        if (r.plot) {
            document.getElementById("pref-plot-img").innerHTML = '<img src="data:image/png;base64,' + r.plot + '" class="img-fluid">';
        }

        var tt = r.type_trust || {};
        if (Object.keys(tt).length > 0) {
            var tbody = document.getElementById("pref-trust-body");
            tbody.innerHTML = "";
            Object.keys(tt).forEach(function(t) {
                var v = tt[t];
                var tc = v.trust_score >= 0.7 ? "success" : v.trust_score >= 0.4 ? "warning" : "danger";
                tbody.innerHTML += '<tr><td>' + t + '</td><td>' + v.accepted + '</td><td>' + v.rejected + '</td>' +
                    '<td><span class="badge bg-' + tc + '">' + (v.trust_score * 100).toFixed(0) + '%</span></td></tr>';
            });
            document.getElementById("pref-trust-section").classList.remove("d-none");
        }

        el.classList.remove("d-none");
        showToast("Preference model: " + r.n_reviews + " reviews, " + (r.improvement >= 0 ? "+" : "") + (r.improvement * 100).toFixed(1) + "%");
    } catch (err) {
        showToast("Preference model error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Pore Pressure Coupling ─────────────────────────

async function runPorePressureCoupling() {
    showLoading("Running pore pressure coupling analysis...");
    try {
        var r = await api("/api/analysis/pore-pressure-coupling", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, source: currentSource })
        }, 120);
        var el = document.getElementById("pp-results");
        if (!el) return;

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("pp-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        var cur = r.current_estimate || {};
        document.getElementById("pp-current-cs").textContent = (cur.cs_pct || 0).toFixed(1) + "%";
        var fGrade = cur.fif_grade || "?";
        var fColor = fGrade === "STABLE" ? "text-success" : fGrade === "MARGINAL" ? "text-warning" : "text-danger";
        document.getElementById("pp-fif-grade").className = "metric-value " + fColor;
        document.getElementById("pp-fif-grade").textContent = fGrade;
        document.getElementById("pp-sensitivity").textContent = (r.sensitivity_cs_per_mpa || 0).toFixed(1) + "%/MPa";
        document.getElementById("pp-sv").textContent = r.sv_mpa;

        var tbody = document.getElementById("pp-table-body");
        tbody.innerHTML = "";
        (r.pp_sweep || []).forEach(function(row) {
            var gc = row.fif_grade === "STABLE" ? "success" : row.fif_grade === "MARGINAL" ? "warning" : "danger";
            tbody.innerHTML += '<tr><td>' + row.pp_fraction_sv + '</td>' +
                '<td>' + row.pp_mpa + '</td>' +
                '<td>' + row.cs_pct + '%</td>' +
                '<td>' + row.fif + '</td>' +
                '<td><span class="badge bg-' + gc + '">' + row.fif_grade + '</span></td>' +
                '<td>' + row.s1_eff_mpa + '</td>' +
                '<td>' + row.s3_eff_mpa + '</td></tr>';
        });

        if (r.plot) {
            document.getElementById("pp-plot-img").innerHTML = '<img src="' + r.plot + '" class="img-fluid">';
        }

        el.classList.remove("d-none");
        showToast("Pp coupling: CS% = " + (cur.cs_pct || 0).toFixed(1) + "%, FIF = " + fGrade);
    } catch (err) {
        showToast("Pp coupling error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Heterogeneous Ensemble ────────────────────────

async function runHeteroEnsemble() {
    showLoading("Building heterogeneous ensemble (5 model families)...");
    try {
        var r = await api("/api/analysis/hetero-ensemble", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, source: currentSource })
        }, 180);
        var el = document.getElementById("he-results");
        if (!el) return;

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("he-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        document.getElementById("he-acc").textContent = (r.ensemble_accuracy * 100).toFixed(1) + "%";
        var impColor = r.ensemble_improvement >= 0 ? "text-success" : "text-danger";
        document.getElementById("he-improvement").className = "metric-value " + impColor;
        document.getElementById("he-improvement").textContent = (r.ensemble_improvement >= 0 ? "+" : "") + (r.ensemble_improvement * 100).toFixed(1) + "%";
        document.getElementById("he-agreement").textContent = (r.mean_agreement * 100).toFixed(0) + "%";
        document.getElementById("he-contested").textContent = r.contested_predictions;

        var tbody = document.getElementById("he-models-body");
        tbody.innerHTML = "";
        var ba = r.base_accuracies || {};
        var mc = r.meta_contributions || {};
        Object.keys(ba).forEach(function(name) {
            tbody.innerHTML += '<tr><td>' + name + '</td>' +
                '<td>' + (ba[name] * 100).toFixed(1) + '%</td>' +
                '<td>' + ((mc[name] || 0) * 100).toFixed(0) + '%</td></tr>';
        });
        // Add ensemble row
        tbody.innerHTML += '<tr class="table-success fw-bold"><td>ENSEMBLE</td>' +
            '<td>' + (r.ensemble_accuracy * 100).toFixed(1) + '%</td><td>-</td></tr>';

        if (r.plot) {
            document.getElementById("he-plot-img").innerHTML = '<img src="' + r.plot + '" class="img-fluid">';
        }

        el.classList.remove("d-none");
        showToast("Hetero-ensemble: " + (r.ensemble_accuracy * 100).toFixed(1) + "% accuracy");
    } catch (err) {
        showToast("Hetero-ensemble error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── ML Anomaly Detection ──────────────────────────

async function runMLAnomalyDetection() {
    showLoading("Running ML-based anomaly detection...");
    try {
        var r = await api("/api/analysis/anomaly-detection", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, source: currentSource })
        }, 120);
        var el = document.getElementById("anomaly-results");
        if (!el) return;

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("anom-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        document.getElementById("anom-count").textContent = r.n_anomalies;
        document.getElementById("anom-rate").textContent = r.anomaly_rate_pct + "%";
        document.getElementById("anom-total").textContent = r.n_samples;
        document.getElementById("anom-threshold").textContent = r.maha_threshold;

        var tbody = document.getElementById("anom-table-body");
        tbody.innerHTML = "";
        (r.anomalies || []).forEach(function(a) {
            var uf = (a.unusual_features || []).map(function(f) { return f.feature + " (z=" + f.z_score + ")"; }).join(", ");
            tbody.innerHTML += '<tr><td>' + a.index + '</td>' +
                '<td>' + (a.depth ? a.depth.toFixed(1) : "-") + '</td>' +
                '<td>' + (a.azimuth ? a.azimuth.toFixed(0) : "-") + '</td>' +
                '<td>' + (a.dip ? a.dip.toFixed(0) : "-") + '</td>' +
                '<td>' + (a.fracture_type || "-") + '</td>' +
                '<td>' + a.iso_score + '</td>' +
                '<td>' + a.mahalanobis + '</td>' +
                '<td class="small">' + uf + '</td></tr>';
        });

        if (r.plot) {
            document.getElementById("anom-plot-img").innerHTML = '<img src="' + r.plot + '" class="img-fluid">';
        }

        el.classList.remove("d-none");
        showToast("Anomaly detection: " + r.n_anomalies + " flagged (" + r.anomaly_rate_pct + "%)");
    } catch (err) {
        showToast("Anomaly detection error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Balanced Classification (SMOTE) ───────────────

async function runBalancedClassify() {
    showLoading("Running balanced classification with SMOTE oversampling...");
    try {
        var r = await api("/api/analysis/balanced-classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, source: currentSource })
        }, 120);
        var el = document.getElementById("bc-results");
        if (!el) return;

        // Brief
        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("bc-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        // Metrics
        var bestMethod = r.best_method || "smote";
        var bestBal = r.methods[bestMethod] ? (r.methods[bestMethod].balanced_accuracy * 100).toFixed(1) + "%" : "--";
        document.getElementById("bc-best-method").textContent = bestMethod.replace(/_/g, " ");
        document.getElementById("bc-best-balacc").textContent = bestBal;
        document.getElementById("bc-samples").textContent = r.n_samples;
        document.getElementById("bc-has-smote").textContent = r.has_smote ? "Yes" : "No";

        // Methods table
        var tbody = document.getElementById("bc-methods-body");
        tbody.innerHTML = "";
        Object.keys(r.methods).forEach(function(m) {
            var v = r.methods[m];
            var isBest = m === bestMethod;
            tbody.innerHTML += '<tr' + (isBest ? ' class="table-success"' : '') + '>' +
                '<td>' + m.replace(/_/g, " ") + (isBest ? ' <i class="bi bi-star-fill text-warning"></i>' : '') + '</td>' +
                '<td>' + (v.accuracy * 100).toFixed(1) + '%</td>' +
                '<td>' + (v.balanced_accuracy * 100).toFixed(1) + '%</td>' +
                '<td>' + (v.f1 * 100).toFixed(1) + '%</td></tr>';
        });

        // Minority improvements
        var mtbody = document.getElementById("bc-minority-body");
        mtbody.innerHTML = "";
        (r.minority_class_improvements || []).forEach(function(mi) {
            var impColor = mi.improvement > 0 ? "text-success" : "text-muted";
            mtbody.innerHTML += '<tr><td>' + mi["class"] + '</td><td>' + mi.count + '</td>' +
                '<td>' + (mi.baseline_recall * 100).toFixed(1) + '%</td>' +
                '<td>' + (mi.best_recall * 100).toFixed(1) + '%</td>' +
                '<td class="' + impColor + '">' + (mi.improvement > 0 ? "+" : "") + (mi.improvement * 100).toFixed(1) + '%</td></tr>';
        });

        // Plot
        if (r.plot) {
            document.getElementById("bc-plot-img").innerHTML = '<img src="' + r.plot + '" class="img-fluid">';
        }

        el.classList.remove("d-none");
        showToast("Balanced classify: best method=" + bestMethod + ", bal.acc=" + bestBal);
    } catch (err) {
        showToast("Balanced classify error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Industrial Readiness Scorecard ────────────────

async function runReadinessScorecard() {
    showLoading("Generating industrial readiness scorecard...");
    try {
        var r = await api("/api/report/readiness-scorecard", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, source: currentSource })
        }, 120);
        var el = document.getElementById("rs-results");
        if (!el) return;

        // Brief
        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("rs-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        // Metrics
        var rColor = r.readiness === "PRODUCTION" ? "text-success" : r.readiness === "PILOT" ? "text-info" :
                     r.readiness === "DEVELOPMENT" ? "text-warning" : "text-danger";
        document.getElementById("rs-readiness").className = "metric-value fs-5 " + rColor;
        document.getElementById("rs-readiness").textContent = r.readiness;
        document.getElementById("rs-score").textContent = r.overall_score + "/100";
        document.getElementById("rs-samples").textContent = r.n_samples;
        document.getElementById("rs-wells").textContent = r.n_wells;

        // Dimensions table
        var tbody = document.getElementById("rs-dimensions-body");
        tbody.innerHTML = "";
        (r.dimensions || []).forEach(function(d) {
            var gc = d.grade === "A" ? "success" : d.grade === "B" ? "info" : d.grade === "C" ? "warning" :
                     d.grade === "D" ? "secondary" : "danger";
            tbody.innerHTML += '<tr><td><strong>' + d.dimension + '</strong></td>' +
                '<td><span class="badge bg-' + gc + '">' + d.grade + '</span></td>' +
                '<td>' + (d.score * 100).toFixed(0) + '%</td>' +
                '<td class="small">' + d.detail + '</td>' +
                '<td class="small text-muted">' + d.action + '</td>' +
                '<td>' + d.weight + '%</td></tr>';
        });

        // Priority actions
        if (r.priority_actions && r.priority_actions.length > 0) {
            var pList = document.getElementById("rs-priority-list");
            pList.innerHTML = "";
            r.priority_actions.forEach(function(a) {
                pList.innerHTML += '<li class="text-danger">' + a + '</li>';
            });
            document.getElementById("rs-priority-section").classList.remove("d-none");
        }

        // Plot
        if (r.plot) {
            document.getElementById("rs-plot-img").innerHTML = '<img src="' + r.plot + '" class="img-fluid">';
        }

        el.classList.remove("d-none");
        showToast("Readiness: " + r.readiness + " (" + r.overall_score + "/100)");
    } catch (err) {
        showToast("Readiness scorecard error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Feature Ablation Study ─────────────────────────

async function runFeatureAblation() {
    showLoading("Running feature ablation study...");
    try {
        var clf = document.getElementById("classifier-select").value;
        var r = await api("/api/analysis/feature-ablation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, source: currentSource, classifier: clf })
        }, 120);
        var el = document.getElementById("fa-results");
        if (!el) return;

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("fa-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        document.getElementById("fa-most-important").textContent = r.most_important_group || "--";
        document.getElementById("fa-baseline").textContent = (r.baseline_accuracy * 100).toFixed(1) + "%";
        document.getElementById("fa-n-groups").textContent = r.n_groups;
        document.getElementById("fa-n-features").textContent = r.n_features_total;

        var tbody = document.getElementById("fa-table-body");
        tbody.innerHTML = "";
        (r.ablation_results || []).forEach(function(ar) {
            var dropColor = ar.accuracy_drop > 0.05 ? "text-danger fw-bold" : ar.accuracy_drop > 0.01 ? "text-warning" : "text-success";
            tbody.innerHTML += '<tr><td>' + ar.importance_rank + '</td>' +
                '<td>' + ar.group + '</td>' +
                '<td>' + ar.n_features_removed + '</td>' +
                '<td>' + (ar.accuracy_without * 100).toFixed(1) + '%</td>' +
                '<td class="' + dropColor + '">' + (ar.accuracy_drop > 0 ? "-" : "") + (ar.accuracy_drop * 100).toFixed(1) + '%</td></tr>';
        });

        if (r.plot) {
            document.getElementById("fa-plot-img").innerHTML = '<img src="' + r.plot + '" class="img-fluid">';
        }

        el.classList.remove("d-none");
        showToast("Feature ablation: most important = " + r.most_important_group);
    } catch (err) {
        showToast("Feature ablation error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Hyperparameter Optimization ───────────────────

async function runOptimizeModel() {
    var nIter = parseInt(document.getElementById("opt-n-iter").value) || 20;
    showLoading("Optimizing hyperparameters (" + nIter + " iterations)...");
    try {
        var clf = document.getElementById("classifier-select").value;
        var r = await api("/api/analysis/optimize-model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ well: currentWell, source: currentSource, classifier: clf, n_iter: nIter })
        }, 300);
        var el = document.getElementById("opt-results");
        if (!el) return;

        var sb = r.stakeholder_brief || {};
        var bColor = sb.risk_level === "GREEN" ? "success" : sb.risk_level === "AMBER" ? "warning" : "danger";
        document.getElementById("opt-brief").innerHTML =
            '<div class="alert alert-' + bColor + ' py-1 small mb-0"><strong>' + sb.headline + '</strong><br>' + sb.confidence_sentence + '</div>';

        document.getElementById("opt-default").textContent = (r.default_accuracy * 100).toFixed(1) + "%";
        document.getElementById("opt-best").textContent = (r.best_accuracy * 100).toFixed(1) + "%";
        var impColor = r.improvement >= 0 ? "text-success" : "text-danger";
        document.getElementById("opt-improvement").className = "metric-value " + impColor;
        document.getElementById("opt-improvement").textContent = (r.improvement >= 0 ? "+" : "") + (r.improvement * 100).toFixed(1) + "%";
        document.getElementById("opt-n-iter-val").textContent = r.n_iterations;

        document.getElementById("opt-best-params").textContent = JSON.stringify(r.best_params, null, 2);

        var tbody = document.getElementById("opt-table-body");
        tbody.innerHTML = "";
        (r.top_configurations || []).forEach(function(cfg) {
            tbody.innerHTML += '<tr><td>' + cfg.rank + '</td>' +
                '<td>' + (cfg.mean_score * 100).toFixed(1) + '%</td>' +
                '<td>±' + (cfg.std_score * 100).toFixed(1) + '%</td>' +
                '<td class="small">' + JSON.stringify(cfg.params) + '</td></tr>';
        });

        if (r.plot) {
            document.getElementById("opt-plot-img").innerHTML = '<img src="' + r.plot + '" class="img-fluid">';
        }

        el.classList.remove("d-none");
        showToast("Optimization: " + (r.improvement >= 0 ? "+" : "") + (r.improvement * 100).toFixed(1) + "% improvement");
    } catch (err) {
        showToast("Optimization error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Clustering ────────────────────────────────────

async function runClustering() {
    showLoading("Running fracture clustering...");
    try {
        var nSel = document.getElementById("n-clusters-select").value;
        var body = {
            well: currentWell,
            n_clusters: nSel === "auto" ? null : parseInt(nSel),
            source: currentSource
        };

        var r = await api("/api/analysis/cluster", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });

        document.getElementById("cluster-results").classList.remove("d-none");
        val("n-clusters-val", r.n_clusters);
        setImg("cluster-img", r.cluster_img);

        var tbody = document.querySelector("#cluster-stats-table tbody");
        clearChildren(tbody);
        r.cluster_stats.forEach(function(s, i) {
            var tr = document.createElement("tr");
            var tdSet = document.createElement("td");
            var strong = document.createElement("strong");
            strong.textContent = "Set " + i;
            tdSet.appendChild(strong);
            tr.appendChild(tdSet);
            tr.appendChild(createCell("td", (s.mean_azimuth != null ? s.mean_azimuth.toFixed(0) : "--") + "\u00b0"));
            tr.appendChild(createCell("td", (s.mean_dip != null ? s.mean_dip.toFixed(0) : "--") + "\u00b0"));
            tr.appendChild(createCell("td", s.count != null ? String(s.count) : "--"));
            tbody.appendChild(tr);
        });

        showToast("Found " + r.n_clusters + " fracture sets in Well " + r.well);
    } catch (err) {
        showToast("Clustering error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Feedback (NEW) ────────────────────────────────

// Star rating buttons
document.querySelectorAll(".fb-star").forEach(function(btn) {
    btn.addEventListener("click", function() {
        feedbackRating = parseInt(btn.dataset.val);
        document.querySelectorAll(".fb-star").forEach(function(b) {
            if (parseInt(b.dataset.val) <= feedbackRating) {
                b.classList.remove("btn-outline-warning");
                b.classList.add("btn-warning");
            } else {
                b.classList.remove("btn-warning");
                b.classList.add("btn-outline-warning");
            }
        });
    });
});

async function submitFeedback() {
    try {
        var body = {
            well: currentWell,
            analysis_type: document.getElementById("fb-type").value,
            rating: feedbackRating,
            comment: document.getElementById("fb-comment").value,
            expert_name: document.getElementById("fb-name").value || "anonymous"
        };

        var fbResult = await api("/api/feedback/submit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });

        showToast("Feedback submitted. Thank you!");
        document.getElementById("fb-comment").value = "";

        // Show feedback receipt (visible confirmation)
        var receiptEl = document.getElementById("feedback-receipt");
        var receiptBody = document.getElementById("feedback-receipt-body");
        if (receiptEl && receiptBody && fbResult.feedback_receipt) {
            var fr = fbResult.feedback_receipt;
            receiptBody.innerHTML =
                '<p class="small mb-1">' + fr.what_happens_next + '</p>' +
                '<p class="small mb-0 text-muted">Overall ratings: ' +
                fr.current_average_rating + '/5 avg across ' + fr.n_ratings_total + ' submission(s).</p>';
            receiptEl.classList.remove("d-none");
        }

        // Refresh summary
        loadFeedbackSummary();
    } catch (err) {
        showToast("Feedback error: " + err.message, "Error");
    }
}

async function loadFeedbackSummary() {
    try {
        var r = await api("/api/feedback/summary");
        var body = document.getElementById("fb-summary-body");
        clearChildren(body);

        if (r.total_feedback === 0) {
            var p = document.createElement("p");
            p.className = "text-muted";
            p.textContent = "No feedback collected yet.";
            body.appendChild(p);
            return;
        }

        var statsDiv = document.createElement("div");
        statsDiv.className = "row g-2 mb-3";

        var col1 = document.createElement("div");
        col1.className = "col-6";
        var card1 = document.createElement("div");
        card1.className = "text-center p-2 bg-light rounded";
        var label1 = document.createElement("div");
        label1.className = "small text-muted";
        label1.textContent = "Total Feedback";
        card1.appendChild(label1);
        var val1 = document.createElement("div");
        val1.className = "fw-bold fs-5";
        val1.textContent = r.total_feedback;
        card1.appendChild(val1);
        col1.appendChild(card1);
        statsDiv.appendChild(col1);

        var col2 = document.createElement("div");
        col2.className = "col-6";
        var card2 = document.createElement("div");
        card2.className = "text-center p-2 bg-light rounded";
        var label2 = document.createElement("div");
        label2.className = "small text-muted";
        label2.textContent = "Avg Rating";
        card2.appendChild(label2);
        var val2 = document.createElement("div");
        val2.className = "fw-bold fs-5";
        val2.textContent = r.avg_rating ? r.avg_rating + " / 5" : "--";
        card2.appendChild(val2);
        col2.appendChild(card2);
        statsDiv.appendChild(col2);

        body.appendChild(statsDiv);

        // By type
        if (r.feedback_by_type && Object.keys(r.feedback_by_type).length > 0) {
            var typeTitle = document.createElement("h6");
            typeTitle.className = "small";
            typeTitle.textContent = "By Analysis Type:";
            body.appendChild(typeTitle);

            Object.keys(r.feedback_by_type).forEach(function(type) {
                var info = r.feedback_by_type[type];
                var row = document.createElement("div");
                row.className = "d-flex justify-content-between small";
                var nameSpan = document.createElement("span");
                nameSpan.textContent = type;
                row.appendChild(nameSpan);
                var valSpan = document.createElement("span");
                valSpan.textContent = info.avg_rating + "/5 (" + info.count + " reviews)";
                row.appendChild(valSpan);
                body.appendChild(row);
            });
        }

        // Label corrections count
        if (r.label_corrections > 0) {
            correctionCount = r.label_corrections;
            val("corr-count", correctionCount + " correction" + (correctionCount !== 1 ? "s" : "") + " pending");
            document.getElementById("btn-retrain").disabled = false;

            // Correction patterns
            if (r.correction_patterns && Object.keys(r.correction_patterns).length > 0) {
                var cpTitle = document.createElement("h6");
                cpTitle.className = "small mt-3";
                cpTitle.textContent = "Correction Patterns:";
                body.appendChild(cpTitle);
                Object.keys(r.correction_patterns).forEach(function(pattern) {
                    var pLine = document.createElement("div");
                    pLine.className = "d-flex justify-content-between small text-danger";
                    var pName = document.createElement("span");
                    pName.textContent = pattern;
                    pLine.appendChild(pName);
                    var pVal = document.createElement("span");
                    pVal.textContent = r.correction_patterns[pattern] + "x";
                    pLine.appendChild(pVal);
                    body.appendChild(pLine);
                });
            }
        }

        // Actionable insights
        var insightsSection = document.getElementById("fb-insights-section");
        var insightsBody = document.getElementById("fb-insights-body");
        clearChildren(insightsBody);
        if (r.actionable_insights && r.actionable_insights.length > 0) {
            insightsSection.classList.remove("d-none");
            r.actionable_insights.forEach(function(insight) {
                var div = document.createElement("div");
                var typeMap = { critical: "alert-danger", warning: "alert-warning", info: "alert-info" };
                div.className = "alert py-2 mb-2 small " + (typeMap[insight.type] || "alert-info");
                div.textContent = insight.message;
                insightsBody.appendChild(div);
            });
        } else {
            insightsSection.classList.add("d-none");
        }
    } catch (err) {
        // Silently fail - feedback summary is not critical
    }
}

// ── Trust Score ──────────────────────────────────

async function runTrustScore() {
    showLoading("Computing trust score...");
    try {
        var r = await api("/api/feedback/trust-score", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, well: currentWell || null})
        });

        var body = document.getElementById("trust-score-body");

        // Color by trust level
        var levelColors = {HIGH: "success", MODERATE: "warning", LOW: "danger", "VERY LOW": "danger"};
        var levelColor = levelColors[r.trust_level] || "secondary";

        var html = '<div class="row g-3 mb-3">';
        // Main score
        html += '<div class="col-md-4"><div class="text-center p-3 rounded bg-' + levelColor + ' bg-opacity-10">';
        html += '<div class="display-4 fw-bold text-' + levelColor + '">' + r.trust_score + '</div>';
        html += '<div class="small text-muted">out of 100</div>';
        html += '<span class="badge bg-' + levelColor + ' mt-1">' + r.trust_level + ' TRUST</span>';
        html += '</div></div>';

        // Trust message + feedback status
        html += '<div class="col-md-8"><div class="alert alert-' + levelColor + ' py-2 mb-2 small">';
        html += '<i class="bi bi-info-circle"></i> ' + r.trust_message + '</div>';
        if (r.feedback_loop_active) {
            html += '<div class="small text-success"><i class="bi bi-check-circle"></i> Expert feedback loop ACTIVE (' + r.corrections_count + ' corrections applied)</div>';
        } else {
            html += '<div class="small text-muted"><i class="bi bi-dash-circle"></i> No expert feedback yet. Submit ratings below to activate the feedback loop.</div>';
        }
        html += '</div></div>';

        // Signal breakdown
        html += '<div class="col-12"><h6 class="small fw-bold mb-2">Trust Signal Breakdown</h6>';
        html += '<table class="table table-sm small"><thead><tr><th>Signal</th><th>Score</th><th>Weight</th><th>Detail</th></tr></thead><tbody>';
        var signals = r.signals || {};
        Object.keys(signals).forEach(function(key) {
            var s = signals[key];
            var barColor = s.score >= 70 ? "success" : s.score >= 40 ? "warning" : "danger";
            html += '<tr><td class="fw-semibold">' + key.replace(/_/g, ' ').replace(/\b\w/g, function(l) { return l.toUpperCase(); }) + '</td>';
            html += '<td><div class="d-flex align-items-center gap-2">';
            html += '<div class="progress flex-grow-1" style="height:6px;width:60px"><div class="progress-bar bg-' + barColor + '" style="width:' + s.score + '%"></div></div>';
            html += '<span>' + s.score + '</span></div></td>';
            html += '<td>' + (s.weight * 100).toFixed(0) + '%</td>';
            html += '<td class="text-muted">' + s.detail + '</td></tr>';
        });
        html += '</tbody></table></div>';

        // Improvements
        if (r.improvements && r.improvements.length > 0) {
            html += '<div class="col-12"><h6 class="small fw-bold mb-2"><i class="bi bi-arrow-up-circle"></i> How to Improve Trust</h6>';
            r.improvements.forEach(function(imp) {
                html += '<div class="alert alert-light py-2 mb-1 small border-start border-3 border-warning">';
                html += '<strong>' + imp.factor + '</strong> (score: ' + imp.current_score + '): ' + imp.action + '</div>';
            });
            html += '</div>';
        }

        html += '</div>';
        body.innerHTML = html;

        // Update card border based on trust level
        var card = document.getElementById("trust-score-card");
        card.className = "card mb-4 shadow-sm border-" + levelColor;

        showToast("Trust score: " + r.trust_score + " (" + r.trust_level + ")");
    } catch (err) {
        showToast("Trust score error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Label Correction & Retrain (NEW) ──────────────

var correctionCount = 0;

async function submitCorrection() {
    try {
        var body = {
            well: currentWell,
            fracture_idx: parseInt(document.getElementById("corr-idx").value),
            original_type: document.getElementById("corr-original").value,
            corrected_type: document.getElementById("corr-correct").value,
            expert_name: document.getElementById("fb-name").value || "anonymous"
        };

        var r = await api("/api/feedback/correct-label", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });

        correctionCount = r.total_corrections;
        val("corr-count", correctionCount + " correction" + (correctionCount !== 1 ? "s" : "") + " pending");
        document.getElementById("btn-retrain").disabled = false;
        showToast("Correction recorded: fracture #" + body.fracture_idx + " -> " + body.corrected_type);
        loadFeedbackSummary();
    } catch (err) {
        showToast("Correction error: " + err.message, "Error");
    }
}

async function retrainModel() {
    showLoading("Retraining model with expert corrections...");
    try {
        var r = await api("/api/feedback/retrain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource, classifier: "xgboost" })
        });

        var resultDiv = document.getElementById("retrain-result");
        resultDiv.classList.remove("d-none");
        clearChildren(resultDiv);

        if (r.status === "retrained") {
            var alert = document.createElement("div");
            alert.className = r.improvement > 0 ? "alert alert-success small" : "alert alert-info small";
            alert.textContent = r.message;
            resultDiv.appendChild(alert);

            var stats = document.createElement("div");
            stats.className = "small";
            stats.textContent = "Original: " + (r.original_accuracy * 100).toFixed(1) +
                "% -> Corrected: " + (r.corrected_accuracy * 100).toFixed(1) + "%";
            resultDiv.appendChild(stats);
        } else {
            var noCorr = document.createElement("div");
            noCorr.className = "alert alert-warning small";
            noCorr.textContent = r.message;
            resultDiv.appendChild(noCorr);
        }

        showToast("Retrain complete: " + r.message);
    } catch (err) {
        showToast("Retrain error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── SHAP Explainability (NEW) ─────────────────────

async function runShapExplanation() {
    showLoading("Computing SHAP explanations (analyzing feature contributions)...");
    try {
        var classifier = document.getElementById("shap-classifier-select").value;
        var r = await api("/api/analysis/shap", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource, classifier: classifier })
        });

        document.getElementById("shap-results").classList.remove("d-none");
        val("shap-method", r.has_shap ? "SHAP (TreeExplainer)" : "Feature Importance");
        val("shap-model", r.classifier ? r.classifier.replace("_", " ") : "--");
        val("shap-n-features", r.n_features);
        val("shap-n-samples", r.n_samples);

        // Global importance bars
        var container = document.getElementById("shap-global-container");
        clearChildren(container);
        if (r.top_features && r.top_features.length > 0) {
            var maxImp = r.top_features[0].importance || 1;
            r.top_features.forEach(function(feat) {
                var pct = (feat.importance / maxImp * 100).toFixed(0);

                var wrapper = document.createElement("div");
                wrapper.className = "feat-bar-container";

                var label = document.createElement("div");
                label.className = "feat-bar-label";
                label.textContent = "#" + feat.rank + " " + feat.feature;
                wrapper.appendChild(label);

                var desc = document.createElement("div");
                desc.className = "small text-muted mb-1";
                desc.textContent = feat.description;
                wrapper.appendChild(desc);

                var bg = document.createElement("div");
                bg.className = "feat-bar-bg";

                var fill = document.createElement("div");
                fill.className = "feat-bar-fill";
                fill.style.width = pct + "%";
                fill.style.backgroundColor = "#6366f1";
                bg.appendChild(fill);

                var valSpan = document.createElement("span");
                valSpan.className = "feat-bar-value";
                valSpan.textContent = feat.importance.toFixed(4);
                bg.appendChild(valSpan);

                wrapper.appendChild(bg);
                container.appendChild(wrapper);
            });
        }

        // Generate stakeholder explanation text
        var explainDiv = document.getElementById("shap-explanation-text");
        clearChildren(explainDiv);

        if (r.top_features && r.top_features.length >= 3) {
            var intro = document.createElement("p");
            intro.textContent = "The AI model makes fracture classification decisions based on these factors, ranked by actual impact:";
            explainDiv.appendChild(intro);

            var ol = document.createElement("ol");
            r.top_features.slice(0, 5).forEach(function(feat) {
                var li = document.createElement("li");
                li.className = "mb-2";
                var strong = document.createElement("strong");
                strong.textContent = feat.description;
                li.appendChild(strong);
                var detail = document.createElement("span");
                detail.textContent = " (importance: " + feat.importance.toFixed(4) + ")";
                li.appendChild(detail);
                ol.appendChild(li);
            });
            explainDiv.appendChild(ol);

            var note = document.createElement("div");
            note.className = "alert alert-info mt-3 small";
            var noteIcon = document.createElement("i");
            noteIcon.className = "bi bi-info-circle";
            note.appendChild(noteIcon);
            var noteText = document.createTextNode(
                " These importance values are based on " + (r.has_shap ? "SHAP Shapley values" : "model-native feature importance") +
                ". Higher values mean the feature has more influence on the classification decision. " +
                "If a critical feature seems wrong (e.g., depth shouldn't matter for your field), " +
                "submit feedback on the Feedback tab."
            );
            note.appendChild(noteText);
            explainDiv.appendChild(note);
        }

        // Per-class importance
        var classSection = document.getElementById("shap-class-section");
        var classContainer = document.getElementById("shap-class-container");
        clearChildren(classContainer);
        if (r.class_importance && Object.keys(r.class_importance).length > 0) {
            classSection.classList.remove("d-none");

            Object.keys(r.class_importance).forEach(function(className) {
                var feats = r.class_importance[className];
                var div = document.createElement("div");
                div.className = "mb-3 p-3 bg-light rounded";

                var title = document.createElement("h6");
                title.className = "mb-2";
                title.textContent = className + " fractures are driven by:";
                div.appendChild(title);

                feats.forEach(function(f) {
                    var line = document.createElement("div");
                    line.className = "d-flex justify-content-between small";
                    var nameSpan = document.createElement("span");
                    nameSpan.textContent = f.feature;
                    line.appendChild(nameSpan);
                    var valSpan = document.createElement("span");
                    valSpan.className = "text-muted";
                    valSpan.textContent = f.importance.toFixed(4);
                    line.appendChild(valSpan);
                    div.appendChild(line);
                });

                classContainer.appendChild(div);
            });
        } else {
            classSection.classList.add("d-none");
        }

        // Sample explanations
        var sampleSection = document.getElementById("shap-sample-section");
        var sampleContainer = document.getElementById("shap-sample-container");
        clearChildren(sampleContainer);
        if (r.sample_explanations && r.sample_explanations.length > 0) {
            sampleSection.classList.remove("d-none");

            r.sample_explanations.forEach(function(sample) {
                var card = document.createElement("div");
                card.className = "mb-2 p-2 border rounded d-flex align-items-start gap-3";

                var badge = document.createElement("span");
                badge.className = "badge bg-secondary";
                badge.textContent = "Fracture #" + sample.sample_index;
                card.appendChild(badge);

                var predBadge = document.createElement("span");
                predBadge.className = "badge bg-primary";
                predBadge.textContent = sample.predicted_class;
                card.appendChild(predBadge);

                var drivers = document.createElement("div");
                drivers.className = "small";
                sample.top_drivers.forEach(function(d) {
                    var line = document.createElement("span");
                    line.className = "me-3";
                    line.textContent = d.feature + ": " + d.shap_value.toFixed(4);
                    drivers.appendChild(line);
                });
                card.appendChild(drivers);

                sampleContainer.appendChild(card);
            });
        } else {
            sampleSection.classList.add("d-none");
        }

        showToast("SHAP explanations computed for " + r.n_samples + " samples using " + classifier.replace("_", " "));

        // Fetch SHAP visualization plots (server-rendered images)
        fetchShapPlots(classifier);
    } catch (err) {
        showToast("SHAP error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function fetchShapPlots(classifier) {
    try {
        var plotSection = document.getElementById("shap-plots-section");
        if (!plotSection) return;
        plotSection.classList.add("d-none");

        var r = await api("/api/shap/plots", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource, classifier: classifier })
        });

        plotSection.classList.remove("d-none");

        // Global importance plot
        var globalImg = document.getElementById("shap-global-plot-img");
        if (globalImg && r.global_importance_plot) {
            globalImg.innerHTML = '<img src="' + r.global_importance_plot + '" class="img-fluid rounded" alt="SHAP Global Importance" style="max-height:500px">';
        }

        // Waterfall plot
        var wfImg = document.getElementById("shap-waterfall-plot-img");
        if (wfImg && r.waterfall_plot) {
            var caption = '';
            if (r.waterfall_sample) {
                caption = '<p class="small text-muted mt-2 mb-0">Sample #' + r.waterfall_sample.index +
                    ' at depth ' + r.waterfall_sample.depth.toFixed(0) + 'm, predicted as <strong>' +
                    r.waterfall_sample.predicted_class + '</strong> (uncertainty: ' +
                    r.waterfall_sample.uncertainty.toFixed(3) + ')</p>';
            }
            wfImg.innerHTML = '<img src="' + r.waterfall_plot + '" class="img-fluid rounded" alt="SHAP Waterfall" style="max-height:400px">' + caption;
        }

        // Feature scatter plot
        var scatterImg = document.getElementById("shap-scatter-plot-img");
        if (scatterImg && r.feature_scatter_plot) {
            scatterImg.innerHTML = '<img src="' + r.feature_scatter_plot + '" class="img-fluid rounded" alt="Feature vs SHAP" style="max-height:400px">';
        }

        // Per-class plots
        var perClassDiv = document.getElementById("shap-per-class-plots");
        if (perClassDiv && r.per_class_plots) {
            perClassDiv.innerHTML = '';
            Object.keys(r.per_class_plots).forEach(function(cls) {
                var col = document.createElement("div");
                col.className = "col-lg-6 col-xl-4";
                col.innerHTML = '<div class="card"><div class="card-header small">' +
                    '<i class="bi bi-diagram-3"></i> ' + cls + '</div>' +
                    '<div class="card-body text-center p-2"><img src="' + r.per_class_plots[cls] +
                    '" class="img-fluid rounded" alt="SHAP ' + cls + '" style="max-height:300px"></div></div>';
                perClassDiv.appendChild(col);
            });
        }

        // Stakeholder brief
        if (r.stakeholder_brief) {
            renderStakeholderBrief('shap-plots-brief', r.stakeholder_brief, 'shap-explain-detail');
        }
    } catch (err) {
        console.warn("SHAP plots error:", err.message);
    }
}

// ── Near-Miss Safety Analysis ─────────────────────

async function runNearMissAnalysis() {
    showLoading("Analyzing near-misses and blind spots (API RP 580)...");
    try {
        var classifier = document.getElementById("nm-classifier-select").value;
        var r = await api("/api/analysis/near-misses", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource, classifier: classifier })
        });

        document.getElementById("nm-results").classList.remove("d-none");
        val("nm-count", r.n_near_misses);
        val("nm-blind-spots", r.n_blind_spots);
        val("nm-red-risk", r.n_red_risk);
        val("nm-accuracy", (r.overall_accuracy * 100).toFixed(1) + "%");

        // Stakeholder brief
        if (r.stakeholder_brief) {
            renderStakeholderBrief('nm-brief', r.stakeholder_brief, 'nm-detail');
        }

        // Plot
        var plotImg = document.getElementById("nm-plot-img");
        if (plotImg && r.plot) {
            plotImg.innerHTML = '<img src="' + r.plot + '" class="img-fluid rounded" alt="Near-Miss Risk Matrix" style="max-height:450px">';
        }

        // Near-miss table
        var tbody = document.getElementById("nm-table-body");
        tbody.innerHTML = '';
        (r.near_misses || []).slice(0, 30).forEach(function(nm) {
            var riskBadge = nm.risk_level === 'HIGH'
                ? '<span class="badge bg-danger">HIGH</span>'
                : '<span class="badge bg-warning text-dark">MED</span>';
            var row = document.createElement("tr");
            row.innerHTML = '<td>' + nm.index + '</td><td>' + nm.depth + 'm</td><td>' + nm.well + '</td>' +
                '<td>' + nm.true_class + '</td><td>' + nm.predicted_class + '</td>' +
                '<td>' + (nm.confidence * 100).toFixed(1) + '%</td>' +
                '<td><strong>' + (nm.margin * 100).toFixed(1) + '%</strong></td>' +
                '<td>' + nm.runner_up + ' (' + (nm.runner_up_prob * 100).toFixed(1) + '%)</td>' +
                '<td>' + riskBadge + '</td>';
            tbody.appendChild(row);
        });

        // Blind spots table
        var bsTbody = document.getElementById("nm-bs-table-body");
        bsTbody.innerHTML = '';
        (r.blind_spots || []).slice(0, 15).forEach(function(bs) {
            var sevBadge = bs.severity === 'HIGH'
                ? '<span class="badge bg-danger">HIGH</span>'
                : '<span class="badge bg-warning text-dark">MED</span>';
            var row = document.createElement("tr");
            row.innerHTML = '<td>' + bs.feature_label + '</td>' +
                '<td>[' + bs.range_low.toFixed(2) + ', ' + bs.range_high.toFixed(2) + ')</td>' +
                '<td><strong>' + (bs.error_rate * 100).toFixed(1) + '%</strong></td>' +
                '<td>' + (bs.baseline_error_rate * 100).toFixed(1) + '%</td>' +
                '<td>' + bs.n_samples + '</td>' +
                '<td>' + sevBadge + '</td>';
            bsTbody.appendChild(row);
        });

        showToast("Near-miss analysis: " + r.n_near_misses + " near-misses, " + r.n_blind_spots + " blind spots");
    } catch (err) {
        showToast("Near-miss error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Failure Dashboard (API RP 580) ───────────────────

async function runFailureDashboard() {
    showLoading("Computing industrial safety assessment (API RP 580/581)...");
    try {
        var classifier = document.getElementById("nm-classifier-select").value;
        var r = await api("/api/analysis/failure-dashboard", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource, classifier: classifier })
        });

        document.getElementById("safety-dashboard").classList.remove("d-none");

        var scoreEl = document.getElementById("safety-score");
        scoreEl.textContent = r.safety_score + "/100";
        scoreEl.className = "metric-value fs-2 " + (r.safety_score >= 80 ? "text-success" : r.safety_score >= 60 ? "text-warning" : "text-danger");

        var decEl = document.getElementById("safety-decision");
        decEl.textContent = r.decision;
        decEl.className = "metric-value " + (r.decision === "GO" ? "text-success" : r.decision === "NO-GO" ? "text-danger" : "text-warning");

        val("safety-factors", r.n_fail + " FAIL, " + r.n_warn + " WARN");

        if (r.stakeholder_brief) {
            renderStakeholderBrief('safety-brief', r.stakeholder_brief, 'safety-detail');
        }

        var plotImg = document.getElementById("safety-plot-img");
        if (plotImg && r.plot) {
            plotImg.innerHTML = '<img src="' + r.plot + '" class="img-fluid rounded" alt="Safety Dashboard" style="max-height:400px">';
        }

        var tbody = document.getElementById("safety-factors-body");
        tbody.innerHTML = '';
        (r.risk_factors || []).forEach(function(rf) {
            var statusBadge = rf.status === 'PASS' ? '<span class="badge bg-success">PASS</span>'
                : rf.status === 'WARN' ? '<span class="badge bg-warning text-dark">WARN</span>'
                : '<span class="badge bg-danger">FAIL</span>';
            var row = document.createElement("tr");
            row.innerHTML = '<td><strong>' + rf.factor + '</strong></td>' +
                '<td>' + rf.value + '</td>' +
                '<td>' + rf.threshold + '</td>' +
                '<td>' + rf.score + '</td>' +
                '<td>' + statusBadge + '</td>';
            tbody.appendChild(row);
        });

        showToast("Safety Score: " + r.safety_score + "/100 - " + r.decision);
    } catch (err) {
        showToast("Safety dashboard error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Feedback Effectiveness ────────────────────────

async function runFeedbackEffectiveness() {
    showLoading("Computing feedback effectiveness...");
    try {
        var r = await apiPost("/api/feedback/effectiveness", {
            source: currentSource, well: getWell(), classifier: "random_forest"
        });
        var el = document.getElementById("effectiveness-results");
        el.classList.remove("d-none");
        var body = document.getElementById("effectiveness-body");

        var counts = r.feedback_counts || {};
        var html = '<div class="row g-2 mb-3">';
        html += '<div class="col-md-3"><div class="card text-center"><div class="card-body py-2">' +
            '<h3>' + (counts.ratings || 0) + '</h3><small class="text-muted">Ratings</small></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center"><div class="card-body py-2">' +
            '<h3>' + (counts.corrections || 0) + '</h3><small class="text-muted">Corrections</small></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center"><div class="card-body py-2">' +
            '<h3>' + (counts.flagged || 0) + '</h3><small class="text-muted">Flagged</small></div></div></div>';
        var baseAcc = r.baseline ? (r.baseline.accuracy * 100).toFixed(1) : "--";
        html += '<div class="col-md-3"><div class="card text-center border-primary"><div class="card-body py-2">' +
            '<h3 class="text-primary">' + baseAcc + '%</h3><small class="text-muted">Baseline Accuracy</small></div></div></div>';
        html += '</div>';

        // Before/After comparison
        if (r.with_corrections) {
            var wc = r.with_corrections;
            var color = wc.improvement > 0 ? "success" : "warning";
            html += '<div class="alert alert-' + color + ' p-3 mb-3">' +
                '<h6><i class="bi bi-arrow-up-circle"></i> Impact of ' + wc.corrections_applied + ' corrections</h6>' +
                '<p class="mb-0">Accuracy: ' + (r.baseline.accuracy * 100).toFixed(1) + '% → ' +
                (wc.accuracy * 100).toFixed(1) + '% (' +
                (wc.improvement > 0 ? '+' : '') + wc.improvement_pct.toFixed(2) + '%)</p></div>';
        } else {
            html += '<div class="alert alert-info p-3 mb-3">' +
                '<i class="bi bi-info-circle"></i> No corrections submitted yet. ' +
                'Submit label corrections on misclassified fractures to start improving the model.</div>';
        }

        // ROI
        if (r.roi && r.roi.corrections_for_1pct) {
            html += '<div class="alert alert-light p-2 mb-3">' +
                '<small><strong>ROI:</strong> ~' + r.roi.corrections_for_1pct +
                ' corrections needed per 1% accuracy improvement</small></div>';
        }

        // Per-class breakdown
        if (r.per_class && r.per_class.length > 0) {
            html += '<h6><i class="bi bi-list-check"></i> Per-Class Correction Priority</h6>';
            html += '<table class="table table-sm"><thead><tr>' +
                '<th>Class</th><th>Accuracy</th><th>Misclassified</th><th>Priority</th></tr></thead><tbody>';
            r.per_class.forEach(function(pc) {
                var pColor = pc.priority === "HIGH" ? "danger" : (pc.priority === "MEDIUM" ? "warning" : "success");
                html += '<tr><td>' + pc["class"] + '</td>' +
                    '<td>' + (pc.accuracy * 100).toFixed(0) + '%</td>' +
                    '<td>' + pc.misclassified + ' / ' + pc.total + '</td>' +
                    '<td><span class="badge bg-' + pColor + '">' + pc.priority + '</span></td></tr>';
            });
            html += '</tbody></table>';
        }

        // Expert sentiment
        if (r.expert_sentiment) {
            var es = r.expert_sentiment;
            var sentColor = es.satisfaction === "LOW" ? "danger" : (es.satisfaction === "HIGH" ? "success" : "warning");
            html += '<div class="mt-2"><small><strong>Expert sentiment:</strong> ' +
                '<span class="badge bg-' + sentColor + '">' + es.satisfaction + '</span> ' +
                '(' + es.average_rating + '/5 from ' + es.n_ratings + ' ratings)</small></div>';
        }

        // Recommendations
        if (r.recommendations && r.recommendations.length > 0) {
            html += '<h6 class="mt-3"><i class="bi bi-lightbulb"></i> Next Steps</h6>';
            r.recommendations.forEach(function(rec) {
                var rColor = rec.priority === "HIGH" ? "danger" : "info";
                html += '<div class="alert alert-' + rColor + ' py-2 px-3 mb-2">' +
                    '<small>' + rec.message + '</small></div>';
            });
        }

        body.innerHTML = html;
        showToast("Effectiveness: " + counts.corrections + " corrections, baseline " + baseAcc + "%");
    } catch (err) {
        showToast("Effectiveness error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Query-by-Committee Active Learning ──────────────

async function runQbcActiveLearning() {
    showLoading("Running Query-by-Committee with all classifiers...");
    try {
        var r = await api("/api/analysis/active-learning-qbc", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource, n_suggest: 20 })
        });

        document.getElementById("qbc-results").classList.remove("d-none");
        val("qbc-committee-size", r.committee_size);
        val("qbc-high-disagree", r.stats ? r.stats.high_disagreement_count : "--");
        val("qbc-mean-ve", r.stats ? r.stats.mean_vote_entropy : "--");
        val("qbc-n-suggestions", r.n_suggestions);

        if (r.stakeholder_brief) {
            renderStakeholderBrief('qbc-brief', r.stakeholder_brief, 'qbc-detail');
        }

        var plotImg = document.getElementById("qbc-plot-img");
        if (plotImg && r.plot) {
            plotImg.innerHTML = '<img src="' + r.plot + '" class="img-fluid rounded" alt="QBC Analysis" style="max-height:400px">';
        }

        var tbody = document.getElementById("qbc-table-body");
        tbody.innerHTML = '';
        (r.suggestions || []).forEach(function(s) {
            var preds = s.model_predictions || {};
            var predBadges = Object.entries(preds).map(function(kv) {
                var isMatch = kv[1] === s.current_label;
                var cls = isMatch ? 'bg-success' : 'bg-danger';
                return '<span class="badge ' + cls + ' me-1" title="' + kv[0] + '">' + kv[1] + '</span>';
            }).join('');

            var row = document.createElement("tr");
            row.innerHTML = '<td>' + s.index + '</td><td>' + s.depth + 'm</td><td>' + s.well + '</td>' +
                '<td><strong>' + s.current_label + '</strong></td>' +
                '<td>' + predBadges + '</td>' +
                '<td>' + s.agreement + '</td>' +
                '<td>' + s.qbc_score.toFixed(3) + '</td>';
            tbody.appendChild(row);
        });

        showToast("QBC: " + r.committee_size + " classifiers, " + r.n_suggestions + " suggestions");
    } catch (err) {
        showToast("QBC error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Active Learning ───────────────────────────────

async function runActiveLearning() {
    showLoading("Identifying most uncertain fractures...");
    try {
        var r = await api("/api/analysis/active-learning", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, n_suggest: 20, classifier: "xgboost"})
        });

        document.getElementById("al-results").classList.remove("d-none");

        // Summary
        document.getElementById("al-summary-text").innerHTML =
            '<div class="alert alert-info small">' + (r.summary || '') + '</div>';

        // Learning curve
        var curveEl = document.getElementById("al-learning-curve");
        var curveBody = document.getElementById("al-curve-body");
        clearChildren(curveBody);
        if (r.learning_curve && r.learning_curve.length > 1) {
            curveEl.classList.remove("d-none");
            var tbl = '<table class="table table-sm"><thead><tr><th>Data Fraction</th><th>Samples</th><th>Accuracy</th></tr></thead><tbody>';
            r.learning_curve.forEach(function(lc) {
                tbl += '<tr><td>' + (lc.fraction * 100).toFixed(0) + '%</td><td>' + lc.n_samples + '</td>' +
                    '<td>' + (lc.accuracy * 100).toFixed(1) + '% ± ' + (lc.std * 100).toFixed(1) + '%</td></tr>';
            });
            tbl += '</tbody></table>';
            if (r.projected_accuracy) {
                tbl += '<div class="small text-muted">Projected with 50% more data: <strong>' +
                    (r.projected_accuracy * 100).toFixed(1) + '%</strong></div>';
            }
            curveBody.innerHTML = tbl;
        }

        // Coverage gaps
        var covEl = document.getElementById("al-coverage");
        var covBody = document.getElementById("al-coverage-body");
        clearChildren(covBody);
        if (r.coverage_gaps && r.coverage_gaps.length > 0) {
            covEl.classList.remove("d-none");
            r.coverage_gaps.forEach(function(g) {
                var div = document.createElement("div");
                div.className = "alert alert-warning py-2 small mb-2";
                div.innerHTML = '<strong>' + g.type + '</strong> (' + g.count + ' samples): ' +
                    g.issue + '<br><i class="bi bi-arrow-right"></i> ' + g.recommendation;
                covBody.appendChild(div);
            });
        }

        // Suggestions table
        var sugEl = document.getElementById("al-suggestions");
        var tbody = document.getElementById("al-table-body");
        clearChildren(tbody);
        if (r.suggestions && r.suggestions.length > 0) {
            sugEl.classList.remove("d-none");
            r.suggestions.forEach(function(s) {
                var tr = document.createElement("tr");
                if (s.mismatch) tr.className = "table-warning";
                tr.innerHTML = '<td>' + s.index + '</td><td>' + s.well + '</td>' +
                    '<td>' + s.depth + '</td><td>' + s.azimuth + '°</td><td>' + s.dip + '°</td>' +
                    '<td>' + s.current_label + '</td><td>' + s.predicted_label + '</td>' +
                    '<td><span class="badge bg-' + (s.confidence >= 50 ? 'warning' : 'danger') + '">' +
                    s.confidence + '%</span></td>' +
                    '<td>' + (s.mismatch ? '<i class="bi bi-exclamation-triangle text-danger"></i>' : '') + '</td>';
                tbody.appendChild(tr);
            });
        }

        showToast("Active learning: " + r.n_suggested + " samples suggested for review");
    } catch (err) {
        showToast("Active learning error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Sensitivity Analysis ─────────────────────────

// ── What-If Interactive Explorer ──────────────────────

// Update slider value displays + debounced auto-run
var _whatIfDebounceTimer = null;
function _debounceWhatIf() {
    // Auto-run what-if 600ms after user stops dragging
    if (_whatIfDebounceTimer) clearTimeout(_whatIfDebounceTimer);
    _whatIfDebounceTimer = setTimeout(function() {
        runWhatIf();
    }, 600);
}

document.addEventListener("DOMContentLoaded", function() {
    var frictionSlider = document.getElementById("whatif-friction");
    var ppSlider = document.getElementById("whatif-pp");
    var depthSlider = document.getElementById("whatif-depth");
    if (frictionSlider) {
        frictionSlider.oninput = function() {
            document.getElementById("whatif-friction-val").textContent = this.value;
            _debounceWhatIf();
        };
    }
    if (ppSlider) {
        ppSlider.oninput = function() {
            document.getElementById("whatif-pp-val").textContent = this.value + " MPa";
            _debounceWhatIf();
        };
    }
    if (depthSlider) {
        depthSlider.oninput = function() {
            document.getElementById("whatif-depth-val").textContent = this.value + " m";
            _debounceWhatIf();
        };
    }
    var abstentionSlider = document.getElementById("abstention-threshold");
    if (abstentionSlider) {
        abstentionSlider.oninput = function() {
            document.getElementById("abstention-threshold-val").textContent =
                Math.round(this.value * 100) + "%";
        };
    }
});

async function runWhatIf() {
    var friction = parseFloat(document.getElementById("whatif-friction").value);
    var pp = parseFloat(document.getElementById("whatif-pp").value);
    var depth = parseFloat(document.getElementById("whatif-depth").value);

    showLoading("Running what-if scenario...");
    try {
        var r = await apiPost("/api/analysis/what-if", {
            source: currentSource, well: getWell(),
            friction: friction, pore_pressure: pp, depth: depth
        });

        var container = document.getElementById("whatif-results");
        container.classList.remove("d-none");

        var riskColors = {GREEN: "success", AMBER: "warning", RED: "danger"};
        var riskColor = riskColors[r.risk_level] || "secondary";

        var html = '<div class="alert alert-' + riskColor + ' py-2">' +
            '<div class="d-flex justify-content-between align-items-center">' +
            '<span><strong>Risk: ' + r.risk_level + '</strong> — ' +
            r.critically_stressed_pct + '% critically stressed</span>' +
            '<span class="badge bg-' + riskColor + ' fs-6">' + r.risk_level + '</span></div></div>';

        html += '<div class="row g-2">';
        html += '<div class="col-md-3"><div class="card text-center"><div class="card-body py-2">' +
            '<div class="small text-muted">SHmax</div>' +
            '<div class="fw-bold">' + r.shmax_deg + '°</div></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center"><div class="card-body py-2">' +
            '<div class="small text-muted">Regime</div>' +
            '<div class="fw-bold">' + r.regime + '</div></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center"><div class="card-body py-2">' +
            '<div class="small text-muted">Critically Stressed</div>' +
            '<div class="fw-bold text-' + riskColor + '">' + r.critically_stressed_pct + '%</div></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center"><div class="card-body py-2">' +
            '<div class="small text-muted">High Risk</div>' +
            '<div class="fw-bold">' + r.high_risk_count + ' fractures</div></div></div></div>';
        html += '</div>';

        html += '<div class="small text-muted mt-2">σ1=' + r.sigma1 + ' MPa, σ3=' + r.sigma3 +
            ' MPa, R=' + r.R_ratio + ' | μ=' + r.friction_used + ', Pp=' + r.pore_pressure_mpa +
            ' MPa, depth=' + r.depth_m + 'm</div>';

        container.innerHTML = html;
        showToast("What-if: " + r.risk_level + " (" + r.critically_stressed_pct + "% CS)");
    } catch (err) {
        showToast("What-if error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


async function runSensitivityHeatmap() {
    showLoading("Generating sensitivity heatmap (81 scenarios)...");
    try {
        var r = await apiPost("/api/analysis/sensitivity-heatmap", {
            source: currentSource, well: getWell(), depth: getDepth()
        });
        var el = document.getElementById("heatmap-results");
        el.classList.remove("d-none");
        var body = document.getElementById("heatmap-body");

        var html = '';
        if (r.chart_img) {
            html += '<div class="text-center mb-3"><img src="data:image/png;base64,' + r.chart_img +
                '" class="img-fluid" alt="Sensitivity heatmap"></div>';
        }

        // Summary info
        html += '<div class="row g-2 mb-2">';
        html += '<div class="col-md-4"><small><strong>Well:</strong> ' + r.well + '</small></div>';
        html += '<div class="col-md-4"><small><strong>Depth:</strong> ' + r.depth_m + 'm</small></div>';
        html += '<div class="col-md-4"><small><strong>Regime:</strong> ' + r.regime + '</small></div>';
        html += '</div>';

        // Quick risk summary from matrix
        if (r.cs_matrix) {
            var allVals = [];
            r.cs_matrix.forEach(function(row) { row.forEach(function(v) { allVals.push(v); }); });
            var minCS = Math.min.apply(null, allVals);
            var maxCS = Math.max.apply(null, allVals);
            html += '<div class="alert alert-info py-2 px-3">' +
                '<small>Critically stressed ranges from <strong>' + minCS + '%</strong> to <strong>' + maxCS +
                '%</strong> across the parameter space (' + r.n_fractures + ' fractures)</small></div>';
        }

        body.innerHTML = html;
        showToast("Heatmap: " + (r.friction_values || []).length + "×" + (r.pp_values_mpa || []).length + " grid computed");
    } catch (err) {
        showToast("Heatmap error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function runWorstCase() {
    showLoading("Running worst-case scenario analysis (5 scenarios)...");
    try {
        var r = await apiPost("/api/analysis/worst-case", {
            source: currentSource, well: getWell(), depth: getDepth()
        });
        var el = document.getElementById("worst-case-results");
        el.classList.remove("d-none");
        var body = document.getElementById("worst-case-body");

        var s = r.summary || {};
        var sensColor = s.sensitivity === "HIGH_SENSITIVITY" ? "danger" :
                        (s.sensitivity === "MODERATE_SENSITIVITY" ? "warning" : "success");

        // Verdict banner
        var html = '<div class="alert alert-' + sensColor + ' p-3 mb-3">' +
            '<h6><i class="bi bi-exclamation-octagon"></i> ' + (s.sensitivity || "").replace(/_/g, " ") + '</h6>' +
            '<p class="mb-1">' + (r.interpretation || "") + '</p>' +
            '<small class="text-muted">' + (r.guidance || "") + '</small></div>';

        // Summary metrics
        html += '<div class="row g-2 mb-3">';
        html += '<div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">CS Range</small>' +
            '<div class="fw-bold">' + (s.cs_range_pct ? s.cs_range_pct[0] + '% – ' + s.cs_range_pct[1] + '%' : '--') + '</div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Spread</small>' +
            '<div class="fw-bold">' + (s.cs_spread_pp || 0) + ' pp</div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Worst Risk</small>' +
            '<div class="fw-bold"><span class="badge bg-' + (s.worst_risk === "RED" ? "danger" : s.worst_risk === "AMBER" ? "warning" : "success") + '">' +
            (s.worst_risk || "--") + '</span></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center p-2"><small class="text-muted">Scenarios</small>' +
            '<div class="fw-bold">' + (r.scenarios || []).length + '</div></div></div>';
        html += '</div>';

        // Scenario table
        html += '<div class="table-responsive"><table class="table table-sm table-hover">';
        html += '<thead class="table-dark"><tr><th>Scenario</th><th>Regime</th><th>SHmax</th>' +
            '<th>mu</th><th>Pp (MPa)</th><th>CS %</th><th>Risk</th></tr></thead><tbody>';
        (r.scenarios || []).forEach(function(sc) {
            if (sc.error) {
                html += '<tr class="table-danger"><td>' + sc.name + '</td><td colspan="6" class="text-danger">' + sc.error + '</td></tr>';
            } else {
                var rc = sc.risk_level === "RED" ? "danger" : (sc.risk_level === "AMBER" ? "warning" : "success");
                html += '<tr><td><strong>' + sc.name + '</strong></td><td>' + sc.regime + '</td>' +
                    '<td>' + sc.shmax + '°</td><td>' + sc.mu + '</td><td>' + sc.pore_pressure + '</td>' +
                    '<td>' + sc.pct_critical + '%</td>' +
                    '<td><span class="badge bg-' + rc + '">' + sc.risk_level + '</span></td></tr>';
            }
        });
        html += '</tbody></table></div>';

        body.innerHTML = html;
        showToast("Worst-case: CS ranges " + s.cs_range_pct[0] + "%-" + s.cs_range_pct[1] + "% (" + s.sensitivity + ")");
    } catch (err) {
        showToast("Worst-case error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function runSensitivity() {
    showLoading("Running sensitivity analysis");
    try {
        var well = document.getElementById("well-select").value || null;
        var regime = document.getElementById("regime-select").value;
        var depth = parseFloat(document.getElementById("depth-input").value) || 3000;
        var pp = getPorePresure();

        var r = await api("/api/analysis/sensitivity", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, well: well, regime: regime, depth: depth, pore_pressure: pp})
        });

        // Tornado diagram
        var tornadoEl = document.getElementById("sens-tornado");
        var tornadoBody = document.getElementById("sens-tornado-body");
        clearChildren(tornadoBody);
        if (r.tornado && r.tornado.length) {
            tornadoEl.classList.remove("d-none");
            r.tornado.forEach(function(t) {
                var row = document.createElement("div");
                row.className = "mb-3";
                row.innerHTML = '<div class="d-flex justify-content-between"><strong>' + t.parameter + '</strong><span class="text-muted">' + t.min_pct_critical + '% – ' + t.max_pct_critical + '% (range: ' + t.range + '%)</span></div>' +
                    '<div class="feat-bar-bg"><div class="feat-bar-fill" style="width:' + Math.min(100, t.range * 2) + '%;background:#dc3545"></div></div>' +
                    '<small class="text-muted">' + t.description + '</small>';
                tornadoBody.appendChild(row);
            });
        }

        // Risk implications
        var risksEl = document.getElementById("sens-risks");
        var risksBody = document.getElementById("sens-risks-body");
        clearChildren(risksBody);
        if (r.risk_implications && r.risk_implications.length) {
            risksEl.classList.remove("d-none");
            r.risk_implications.forEach(function(ri) {
                var div = document.createElement("div");
                div.className = "mb-2";
                var icon = ri.severity === "high" ? "exclamation-triangle text-danger" : "info-circle text-info";
                div.innerHTML = '<i class="bi bi-' + icon + '"></i> ' + ri.message;
                risksBody.appendChild(div);
            });
        }

        // Regime comparison
        var regimeEl = document.getElementById("sens-regime");
        var regimeBody = document.getElementById("sens-regime-body");
        clearChildren(regimeBody);
        var regimes = (r.results || {}).stress_regime || [];
        if (regimes.length) {
            regimeEl.classList.remove("d-none");
            var tbl = '<table class="table table-sm table-hover"><thead><tr><th>Regime</th><th>σ1</th><th>σ3</th><th>SHmax</th><th>μ</th><th>R</th><th>Crit. Stressed</th><th>Misfit</th></tr></thead><tbody>';
            regimes.forEach(function(rg) {
                if (rg.error) {
                    tbl += '<tr><td>' + rg.regime + '</td><td colspan="7" class="text-danger">' + rg.error + '</td></tr>';
                } else {
                    var isBest = rg.regime === (r.base_result || {}).regime;
                    tbl += '<tr' + (isBest ? ' class="table-active"' : '') + '><td><strong>' + rg.regime.replace("_"," ") + '</strong>' + (isBest ? ' ★' : '') + '</td>' +
                        '<td>' + fmt(rg.sigma1) + '</td><td>' + fmt(rg.sigma3) + '</td><td>' + fmt(rg.shmax) + '°</td>' +
                        '<td>' + fmt(rg.mu, 3) + '</td><td>' + fmt(rg.R, 3) + '</td><td>' + fmt(rg.pct_critically_stressed) + '%</td><td>' + fmt(rg.misfit) + '</td></tr>';
                }
            });
            tbl += '</tbody></table>';
            regimeBody.innerHTML = tbl;
        }

        // Friction detail
        var frictionEl = document.getElementById("sens-friction");
        var frictionBody = document.getElementById("sens-friction-body");
        clearChildren(frictionBody);
        var muData = (r.results || {}).friction_coefficient || [];
        if (muData.length) {
            frictionEl.classList.remove("d-none");
            muData.forEach(function(m) {
                var isBase = Math.abs(m.value - (r.base_result || {}).mu) < 0.01;
                var bar = document.createElement("div");
                bar.className = "mb-2" + (isBase ? " fw-bold" : "");
                bar.innerHTML = '<div class="d-flex justify-content-between"><span>μ = ' + m.value.toFixed(1) + (isBase ? ' (current)' : '') + '</span><span>' + m.pct_critically_stressed + '%</span></div>' +
                    '<div class="feat-bar-bg"><div class="feat-bar-fill" style="width:' + m.pct_critically_stressed + '%;background:' + (m.pct_critically_stressed > 50 ? '#dc3545' : '#3b82f6') + '"></div></div>';
                frictionBody.appendChild(bar);
            });
        }

        // PP detail
        var ppEl = document.getElementById("sens-pp");
        var ppBody = document.getElementById("sens-pp-body");
        clearChildren(ppBody);
        var ppData = (r.results || {}).pore_pressure || [];
        if (ppData.length) {
            ppEl.classList.remove("d-none");
            ppData.forEach(function(p) {
                var isBase = Math.abs(p.value - (r.base_result || {}).pore_pressure) < 0.5;
                var bar = document.createElement("div");
                bar.className = "mb-2" + (isBase ? " fw-bold" : "");
                bar.innerHTML = '<div class="d-flex justify-content-between"><span>Pp = ' + p.value.toFixed(1) + ' MPa' + (isBase ? ' (current)' : '') + '</span><span>' + p.pct_critically_stressed + '%</span></div>' +
                    '<div class="feat-bar-bg"><div class="feat-bar-fill" style="width:' + p.pct_critically_stressed + '%;background:' + (p.pct_critically_stressed > 50 ? '#dc3545' : '#3b82f6') + '"></div></div>';
                ppBody.appendChild(bar);
            });
        }

        showToast("Sensitivity analysis complete");
    } catch (err) {
        showToast("Sensitivity error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Risk Matrix ──────────────────────────────────

async function runRiskMatrix() {
    showLoading("Computing risk assessment");
    try {
        var well = document.getElementById("well-select").value || null;
        var regime = document.getElementById("regime-select").value;
        var depth = parseFloat(document.getElementById("depth-input").value) || 3000;
        var pp = getPorePresure();

        var r = await api("/api/analysis/risk-matrix", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, well: well, regime: regime, depth: depth, pore_pressure: pp})
        });

        // Overall banner
        var overallEl = document.getElementById("risk-overall");
        overallEl.classList.remove("d-none");
        var card = document.getElementById("risk-overall-card");
        card.className = "card border-" + (r.overall_color || "secondary");

        document.getElementById("risk-go-nogo").textContent = r.go_nogo || "N/A";
        document.getElementById("risk-go-nogo").className = "mb-2 text-" + (r.overall_color || "secondary");
        document.getElementById("risk-score-badge").textContent = "Risk Score: " + r.overall_score + "/100 (" + r.overall_level + ")";
        document.getElementById("risk-detail").textContent = r.go_nogo_detail || "";

        // Factors table
        var factorsEl = document.getElementById("risk-factors");
        factorsEl.classList.remove("d-none");
        var tbody = document.getElementById("risk-factors-tbody");
        clearChildren(tbody);
        (r.factors || []).forEach(function(f) {
            var tr = document.createElement("tr");
            var scoreColor = f.score >= 60 ? "danger" : f.score >= 40 ? "warning" : "success";
            tr.innerHTML = '<td><strong>' + f.factor + '</strong><br><small class="text-muted">' + f.impact + '</small></td>' +
                '<td><span class="badge bg-' + scoreColor + '">' + f.score + '</span></td>' +
                '<td><small>' + f.detail + '</small></td>' +
                '<td><small>' + f.mitigation + '</small></td>';
            tbody.appendChild(tr);
        });

        showToast("Risk assessment: " + r.overall_level + " — " + r.go_nogo);
    } catch (err) {
        showToast("Risk matrix error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Well Comparison ──────────────────────────────

async function runWellComparison() {
    showLoading("Comparing wells (running inversions)");
    try {
        var depth = parseFloat(document.getElementById("depth-input").value) || 3000;
        var r = await api("/api/analysis/compare-wells", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, depth: depth})
        });

        if (r.status === "insufficient_wells") {
            showToast(r.message, "Info");
            return;
        }

        // Consistency checks
        var consEl = document.getElementById("wells-consistency");
        var consBody = document.getElementById("wells-consistency-body");
        clearChildren(consBody);
        if (r.consistency_checks && r.consistency_checks.length) {
            consEl.classList.remove("d-none");
            r.consistency_checks.forEach(function(c) {
                var div = document.createElement("div");
                div.className = "mb-2";
                var icon = c.status === "OK" ? "check-circle text-success" : "exclamation-triangle text-warning";
                div.innerHTML = '<i class="bi bi-' + icon + '"></i> <strong>' + c.check + '</strong>: ' + c.detail;
                consBody.appendChild(div);
            });
        }

        // Well results table
        var tableEl = document.getElementById("wells-table");
        var tableBody = document.getElementById("wells-table-body");
        clearChildren(tableBody);
        if (r.wells) {
            tableEl.classList.remove("d-none");
            var tbl = '<table class="table table-sm table-hover"><thead><tr><th>Well</th><th>Fractures</th><th>Quality</th><th>σ1</th><th>SHmax</th><th>μ</th><th>Crit%</th><th>ML Acc</th></tr></thead><tbody>';
            for (var wName in r.wells) {
                var w = r.wells[wName];
                tbl += '<tr><td><strong>' + wName + '</strong></td><td>' + w.n_fractures + '</td>' +
                    '<td>' + w.data_quality_grade + ' (' + w.data_quality_score + ')</td>' +
                    '<td>' + (w.sigma1 ? fmt(w.sigma1) : 'N/A') + '</td>' +
                    '<td>' + (w.shmax ? fmt(w.shmax) + '°' : 'N/A') + '</td>' +
                    '<td>' + (w.mu ? fmt(w.mu, 3) : 'N/A') + '</td>' +
                    '<td>' + (w.pct_critically_stressed != null ? fmt(w.pct_critically_stressed) + '%' : 'N/A') + '</td>' +
                    '<td>' + (w.classification_accuracy != null ? fmt(w.classification_accuracy) + '%' : 'N/A') + '</td></tr>';
            }
            tbl += '</tbody></table>';
            tableBody.innerHTML = tbl;
        }

        // Cross-validation
        var cvEl = document.getElementById("wells-crossval");
        var cvBody = document.getElementById("wells-crossval-body");
        clearChildren(cvBody);
        if (r.cross_validation && Object.keys(r.cross_validation).length) {
            cvEl.classList.remove("d-none");
            var cvTbl = '<table class="table table-sm"><thead><tr><th>Train → Test</th><th>Accuracy</th><th>Train Size</th><th>Test Size</th></tr></thead><tbody>';
            for (var key in r.cross_validation) {
                var cv = r.cross_validation[key];
                var accColor = cv.accuracy >= 70 ? "success" : cv.accuracy >= 50 ? "warning" : "danger";
                cvTbl += '<tr><td>' + key + '</td>' +
                    '<td><span class="badge bg-' + accColor + '">' + (cv.accuracy != null ? cv.accuracy + '%' : (cv.error || 'N/A')) + '</span>' +
                    (cv.note ? '<br><small class="text-muted">' + cv.note + '</small>' : '') + '</td>' +
                    '<td>' + (cv.train_size || '') + '</td><td>' + (cv.test_size || '') + '</td></tr>';
            }
            cvTbl += '</tbody></table>';
            cvBody.innerHTML = cvTbl;
        }

        showToast("Well comparison complete: " + r.n_wells + " wells analyzed");
    } catch (err) {
        showToast("Well comparison error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Well Report ──────────────────────────────────

async function generateReport() {
    showLoading("Generating stakeholder report");
    try {
        var well = document.getElementById("well-select").value || null;
        var regime = document.getElementById("regime-select").value;
        var depth = parseFloat(document.getElementById("depth-input").value) || 3000;
        var pp = getPorePresure();

        var r = await api("/api/report/well", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, well: well, regime: regime, depth: depth, pore_pressure: pp})
        });

        document.getElementById("report-content").classList.remove("d-none");
        document.getElementById("btn-print-report").classList.remove("d-none");

        // Executive summary
        var execHtml = '<p class="lead">' + (r.executive_summary || '') + '</p>';

        // Auto-regime detection info
        if (r.regime_detection) {
            var rd = r.regime_detection;
            var confBadge = rd.confidence === "HIGH" ? "bg-success" : rd.confidence === "MODERATE" ? "bg-warning text-dark" : "bg-danger";
            execHtml += '<div class="alert alert-info py-2 mb-2"><i class="bi bi-robot me-1"></i>' +
                '<strong>Auto-detected regime:</strong> ' + rd.best_regime.replace("_", "-") +
                ' <span class="badge ' + confBadge + '">' + rd.confidence + '</span> ' +
                '<span class="small text-muted">(' + rd.summary.substring(0, 200) + ')</span></div>';
        }

        // Classification warning
        if (r.classification && r.classification.imbalance_warning) {
            execHtml += '<div class="alert alert-danger py-2 mb-2"><i class="bi bi-exclamation-triangle me-1"></i>' +
                r.classification.imbalance_warning + '</div>';
        }

        execHtml += '<small class="text-muted">Generated: ' + (r.generated_at || '') + ' | Version: ' + (r.version || '') + '</small>';
        document.getElementById("report-exec-summary").innerHTML = execHtml;

        // Stress state
        var ss = r.stress_state || {};
        document.getElementById("report-stress").innerHTML =
            '<div class="row g-3">' +
            '<div class="col-md-4"><div class="metric-card"><div class="metric-label">σ1 (Max)</div><div class="metric-value">' + fmt(ss.sigma1_mpa) + ' MPa</div></div></div>' +
            '<div class="col-md-4"><div class="metric-card"><div class="metric-label">σ2 (Int)</div><div class="metric-value">' + fmt(ss.sigma2_mpa) + ' MPa</div></div></div>' +
            '<div class="col-md-4"><div class="metric-card"><div class="metric-label">σ3 (Min)</div><div class="metric-value">' + fmt(ss.sigma3_mpa) + ' MPa</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">SHmax</div><div class="metric-value">' + fmt(ss.shmax_azimuth) + '° ' + (ss.shmax_compass || '') + '</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">R Ratio</div><div class="metric-value">' + fmt(ss.R_ratio, 3) + '</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Regime</div><div class="metric-value">' + (ss.regime || '').replace('_',' ') + '</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Pore Pressure</div><div class="metric-value">' + fmt(ss.pore_pressure_mpa) + ' MPa</div></div></div>' +
            '</div>';

        // Risk assessment
        var ra = r.risk_assessment || {};
        var riskHtml = '<div class="text-center mb-3"><h3 class="text-' + (ra.overall_color || 'secondary') + '">' + (ra.go_nogo || 'N/A') + '</h3>' +
            '<p>Risk Score: ' + (ra.overall_score || 0) + '/100 (' + (ra.overall_level || '') + ')</p>' +
            '<p class="text-muted">' + (ra.go_nogo_detail || '') + '</p></div>';
        if (ra.factors) {
            riskHtml += '<table class="table table-sm"><thead><tr><th>Factor</th><th>Score</th><th>Detail</th></tr></thead><tbody>';
            ra.factors.forEach(function(f) {
                var sc = f.score >= 60 ? "danger" : f.score >= 40 ? "warning" : "success";
                riskHtml += '<tr><td>' + f.factor + '</td><td><span class="badge bg-' + sc + '">' + f.score + '</span></td><td>' + f.detail + '</td></tr>';
            });
            riskHtml += '</tbody></table>';
        }
        document.getElementById("report-risk").innerHTML = riskHtml;

        // Recommendations
        var recs = r.recommendations || {};
        var recHtml = '';
        if (recs.drilling && recs.drilling.length) {
            recHtml += '<h6><i class="bi bi-gear"></i> Drilling</h6><ul>';
            recs.drilling.forEach(function(d) { recHtml += '<li>' + d + '</li>'; });
            recHtml += '</ul>';
        }
        if (recs.completion && recs.completion.length) {
            recHtml += '<h6><i class="bi bi-wrench"></i> Completion</h6><ul>';
            recs.completion.forEach(function(c) { recHtml += '<li>' + c + '</li>'; });
            recHtml += '</ul>';
        }
        if (recs.monitoring && recs.monitoring.length) {
            recHtml += '<h6><i class="bi bi-eye"></i> Monitoring</h6><ul>';
            recs.monitoring.forEach(function(m) { recHtml += '<li>' + m + '</li>'; });
            recHtml += '</ul>';
        }
        if (!recHtml) recHtml = '<p class="text-muted">No specific recommendations.</p>';
        document.getElementById("report-recommendations").innerHTML = recHtml;

        // Data quality
        var dq = r.data_quality || {};
        var dqHtml = '<div class="d-flex align-items-center gap-3 mb-2">' +
            '<span class="badge bg-' + (dq.grade === 'A' ? 'success' : dq.grade === 'B' ? 'primary' : dq.grade === 'C' ? 'warning' : 'danger') + ' fs-5">Grade: ' + (dq.grade || '?') + '</span>' +
            '<span>Score: ' + (dq.score || 0) + '/100</span></div>';
        if (dq.issues && dq.issues.length) {
            dqHtml += '<div class="text-danger mb-1"><strong>Issues:</strong></div><ul>';
            dq.issues.forEach(function(i) { dqHtml += '<li class="text-danger small">' + i + '</li>'; });
            dqHtml += '</ul>';
        }
        if (dq.warnings && dq.warnings.length) {
            dqHtml += '<div class="text-warning mb-1"><strong>Warnings:</strong></div><ul>';
            dq.warnings.forEach(function(w) { dqHtml += '<li class="text-warning small">' + w + '</li>'; });
            dqHtml += '</ul>';
        }
        document.getElementById("report-quality").innerHTML = dqHtml;

        // Calibration section (append to quality card)
        if (r.calibration) {
            var calBadge = {"EXCELLENT": "success", "GOOD": "primary", "FAIR": "warning", "POOR": "danger"}[r.calibration.reliability] || "secondary";
            dqHtml += '<hr><h6><i class="bi bi-bullseye"></i> Model Calibration</h6>' +
                '<p><span class="badge bg-' + calBadge + '">' + r.calibration.reliability + '</span> ' +
                'ECE: ' + r.calibration.ece_pct + '% | Brier: ' + r.calibration.brier_score + '</p>' +
                '<p class="small text-muted">' + (r.calibration.summary || '') + '</p>';
            document.getElementById("report-quality").innerHTML = dqHtml;
        }

        // Data collection roadmap (append to recommendations)
        if (r.data_roadmap) {
            var roadHtml = '<hr><h6><i class="bi bi-lightbulb"></i> Data Collection Roadmap ' +
                '<span class="badge bg-info">' + r.data_roadmap.completeness_pct + '% Complete</span></h6>';
            if (r.data_roadmap.priority_actions && r.data_roadmap.priority_actions.length > 0) {
                roadHtml += '<div class="text-danger small"><strong>Critical Actions:</strong></div><ul>';
                r.data_roadmap.priority_actions.forEach(function(a) { roadHtml += '<li class="small text-danger">' + a + '</li>'; });
                roadHtml += '</ul>';
            }
            if (r.data_roadmap.recommendations && r.data_roadmap.recommendations.length > 0) {
                roadHtml += '<div class="text-warning small"><strong>Recommendations:</strong></div><ul>';
                r.data_roadmap.recommendations.forEach(function(a) { roadHtml += '<li class="small">' + a + '</li>'; });
                roadHtml += '</ul>';
            }
            document.getElementById("report-recommendations").innerHTML += roadHtml;
        }

        showToast("Report generated for " + (r.well_name || "well"));
    } catch (err) {
        showToast("Report error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Uncertainty Budget ────────────────────────────

async function runCalibrationReport() {
    showLoading("Computing calibration report + OOD detection (Platt scaling)...");
    try {
        var r = await api("/api/analysis/calibration-report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource, classifier: "random_forest" })
        });

        document.getElementById("cal-report").classList.remove("d-none");
        val("cal-ece", r.ece_calibrated ? r.ece_calibrated.toFixed(4) : "--");
        val("cal-improvement", r.ece_improvement_pct ? r.ece_improvement_pct.toFixed(1) + "%" : "--");
        val("cal-brier", r.brier_calibrated ? r.brier_calibrated.toFixed(4) : "--");

        var qBadge = r.calibration_quality;
        var qColor = qBadge === 'GOOD' ? 'text-success' : (qBadge === 'FAIR' ? 'text-warning' : 'text-danger');
        document.getElementById("cal-quality").className = "metric-value " + qColor;
        val("cal-quality", qBadge);

        if (r.stakeholder_brief) {
            renderStakeholderBrief('cal-brief', r.stakeholder_brief, 'cal-detail');
        }

        var plotImg = document.getElementById("cal-plot-img");
        if (plotImg && r.plot) {
            plotImg.innerHTML = '<img src="' + r.plot + '" class="img-fluid rounded" alt="Calibration Diagram" style="max-height:450px">';
        }

        // OOD per well
        var oodBody = document.getElementById("cal-ood-body");
        oodBody.innerHTML = '';
        if (r.ood_per_well) {
            Object.entries(r.ood_per_well).forEach(function(kv) {
                var w = kv[0], d = kv[1];
                var sevBadge = d.ood_severity === 'HIGH' ? '<span class="badge bg-danger">HIGH</span>'
                    : d.ood_severity === 'MEDIUM' ? '<span class="badge bg-warning text-dark">MED</span>'
                    : '<span class="badge bg-success">LOW</span>';
                var row = document.createElement("tr");
                row.innerHTML = '<td><strong>' + w + '</strong></td>' +
                    '<td>' + d.mean_mahalanobis + '</td>' +
                    '<td>' + d.max_mahalanobis + '</td>' +
                    '<td>' + d.pct_above_threshold + '%</td>' +
                    '<td>' + d.n_samples + '</td>' +
                    '<td>' + sevBadge + '</td>';
                oodBody.appendChild(row);
            });
        }

        showToast("Calibration: " + r.calibration_quality + ", ECE=" + (r.ece_calibrated || 0).toFixed(4));
    } catch (err) {
        showToast("Calibration error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function runUncertaintyBudget(includeBayesian) {
    showLoading(includeBayesian ? "Computing uncertainty budget with Bayesian MCMC..." : "Computing uncertainty budget...");
    try {
        var well = document.getElementById("well-select").value || null;
        var regime = document.getElementById("regime-select").value;
        var depth = parseFloat(document.getElementById("depth-input").value) || 3000;
        var pp = getPorePresure();

        var r = await api("/api/analysis/uncertainty-budget", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                source: currentSource, well: well, regime: regime,
                depth: depth, pore_pressure: pp,
                include_bayesian: !!includeBayesian
            })
        });

        // Overall banner
        var overallEl = document.getElementById("ub-overall");
        overallEl.classList.remove("d-none");
        var card = document.getElementById("ub-overall-card");
        var levelColor = r.uncertainty_level === "LOW" ? "success" : r.uncertainty_level === "MODERATE" ? "warning" : "danger";
        card.className = "card border-" + levelColor;

        document.getElementById("ub-level").textContent = r.uncertainty_level + " UNCERTAINTY";
        document.getElementById("ub-level").className = "mb-2 text-" + levelColor;
        document.getElementById("ub-total-score").textContent = "Score: " + r.total_score + "/100";
        document.getElementById("ub-dominant").textContent = r.dominant_source ? "Largest contributor: " + r.dominant_source : "";

        // Stakeholder summary
        var summaryEl = document.getElementById("ub-summary");
        summaryEl.classList.remove("d-none");
        document.getElementById("ub-summary-body").innerHTML = '<p class="lead mb-0">' + (r.stakeholder_summary || '') + '</p>';

        // Source rankings
        var sourcesEl = document.getElementById("ub-sources");
        var sourcesBody = document.getElementById("ub-sources-body");
        clearChildren(sourcesBody);
        if (r.sources && r.sources.length) {
            sourcesEl.classList.remove("d-none");
            var maxScore = r.sources[0].score || 1;

            r.sources.forEach(function(s, idx) {
                var scoreColor = s.score >= 60 ? "#dc3545" : s.score >= 40 ? "#ffc107" : "#198754";
                var row = document.createElement("div");
                row.className = "mb-4 p-3 rounded " + (idx === 0 ? "bg-light border" : "");

                var header = '<div class="d-flex justify-content-between align-items-center mb-1">' +
                    '<strong>' + (idx + 1) + '. ' + s.source + '</strong>' +
                    '<span class="badge" style="background:' + scoreColor + '">' + s.score + '/100</span></div>';

                var bar = '<div class="feat-bar-bg mb-2"><div class="feat-bar-fill" style="width:' +
                    Math.round(s.score / maxScore * 100) + '%;background:' + scoreColor + '"></div></div>';

                var detail = '<div class="small text-muted mb-1">' + s.detail + '</div>';
                var driver = '<div class="small"><strong>Key driver:</strong> ' + s.driver + '</div>';
                var rec = '<div class="small text-primary mt-1"><i class="bi bi-arrow-right-circle"></i> ' + s.recommendation + '</div>';

                row.innerHTML = header + bar + detail + driver + rec;
                sourcesBody.appendChild(row);
            });
        }

        // Recommended actions
        var actionsEl = document.getElementById("ub-actions");
        var actionsBody = document.getElementById("ub-actions-body");
        clearChildren(actionsBody);
        if (r.recommended_actions && r.recommended_actions.length) {
            actionsEl.classList.remove("d-none");
            r.recommended_actions.forEach(function(a) {
                var div = document.createElement("div");
                div.className = "mb-3 p-3 bg-light rounded d-flex gap-3 align-items-start";
                div.innerHTML = '<div class="badge bg-success fs-6 px-3 py-2">#' + a.priority + '</div>' +
                    '<div><strong>' + a.source + '</strong><br>' +
                    '<span>' + a.action + '</span>' +
                    (a.impact ? '<br><small class="text-muted">' + a.impact + '</small>' : '') +
                    '</div>';
                actionsBody.appendChild(div);
            });
        }

        showToast("Uncertainty budget: " + r.uncertainty_level + " (" + r.total_score + "/100)");
    } catch (err) {
        showToast("Uncertainty budget error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Bayesian MCMC ────────────────────────────────

async function runBayesian() {
    showLoading("Running Bayesian MCMC (may take 30-60s)");
    try {
        var well = document.getElementById("well-select").value || null;
        var regime = document.getElementById("regime-select").value;
        var depth = parseFloat(document.getElementById("depth-input").value) || 3000;
        var pp = getPorePresure();

        var r = await api("/api/analysis/bayesian", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, well: well, regime: regime, depth: depth, pore_pressure: pp, fast: true})
        });

        if (!r.available) {
            showToast(r.error || "Bayesian MCMC not available", "Error");
            return;
        }

        var resultsEl = document.getElementById("bayesian-results");
        resultsEl.classList.remove("d-none");

        // Convergence badge
        var convBadge = document.getElementById("bayes-convergence");
        convBadge.textContent = r.converged ? "Converged" : "May need more steps";
        convBadge.className = "badge ms-2 bg-" + (r.converged ? "success" : "warning");

        // Stakeholder summary
        document.getElementById("bayes-summary").textContent = r.stakeholder_summary || "";

        // Parameter table
        var paramsBody = document.getElementById("bayes-params-body");
        clearChildren(paramsBody);

        var paramLabels = {
            sigma1: "Maximum stress (sigma1)",
            sigma3: "Minimum stress (sigma3)",
            sigma2: "Intermediate stress (sigma2)",
            R: "R ratio",
            SHmax_azimuth: "SHmax direction",
            mu: "Friction coefficient"
        };
        var paramUnits = {sigma1: " MPa", sigma3: " MPa", sigma2: " MPa", R: "", SHmax_azimuth: "deg", mu: ""};

        var tbl = '<table class="table table-sm"><thead><tr><th>Parameter</th><th>Best Fit</th><th>Median</th><th>68% CI</th><th>90% CI</th></tr></thead><tbody>';
        for (var pName in r.parameters) {
            var p = r.parameters[pName];
            var unit = paramUnits[pName] || "";
            var label = paramLabels[pName] || pName;
            tbl += '<tr><td>' + label + '</td>' +
                '<td>' + (p.best_fit != null ? fmt(p.best_fit, 2) + unit : '-') + '</td>' +
                '<td>' + fmt(p.median, 2) + unit + '</td>' +
                '<td>' + fmt(p.ci_68[0], 2) + ' – ' + fmt(p.ci_68[1], 2) + unit + '</td>' +
                '<td>' + fmt(p.ci_90[0], 2) + ' – ' + fmt(p.ci_90[1], 2) + unit + '</td></tr>';
        }
        tbl += '</tbody></table>';
        paramsBody.innerHTML = tbl;

        // Metadata
        document.getElementById("bayes-meta").textContent =
            r.n_samples + " posterior samples | " +
            r.nwalkers + " walkers x " + r.nsteps + " steps | " +
            "Acceptance: " + (r.acceptance_fraction * 100).toFixed(1) + "%";

        // Inject ± ranges into main inversion metric cards
        if (r.parameters) {
            var params = r.parameters;
            var _addCI = function(elId, pName, unit) {
                var el = document.getElementById(elId);
                if (el && params[pName]) {
                    var ci = params[pName].ci_90;
                    var current = el.textContent.split("±")[0].trim();
                    el.innerHTML = current + '<br><small class="text-muted">90% CI: ' +
                        ci[0].toFixed(1) + '–' + ci[1].toFixed(1) + (unit || "") + '</small>';
                }
            };
            _addCI("inv-sigma1", "sigma1", " MPa");
            _addCI("inv-sigma3", "sigma3", " MPa");
            _addCI("inv-R", "R", "");
            _addCI("inv-shmax", "SHmax_azimuth", "°");
            // Hide the hint badge since we now have CI
            var hint = document.getElementById("inv-ci-hint");
            if (hint) hint.classList.add("d-none");
        }

        showToast("Bayesian uncertainty computed: " + r.n_samples + " posterior samples");
    } catch (err) {
        showToast("Bayesian error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Auto-Analysis Overview ───────────────────────

async function runOverview() {
    try {
        var well = document.getElementById("well-select").value || null;
        var regime = document.getElementById("regime-select").value;
        var depth = parseFloat(document.getElementById("depth-input").value) || 3000;

        var r = await api("/api/analysis/overview", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, well: well, regime: regime, depth: depth})
        });

        var panel = document.getElementById("overview-panel");
        panel.classList.remove("d-none");

        // SHmax
        var stress = r.stress || {};
        val("ov-shmax", stress.shmax ? stress.shmax + "\u00B0" : "N/A");

        // Regime (auto-detected)
        var regimeEl = document.getElementById("ov-regime");
        if (r.regime_detection) {
            var rd = r.regime_detection;
            regimeEl.textContent = rd.best_regime.replace("_", "-");
            var confColor = rd.confidence === "HIGH" ? "text-success" : rd.confidence === "MODERATE" ? "text-warning" : "text-danger";
            regimeEl.className = "metric-value " + confColor;
            regimeEl.title = rd.confidence + " confidence (misfit ratio: " + rd.misfit_ratio + ")";
        } else {
            val("ov-regime", stress.regime ? stress.regime.replace("_", "-") : "N/A");
        }

        // Critically stressed
        var cs = r.critically_stressed || {};
        val("ov-cs", cs.pct != null ? cs.pct + "%" : "N/A");
        // Color-code CS% based on risk thresholds
        var csEl = document.getElementById("ov-cs");
        if (csEl && cs.pct != null) {
            csEl.className = "metric-value " + (cs.pct < 10 ? "text-success" : cs.pct <= 30 ? "text-warning" : "text-danger");
        }

        // Risk
        var risk = r.risk || {};
        var riskEl = document.getElementById("ov-risk");
        riskEl.textContent = risk.level || "N/A";
        riskEl.className = "metric-value text-" + (risk.level === "LOW" ? "success" : risk.level === "MODERATE" ? "warning" : "danger");

        // Quality
        var dq = r.data_quality || {};
        val("ov-quality", dq.grade || "N/A");

        // Go/No-Go
        var gonogoEl = document.getElementById("ov-gonogo");
        gonogoEl.textContent = risk.go_nogo || "N/A";
        gonogoEl.className = "metric-value " + (risk.go_nogo === "GO" ? "text-success" : risk.go_nogo === "CONDITIONAL" ? "text-warning" : "text-danger");

        // Render stakeholder brief
        if (r.stakeholder_brief) {
            renderStakeholderBrief("ov-brief", r.stakeholder_brief, "ov-brief-detail");
        }

        // Warnings and disclaimers
        var warnings = document.getElementById("ov-warnings");
        clearChildren(warnings);
        var warningsList = [];

        // Sample size warning
        if (r.n_fractures < 50) {
            warningsList.push({
                type: "danger",
                icon: "exclamation-triangle",
                text: "Very small sample size (" + r.n_fractures + " fractures). Results have high uncertainty and should NOT be used for operational decisions without additional data."
            });
        } else if (r.n_fractures < 200) {
            warningsList.push({
                type: "warning",
                icon: "exclamation-circle",
                text: "Limited sample size (" + r.n_fractures + " fractures). Results should be validated with additional wellbore data or regional models."
            });
        }

        // Regime confidence warning
        if (r.regime_detection && r.regime_detection.confidence === "LOW") {
            warningsList.push({
                type: "warning",
                icon: "question-circle",
                text: "Stress regime is poorly constrained (all regimes fit similarly). Use regional tectonic knowledge or independent data (breakouts, focal mechanisms) to confirm the " + r.regime_detection.best_regime.replace("_", "-") + " regime."
            });
        }

        // Data quality warning
        if (dq.score != null && dq.score < 50) {
            warningsList.push({
                type: "danger",
                icon: "shield-exclamation",
                text: "Data quality is " + dq.grade + " (score " + dq.score + "/100). Significant data issues detected. Review data quality tab before relying on results."
            });
        }

        // QC summary from WSM/EAGE standards
        if (r.qc_summary) {
            var qc = r.qc_summary;
            var qcType = qc.pass_rate_pct >= 80 ? "success" : qc.pass_rate_pct >= 50 ? "warning" : "danger";
            var qcIcon = qcType === "success" ? "check-circle" : qcType === "warning" ? "exclamation-triangle" : "x-circle";
            var qcText = "Fracture QC (WSM/EAGE): " + qc.passed + "/" + qc.total + " pass (" + qc.pass_rate_pct + "%).";
            if (qc.top_flags && qc.top_flags.length > 0) {
                qcText += " Issues: " + qc.top_flags.join("; ") + ".";
            }
            if (qc.pass_rate_pct < 50) {
                qcText += " Consider reviewing data quality before relying on results.";
            }
            warningsList.push({type: qcType === "success" ? "info" : qcType, icon: qcIcon, text: qcText});
        }

        // High risk warning
        if (risk.level === "HIGH" || risk.go_nogo === "NO-GO") {
            warningsList.push({
                type: "danger",
                icon: "sign-stop",
                text: "HIGH RISK assessment. " + (cs.pct || 0) + "% of fractures are critically stressed. Operations near these fractures may trigger fault reactivation or induced seismicity."
            });
        }

        // Calibration info
        if (r.calibration) {
            var calColor = {"EXCELLENT": "success", "GOOD": "info", "FAIR": "warning", "POOR": "danger"}[r.calibration.reliability] || "secondary";
            warningsList.push({
                type: calColor === "danger" ? "danger" : calColor === "warning" ? "warning" : "info",
                icon: "bullseye",
                text: "Model calibration: " + r.calibration.reliability +
                    " (ECE=" + (r.calibration.ece * 100).toFixed(1) + "%). " +
                    (r.calibration.reliability === "POOR"
                        ? "DO NOT rely on confidence percentages for decisions."
                        : "Probability estimates are trustworthy.")
            });
        }

        // Data recommendations summary
        if (r.data_recommendations && r.data_recommendations.n_priority > 0) {
            warningsList.push({
                type: "warning",
                icon: "lightbulb",
                text: r.data_recommendations.n_priority + " critical data gap(s) detected. " +
                    "Visit the Calibration tab > Data Recommendations for specific actions to improve accuracy. " +
                    "Dataset completeness: " + r.data_recommendations.completeness_pct + "%."
            });
        }

        // General operational disclaimer
        if (warningsList.length > 0 || (risk.level !== "LOW")) {
            warningsList.push({
                type: "info",
                icon: "info-circle",
                text: "These results are model estimates based on fracture orientation data only. For operational decisions, integrate with regional stress models, wellbore stability analysis, and local geological knowledge. All analyses should be reviewed by a qualified geomechanics engineer."
            });
        }

        warningsList.forEach(function(w) {
            var div = document.createElement("div");
            div.className = "alert alert-" + w.type + " py-2 mb-2 small";
            div.innerHTML = '<i class="bi bi-' + w.icon + ' me-1"></i>' + w.text;
            warnings.appendChild(div);
        });

    } catch (err) {
        // Overview is not critical, don't show error toast
        console.warn("Overview failed:", err.message);
    }
}


// ── Calibration Assessment ────────────────────────

async function runCalibration() {
    showLoading("Assessing model calibration...");
    try {
        var r = await api("/api/analysis/calibration", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource, fast: true })
        });

        document.getElementById("calibration-results").classList.remove("d-none");

        // Metrics row
        var metrics = document.getElementById("calibration-metrics");
        var reliabilityColor = {
            "EXCELLENT": "success", "GOOD": "primary",
            "FAIR": "warning", "POOR": "danger"
        }[r.reliability] || "secondary";

        metrics.innerHTML =
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Reliability</div>' +
            '<div class="metric-value"><span class="badge bg-' + reliabilityColor + '">' + r.reliability + '</span></div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">ECE</div>' +
            '<div class="metric-value">' + (r.ece * 100).toFixed(1) + '%</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Brier Score</div>' +
            '<div class="metric-value">' + r.brier_score.toFixed(3) + '</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Samples</div>' +
            '<div class="metric-value">' + r.n_samples + '</div></div></div>';

        // Reliability message
        metrics.innerHTML += '<div class="col-12 mt-2"><div class="alert alert-' + reliabilityColor + ' py-2 small">' +
            '<i class="bi bi-info-circle me-1"></i>' + r.reliability_message + '</div></div>';

        // Confidence bins table
        var binsEl = document.getElementById("calibration-bins");
        if (r.confidence_bins && r.confidence_bins.length > 0) {
            var html = '<table class="table table-sm table-hover"><thead><tr>' +
                '<th>Confidence Range</th><th>Samples</th><th>Actual Accuracy</th><th>Avg Confidence</th><th>Gap</th></tr></thead><tbody>';
            r.confidence_bins.forEach(function(b) {
                var gapColor = b.gap > 0.15 ? "text-danger" : b.gap > 0.05 ? "text-warning" : "text-success";
                html += '<tr><td>' + b.range + '</td><td>' + b.n_samples + '</td>' +
                    '<td>' + (b.accuracy * 100).toFixed(1) + '%</td>' +
                    '<td>' + (b.avg_confidence * 100).toFixed(1) + '%</td>' +
                    '<td class="' + gapColor + '">' + (b.gap * 100).toFixed(1) + '%</td></tr>';
            });
            html += '</tbody></table>';
            binsEl.innerHTML = html;
        }

        // Per-class calibration
        var perClassEl = document.getElementById("calibration-per-class");
        if (r.per_class) {
            var pcHtml = '<table class="table table-sm"><thead><tr>' +
                '<th>Fracture Type</th><th>Brier Score</th><th>Samples</th></tr></thead><tbody>';
            Object.keys(r.per_class).forEach(function(cname) {
                var c = r.per_class[cname];
                pcHtml += '<tr><td>' + cname + '</td>' +
                    '<td>' + (c.brier_score !== null ? c.brier_score.toFixed(4) : 'N/A') + '</td>' +
                    '<td>' + c.n_samples + '</td></tr>';
            });
            pcHtml += '</tbody></table>';
            perClassEl.innerHTML = pcHtml;
        }

        showToast("Calibration: " + r.reliability + " (ECE=" + (r.ece * 100).toFixed(1) + "%)");
    } catch (err) {
        showToast("Calibration error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


async function runOODCheck() {
    showLoading("Running distribution check...");
    try {
        var r = await api("/api/analysis/ood-check", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource })
        });

        document.getElementById("ood-results").classList.remove("d-none");
        var card = document.getElementById("ood-card");
        var sevColor = {"HIGH": "danger", "MODERATE": "warning", "LOW": "success"}[r.severity] || "secondary";

        var html = '<div class="card-header bg-' + sevColor + ' text-white">' +
            '<i class="bi bi-exclamation-triangle me-1"></i> Distribution Check: ' + r.severity + '</div>' +
            '<div class="card-body">' +
            '<p>' + r.message + '</p>';

        if (r.note) {
            html += '<p class="text-muted small">' + r.note + '</p>';
        }

        // Detection metrics
        html += '<div class="row">';
        if (r.mahalanobis_pct_ood !== null && r.mahalanobis_pct_ood !== undefined) {
            html += '<div class="col-md-4"><div class="metric-card"><div class="metric-label">Mahalanobis OOD %</div>' +
                '<div class="metric-value">' + r.mahalanobis_pct_ood + '%</div></div></div>';
        }
        if (r.isolation_forest_pct_outlier !== null && r.isolation_forest_pct_outlier !== undefined) {
            html += '<div class="col-md-4"><div class="metric-card"><div class="metric-label">Isolation Forest Outlier %</div>' +
                '<div class="metric-value">' + r.isolation_forest_pct_outlier + '%</div></div></div>';
        }
        if (r.drift_features && r.drift_features.length > 0) {
            html += '<div class="col-md-4"><div class="metric-card"><div class="metric-label">Drifting Features</div>' +
                '<div class="metric-value">' + r.drift_features.length + '</div></div></div>';
        }
        html += '</div>';

        // Drift features table
        if (r.drift_features && r.drift_features.length > 0) {
            html += '<h6 class="mt-3">Feature Drift Details</h6>' +
                '<table class="table table-sm"><thead><tr><th>Feature</th><th>Shift (sigma)</th></tr></thead><tbody>';
            r.drift_features.forEach(function(f) {
                html += '<tr><td>' + f.feature + '</td><td>' + f.shift_sigma + ' SD</td></tr>';
            });
            html += '</tbody></table>';
        }

        // Range warnings
        if (r.range_warnings && r.range_warnings.length > 0) {
            html += '<h6>Range Warnings</h6>';
            r.range_warnings.forEach(function(w) {
                html += '<div class="alert alert-warning py-1 small"><strong>' + w.column + ':</strong> ' +
                    'Reference: ' + w.ref_range + ', New: ' + w.new_range + '</div>';
            });
        }

        html += '</div>';
        card.innerHTML = html;

        showToast("OOD Check: " + r.severity);
    } catch (err) {
        showToast("OOD check error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


async function runDataRecommendations() {
    showLoading("Generating data collection recommendations...");
    try {
        var r = await api("/api/data/recommendations", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: currentSource })
        });

        document.getElementById("data-rec-results").classList.remove("d-none");
        var content = document.getElementById("data-rec-content");

        var html = '<div class="card mb-3 border-info">' +
            '<div class="card-header bg-info text-white"><i class="bi bi-lightbulb me-1"></i> Data Collection Roadmap</div>' +
            '<div class="card-body">' +
            '<p class="text-muted small">' + r.summary + '</p>';

        // Progress bar
        html += '<div class="mb-3"><strong>Dataset Completeness</strong>' +
            '<div class="progress mt-1" style="height:20px"><div class="progress-bar ' +
            (r.data_completeness_pct >= 100 ? 'bg-success' : r.data_completeness_pct >= 50 ? 'bg-warning' : 'bg-danger') +
            '" style="width:' + r.data_completeness_pct + '%">' + r.data_completeness_pct + '%</div></div></div>';

        // Priority actions
        if (r.priority_actions && r.priority_actions.length > 0) {
            html += '<h6 class="text-danger"><i class="bi bi-exclamation-circle me-1"></i>Priority Actions</h6>';
            r.priority_actions.forEach(function(a) {
                html += '<div class="alert alert-danger py-2 mb-2"><strong>' + a.action + '</strong>' +
                    '<br><small>' + a.reason + '</small>';
                if (a.expected_impact) {
                    html += '<br><small class="text-success"><i class="bi bi-graph-up me-1"></i>' + a.expected_impact + '</small>';
                }
                html += '</div>';
            });
        }

        // Recommendations
        if (r.recommendations && r.recommendations.length > 0) {
            html += '<h6 class="text-warning mt-3"><i class="bi bi-arrow-right-circle me-1"></i>Recommendations</h6>';
            r.recommendations.forEach(function(rec) {
                var color = rec.priority === "MODERATE" ? "warning" : "info";
                html += '<div class="alert alert-' + color + ' py-2 mb-2"><strong>' + rec.action + '</strong>' +
                    '<br><small>' + rec.reason + '</small></div>';
            });
        }

        // Minimum viable dataset
        if (r.min_viable_dataset && r.current_meets_minimum) {
            html += '<h6 class="mt-3"><i class="bi bi-check2-square me-1"></i>Minimum Viable Dataset</h6>' +
                '<table class="table table-sm"><thead><tr><th>Criterion</th><th>Required</th><th>Met?</th></tr></thead><tbody>';
            var mvd = r.min_viable_dataset;
            var cm = r.current_meets_minimum;
            var criteria = [
                ["Total Samples", ">= " + mvd.min_total_samples, cm.total_samples],
                ["Per-Class Minimum", ">= " + mvd.min_per_class, cm.per_class],
                ["Number of Wells", ">= " + mvd.min_wells, cm.wells],
            ];
            criteria.forEach(function(c) {
                html += '<tr><td>' + c[0] + '</td><td>' + c[1] + '</td>' +
                    '<td>' + (c[2] ? '<i class="bi bi-check-circle text-success"></i>' : '<i class="bi bi-x-circle text-danger"></i>') + '</td></tr>';
            });
            html += '</tbody></table>';
        }

        html += '</div></div>';
        content.innerHTML = html;

        showToast("Data recommendations generated");
    } catch (err) {
        showToast("Recommendations error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Field Calibration ─────────────────────────────

async function addFieldMeasurement() {
    var testType = document.getElementById("field-test-type").value;
    var depth = parseFloat(document.getElementById("field-depth").value);
    var stress = parseFloat(document.getElementById("field-stress").value);
    var direction = document.getElementById("field-direction").value;
    var azimuth = document.getElementById("field-azimuth").value;
    var notes = document.getElementById("field-notes").value;

    if (!depth || !stress) {
        showToast("Depth and stress magnitude are required", "Error");
        return;
    }

    var payload = {
        well: currentWell,
        test_type: testType,
        depth_m: depth,
        measured_stress_mpa: stress,
        stress_direction: direction,
        notes: notes
    };
    if (azimuth) payload.azimuth_deg = parseFloat(azimuth);

    try {
        var r = await api("/api/calibration/add-measurement", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        showToast("Measurement added (" + r.total_for_well + " total for " + currentWell + ")");
        document.getElementById("field-notes").value = "";
    } catch (err) {
        showToast("Error adding measurement: " + err.message, "Error");
    }
}

async function loadFieldMeasurements() {
    try {
        var r = await api("/api/calibration/measurements?well=" + currentWell);
        var measurements = r.measurements || [];
        if (measurements.length === 0) {
            showToast("No measurements recorded for well " + currentWell);
            return;
        }

        var resultsDiv = document.getElementById("field-validation-results");
        resultsDiv.classList.remove("d-none");

        var tableDiv = document.getElementById("field-cal-table");
        var html = '<table class="table table-sm table-striped"><thead><tr>' +
            '<th>Type</th><th>Depth (m)</th><th>Stress (MPa)</th><th>Direction</th><th>Azimuth</th><th>Notes</th>' +
            '</tr></thead><tbody>';
        measurements.forEach(function(m) {
            html += '<tr><td>' + m.test_type + '</td><td>' + m.depth_m +
                '</td><td>' + m.measured_stress_mpa + '</td><td>' + m.stress_direction +
                '</td><td>' + (m.azimuth_deg || '-') + '</td><td>' + (m.notes || '-') + '</td></tr>';
        });
        html += '</tbody></table>';
        tableDiv.innerHTML = html;
        document.getElementById("field-cal-metrics").innerHTML = '';
        document.getElementById("field-cal-recommendations").innerHTML =
            '<p class="text-muted small">' + measurements.length + ' measurement(s) recorded. Click "Validate" to compare against model predictions.</p>';

        showToast(measurements.length + " measurement(s) loaded");
    } catch (err) {
        showToast("Error loading measurements: " + err.message, "Error");
    }
}

async function runFieldValidation() {
    showLoading("Validating model against field measurements...");
    try {
        var depth = parseFloat(document.getElementById("field-depth").value) || 3000;
        var r = await api("/api/calibration/validate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                well: currentWell,
                source: currentSource,
                depth_m: depth,
                pp_mpa: 30
            })
        });

        var resultsDiv = document.getElementById("field-validation-results");
        resultsDiv.classList.remove("d-none");

        if (r.status === "no_measurements") {
            document.getElementById("field-cal-metrics").innerHTML =
                '<div class="col-12"><div class="alert alert-info">' +
                '<i class="bi bi-info-circle me-1"></i> ' + r.message +
                '</div><p class="small text-muted">' + r.recommendation + '</p></div>';
            document.getElementById("field-cal-table").innerHTML = '';
            document.getElementById("field-cal-recommendations").innerHTML = '';
            showToast("No field measurements — add some first");
            return;
        }

        // Metrics
        var ratingColor = r.overall_rating === "CALIBRATED" ? "success" :
                         r.overall_rating === "ACCEPTABLE" ? "info" :
                         r.overall_rating === "NEEDS_RECALIBRATION" ? "warning" : "danger";
        var metricsHtml = '<div class="col-md-3"><div class="metric-card">' +
            '<div class="metric-label">Calibration Score</div>' +
            '<div class="metric-value">' + r.calibration_score + '/100</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card">' +
            '<div class="metric-label">Rating</div>' +
            '<div class="metric-value text-' + ratingColor + '">' + r.overall_rating + '</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card">' +
            '<div class="metric-label">Avg Stress Error</div>' +
            '<div class="metric-value">' + r.avg_stress_error_pct + '%</div></div></div>' +
            '<div class="col-md-3"><div class="metric-card">' +
            '<div class="metric-label">Measurements</div>' +
            '<div class="metric-value">' + r.n_measurements + '</div></div></div>';
        document.getElementById("field-cal-metrics").innerHTML = metricsHtml;

        // Comparison table
        var tableHtml = '<table class="table table-sm"><thead><tr>' +
            '<th>Test</th><th>Depth</th><th>Direction</th>' +
            '<th>Measured (MPa)</th><th>Predicted (MPa)</th><th>Error</th><th>Rating</th>' +
            '</tr></thead><tbody>';
        r.comparisons.forEach(function(c) {
            var rowClass = c.rating === "EXCELLENT" ? "table-success" :
                          c.rating === "GOOD" ? "" :
                          c.rating === "FAIR" ? "table-warning" : "table-danger";
            tableHtml += '<tr class="' + rowClass + '"><td>' + c.test_type + '</td>' +
                '<td>' + c.depth_m + 'm</td><td>' + c.stress_direction + '</td>' +
                '<td>' + c.measured_mpa + '</td><td>' + (c.predicted_mpa || '-') + '</td>' +
                '<td>' + (c.error_pct ? c.error_pct + '%' : '-') + '</td>' +
                '<td>' + (c.rating || '-') + '</td></tr>';
        });
        tableHtml += '</tbody></table>';

        // Model predictions summary
        var mp = r.model_predictions;
        tableHtml += '<div class="mt-2 small text-muted">Model: regime=' + r.regime +
            ', sigma1=' + mp.sigma1_mpa + 'MPa, sigma3=' + mp.sigma3_mpa +
            'MPa, SHmax=' + mp.shmax_azimuth_deg + '°, Sv=' + mp.sv_mpa + 'MPa</div>';

        document.getElementById("field-cal-table").innerHTML = tableHtml;

        // Recommendations
        var recHtml = '';
        r.recommendations.forEach(function(rec) {
            recHtml += '<div class="alert alert-info py-1 small"><i class="bi bi-lightbulb me-1"></i> ' + rec + '</div>';
        });
        recHtml += '<p class="small text-muted mt-2"><i class="bi bi-info-circle me-1"></i> ' +
            r.industry_context + '</p>';
        document.getElementById("field-cal-recommendations").innerHTML = recHtml;

        showToast("Calibration: " + r.overall_rating + " (score " + r.calibration_score + "/100)");
    } catch (err) {
        showToast("Validation error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Learning Curve ────────────────────────────────

async function runLearningCurve() {
    showLoading("Computing learning curve...");
    try {
        var data = await api("/api/analysis/learning-curve", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ source: currentSource, well: currentWell })
        });

        if (data.error) {
            showToast(data.error, "Error");
            return;
        }

        document.getElementById("lc-results").classList.remove("d-none");

        // Summary metrics
        var convColors = { PLATEAU: "text-warning", SLOWING: "text-info", GROWING: "text-success", INSUFFICIENT: "text-muted" };
        var convEl = document.getElementById("lc-convergence");
        convEl.textContent = data.convergence;
        convEl.className = "metric-value " + (convColors[data.convergence] || "");

        var lastAcc = data.val_scores[data.val_scores.length - 1];
        val("lc-current-acc", (lastAcc * 100).toFixed(1) + "%");
        val("lc-n-samples", data.n_samples);

        document.getElementById("lc-convergence-msg").textContent = data.convergence_message;

        // Accuracy table
        var tbody = document.querySelector("#lc-table tbody");
        clearChildren(tbody);
        for (var i = 0; i < data.train_sizes.length; i++) {
            var tr = document.createElement("tr");
            tr.appendChild(createCell("td", data.train_sizes[i]));
            tr.appendChild(createCell("td", (data.train_scores[i] * 100).toFixed(1) + "%"));
            tr.appendChild(createCell("td", (data.val_scores[i] * 100).toFixed(1) + "%"));
            tr.appendChild(createCell("td", (data.balanced_scores[i] * 100).toFixed(1) + "%"));
            tbody.appendChild(tr);
        }

        // Projections
        if (data.projection && data.projection.available && data.projection.targets) {
            document.getElementById("lc-projections").classList.remove("d-none");
            var projList = document.getElementById("lc-projection-list");
            clearChildren(projList);
            data.projection.targets.forEach(function(t) {
                var div = document.createElement("div");
                div.className = "alert py-2 mb-2 small " +
                    (t.status === "ACHIEVED" ? "alert-success" :
                     t.status === "PROJECTED" ? "alert-info" : "alert-warning");
                var text = (t.target_accuracy * 100) + "% accuracy: ";
                if (t.status === "ACHIEVED") {
                    text += "Already achieved!";
                } else if (t.status === "PROJECTED") {
                    text += "~" + t.samples_needed + " total samples needed (" + t.additional_needed + " more)";
                } else {
                    text += t.reason || t.status;
                }
                div.textContent = text;
                projList.appendChild(div);
            });
        }

        // Per-class table
        if (data.per_class && data.class_names) {
            document.getElementById("lc-per-class").classList.remove("d-none");
            var header = document.getElementById("lc-class-header");
            clearChildren(header);
            header.appendChild(createCell("th", "Samples"));
            data.class_names.forEach(function(cls) {
                header.appendChild(createCell("th", cls));
            });

            var classBody = document.getElementById("lc-class-body");
            clearChildren(classBody);
            for (var j = 0; j < data.train_sizes.length; j++) {
                var row = document.createElement("tr");
                row.appendChild(createCell("td", data.train_sizes[j]));
                data.class_names.forEach(function(cls) {
                    var v = data.per_class[cls][j];
                    row.appendChild(createCell("td", v != null ? (v * 100).toFixed(1) + "%" : "N/A"));
                });
                classBody.appendChild(row);
            }
        }

        // Show chart image
        if (data.chart_img) {
            setImg("lc-chart-img", data.chart_img);
            document.getElementById("lc-chart-container").classList.remove("d-none");
        }

        showToast("Learning curve computed in " + (data.elapsed_s || "?") + "s");
    } catch (err) {
        showToast("Learning curve error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Bootstrap Confidence Intervals ────────────────

async function runBootstrapCI() {
    showLoading("Computing bootstrap CIs (200 resamples)...");
    try {
        var data = await api("/api/analysis/bootstrap-ci", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ source: currentSource, well: currentWell })
        });

        if (data.error) {
            showToast(data.error, "Error");
            return;
        }

        document.getElementById("ci-results").classList.remove("d-none");

        var relColors = { HIGH: "text-success", MODERATE: "text-warning", LOW: "text-danger" };
        var relEl = document.getElementById("ci-reliability");
        relEl.textContent = data.reliability;
        relEl.className = "metric-value " + (relColors[data.reliability] || "");

        val("ci-accuracy", (data.accuracy.mean * 100).toFixed(1) + "%");
        val("ci-accuracy-range", "[" + (data.accuracy.ci_low * 100).toFixed(1) + "% - " + (data.accuracy.ci_high * 100).toFixed(1) + "%]");
        val("ci-n-boot", data.n_bootstrap);

        var msgEl = document.getElementById("ci-reliability-msg");
        msgEl.textContent = data.reliability_message;
        msgEl.className = "alert mb-3 small " +
            (data.reliability === "HIGH" ? "alert-success" :
             data.reliability === "MODERATE" ? "alert-warning" : "alert-danger");

        // Per-class table
        var tbody = document.querySelector("#ci-class-table tbody");
        clearChildren(tbody);
        data.class_names.forEach(function(cls) {
            var pc = data.per_class[cls];
            var tr = document.createElement("tr");
            tr.appendChild(createCell("td", cls, { fontWeight: "600" }));

            function ciCell(metric) {
                if (!metric) return createCell("td", "N/A");
                return createCell("td", (metric.mean * 100).toFixed(1) + "%");
            }
            function ciRangeCell(metric) {
                if (!metric) return createCell("td", "N/A");
                var width = metric.width;
                var style = { fontSize: "0.8rem" };
                if (width > 0.3) style.color = "#dc2626";
                else if (width > 0.15) style.color = "#d97706";
                else style.color = "#16a34a";
                return createCell("td", "[" + (metric.ci_low * 100).toFixed(1) + " - " + (metric.ci_high * 100).toFixed(1) + "]", style);
            }

            tr.appendChild(ciCell(pc.f1));
            tr.appendChild(ciRangeCell(pc.f1));
            tr.appendChild(ciCell(pc.precision));
            tr.appendChild(ciRangeCell(pc.precision));
            tr.appendChild(ciCell(pc.recall));
            tr.appendChild(ciRangeCell(pc.recall));
            tbody.appendChild(tr);
        });

        // Show CI chart
        if (data.chart_img) {
            setImg("ci-chart-img", data.chart_img);
            document.getElementById("ci-chart-container").classList.remove("d-none");
        }

        showToast("Bootstrap CIs computed (" + data.n_bootstrap + " resamples, " + (data.elapsed_s || "?") + "s)");
    } catch (err) {
        showToast("Bootstrap CI error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Decision Support Matrix ───────────────────────

async function runDecisionMatrix() {
    showLoading("Generating decision matrix (all regimes)...");
    try {
        var depth = document.getElementById("depth-input").value || 3000;
        var data = await api("/api/analysis/decision-matrix", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                source: currentSource,
                well: currentWell,
                depth: parseFloat(depth)
            })
        });

        document.getElementById("dm-results").classList.remove("d-none");

        // Recommended action
        document.getElementById("dm-action-text").textContent = data.recommended_action;

        // Confidence
        var conf = data.confidence || {};
        var confColors = { HIGH: "text-success", MODERATE: "text-warning", LOW: "text-danger", UNKNOWN: "text-muted" };
        var confEl = document.getElementById("dm-confidence");
        confEl.textContent = conf.overall || "UNKNOWN";
        confEl.className = "metric-value " + (confColors[conf.overall] || "");
        val("dm-quality", "Grade " + (conf.data_quality || "?"));
        val("dm-regime-cert", conf.regime_certainty || "?");
        val("dm-n-frac", conf.n_fractures || 0);

        // Options table
        var tbody = document.querySelector("#dm-options-table tbody");
        clearChildren(tbody);
        (data.options || []).forEach(function(o) {
            var tr = document.createElement("tr");
            if (o.status === "ERROR") {
                tr.appendChild(createCell("td", o.regime, { fontWeight: "600" }));
                var errTd = document.createElement("td");
                errTd.colSpan = 7;
                errTd.textContent = "Error: " + o.error;
                errTd.style.color = "#dc2626";
                tr.appendChild(errTd);
            } else {
                tr.appendChild(createCell("td", o.regime_label || o.regime, { fontWeight: "600" }));
                tr.appendChild(createCell("td", fmt(o.sigma1_mpa, 1)));
                tr.appendChild(createCell("td", fmt(o.sigma3_mpa, 1)));
                tr.appendChild(createCell("td", fmt(o.shmax_deg, 0) + "\u00B0"));
                tr.appendChild(createCell("td", fmt(o.misfit, 4)));

                var csStyle = {};
                if (o.critically_stressed_pct > 50) csStyle.color = "#dc2626";
                else if (o.critically_stressed_pct > 25) csStyle.color = "#d97706";
                else csStyle.color = "#16a34a";
                tr.appendChild(createCell("td", fmt(o.critically_stressed_pct, 1) + "%", csStyle));

                var riskColors = { HIGH: "#dc2626", MODERATE: "#d97706", LOW: "#16a34a" };
                tr.appendChild(createCell("td", o.risk_level, { color: riskColors[o.risk_level] || "#000", fontWeight: "600" }));

                var goStyle = o.go_nogo && o.go_nogo.includes("NO-GO") ? { color: "#dc2626", fontWeight: "700" } :
                              o.go_nogo && o.go_nogo.includes("CAUTION") ? { color: "#d97706", fontWeight: "600" } :
                              { color: "#16a34a", fontWeight: "600" };
                tr.appendChild(createCell("td", o.go_nogo || "?", goStyle));
            }
            tbody.appendChild(tr);
        });

        // Risk comparison
        var rc = data.risk_comparison || {};
        if (rc.safest || rc.best_fit) {
            document.getElementById("dm-risk-comparison").classList.remove("d-none");
            val("dm-safest", rc.safest || "N/A");
            val("dm-best-fit", rc.best_fit || "N/A");
            val("dm-conservative", rc.most_conservative || "N/A");
        }

        // Trade-offs
        if (data.trade_offs && data.trade_offs.length > 0) {
            document.getElementById("dm-tradeoffs").classList.remove("d-none");
            var tfList = document.getElementById("dm-tradeoff-list");
            clearChildren(tfList);
            data.trade_offs.forEach(function(tf) {
                var color = tf.impact === "HIGH" ? "danger" : "warning";
                var div = document.createElement("div");
                div.className = "alert alert-" + color + " py-2 mb-2";
                div.innerHTML = '<strong>' + tf.factor + '</strong> (' + tf.range + ')<br>' +
                    '<small>' + tf.implication + '</small>';
                tfList.appendChild(div);
            });
        }

        showToast("Decision matrix generated (" + (data.elapsed_s || "?") + "s)");
    } catch (err) {
        showToast("Decision matrix error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Depth-Zone Classification ─────────────────────

async function runDepthZoneClassify() {
    showLoading("Running depth-zone classification...");
    try {
        var r = await apiPost("/api/analysis/depth-zone", {
            source: currentSource, well: getWell(), n_zones: 3, classifier: "random_forest"
        });
        var el = document.getElementById("depthzone-results");
        el.classList.remove("d-none");

        if (r.error) {
            el.innerHTML = '<div class="alert alert-warning">' + r.error + '</div>';
            return;
        }

        var rec = r.recommendation || {};
        var vColors = {
            ZONE_MODELS_BETTER: "success", SIMILAR_PERFORMANCE: "info", GLOBAL_MODEL_BETTER: "warning"
        };
        var html = '<div class="alert alert-' + (vColors[rec.verdict] || "secondary") + ' p-3 mb-3">' +
            '<h6>' + (rec.verdict || "").replace(/_/g, " ") + '</h6>' +
            '<p class="mb-0">' + (rec.message || "") + '</p></div>';

        // Accuracy comparison
        html += '<div class="row g-2 mb-3">';
        html += '<div class="col-md-4"><div class="card text-center"><div class="card-body py-2">' +
            '<h4>' + (r.global_accuracy * 100).toFixed(1) + '%</h4>' +
            '<small class="text-muted">Global Model</small></div></div></div>';
        html += '<div class="col-md-4"><div class="card text-center border-primary"><div class="card-body py-2">' +
            '<h4 class="text-primary">' + (r.weighted_zone_accuracy * 100).toFixed(1) + '%</h4>' +
            '<small class="text-muted">Zone-Weighted</small></div></div></div>';
        var impColor = r.improvement > 0 ? "success" : "warning";
        html += '<div class="col-md-4"><div class="card text-center border-' + impColor + '"><div class="card-body py-2">' +
            '<h4 class="text-' + impColor + '">' + (r.improvement > 0 ? '+' : '') +
            (r.improvement * 100).toFixed(1) + '%</h4>' +
            '<small class="text-muted">Improvement</small></div></div></div>';
        html += '</div>';

        // Per-zone table
        if (r.zones && r.zones.length > 0) {
            html += '<h6><i class="bi bi-layers"></i> Depth Zones (' + r.n_zones + ')</h6>';
            html += '<table class="table table-sm"><thead><tr>' +
                '<th>Zone</th><th>Depth Range</th><th>Samples</th><th>Classes</th><th>Accuracy</th></tr></thead><tbody>';
            r.zones.forEach(function(z) {
                var acc = z.accuracy !== null ? (z.accuracy * 100).toFixed(1) + '%' : (z.note || 'N/A');
                html += '<tr><td>' + z.zone + '</td>' +
                    '<td>' + z.depth_range_m[0] + '–' + z.depth_range_m[1] + ' m</td>' +
                    '<td>' + z.n_samples + '</td>' +
                    '<td>' + (z.n_classes || '-') + '</td>' +
                    '<td>' + acc + '</td></tr>';
            });
            html += '</tbody></table>';
        }

        el.innerHTML = html;
        showToast("Depth-zone: " + (r.improvement > 0 ? '+' : '') + (r.improvement * 100).toFixed(1) + "% vs global");
    } catch (err) {
        showToast("Depth-zone error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Hierarchical Classification ───────────────────

async function runHierarchical() {
    showLoading("Running hierarchical classification...");
    try {
        var data = await api("/api/analysis/hierarchical", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ source: currentSource })
        });

        if (!data.applicable) {
            showToast(data.reason || "Not applicable", "Info");
            return;
        }

        document.getElementById("hier-results").classList.remove("d-none");

        var rec = data.recommendation || {};
        var recEl = document.getElementById("hier-recommendation");
        var recColor = rec.approach === "HIERARCHICAL RECOMMENDED" ? "alert-success" :
                       rec.approach === "FLAT RECOMMENDED" ? "alert-info" : "alert-warning";
        recEl.className = "alert mb-3 " + recColor;
        recEl.innerHTML = '<strong>' + rec.approach + ':</strong> ' + rec.message;

        val("hier-approach", rec.approach);
        var comp = data.comparison || {};
        val("hier-flat-bal", (comp.flat_balanced * 100).toFixed(1) + "%");
        val("hier-hier-bal", (comp.hierarchical_balanced * 100).toFixed(1) + "%");
        val("hier-l1-acc", ((data.level1 || {}).accuracy * 100).toFixed(1) + "%");

        // Per-class table
        var tbody = document.querySelector("#hier-class-table tbody");
        clearChildren(tbody);
        if (data.per_class) {
            for (var cls in data.per_class) {
                var pc = data.per_class[cls];
                var tr = document.createElement("tr");
                tr.appendChild(createCell("td", cls, { fontWeight: "600" }));
                tr.appendChild(createCell("td", pc.n_samples));
                var typeStyle = pc.is_rare ? { color: "#dc2626", fontWeight: "600" } : {};
                tr.appendChild(createCell("td", pc.is_rare ? "RARE" : "Common", typeStyle));
                tr.appendChild(createCell("td", (pc.flat_f1 * 100).toFixed(1) + "%"));
                tr.appendChild(createCell("td", (pc.hierarchical_f1 * 100).toFixed(1) + "%"));
                var impStyle = {};
                if (pc.improvement > 0.05) impStyle.color = "#16a34a";
                else if (pc.improvement < -0.05) impStyle.color = "#dc2626";
                var impText = (pc.improvement >= 0 ? "+" : "") + (pc.improvement * 100).toFixed(1) + "%";
                tr.appendChild(createCell("td", impText, impStyle));
                tbody.appendChild(tr);
            }
        }

        showToast("Hierarchical classification completed (" + (data.elapsed_s || "?") + "s)");
    } catch (err) {
        showToast("Hierarchical error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Scenario Comparison ───────────────────────────

function addScenario() {
    var list = document.getElementById("scenario-list");
    var rows = list.querySelectorAll(".scenario-row");
    if (rows.length >= 6) {
        showToast("Maximum 6 scenarios", "Limit");
        return;
    }
    var html = '<div class="row g-2 mb-2 scenario-row">' +
        '<div class="col-md-3"><input type="text" class="form-control form-control-sm scenario-name" placeholder="Scenario name" value="Scenario ' + (rows.length + 1) + '"></div>' +
        '<div class="col-md-3"><select class="form-select form-select-sm scenario-regime">' +
        '<option value="normal">Normal</option><option value="strike_slip">Strike-Slip</option><option value="thrust">Thrust</option></select></div>' +
        '<div class="col-md-3"><input type="number" class="form-control form-control-sm scenario-pp" placeholder="Pore pressure (MPa)" value=""></div>' +
        '<div class="col-md-3"><button class="btn btn-outline-danger btn-sm" onclick="removeScenario(this)"><i class="bi bi-trash"></i></button></div></div>';
    list.insertAdjacentHTML("beforeend", html);
}

function removeScenario(btn) {
    var list = document.getElementById("scenario-list");
    if (list.querySelectorAll(".scenario-row").length <= 2) {
        showToast("Need at least 2 scenarios", "Minimum");
        return;
    }
    btn.closest(".scenario-row").remove();
}

async function runScenarios() {
    var rows = document.querySelectorAll("#scenario-list .scenario-row");
    var scenarios = [];
    rows.forEach(function(row) {
        var name = row.querySelector(".scenario-name").value || "Unnamed";
        var regime = row.querySelector(".scenario-regime").value;
        var ppVal = row.querySelector(".scenario-pp").value;
        var s = { name: name, regime: regime };
        if (ppVal !== "") s.pore_pressure = parseFloat(ppVal);
        scenarios.push(s);
    });

    if (scenarios.length < 2) {
        showToast("Need at least 2 scenarios", "Error");
        return;
    }

    showLoading("Comparing " + scenarios.length + " scenarios...");
    try {
        var depth = document.getElementById("depth-input").value || 3000;
        var data = await api("/api/analysis/scenarios", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                source: currentSource,
                well: currentWell,
                depth: parseFloat(depth),
                scenarios: scenarios
            })
        });

        document.getElementById("scenario-results").classList.remove("d-none");

        // Recommendation
        document.getElementById("scenario-recommendation").innerHTML =
            '<i class="bi bi-lightbulb me-1"></i><strong>Recommendation:</strong> ' + data.recommendation;

        // Sensitivity message
        if (data.sensitivity_message) {
            document.getElementById("scenario-sensitivity-msg").textContent = data.sensitivity_message;
            document.getElementById("scenario-sensitivity-msg").classList.remove("d-none");
        }

        // Results table
        var tbody = document.querySelector("#scenario-table tbody");
        clearChildren(tbody);
        data.scenarios.forEach(function(s) {
            var tr = document.createElement("tr");
            if (s.status === "ERROR") {
                tr.appendChild(createCell("td", s.name, { fontWeight: "600" }));
                tr.appendChild(createCell("td", s.regime));
                var errCell = document.createElement("td");
                errCell.colSpan = 8;
                errCell.textContent = "Error: " + s.error;
                errCell.style.color = "#dc2626";
                tr.appendChild(errCell);
            } else {
                tr.appendChild(createCell("td", s.name, { fontWeight: "600" }));
                tr.appendChild(createCell("td", s.regime));
                tr.appendChild(createCell("td", fmt(s.sigma1, 1)));
                tr.appendChild(createCell("td", fmt(s.sigma3, 1)));
                tr.appendChild(createCell("td", fmt(s.shmax, 0) + "\u00B0"));
                tr.appendChild(createCell("td", fmt(s.R_ratio, 3)));
                tr.appendChild(createCell("td", fmt(s.mu, 3)));
                tr.appendChild(createCell("td", fmt(s.pore_pressure, 1)));
                tr.appendChild(createCell("td", fmt(s.misfit, 4)));
                var csPct = s.critically_stressed_pct;
                var csStyle = { fontWeight: "600" };
                if (csPct > 50) csStyle.color = "#dc2626";
                else if (csPct > 25) csStyle.color = "#d97706";
                else csStyle.color = "#16a34a";
                tr.appendChild(createCell("td", fmt(csPct, 1) + "%", csStyle));
            }
            tbody.appendChild(tr);
        });

        // Metrics spread
        if (data.metrics_spread && Object.keys(data.metrics_spread).length > 0) {
            document.getElementById("scenario-spread").classList.remove("d-none");
            var sTbody = document.querySelector("#scenario-spread-table tbody");
            clearChildren(sTbody);
            var paramLabels = {
                sigma1: "\u03C31 (MPa)", sigma3: "\u03C33 (MPa)", shmax: "SHmax (\u00B0)",
                R_ratio: "R ratio", mu: "Friction \u03BC", critically_stressed_pct: "Crit. Stressed %"
            };
            for (var key in data.metrics_spread) {
                var m = data.metrics_spread[key];
                var sr = document.createElement("tr");
                sr.appendChild(createCell("td", paramLabels[key] || key, { fontWeight: "600" }));
                sr.appendChild(createCell("td", fmt(m.min, 2)));
                sr.appendChild(createCell("td", fmt(m.max, 2)));
                sr.appendChild(createCell("td", fmt(m.range, 2)));
                var cvStyle = {};
                if (m.cv_pct > 20) cvStyle.color = "#dc2626";
                else if (m.cv_pct > 10) cvStyle.color = "#d97706";
                sr.appendChild(createCell("td", fmt(m.cv_pct, 1) + "%", cvStyle));
                sTbody.appendChild(sr);
            }
        }

        showToast("Compared " + data.n_successful + "/" + data.n_scenarios + " scenarios in " + (data.elapsed_s || "?") + "s");
    } catch (err) {
        showToast("Scenario error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Monte Carlo Uncertainty ───────────────────────

async function runMonteCarlo() {
    showLoading("Running Monte Carlo uncertainty propagation...");
    try {
        var r = await api("/api/analysis/monte-carlo", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                source: currentSource,
                well: currentWell || null,
                regime: document.getElementById("regime-select").value === "auto" ? "normal" : document.getElementById("regime-select").value,
                depth: parseFloat(document.getElementById("depth-input").value) || 3000,
                n_simulations: parseInt(document.getElementById("mc-nsims").value) || 200,
                azimuth_std: parseFloat(document.getElementById("mc-az-std").value) || 5,
                dip_std: parseFloat(document.getElementById("mc-dip-std").value) || 3,
                depth_std: parseFloat(document.getElementById("mc-dep-std").value) || 2,
            })
        });

        if (r.error) { showToast(r.error, "Error"); return; }

        document.getElementById("mc-results").classList.remove("d-none");

        // Reliability
        var relColors = {HIGH: "success", MODERATE: "warning", LOW: "danger"};
        var relEl = document.getElementById("mc-reliability");
        relEl.className = "alert alert-" + (relColors[r.reliability] || "secondary") + " mb-3";
        relEl.innerHTML = '<strong>' + r.reliability + ' RELIABILITY</strong> — ' + r.reliability_message +
            '<br><small class="text-muted">' + r.n_successful + '/' + r.n_simulations + ' simulations completed</small>';

        // Stats table
        var tbody = document.getElementById("mc-stats-body");
        clearChildren(tbody);
        var paramLabels = {shmax: "SHmax (°)", sigma1: "σ1 (MPa)", sigma3: "σ3 (MPa)", R: "R ratio", mu: "Friction μ", cs_pct: "Critically Stressed %", misfit: "Misfit"};
        var stats = r.statistics || {};
        Object.keys(stats).forEach(function(key) {
            var s = stats[key];
            var tr = document.createElement("tr");
            tr.innerHTML = '<td class="fw-semibold">' + (paramLabels[key] || key) + '</td>' +
                '<td>' + s.mean + '</td><td>' + s.median + '</td><td>' + s.std + '</td>' +
                '<td>' + s.ci_lower + '</td><td>' + s.ci_upper + '</td>' +
                '<td><span class="badge bg-' + (s.ci_width < 20 ? 'success' : s.ci_width < 45 ? 'warning' : 'danger') + '">' + s.ci_width + '</span></td>';
            tbody.appendChild(tr);
        });

        // Sensitivity ranking
        var sensBody = document.getElementById("mc-sensitivity-body");
        clearChildren(sensBody);
        if (r.sensitivity_ranking && r.sensitivity_ranking.length > 0) {
            var maxStd = Math.max.apply(null, r.sensitivity_ranking.map(function(s) { return s.shmax_std; }));
            r.sensitivity_ranking.forEach(function(s, i) {
                var pct = maxStd > 0 ? (s.shmax_std / maxStd * 100) : 0;
                var div = document.createElement("div");
                div.className = "mb-2";
                div.innerHTML = '<div class="d-flex justify-content-between small"><span>' + (i + 1) + '. ' + s.label + '</span>' +
                    '<span>SHmax σ = ' + s.shmax_std + '°, range = ' + s.shmax_range + '°</span></div>' +
                    '<div class="progress" style="height:8px"><div class="progress-bar bg-' + (i === 0 ? 'danger' : 'warning') +
                    '" style="width:' + pct + '%"></div></div>';
                sensBody.appendChild(div);
            });
        }

        showToast("Monte Carlo: " + r.reliability + " reliability, SHmax ±" + (stats.shmax ? stats.shmax.ci_width / 2 : "?") + "°");
    } catch (err) {
        showToast("Monte Carlo error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Domain Validation ────────────────────────────

async function runDomainValidation() {
    showLoading("Validating domain constraints...");
    try {
        var r = await api("/api/data/validate-constraints", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, well: currentWell || null})
        });

        document.getElementById("val-results").classList.remove("d-none");

        var statusColors = {PASS: "success", OK: "success", CAUTION: "warning", FAIL: "danger"};
        var statusEl = document.getElementById("val-status");
        statusEl.className = "alert alert-" + (statusColors[r.status] || "info") + " mb-3";
        statusEl.innerHTML = '<strong>' + r.status + '</strong> — ' + r.status_message +
            ' (' + r.n_records + ' records, ' + r.n_errors + ' errors, ' + r.n_warnings + ' warnings)';

        var issuesBody = document.getElementById("val-issues-body");
        clearChildren(issuesBody);
        if (r.issues && r.issues.length > 0) {
            r.issues.forEach(function(issue) {
                var sevColors = {ERROR: "danger", WARNING: "warning", INFO: "info"};
                var div = document.createElement("div");
                div.className = "alert alert-" + (sevColors[issue.severity] || "info") + " py-2 mb-2 small";
                div.innerHTML = '<strong>' + issue.severity + ' [' + issue.field + ']:</strong> ' + issue.message;
                issuesBody.appendChild(div);
            });
        }

        showToast("Validation: " + r.status + " — " + r.n_errors + " errors, " + r.n_warnings + " warnings");
    } catch (err) {
        showToast("Validation error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Cross-Well Validation ────────────────────────

async function runCrossWellCV() {
    showLoading("Running leave-one-well-out cross-validation...");
    try {
        var r = await api("/api/analysis/cross-well-cv", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource, classifier: "random_forest"})
        });

        if (r.error) { showToast(r.error, "Error"); return; }

        document.getElementById("cwcv-results").classList.remove("d-none");

        var tColors = {GOOD: "success", MODERATE: "warning", POOR: "danger", UNKNOWN: "secondary"};
        var transEl = document.getElementById("cwcv-transfer");
        transEl.className = "alert alert-" + (tColors[r.transferability] || "info") + " mb-3";
        transEl.innerHTML = '<strong>Transferability: ' + r.transferability + '</strong> — ' + r.transferability_message;

        // LOWO table
        var tbody = document.getElementById("cwcv-table-body");
        clearChildren(tbody);
        (r.leave_one_well_out || []).forEach(function(row) {
            var tr = document.createElement("tr");
            if (row.error) {
                tr.innerHTML = '<td>' + row.well + '</td><td colspan="5" class="text-danger">' + row.error + '</td>';
            } else {
                var f1html = '';
                if (row.per_class_f1) {
                    Object.keys(row.per_class_f1).forEach(function(c) {
                        f1html += c + ': ' + (row.per_class_f1[c] * 100).toFixed(0) + '% ';
                    });
                }
                tr.innerHTML = '<td>' + row.well + '</td><td>' + row.n_train + '</td><td>' + row.n_test + '</td>' +
                    '<td>' + (row.accuracy * 100).toFixed(1) + '%</td>' +
                    '<td>' + (row.balanced_accuracy * 100).toFixed(1) + '%</td>' +
                    '<td class="small">' + f1html + '</td>';
            }
            tbody.appendChild(tr);
        });

        // Within-well table
        var withinBody = document.getElementById("cwcv-within-body");
        clearChildren(withinBody);
        (r.within_well_cv || []).forEach(function(row) {
            var tr = document.createElement("tr");
            if (row.error) {
                tr.innerHTML = '<td>' + row.well + '</td><td>' + row.n_samples + '</td><td colspan="2" class="text-muted">' + row.error + '</td>';
            } else {
                tr.innerHTML = '<td>' + row.well + '</td><td>' + row.n_samples + '</td><td>' + row.n_classes + '</td>' +
                    '<td>' + (row.cv_accuracy * 100).toFixed(1) + '% ± ' + (row.cv_std * 100).toFixed(1) + '%</td>';
            }
            withinBody.appendChild(tr);
        });

        val("cwcv-recommendation", r.recommendation || '');

        showToast("Cross-well: " + r.transferability + " transferability");
    } catch (err) {
        showToast("Cross-well CV error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Expert Ensemble ──────────────────────────────

async function runExpertEnsemble() {
    showLoading("Training expert-weighted ensemble...");
    try {
        var r = await api("/api/analysis/expert-ensemble", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source: currentSource})
        });

        document.getElementById("ee-results").classList.remove("d-none");

        // Summary
        var summaryEl = document.getElementById("ee-summary");
        var impColor = r.improvement > 0 ? "success" : r.improvement < 0 ? "danger" : "info";
        summaryEl.innerHTML = '<div class="row g-2">' +
            '<div class="col-md-3"><div class="text-center p-2 bg-light rounded">' +
            '<div class="small text-muted">Equal-Weight Acc</div><div class="fw-bold">' + (r.equal_weight_accuracy * 100).toFixed(1) + '%</div></div></div>' +
            '<div class="col-md-3"><div class="text-center p-2 bg-light rounded">' +
            '<div class="small text-muted">Expert-Weight Acc</div><div class="fw-bold">' + (r.expert_weight_accuracy * 100).toFixed(1) + '%</div></div></div>' +
            '<div class="col-md-3"><div class="text-center p-2 bg-light rounded">' +
            '<div class="small text-muted">Disagreement Rate</div><div class="fw-bold">' + (r.disagreement_rate * 100).toFixed(1) + '%</div></div></div>' +
            '<div class="col-md-3"><div class="text-center p-2 bg-light rounded">' +
            '<div class="small text-muted">Low Confidence</div><div class="fw-bold">' + r.n_low_confidence + '/' + r.n_total + '</div></div></div>' +
            '</div>' +
            (r.expert_weights_applied ?
                '<div class="alert alert-success py-2 mt-2 small"><i class="bi bi-check-circle"></i> Expert feedback incorporated into ensemble weights.</div>' :
                '<div class="alert alert-info py-2 mt-2 small"><i class="bi bi-info-circle"></i> No expert feedback yet — using accuracy-proportional weights. Submit 3+ feedback ratings to activate RLHF-style weighting.</div>');

        // Weights table
        var wBody = document.getElementById("ee-weights-body");
        clearChildren(wBody);
        var weights = r.model_weights || {};
        var accs = r.model_accuracies || {};
        Object.keys(weights).forEach(function(name) {
            var w = weights[name];
            var a = accs[name] || {};
            var tr = document.createElement("tr");
            tr.innerHTML = '<td>' + name + '</td>' +
                '<td>' + (w.base * 100).toFixed(1) + '%</td>' +
                '<td>' + (w.adjusted * 100).toFixed(1) + '%</td>' +
                '<td>' + ((a.cv_accuracy || 0) * 100).toFixed(1) + '%</td>';
            wBody.appendChild(tr);
        });

        // Per-class table
        var cBody = document.getElementById("ee-class-body");
        clearChildren(cBody);
        var perClass = r.per_class || {};
        Object.keys(perClass).forEach(function(cname) {
            var c = perClass[cname];
            var deltaColor = c.delta_f1 > 0 ? "text-success" : c.delta_f1 < 0 ? "text-danger" : "";
            var tr = document.createElement("tr");
            tr.innerHTML = '<td>' + cname + '</td>' +
                '<td>' + (c.equal_f1 * 100).toFixed(1) + '%</td>' +
                '<td>' + (c.expert_f1 * 100).toFixed(1) + '%</td>' +
                '<td class="' + deltaColor + '">' + (c.delta_f1 > 0 ? '+' : '') + (c.delta_f1 * 100).toFixed(1) + '%</td>';
            cBody.appendChild(tr);
        });

        showToast("Expert ensemble: " + (r.expert_weight_accuracy * 100).toFixed(1) + "% accuracy, " + r.n_models + " models");
    } catch (err) {
        showToast("Expert ensemble error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Research & Physics ────────────────────────────

async function loadResearchMethods() {
    try {
        var r = await api("/api/research/methods");
        var body = document.getElementById("research-body");
        clearChildren(body);

        // Title
        var h = document.createElement("h5");
        h.className = "mb-3";
        h.textContent = r.title || "Scientific Methods";
        body.appendChild(h);

        // Methods cards
        (r.methods || []).forEach(function(m) {
            var card = document.createElement("div");
            card.className = "card mb-3";
            var html = '<div class="card-body"><h6 class="card-title">' + m.name + '</h6>';
            html += '<p class="card-text small">' + m.description + '</p>';
            if (m.reference) {
                html += '<small class="text-muted"><i class="bi bi-journal"></i> ' + m.reference + '</small>';
            }
            if (m.factors && m.factors.length > 0) {
                html += '<div class="mt-2">';
                m.factors.forEach(function(f) {
                    html += '<span class="badge bg-light text-dark me-1 mb-1">' + f + '</span>';
                });
                html += '</div>';
            }
            html += '</div>';
            card.innerHTML = html;
            body.appendChild(card);
        });

        // Factors accounted
        if (r.factors_accounted) {
            var facCard = document.createElement("div");
            facCard.className = "card mt-4 border-success";
            var facHtml = '<div class="card-header bg-success text-white">Factors Accounted For</div><div class="card-body">';
            Object.keys(r.factors_accounted).forEach(function(cat) {
                facHtml += '<h6 class="small fw-bold text-capitalize mt-2">' + cat + '</h6><ul class="small mb-2">';
                r.factors_accounted[cat].forEach(function(f) {
                    facHtml += '<li>' + f + '</li>';
                });
                facHtml += '</ul>';
            });
            facHtml += '</div>';
            facCard.innerHTML = facHtml;
            body.appendChild(facCard);
        }

        // 2025-2026 Research Citations
        if (r.research_2025_2026 && r.research_2025_2026.length > 0) {
            var resCard = document.createElement("div");
            resCard.className = "card mt-4 border-info";
            var resHtml = '<div class="card-header bg-info text-white"><i class="bi bi-mortarboard"></i> 2025-2026 Research Integration</div><div class="card-body">';
            r.research_2025_2026.forEach(function(p) {
                resHtml += '<div class="mb-3 pb-2 border-bottom">';
                resHtml += '<h6 class="mb-1"><span class="badge bg-primary me-1">' + p.year + '</span> ' + p.title + '</h6>';
                resHtml += '<div class="small text-muted mb-1"><i class="bi bi-journal"></i> ' + p.source + '</div>';
                resHtml += '<p class="small mb-1"><strong>Finding:</strong> ' + p.finding + '</p>';
                resHtml += '<p class="small mb-0 text-success"><strong>Our implementation:</strong> ' + p.relevance + '</p>';
                resHtml += '</div>';
            });
            resHtml += '</div>';
            resCard.innerHTML = resHtml;
            body.appendChild(resCard);
        }

        // Limitations
        if (r.limitations && r.limitations.length > 0) {
            var limCard = document.createElement("div");
            limCard.className = "card mt-3 border-warning";
            var limHtml = '<div class="card-header bg-warning text-dark">Known Limitations</div><div class="card-body"><ul class="small mb-0">';
            r.limitations.forEach(function(l) {
                limHtml += '<li>' + l + '</li>';
            });
            limHtml += '</ul></div>';
            limCard.innerHTML = limHtml;
            body.appendChild(limCard);
        }
    } catch (err) {
        showToast("Research methods error: " + err.message, "Error");
    }
}

async function runPhysicsCheck() {
    showLoading("Checking physics constraints...");
    try {
        var r = await api("/api/analysis/physics-check", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                source: currentSource,
                well: currentWell || "3P",
                depth: parseFloat(document.getElementById("depth-input").value) || 3000
            })
        });

        document.getElementById("physics-results").classList.remove("d-none");

        var statusColors = {PASS: "success", CAUTION: "warning", FAIL: "danger"};
        var statusEl = document.getElementById("physics-status");
        statusEl.className = "alert alert-" + (statusColors[r.status] || "info") + " mb-3";
        statusEl.innerHTML = '<strong>' + r.status + '</strong> — ' + r.status_message;

        var issuesEl = document.getElementById("physics-issues");
        clearChildren(issuesEl);
        if (r.violations && r.violations.length > 0) {
            r.violations.forEach(function(v) {
                var div = document.createElement("div");
                div.className = "alert alert-danger py-2 mb-2 small";
                div.innerHTML = '<strong>VIOLATION: ' + v.constraint + '</strong><br>' +
                    'Expected: ' + v.expected + '<br>Actual: ' + v.actual;
                issuesEl.appendChild(div);
            });
        }
        if (r.warnings && r.warnings.length > 0) {
            r.warnings.forEach(function(w) {
                var div = document.createElement("div");
                div.className = "alert alert-warning py-2 mb-2 small";
                div.innerHTML = '<strong>' + w.constraint + '</strong><br>' +
                    'Expected: ' + w.expected + '<br>Actual: ' + w.actual + '<br>' +
                    '<i>' + (w.note || '') + '</i>';
                issuesEl.appendChild(div);
            });
        }

        // Constraints checked
        var constEl = document.getElementById("physics-constraints");
        constEl.innerHTML = '<h6 class="small fw-bold">Constraints Checked</h6>' +
            '<div class="small text-muted">' +
            (r.constraints_checked || []).map(function(c) { return '<i class="bi bi-check-circle text-success"></i> ' + c; }).join('<br>') +
            '</div>';

        showToast("Physics: " + r.status);
    } catch (err) {
        showToast("Physics check error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Physics-Constrained Prediction ────────────────

async function runPhysicsPredict() {
    showLoading("Running physics-constrained prediction...");
    try {
        var r = await apiPost("/api/analysis/physics-predict", {
            source: currentSource, well: getWell(), depth: getDepth(), fast: true
        });
        var el = document.getElementById("physics-predict-results");
        el.classList.remove("d-none");
        var body = document.getElementById("physics-predict-body");

        // Physics score banner
        var scoreColor = r.physics_score >= 0.8 ? "success" : r.physics_score >= 0.5 ? "warning" : "danger";
        var html = '<div class="alert alert-' + scoreColor + '">' +
            '<h5 class="mb-1"><i class="bi bi-shield-check"></i> Physics Score: ' +
            (r.physics_score * 100).toFixed(0) + '%</h5>' +
            '<small>Constraint Status: ' + r.constraint_status + ' | Regime: ' + r.regime_used + '</small>' +
            '</div>';

        // Confidence comparison
        html += '<div class="row mb-3">' +
            '<div class="col-md-4"><div class="card border-primary"><div class="card-body text-center">' +
            '<div class="h5 mb-0">' + (r.ml_confidence_mean * 100).toFixed(1) + '%</div>' +
            '<small class="text-muted">Raw ML Confidence</small></div></div></div>' +
            '<div class="col-md-4"><div class="card border-' + scoreColor + '"><div class="card-body text-center">' +
            '<div class="h5 mb-0">' + (r.adjusted_confidence_mean * 100).toFixed(1) + '%</div>' +
            '<small class="text-muted">Physics-Adjusted</small></div></div></div>' +
            '<div class="col-md-4"><div class="card border-warning"><div class="card-body text-center">' +
            '<div class="h5 mb-0">' + r.low_confidence_count + ' (' + r.low_confidence_pct + '%)</div>' +
            '<small class="text-muted">Low-Confidence Samples</small></div></div></div></div>';

        // Physics flags
        if (r.physics_flags && r.physics_flags.length > 0) {
            html += '<h6><i class="bi bi-flag"></i> Physics Flags</h6>';
            r.physics_flags.forEach(function(f) {
                html += '<div class="alert alert-warning py-2 mb-1 small"><i class="bi bi-exclamation-triangle"></i> ' + f + '</div>';
            });
        }

        // Per-class physics-adjusted confidence
        if (r.per_class) {
            html += '<h6 class="mt-3"><i class="bi bi-bar-chart"></i> Per-Class Physics-Adjusted Metrics</h6>';
            html += '<table class="table table-sm"><thead><tr>' +
                '<th>Fracture Type</th><th>Count</th><th>Accuracy</th><th>Avg Confidence</th></tr></thead><tbody>';
            Object.entries(r.per_class).forEach(function(e) {
                var cls = e[0], info = e[1];
                var confColor = info.avg_confidence >= 0.7 ? "text-success" : info.avg_confidence >= 0.5 ? "text-warning" : "text-danger";
                html += '<tr><td>' + cls + '</td><td>' + info.count + '</td>' +
                    '<td>' + (info.accuracy * 100).toFixed(1) + '%</td>' +
                    '<td class="' + confColor + ' fw-bold">' + (info.avg_confidence * 100).toFixed(1) + '%</td></tr>';
            });
            html += '</tbody></table>';
        }

        body.innerHTML = html;
        showToast("Physics-constrained: " + (r.physics_score * 100).toFixed(0) + '% physics score');
    } catch (err) {
        showToast("Physics predict error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Misclassification Analysis ────────────────────

async function runMisclassification() {
    showLoading("Analyzing misclassifications...");
    try {
        var r = await apiPost("/api/analysis/misclassification", {
            source: currentSource, well: getWell(), fast: true
        });
        var el = document.getElementById("misclass-results");
        el.classList.remove("d-none");
        var body = document.getElementById("misclass-body");

        // Overall accuracy banner
        var accColor = r.overall_accuracy >= 0.85 ? "success" : r.overall_accuracy >= 0.7 ? "warning" : "danger";
        var html = '<div class="alert alert-' + accColor + '">' +
            '<h5 class="mb-1"><i class="bi bi-bullseye"></i> CV Accuracy: ' +
            (r.overall_accuracy * 100).toFixed(1) + '% (' + r.n_errors + ' errors out of ' + r.n_samples + ')</h5>' +
            '</div>';

        // Top confused pairs
        if (r.confused_pairs && r.confused_pairs.length > 0) {
            html += '<h6><i class="bi bi-shuffle"></i> Most Confused Pairs (Where Errors Happen)</h6>';
            html += '<table class="table table-sm table-hover"><thead><tr>' +
                '<th>True Type</th><th>Predicted As</th><th>Count</th><th>% of True</th></tr></thead><tbody>';
            r.confused_pairs.slice(0, 8).forEach(function(p) {
                var severity = p.pct_of_true > 30 ? "table-danger" : p.pct_of_true > 10 ? "table-warning" : "";
                html += '<tr class="' + severity + '"><td>' + p.true_class + '</td>' +
                    '<td>' + p.predicted_as + '</td><td>' + p.count + '</td>' +
                    '<td>' + p.pct_of_true + '%</td></tr>';
            });
            html += '</tbody></table>';
        }

        // Per-class failure breakdown
        if (r.class_failures) {
            html += '<h6 class="mt-3"><i class="bi bi-pie-chart"></i> Per-Class Accuracy & Failure Modes</h6>';
            Object.entries(r.class_failures).forEach(function(e) {
                var cls = e[0], info = e[1];
                var barWidth = Math.max(5, info.accuracy * 100);
                var barColor = info.accuracy >= 0.85 ? "#198754" : info.accuracy >= 0.6 ? "#ffc107" : "#dc3545";
                html += '<div class="mb-2"><div class="d-flex justify-content-between">' +
                    '<strong>' + cls + '</strong><span>' + (info.accuracy * 100).toFixed(0) + '% (' + info.correct + '/' + info.total + ')</span></div>' +
                    '<div class="progress" style="height:8px"><div class="progress-bar" style="width:' + barWidth + '%;background:' + barColor + '"></div></div>';
                if (info.top_confusions && info.top_confusions.length > 0) {
                    html += '<small class="text-muted">Most confused with: ' +
                        info.top_confusions.map(function(c) { return c.confused_with + ' (' + c.count + ')'; }).join(', ') +
                        '</small>';
                }
                html += '</div>';
            });
        }

        // Error profile
        if (r.error_profile) {
            html += '<h6 class="mt-3"><i class="bi bi-graph-down"></i> Error Pattern Analysis</h6><div class="row">';
            if (r.error_profile.depth_analysis) {
                var da = r.error_profile.depth_analysis;
                html += '<div class="col-md-6"><div class="card"><div class="card-body">' +
                    '<h6 class="card-title">Depth Pattern</h6>' +
                    '<p class="small">Errors cluster at <strong>' + da.depth_bias + '</strong> depths</p>' +
                    '<p class="small mb-0">Correct mean: ' + da.correct_mean_depth + 'm | Error mean: ' + da.error_mean_depth + 'm</p>' +
                    '</div></div></div>';
            }
            if (r.error_profile.dip_analysis) {
                var dip = r.error_profile.dip_analysis;
                html += '<div class="col-md-6"><div class="card"><div class="card-body">' +
                    '<h6 class="card-title">Dip Pattern</h6>' +
                    '<p class="small">High-dip errors: ' + dip.high_dip_errors + ' | Low-dip: ' + dip.low_dip_errors + '</p>' +
                    '<p class="small mb-0">Correct mean dip: ' + dip.correct_mean_dip + '° | Error mean: ' + dip.error_mean_dip + '°</p>' +
                    '</div></div></div>';
            }
            html += '</div>';
        }

        // Confusion matrix chart
        if (r.confusion_chart_img) {
            html += '<h6 class="mt-3"><i class="bi bi-grid-3x3"></i> Confusion Matrix</h6>';
            html += '<div class="text-center"><img src="' + r.confusion_chart_img +
                '" class="img-fluid rounded shadow-sm" alt="Confusion Matrix" style="max-width:600px"></div>';
        }

        // Recommendations
        if (r.recommendations && r.recommendations.length > 0) {
            html += '<h6 class="mt-3"><i class="bi bi-lightbulb"></i> Improvement Recommendations</h6>';
            r.recommendations.forEach(function(rec) {
                var recColor = rec.priority === "HIGH" ? "danger" : "warning";
                html += '<div class="alert alert-' + recColor + ' py-2 mb-2 small">' +
                    '<strong>' + rec.priority + ' — ' + rec["class"] + ':</strong> ' + rec.message + '</div>';
            });
        }

        body.innerHTML = html;
        showToast("Misclassification: " + r.n_errors + " errors found");
    } catch (err) {
        showToast("Misclassification error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Model Bias Detection ──────────────────────────

async function runModelBias() {
    showLoading("Detecting model biases...");
    try {
        var r = await apiPost("/api/analysis/model-bias", {
            source: currentSource, well: getWell(), fast: true
        });
        var el = document.getElementById("bias-results");
        el.classList.remove("d-none");
        var body = document.getElementById("bias-body");

        var biasColors = {NONE: "success", MODERATE: "warning", HIGH: "danger"};
        var html = '<div class="alert alert-' + (biasColors[r.bias_level] || "info") + '">' +
            '<h5 class="mb-1"><i class="bi bi-sliders2"></i> Bias Level: ' + r.bias_level + '</h5>' +
            '<p class="mb-0">' + r.bias_message + '</p></div>';

        // Biases found
        if (r.biases && r.biases.length > 0) {
            html += '<h6><i class="bi bi-exclamation-diamond"></i> Biases Detected</h6>';
            r.biases.forEach(function(b) {
                var color = b.severity === "HIGH" ? "danger" : "warning";
                html += '<div class="alert alert-' + color + ' py-2 mb-2 small">' +
                    '<strong>' + b.severity + ' — ' + b.type.replace(/_/g, ' ') + '</strong>: ' + b.message + '</div>';
            });
        }

        // Class distribution comparison
        if (r.class_distribution) {
            html += '<h6 class="mt-3"><i class="bi bi-bar-chart"></i> Class Distribution: True vs Predicted</h6>';
            html += '<table class="table table-sm"><thead><tr>' +
                '<th>Type</th><th>True %</th><th>Predicted %</th><th>Bias</th></tr></thead><tbody>';
            Object.entries(r.class_distribution).forEach(function(e) {
                var cls = e[0], info = e[1];
                var biasColor = Math.abs(info.bias) > 15 ? "text-danger" : Math.abs(info.bias) > 5 ? "text-warning" : "";
                html += '<tr><td>' + cls + '</td><td>' + info.true_pct + '%</td>' +
                    '<td>' + info.predicted_pct + '%</td>' +
                    '<td class="' + biasColor + ' fw-bold">' + (info.bias > 0 ? '+' : '') + info.bias + '%</td></tr>';
            });
            html += '</tbody></table>';
        }

        body.innerHTML = html;
        showToast("Bias detection: " + r.bias_level);
    } catch (err) {
        showToast("Bias detection error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Reliability Report ────────────────────────────

async function runReliabilityReport() {
    showLoading("Generating reliability report...");
    try {
        var r = await apiPost("/api/analysis/reliability-report", {
            source: currentSource, well: getWell(), depth: getDepth(), fast: true
        });
        var el = document.getElementById("reliability-results");
        el.classList.remove("d-none");
        var body = document.getElementById("reliability-body");

        var gradeColors = {A: "success", B: "info", C: "warning", D: "danger"};
        var html = '<div class="alert alert-' + (gradeColors[r.reliability_grade] || "secondary") + ' p-3">' +
            '<h4 class="mb-1">Grade: ' + r.reliability_grade + '</h4>' +
            '<p class="mb-1">' + r.reliability_message + '</p>' +
            '<small>Accuracy: ' + (r.overall_accuracy * 100).toFixed(1) + '% | Bias: ' + r.bias_level +
            ' | Limitations: ' + r.n_limitations + '</small></div>';

        // Limitations
        if (r.limitations && r.limitations.length > 0) {
            html += '<h6><i class="bi bi-exclamation-triangle"></i> Known Limitations (' + r.n_limitations + ')</h6>';
            r.limitations.forEach(function(lim) {
                var color = lim.severity === "HIGH" ? "danger" : "warning";
                html += '<div class="card mb-2 border-' + color + '"><div class="card-body py-2">' +
                    '<div class="d-flex justify-content-between"><strong class="small">' + lim.scope + '</strong>' +
                    '<span class="badge bg-' + color + '">' + lim.severity + '</span></div>' +
                    '<p class="small mb-1">' + lim.limitation + '</p>' +
                    '<small class="text-muted"><i class="bi bi-tools"></i> ' + lim.mitigation + '</small>' +
                    '</div></div>';
            });
        }

        // Improvement roadmap
        if (r.improvement_roadmap && r.improvement_roadmap.length > 0) {
            html += '<h6 class="mt-3"><i class="bi bi-map"></i> Improvement Roadmap</h6>';
            html += '<ol class="list-group list-group-numbered">';
            r.improvement_roadmap.forEach(function(step) {
                html += '<li class="list-group-item d-flex justify-content-between align-items-start">' +
                    '<div class="ms-2 me-auto"><div class="fw-bold">' + step.action + '</div>' +
                    '<small>Expected: ' + step.expected_impact + '</small></div>' +
                    '<span class="badge bg-primary rounded-pill">' + step.effort + '</span></li>';
            });
            html += '</ol>';
        }

        // Top confusion pairs
        if (r.confusion_pairs && r.confusion_pairs.length > 0) {
            html += '<h6 class="mt-3"><i class="bi bi-shuffle"></i> Top Error Patterns</h6>';
            html += '<table class="table table-sm"><thead><tr>' +
                '<th>True</th><th>Predicted As</th><th>Count</th></tr></thead><tbody>';
            r.confusion_pairs.forEach(function(p) {
                html += '<tr><td>' + p.true_class + '</td><td>' + p.predicted_as + '</td><td>' + p.count + '</td></tr>';
            });
            html += '</tbody></table>';
        }

        body.innerHTML = html;
        showToast("Reliability: Grade " + r.reliability_grade);
    } catch (err) {
        showToast("Reliability report error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Prediction Abstention ─────────────────────────

async function runPredictWithAbstention() {
    showLoading("Running prediction with abstention...");
    try {
        var thresholdEl = document.getElementById("abstention-threshold");
        var threshold = thresholdEl ? parseFloat(thresholdEl.value) : 0.60;

        var r = await apiPost("/api/analysis/predict-with-abstention", {
            source: currentSource, well: getWell(), threshold: threshold,
            classifier: "random_forest", fast: true
        });
        var el = document.getElementById("abstention-results");
        el.classList.remove("d-none");
        var body = document.getElementById("abstention-body");

        // Recommendation badge
        var recColors = {
            LOW_ABSTENTION: "success", MODERATE_ABSTENTION: "warning", HIGH_ABSTENTION: "danger"
        };
        var rec = r.recommendation || {};
        var html = '<div class="alert alert-' + (recColors[rec.verdict] || "info") + ' p-3">' +
            '<h5 class="mb-1"><i class="bi bi-shield-check"></i> ' + (rec.verdict || "").replace(/_/g, " ") + '</h5>' +
            '<p class="mb-1">' + (rec.message || "") + '</p>' +
            '<small><strong>Suggested action:</strong> ' + (rec.action || "").replace(/_/g, " ") + '</small></div>';

        // Summary cards
        html += '<div class="row g-2 mb-3">';
        html += '<div class="col-md-3"><div class="card text-center"><div class="card-body py-2">' +
            '<h3>' + r.total_samples + '</h3><small class="text-muted">Total Samples</small></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center border-success"><div class="card-body py-2">' +
            '<h3 class="text-success">' + r.confident_predictions + '</h3><small class="text-muted">Confident</small></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center border-warning"><div class="card-body py-2">' +
            '<h3 class="text-warning">' + r.abstained_predictions + '</h3><small class="text-muted">Abstained (' + r.abstention_rate + '%)</small></div></div></div>';
        html += '<div class="col-md-3"><div class="card text-center border-info"><div class="card-body py-2">' +
            '<h3 class="text-info">+' + (r.accuracy_gain * 100).toFixed(1) + '%</h3><small class="text-muted">Accuracy Gain</small></div></div></div>';
        html += '</div>';

        // Accuracy comparison
        html += '<div class="row g-2 mb-3">';
        html += '<div class="col-md-6"><div class="card"><div class="card-body py-2">' +
            '<h6>Overall Accuracy (all samples)</h6>' +
            '<div class="progress" style="height:25px"><div class="progress-bar bg-secondary" style="width:' + (r.accuracy_overall * 100) + '%">' +
            (r.accuracy_overall * 100).toFixed(1) + '%</div></div></div></div></div>';
        html += '<div class="col-md-6"><div class="card"><div class="card-body py-2">' +
            '<h6>Confident-Only Accuracy</h6>' +
            '<div class="progress" style="height:25px"><div class="progress-bar bg-success" style="width:' + (r.accuracy_confident_only * 100) + '%">' +
            (r.accuracy_confident_only * 100).toFixed(1) + '%</div></div></div></div></div>';
        html += '</div>';

        // Confidence distribution chart (server-rendered or JS fallback)
        if (r.chart_img) {
            html += '<div class="text-center mb-3"><img src="data:image/png;base64,' + r.chart_img +
                '" class="img-fluid" alt="Abstention confidence distribution"></div>';
        } else if (r.confidence_distribution) {
            html += '<h6><i class="bi bi-bar-chart"></i> Confidence Distribution</h6>';
            html += '<div class="d-flex gap-1 mb-3">';
            var maxCount = Math.max.apply(null, r.confidence_distribution.map(function(b) { return b.count; }));
            r.confidence_distribution.forEach(function(bin) {
                var hPct = maxCount > 0 ? Math.max(5, (bin.count / maxCount) * 100) : 5;
                var color = parseFloat(bin.range) < 0.5 ? "#dc3545" : (parseFloat(bin.range) < 0.7 ? "#ffc107" : "#198754");
                html += '<div class="text-center flex-fill">' +
                    '<div style="height:80px;display:flex;align-items:flex-end;justify-content:center">' +
                    '<div style="width:100%;height:' + hPct + '%;background:' + color + ';border-radius:3px 3px 0 0"></div></div>' +
                    '<small class="d-block">' + bin.range + '</small>' +
                    '<small class="text-muted">' + bin.count + '</small></div>';
            });
            html += '</div>';
        }

        // Abstention by class
        if (r.abstain_by_class && Object.keys(r.abstain_by_class).length > 0) {
            html += '<h6><i class="bi bi-exclamation-diamond"></i> Abstentions by Class</h6>';
            html += '<table class="table table-sm"><thead><tr><th>Fracture Type</th><th>Abstained</th></tr></thead><tbody>';
            Object.keys(r.abstain_by_class).forEach(function(cls) {
                html += '<tr><td>' + cls + '</td><td><span class="badge bg-warning">' + r.abstain_by_class[cls] + '</span></td></tr>';
            });
            html += '</tbody></table>';
        }

        // Sample-level table (first 20 abstained + first 10 confident)
        if (r.samples && r.samples.length > 0) {
            var abstained = r.samples.filter(function(s) { return s.status !== "CONFIDENT"; });
            var confident = r.samples.filter(function(s) { return s.status === "CONFIDENT" && !s.correct; });
            html += '<h6><i class="bi bi-person-raised-hand"></i> Samples Requiring Expert Review (' + abstained.length + ')</h6>';
            html += '<p class="small text-muted">Select the correct fracture type for uncertain samples. Corrections improve future predictions.</p>';
            if (abstained.length > 0) {
                // Store class names for dropdowns
                var classNames = r.class_names || [];
                var classOpts = classNames.map(function(c) { return '<option value="' + c + '">' + c + '</option>'; }).join("");
                html += '<div class="table-responsive"><table class="table table-sm table-striped"><thead><tr>' +
                    '<th>Depth</th><th>Az</th><th>Dip</th><th>Current</th><th>Conf</th><th>Top Candidates</th><th>Correct As</th></tr></thead><tbody>';
                abstained.slice(0, 20).forEach(function(s, idx) {
                    var cands = (s.top_candidates || []).map(function(c) {
                        return c["class"] + " (" + (c.probability * 100).toFixed(0) + "%)";
                    }).join(", ");
                    var tentative = s.tentative_prediction || s.true_label;
                    html += '<tr class="table-warning" id="review-row-' + idx + '"><td>' + (s.depth || "-") + '</td><td>' + (s.azimuth || "-") + '</td>' +
                        '<td>' + (s.dip || "-") + '</td><td>' + s.true_label + '</td>' +
                        '<td>' + (s.confidence * 100).toFixed(0) + '%</td><td>' + cands + '</td>' +
                        '<td><select class="form-select form-select-sm review-correction" data-index="' + s.index + '" ' +
                        'data-depth="' + (s.depth || "") + '" data-az="' + (s.azimuth || "") + '" data-dip="' + (s.dip || "") + '" ' +
                        'data-original="' + s.true_label + '" style="width:auto;display:inline-block">' +
                        '<option value="">-- Accept --</option>' + classOpts + '</select></td></tr>';
                });
                html += '</tbody></table></div>';
                html += '<button class="btn btn-primary btn-sm mt-2" onclick="submitReviewCorrections()">' +
                    '<i class="bi bi-check2-all"></i> Submit Corrections</button>';
                if (abstained.length > 20) {
                    html += '<small class="text-muted ms-2">Showing 20 of ' + abstained.length + ' abstained samples</small>';
                }
            }
            // Misclassified confident samples
            if (confident.length > 0) {
                html += '<h6 class="mt-3"><i class="bi bi-x-circle"></i> Confident but Wrong (' + confident.length + ')</h6>';
                html += '<div class="table-responsive"><table class="table table-sm"><thead><tr>' +
                    '<th>Depth</th><th>True</th><th>Predicted</th><th>Conf</th></tr></thead><tbody>';
                confident.slice(0, 10).forEach(function(s) {
                    html += '<tr class="table-danger"><td>' + (s.depth || "-") + '</td>' +
                        '<td>' + s.true_label + '</td><td>' + s.prediction + '</td>' +
                        '<td>' + (s.confidence * 100).toFixed(0) + '%</td></tr>';
                });
                html += '</tbody></table></div>';
            }
        }

        body.innerHTML = html;
        showToast("Abstention: " + r.abstained_predictions + "/" + r.total_samples + " samples flagged for review");
    } catch (err) {
        showToast("Abstention error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


async function submitReviewCorrections() {
    var selects = document.querySelectorAll(".review-correction");
    var corrections = [];
    selects.forEach(function(sel) {
        if (sel.value) {
            corrections.push({
                fracture_index: parseInt(sel.dataset.index),
                depth: parseFloat(sel.dataset.depth) || null,
                azimuth: parseFloat(sel.dataset.az) || null,
                dip: parseFloat(sel.dataset.dip) || null,
                original_type: sel.dataset.original,
                corrected_type: sel.value,
                source: "uncertainty_review"
            });
        }
    });

    if (corrections.length === 0) {
        showToast("No corrections selected", "Info");
        return;
    }

    try {
        var r = await apiPost("/api/feedback/batch-corrections", {
            well: getWell(),
            corrections: corrections,
            reviewer: "expert",
        });
        showToast(corrections.length + " correction(s) submitted — model will improve");
        // Gray out submitted rows
        selects.forEach(function(sel) {
            if (sel.value) {
                var row = sel.closest("tr");
                if (row) {
                    row.classList.remove("table-warning");
                    row.classList.add("table-success");
                    sel.disabled = true;
                }
            }
        });
    } catch (err) {
        showToast("Correction error: " + err.message, "Error");
    }
}


// ── Audit Trail ───────────────────────────────────

async function loadAuditLog() {
    try {
        var data = await api("/api/audit/log?limit=50");
        val("audit-total", data.total);

        if (data.entries.length > 0) {
            var first = data.entries[data.entries.length - 1];
            val("audit-session-start", first.timestamp.substring(0, 19).replace("T", " "));
        }

        var tbody = document.querySelector("#audit-table tbody");
        clearChildren(tbody);
        data.entries.forEach(function(e) {
            var tr = document.createElement("tr");
            tr.appendChild(createCell("td", e.id));
            tr.appendChild(createCell("td", e.timestamp.substring(11, 19)));
            tr.appendChild(createCell("td", e.action, { fontWeight: "600" }));
            tr.appendChild(createCell("td", e.well || "All"));
            tr.appendChild(createCell("td", e.source));
            tr.appendChild(createCell("td", e.elapsed_s + "s"));
            tr.appendChild(createCell("td", e.result_hash, { fontFamily: "monospace", fontSize: "0.75rem" }));
            tbody.appendChild(tr);
        });
    } catch (err) {
        // Silent - audit is non-critical
    }
}

async function exportAuditLog() {
    try {
        var data = await api("/api/audit/export", { method: "POST" });
        if (data.csv && data.rows > 0) {
            var blob = new Blob([data.csv], { type: "text/csv" });
            var url = URL.createObjectURL(blob);
            var a = document.createElement("a");
            a.href = url;
            a.download = data.filename || "audit_trail.csv";
            a.click();
            URL.revokeObjectURL(url);
            showToast("Exported " + data.rows + " audit records");
        } else {
            showToast("No audit records to export");
        }
    } catch (err) {
        showToast("Export error: " + err.message, "Error");
    }
}


// ── Expert Stress Solution Ranking (RLHF) ─────────

async function runExpertRanking() {
    showLoading("Generating stress solutions for expert review...");
    try {
        var depth = document.getElementById("depth-input").value || 3000;
        var pp = document.getElementById("pp-input").value || null;
        var data = await apiPost("/api/analysis/expert-stress-ranking", {
            source: currentSource, well: currentWell,
            depth: parseFloat(depth), pore_pressure: pp ? parseFloat(pp) : null
        });

        document.getElementById("esr-results").classList.remove("d-none");

        // Auto-detection summary
        var confColors = { HIGH: "alert-success", MODERATE: "alert-warning", LOW: "alert-danger" };
        var autoEl = document.getElementById("esr-auto-summary");
        autoEl.className = "alert mb-3 " + (confColors[data.auto_confidence] || "alert-info");
        document.getElementById("esr-auto-text").innerHTML =
            "Best fit: <strong>" + (data.auto_best || "?").replace("_", " ") + "</strong> " +
            "(confidence: <strong>" + (data.auto_confidence || "?") + "</strong>, " +
            "misfit ratio: " + fmt(data.misfit_ratio, 2) + ")";

        // Solution cards
        var container = document.getElementById("esr-solutions");
        clearChildren(container);
        (data.solutions || []).forEach(function(sol) {
            var borderClass = sol.is_auto_best ? "border-primary" : "border-secondary";
            var card = document.createElement("div");
            card.className = "col-md-4";
            var csColor = sol.critically_stressed_pct > 50 ? "text-danger" :
                          sol.critically_stressed_pct > 25 ? "text-warning" : "text-success";

            var html = '<div class="card h-100 ' + borderClass + '">' +
                '<div class="card-header d-flex justify-content-between align-items-center">' +
                '<strong>#' + sol.rank + ' ' + sol.regime_label + '</strong>';
            if (sol.is_auto_best) {
                html += '<span class="badge bg-primary">Auto Best</span>';
            }
            html += '</div><div class="card-body">' +
                '<table class="table table-sm mb-2">' +
                '<tr><td class="text-muted">SHmax</td><td class="fw-bold">' + fmt(sol.shmax_azimuth_deg, 1) + '&deg;</td></tr>' +
                '<tr><td class="text-muted">&sigma;1 / &sigma;3</td><td>' + fmt(sol.sigma1, 1) + ' / ' + fmt(sol.sigma3, 1) + ' MPa</td></tr>' +
                '<tr><td class="text-muted">R-ratio</td><td>' + fmt(sol.R, 4) + '</td></tr>' +
                '<tr><td class="text-muted">Friction (&mu;)</td><td>' + fmt(sol.mu, 3) + '</td></tr>' +
                '<tr><td class="text-muted">Misfit</td><td>' + fmt(sol.misfit, 2) + '</td></tr>' +
                '<tr><td class="text-muted">Crit. Stressed</td><td class="fw-bold ' + csColor + '">' + fmt(sol.critically_stressed_pct, 1) + '%</td></tr>' +
                '<tr><td class="text-muted">Max Slip Tend.</td><td>' + fmt(sol.max_slip_tendency, 3) + '</td></tr>' +
                '</table>';
            if (sol.mohr_img) {
                html += '<img src="' + sol.mohr_img + '" class="img-fluid rounded" alt="Mohr circle">';
            }
            html += '<div class="small text-muted mt-2">' + (sol.description || "") + '</div>';
            html += '</div></div>';
            card.innerHTML = html;
            container.appendChild(card);
        });

        // Pre-select auto-best in the dropdown
        var selectEl = document.getElementById("esr-select-regime");
        selectEl.value = data.auto_best || "";

        showToast("3 stress solutions ready for review (" + (data.elapsed_s || "?") + "s)");
    } catch (err) {
        showToast("Expert ranking error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function submitExpertSelection() {
    var regime = document.getElementById("esr-select-regime").value;
    if (!regime) { showToast("Please select a regime first", "Error"); return; }
    try {
        var data = await apiPost("/api/analysis/expert-stress-select", {
            source: currentSource, well: currentWell,
            regime: regime,
            reason: document.getElementById("esr-reason").value || "",
            expert_confidence: document.getElementById("esr-select-confidence").value
        });
        var msgEl = document.getElementById("esr-submit-msg");
        msgEl.classList.remove("d-none");
        msgEl.className = "small mt-2 text-success";
        msgEl.textContent = data.message || "Selection recorded.";
        showToast("Expert selection saved");
    } catch (err) {
        showToast("Submit error: " + err.message, "Error");
    }
}


// ── Stakeholder Uncertainty Dashboard ─────────────

async function runUncertaintyDashboard() {
    showLoading("Running confidence assessment...");
    try {
        var depth = document.getElementById("depth-input").value || 3000;
        var pp = document.getElementById("pp-input").value || null;
        var regime = document.getElementById("regime-select").value || "auto";
        var data = await apiPost("/api/analysis/uncertainty-dashboard", {
            source: currentSource, well: currentWell,
            depth: parseFloat(depth), pore_pressure: pp ? parseFloat(pp) : null,
            regime: regime
        });

        document.getElementById("ud-results").classList.remove("d-none");

        // Overall grade
        var overall = data.overall || {};
        var gradeColors = { HIGH: "success", MODERATE: "warning", LOW: "danger" };
        var gc = gradeColors[overall.grade] || "secondary";
        document.getElementById("ud-overall-card").className = "card mb-3 border-" + gc;
        document.getElementById("ud-grade").textContent = overall.label || overall.grade;
        document.getElementById("ud-grade").className = "display-4 fw-bold mb-1 text-" + gc;
        document.getElementById("ud-score").textContent = "Score: " + (overall.score || "?") + " / 100";
        document.getElementById("ud-advice").textContent = overall.advice || "";

        // Signal cards
        var container = document.getElementById("ud-signals");
        clearChildren(container);
        var signalColors = { GREEN: "success", AMBER: "warning", RED: "danger", UNKNOWN: "secondary" };
        (data.signals || []).forEach(function(sig) {
            var sc = signalColors[sig.grade] || "secondary";
            var card = document.createElement("div");
            card.className = "col-md-4 col-lg";
            card.innerHTML =
                '<div class="card h-100 border-' + sc + '">' +
                '<div class="card-body text-center">' +
                '<i class="bi ' + (sig.icon || "bi-question-circle") + ' fs-3 text-' + sc + '"></i>' +
                '<div class="fw-bold mt-1">' + sig.name + '</div>' +
                '<span class="badge bg-' + sc + ' mt-1">' + sig.grade + '</span>' +
                '<div class="small text-muted mt-1">' + (sig.detail || "") + '</div>' +
                '</div></div>';
            container.appendChild(card);
        });

        showToast("Confidence check complete (" + (data.elapsed_s || "?") + "s)");
    } catch (err) {
        showToast("Dashboard error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Data Contribution Tracker ─────────────────────

async function runDataTracker() {
    showLoading("Analyzing data gaps...");
    try {
        var data = await apiPost("/api/analysis/data-tracker", {
            source: currentSource, well: currentWell
        });

        document.getElementById("dt-results").classList.remove("d-none");

        // Health grade
        var h = data.health || {};
        var hColors = { EXCELLENT: "success", GOOD: "info", FAIR: "warning", POOR: "danger" };
        var hc = hColors[h.grade] || "secondary";
        document.getElementById("dt-health-card").className = "card mb-3 border-" + hc;
        document.getElementById("dt-health-grade").className = "fw-bold mb-1 text-" + hc;
        document.getElementById("dt-health-grade").textContent = h.grade || "--";
        document.getElementById("dt-health-summary").textContent = h.summary || "";

        // Class analysis table
        var tbody = document.getElementById("dt-class-tbody");
        clearChildren(tbody);
        var pColors = { CRITICAL: "danger", HIGH: "warning", MODERATE: "info", ADEQUATE: "success" };
        (data.class_analysis || []).forEach(function(c) {
            var tr = document.createElement("tr");
            tr.appendChild(createCell("td", c.type, { fontWeight: "600" }));
            tr.appendChild(createCell("td", c.current_count));
            tr.appendChild(createCell("td", c.target_count));
            var defStyle = c.deficit > 0 ? { color: "#dc2626", fontWeight: "600" } : { color: "#16a34a" };
            tr.appendChild(createCell("td", c.deficit > 0 ? "+" + c.deficit + " needed" : "OK", defStyle));
            var badge = document.createElement("td");
            badge.innerHTML = '<span class="badge bg-' + (pColors[c.priority] || "secondary") + '">' + c.priority + '</span>';
            tr.appendChild(badge);
            tbody.appendChild(tr);
        });

        // Depth zones table
        var dtbody = document.getElementById("dt-depth-tbody");
        clearChildren(dtbody);
        (data.depth_zones || []).forEach(function(z) {
            var tr = document.createElement("tr");
            tr.appendChild(createCell("td", z.zone, { fontWeight: "600" }));
            tr.appendChild(createCell("td", z.count));
            tr.appendChild(createCell("td", z.density_pct + "%"));
            var badge = document.createElement("td");
            badge.innerHTML = '<span class="badge bg-' + (pColors[z.priority] || "secondary") + '">' + z.priority + '</span>';
            tr.appendChild(badge);
            dtbody.appendChild(tr);
        });

        // Current accuracy
        if (data.current_accuracy) {
            document.getElementById("dt-current-acc").textContent = (data.current_accuracy * 100).toFixed(1) + "%";
        }

        // Projections
        var projEl = document.getElementById("dt-projections");
        clearChildren(projEl);
        (data.projections || []).forEach(function(p) {
            var div = document.createElement("div");
            div.className = "d-flex justify-content-between align-items-center mb-2";
            div.innerHTML =
                '<span><strong>' + (p.multiplier || p.estimated_samples + " samples") + '</strong> (' +
                (p.estimated_samples || "?") + ' samples)</span>' +
                '<span class="badge bg-info">' + ((p.projected_accuracy || 0) * 100).toFixed(1) + '% projected</span>';
            projEl.appendChild(div);
        });

        // Recommendations
        var recEl = document.getElementById("dt-recommendations");
        clearChildren(recEl);
        (data.recommendations || []).forEach(function(r) {
            var ic = r.impact === "HIGH" ? "danger" : "warning";
            var div = document.createElement("div");
            div.className = "alert alert-" + ic + " py-2 mb-2";
            div.innerHTML = '<strong>' + r.action + '</strong>' +
                ' <span class="badge bg-' + ic + '">' + r.impact + '</span>' +
                '<div class="small mt-1">' + (r.detail || "") + '</div>';
            recEl.appendChild(div);
        });

        showToast("Data gap analysis complete (" + (data.elapsed_s || "?") + "s)");
    } catch (err) {
        showToast("Data tracker error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Preference-Weighted Regime (RLHF Loop) ────────

async function runPreferenceWeightedRegime() {
    showLoading("Computing preference-weighted regime...");
    try {
        var depth = document.getElementById("depth-input").value || 3000;
        var pp = document.getElementById("pp-input").value || null;
        var data = await apiPost("/api/analysis/preference-weighted-regime", {
            source: currentSource, well: currentWell,
            depth: parseFloat(depth), pore_pressure: pp ? parseFloat(pp) : null
        });

        document.getElementById("pwr-results").classList.remove("d-none");

        // Final recommendation
        var confColors = { HIGH: "success", MODERATE: "warning", LOW: "danger" };
        var fc = confColors[data.final_confidence] || "secondary";
        var recEl = document.getElementById("pwr-recommendation");
        recEl.className = "alert alert-" + fc + " mb-3";
        recEl.innerHTML =
            '<i class="bi bi-shield-check me-1"></i>' +
            '<strong>Recommended Regime: ' + (data.final_regime || "?").replace("_", " ").replace(/\b\w/g, function(l) { return l.toUpperCase(); }) + '</strong> ' +
            '<span class="badge bg-' + fc + ' ms-2">' + (data.final_confidence || "?") + '</span>' +
            '<div class="small mt-2">' + (data.adjustment || "") + '</div>';

        // Physics vs Expert comparison
        var physRes = data.physics_result || {};
        var expCon = data.expert_consensus || {};
        var compEl = document.getElementById("pwr-comparison");
        clearChildren(compEl);

        // Physics card
        var physCard = document.createElement("div");
        physCard.className = "col-md-6";
        physCard.innerHTML =
            '<div class="card h-100">' +
            '<div class="card-header"><i class="bi bi-calculator me-1"></i> Physics (Auto-Detection)</div>' +
            '<div class="card-body">' +
            '<div class="fs-5 fw-bold">' + (physRes.regime || "?").replace("_", " ") + '</div>' +
            '<div class="text-muted">Confidence: ' + (physRes.confidence || "?") + '</div>' +
            '<div class="text-muted">Misfit ratio: ' + fmt(physRes.misfit_ratio, 3) + '</div>' +
            '</div></div>';
        compEl.appendChild(physCard);

        // Expert card
        var expCard = document.createElement("div");
        expCard.className = "col-md-6";
        var expStatus = expCon.status || "NONE";
        var expBadge = expStatus === "STRONG" ? "bg-success" : expStatus === "WEAK" ? "bg-warning" : "bg-secondary";
        var expHtml =
            '<div class="card h-100">' +
            '<div class="card-header"><i class="bi bi-people me-1"></i> Expert Consensus</div>' +
            '<div class="card-body">';
        if (expCon.n_selections > 0) {
            expHtml +=
                '<div class="fs-5 fw-bold">' + (expCon.consensus_regime || "none").replace("_", " ") + '</div>' +
                '<div>Status: <span class="badge ' + expBadge + '">' + expStatus + '</span> (' + fmt(expCon.consensus_confidence, 0) + '%)</div>' +
                '<div class="text-muted">' + expCon.n_selections + ' total selections</div>';
            // Show regime breakdown
            var pcts = expCon.regime_pct || {};
            Object.keys(pcts).forEach(function(r) {
                expHtml += '<div class="d-flex align-items-center mt-1">' +
                    '<span class="me-2">' + r.replace("_", " ") + ':</span>' +
                    '<div class="progress flex-grow-1" style="height:14px;">' +
                    '<div class="progress-bar" style="width:' + pcts[r] + '%">' + pcts[r] + '%</div></div></div>';
            });
        } else {
            expHtml += '<div class="text-muted">No expert selections yet.</div>' +
                '<div class="small mt-2">Use <strong>Expert Stress Ranking</strong> above to submit your regime preference.</div>';
        }
        expHtml += '</div></div>';
        expCard.innerHTML = expHtml;
        compEl.appendChild(expCard);

        // Blend source badge
        var blendEl = document.getElementById("pwr-blend-source");
        var blendLabels = {
            "physics_only": ["Physics Only", "secondary"],
            "physics_expert_agreement": ["Physics + Expert Agreement", "success"],
            "expert_override": ["Expert Override", "warning"],
            "physics_with_expert_warning": ["Physics (Expert Disagrees)", "danger"]
        };
        var bl = blendLabels[data.blend_source] || ["Unknown", "secondary"];
        blendEl.innerHTML = 'Source: <span class="badge bg-' + bl[1] + '">' + bl[0] + '</span>';

        showToast("Preference-weighted analysis done (" + (data.elapsed_s || "?") + "s)");
    } catch (err) {
        showToast("Error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Regime Stability Check ────────────────────────

async function runRegimeStability() {
    showLoading("Testing regime stability under parameter variation...");
    try {
        var depth = document.getElementById("depth-input").value || 3000;
        var pp = document.getElementById("pp-input").value || 0;
        var data = await apiPost("/api/analysis/regime-stability", {
            source: currentSource, well: currentWell,
            depth: parseFloat(depth), pore_pressure: parseFloat(pp)
        });

        document.getElementById("rs-results").classList.remove("d-none");

        // Stability grade
        var colorMap = { success: "success", warning: "warning", danger: "danger" };
        var sc = data.stability_color || "secondary";
        var gradeEl = document.getElementById("rs-grade");
        gradeEl.className = "card mb-3 border-" + sc;
        document.getElementById("rs-stability-label").textContent = (data.stability || "?").replace("_", " ");
        document.getElementById("rs-stability-label").className = "display-6 fw-bold text-" + sc;
        document.getElementById("rs-baseline").textContent =
            "Baseline: " + (data.baseline_regime || "?").replace("_", " ") +
            " (" + (data.baseline_confidence || "?") + ")";
        document.getElementById("rs-message").textContent = data.message || "";

        // Perturbation table
        var tbody = document.getElementById("rs-perturbation-tbody");
        clearChildren(tbody);
        (data.perturbations || []).forEach(function(p) {
            var tr = document.createElement("tr");
            if (p.flipped) tr.className = "table-danger";
            tr.appendChild(createCell("td", p.test, { fontWeight: "600" }));
            tr.appendChild(createCell("td", (p.regime || "?").replace("_", " ")));
            tr.appendChild(createCell("td", p.confidence || "?"));
            tr.appendChild(createCell("td", fmt(p.misfit_ratio, 3)));
            var flipTd = document.createElement("td");
            if (p.flipped) {
                flipTd.innerHTML = '<span class="badge bg-danger">FLIPPED</span>';
            } else {
                flipTd.innerHTML = '<span class="badge bg-success">Stable</span>';
            }
            tr.appendChild(flipTd);
            tbody.appendChild(tr);
        });

        // Summary
        document.getElementById("rs-summary").textContent =
            data.flips + " of " + data.total_tests + " tests caused regime flip";

        showToast("Stability check done — " + (data.stability || "?") + " (" + (data.elapsed_s || "?") + "s)");
    } catch (err) {
        showToast("Stability check error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Expert Preference History ─────────────────────

async function loadExpertHistory() {
    showLoading("Loading expert preference history...");
    try {
        var url = "/api/analysis/expert-preference-history";
        if (currentWell) url += "?well=" + encodeURIComponent(currentWell);
        var data = await api(url);

        document.getElementById("eph-results").classList.remove("d-none");

        // Consensus summary
        var con = data.consensus || {};
        var statusBadge = con.status === "STRONG" ? "bg-success" :
                          con.status === "WEAK" ? "bg-warning" : "bg-secondary";
        document.getElementById("eph-consensus").innerHTML =
            '<span class="badge ' + statusBadge + ' me-2">' + (con.status || "NONE") + '</span>' +
            (con.consensus_regime ? con.consensus_regime.replace("_", " ") + ' (' + fmt(con.consensus_confidence, 0) + '%)' : 'No selections yet') +
            ' &mdash; ' + (con.n_selections || 0) + ' total selections';

        // Timeline
        var tbody = document.getElementById("eph-timeline-tbody");
        clearChildren(tbody);
        (data.timeline || []).slice(-20).forEach(function(t) {
            var tr = document.createElement("tr");
            tr.appendChild(createCell("td", t.step));
            tr.appendChild(createCell("td", (t.timestamp || "").substring(0, 19).replace("T", " ")));
            tr.appendChild(createCell("td", (t.regime || "").replace("_", " ")));
            tr.appendChild(createCell("td", (t.dominant_regime || "").replace("_", " ")));
            var pctTd = document.createElement("td");
            pctTd.innerHTML = '<div class="progress" style="height:16px;min-width:60px">' +
                '<div class="progress-bar" style="width:' + t.dominant_pct + '%">' + t.dominant_pct + '%</div></div>';
            tr.appendChild(pctTd);
            tbody.appendChild(tr);
        });

        // Well summaries
        var wsEl = document.getElementById("eph-well-summaries");
        clearChildren(wsEl);
        var ws = data.well_summaries || {};
        Object.keys(ws).forEach(function(w) {
            var s = ws[w];
            var badge = s.status === "STRONG" ? "bg-success" : s.status === "WEAK" ? "bg-warning" : "bg-secondary";
            var div = document.createElement("div");
            div.className = "col-md-4";
            div.innerHTML =
                '<div class="card">' +
                '<div class="card-body text-center">' +
                '<div class="fw-bold">Well ' + w + '</div>' +
                '<span class="badge ' + badge + '">' + s.status + '</span>' +
                '<div class="small text-muted">' + (s.consensus_regime || "none").replace("_", " ") +
                ' (' + fmt(s.consensus_confidence, 0) + '%), ' + s.n_selections + ' votes</div>' +
                '</div></div>';
            wsEl.appendChild(div);
        });

        showToast("Expert history loaded (" + (data.total_all_wells || 0) + " total selections)");
    } catch (err) {
        showToast("Error loading history: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

async function resetExpertPreferences() {
    if (!confirm("Reset all expert preferences for Well " + currentWell + "? This cannot be undone.")) return;
    try {
        var data = await apiPost("/api/analysis/expert-preference-reset", {
            well: currentWell
        });
        showToast(data.message || "Preferences reset");
        loadExpertHistory(); // Refresh
    } catch (err) {
        showToast("Reset error: " + err.message, "Error");
    }
}


// ── One-Click Comprehensive Report ─────────────────

async function runComprehensiveReport() {
    showLoading("Generating comprehensive report (7 modules)...");
    try {
        var depth = document.getElementById("depth-input").value || 3000;
        var pp = document.getElementById("pp-input").value || 0;
        var data = await apiPost("/api/report/comprehensive", {
            source: currentSource, well: currentWell,
            depth: parseFloat(depth), pore_pressure: parseFloat(pp)
        });

        document.getElementById("cr-results").classList.remove("d-none");

        // Verdict banner
        var vc = data.verdict_color || "secondary";
        var verdictEl = document.getElementById("cr-verdict");
        verdictEl.className = "alert alert-" + vc + " mb-3 text-center";
        var verdictLabels = { "GO": "GO \u2014 Safe for Operational Use", "CAUTION": "CAUTION \u2014 Validate Before Commitment", "NO_GO": "NO-GO \u2014 Do Not Use for Decisions" };
        document.getElementById("cr-verdict-label").textContent = verdictLabels[data.verdict] || data.verdict;
        var ss = data.signal_summary || {};
        document.getElementById("cr-signal-summary").innerHTML =
            '<span class="badge bg-success me-1">' + (ss.GREEN || 0) + ' GREEN</span>' +
            '<span class="badge bg-warning me-1">' + (ss.AMBER || 0) + ' AMBER</span>' +
            '<span class="badge bg-danger">' + (ss.RED || 0) + ' RED</span>';

        // Executive brief (render markdown bold)
        var brief = data.executive_brief || "";
        brief = brief.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        document.getElementById("cr-brief").innerHTML = brief;

        // Module cards
        var container = document.getElementById("cr-modules");
        clearChildren(container);
        var modules = data.modules || {};

        if (modules.data_quality) {
            var dq = modules.data_quality;
            var dqc = dq.score >= 70 ? "success" : dq.score >= 40 ? "warning" : "danger";
            addModuleCard(container, "Data Quality", "bi-database-check", dqc,
                "Score: " + dq.score + "/100 (" + dq.grade + ")",
                dq.anomaly_pct > 0 ? dq.anomaly_pct.toFixed(1) + "% anomalous" : "Clean data");
        }
        if (modules.stress_inversion) {
            var si = modules.stress_inversion;
            var sic = si.confidence === "HIGH" ? "success" : si.confidence === "MODERATE" ? "warning" : "danger";
            addModuleCard(container, "Stress Field", "bi-compass", sic,
                si.best_regime.replace("_", " ") + " regime, SHmax " + si.shmax_azimuth + "\u00B0",
                si.confidence + " confidence (misfit " + si.misfit_ratio + ")");
        }
        if (modules.classification) {
            var cl = modules.classification;
            var clc = cl.accuracy >= 0.75 ? "success" : cl.accuracy >= 0.55 ? "warning" : "danger";
            addModuleCard(container, "ML Classification", "bi-cpu", clc,
                (cl.accuracy * 100).toFixed(1) + "% accuracy",
                cl.n_classes + " fracture types classified");
        }
        if (modules.critically_stressed) {
            var cs = modules.critically_stressed;
            var csc = cs.risk_level === "LOW" ? "success" : cs.risk_level === "MODERATE" ? "warning" : "danger";
            addModuleCard(container, "Critically Stressed", "bi-exclamation-triangle", csc,
                cs.pct_critical + "% critical (" + cs.risk_level + " risk)",
                cs.count_critical + " of " + cs.n_total + " fractures");
        }
        if (modules.regime_stability) {
            var rs = modules.regime_stability;
            var rsc = rs.stability === "STABLE" ? "success" : rs.stability === "MOSTLY_STABLE" ? "warning" : "danger";
            addModuleCard(container, "Regime Stability", "bi-shield-check", rsc,
                rs.stability.replace("_", " "),
                rs.flips + " of " + rs.total_tests + " perturbations flip regime");
        }
        if (modules.expert_consensus && modules.expert_consensus.n_selections > 0) {
            var ec = modules.expert_consensus;
            var ecc = ec.status === "STRONG" ? "success" : ec.status === "WEAK" ? "warning" : "secondary";
            addModuleCard(container, "Expert Consensus", "bi-people", ecc,
                (ec.regime || "none").replace("_", " ") + " (" + ec.status + ")",
                ec.n_selections + " selections, " + ec.confidence_pct.toFixed(0) + "% agreement");
        }

        // Errors
        var errEl = document.getElementById("cr-errors");
        clearChildren(errEl);
        if (data.errors && data.errors.length > 0) {
            errEl.classList.remove("d-none");
            data.errors.forEach(function(e) {
                var div = document.createElement("div");
                div.className = "alert alert-warning py-1 mb-1 small";
                div.textContent = e;
                errEl.appendChild(div);
            });
        } else {
            errEl.classList.add("d-none");
        }

        showToast("Comprehensive report: " + data.verdict + " (" + (data.elapsed_s || "?") + "s)");
    } catch (err) {
        showToast("Report error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

function addModuleCard(container, title, icon, color, main, detail) {
    var card = document.createElement("div");
    card.className = "col-md-4 col-lg-3";
    card.innerHTML =
        '<div class="card h-100 border-' + color + '">' +
        '<div class="card-body p-2">' +
        '<div class="d-flex align-items-center mb-1">' +
        '<i class="bi ' + icon + ' text-' + color + ' me-1"></i>' +
        '<strong class="small">' + title + '</strong>' +
        '</div>' +
        '<div class="fw-bold text-' + color + '">' + main + '</div>' +
        '<div class="small text-muted">' + detail + '</div>' +
        '</div></div>';
    container.appendChild(card);
}


// ── Prediction Trustworthiness Report ──────────────

async function runTrustworthinessReport() {
    showLoading("Running comprehensive trustworthiness audit (5 checks)...");
    try {
        var data = await apiPost("/api/analysis/trustworthiness-report", {
            source: currentSource, well: currentWell
        });

        document.getElementById("tr-results").classList.remove("d-none");

        // Overall trust level
        var colorMap = { success: "success", warning: "warning", danger: "danger" };
        var tc = data.trust_color || "secondary";
        document.getElementById("tr-overall").className = "card mb-3 border-" + tc;
        document.getElementById("tr-trust-level").textContent = (data.trust_level || "?") + " Trustworthiness";
        document.getElementById("tr-trust-level").className = "fw-bold mb-1 text-" + tc;
        document.getElementById("tr-score").textContent = "Score: " + (data.overall_score || "?") + " / 100";
        document.getElementById("tr-advice").textContent = data.trust_advice || "";

        // Check cards
        var container = document.getElementById("tr-checks");
        clearChildren(container);
        var gradeColors = { GREEN: "success", AMBER: "warning", RED: "danger" };
        (data.checks || []).forEach(function(c) {
            var gc = gradeColors[c.grade] || "secondary";
            var card = document.createElement("div");
            card.className = "col-md-6 col-lg-4";
            var issuesHtml = "";
            if (c.issues && c.issues.length > 0) {
                issuesHtml = '<ul class="small mb-0 mt-2">';
                c.issues.forEach(function(i) {
                    issuesHtml += '<li class="text-danger">' + i + '</li>';
                });
                issuesHtml += '</ul>';
            }
            card.innerHTML =
                '<div class="card h-100 border-' + gc + '">' +
                '<div class="card-body">' +
                '<div class="d-flex justify-content-between align-items-start">' +
                '<div><i class="bi ' + (c.icon || "bi-check") + ' me-1"></i><strong>' + c.name + '</strong></div>' +
                '<span class="badge bg-' + gc + '">' + c.score + '</span>' +
                '</div>' +
                '<div class="small text-muted mt-1">' + (c.detail || "") + '</div>' +
                '<div class="small mt-1 fw-bold">' + (c.action || "") + '</div>' +
                issuesHtml +
                '</div></div>';
            container.appendChild(card);
        });

        // All issues (aggregated)
        var issues = data.all_issues || [];
        var issuesSection = document.getElementById("tr-issues-section");
        var issuesEl = document.getElementById("tr-issues");
        clearChildren(issuesEl);
        if (issues.length > 0) {
            issuesSection.classList.remove("d-none");
            issues.forEach(function(i) {
                var ic = gradeColors[i.grade] || "secondary";
                var div = document.createElement("div");
                div.className = "alert alert-" + ic + " py-2 mb-2";
                div.innerHTML =
                    '<span class="badge bg-' + ic + ' me-1">' + i.grade + '</span>' +
                    '<strong>' + i.check + ':</strong> ' + i.issue;
                issuesEl.appendChild(div);
            });
        } else {
            issuesSection.classList.add("d-none");
        }

        showToast("Trustworthiness audit complete — " + (data.trust_level || "?") + " (" + (data.elapsed_s || "?") + "s)");
    } catch (err) {
        showToast("Trustworthiness report error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}


// ── Glossary Tooltips ─────────────────────────────

var GLOSSARY = {
    "SHmax": "Maximum horizontal stress direction — the azimuth where the Earth's crust pushes hardest. Critical for deciding well trajectory.",
    "Mohr-Coulomb": "Failure criterion predicting when rock fractures slip. Uses friction and pore pressure to determine critical stress state.",
    "R-ratio": "Stress shape parameter (0-1). Describes how the intermediate stress compares to the maximum and minimum.",
    "Critically stressed": "Fractures close to slipping under current stress — likely fluid conduits that can cause drilling problems.",
    "Pore pressure": "Fluid pressure inside rock pores. Higher pore pressure makes fractures more likely to slip.",
    "Slip tendency": "How close a fracture is to sliding. Higher values mean higher risk.",
    "Dilation tendency": "How likely a fracture is to open. Important for predicting fluid flow paths.",
    "SMOTE": "Synthetic Minority Over-sampling — creates artificial training samples for rare fracture types to improve classification.",
    "Bootstrap CI": "Confidence interval estimated by repeatedly resampling the data — shows how reliable a measurement is.",
    "OOD": "Out-Of-Distribution — when new data looks very different from the training data, meaning predictions may be unreliable.",
    "ECE": "Expected Calibration Error — measures whether predicted probabilities match actual frequencies. Lower is better.",
    "SHAP": "SHapley Additive exPlanations — shows which features influenced each prediction most.",
    "Byerlee": "Byerlee's law — rock friction coefficient typically 0.6-0.85. Values outside this range suggest unusual conditions.",
};

function initTooltips() {
    // Re-initialize tooltips on all [title] and [data-bs-toggle=tooltip] elements
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function(el) {
        bootstrap.Tooltip.getOrCreateInstance(el);
    });
    document.querySelectorAll('[title]:not([data-bs-toggle]):not([data-tooltip-done])').forEach(function(el) {
        new bootstrap.Tooltip(el, { trigger: 'hover', placement: 'top' });
        el.setAttribute("data-tooltip-done", "1");
    });
}

// Auto-initialize tooltips on dynamically added content
var _tooltipObserver = new MutationObserver(function(mutations) {
    var hasNew = false;
    mutations.forEach(function(m) {
        if (m.addedNodes.length > 0) hasNew = true;
    });
    if (hasNew) {
        clearTimeout(_tooltipObserver._timer);
        _tooltipObserver._timer = setTimeout(initTooltips, 200);
    }
});
_tooltipObserver._timer = null;


// ── Calibrated Ensemble Prediction ───────────────────

async function runEnsemblePredict() {
    const well = document.getElementById('well-select')?.value || '3P';
    showLoading('Training ensemble of all models...');
    try {
        const res = await fetch('/api/analysis/ensemble-predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ well, source: 'demo' }),
        });
        const d = await res.json();
        hideLoading();

        const el = id => document.getElementById(id);
        el('ens-results').classList.remove('d-none');

        const ens = d.ensemble;
        el('ens-acc').textContent = (ens.weighted_accuracy * 100).toFixed(1) + '%';
        el('ens-n-models').textContent = ens.n_models + ' models combined';

        const agr = ens.avg_agreement;
        el('ens-agreement').textContent = (agr * 100).toFixed(0) + '%';
        el('ens-agreement').style.color = agr > 0.8 ? '#28a745' : agr > 0.6 ? '#ffc107' : '#dc3545';

        el('ens-uncertain').textContent = d.uncertain_samples?.length || 0;
        el('ens-interpretation').innerHTML = '<i class="bi bi-info-circle"></i> ' + ens.interpretation;

        // Per-model bars
        const modelsEl = el('ens-models');
        let mhtml = '<h6>Per-Model Performance</h6>';
        const sorted = [...d.models].sort((a, b) => b.accuracy - a.accuracy);
        sorted.forEach(m => {
            const pct = (m.accuracy * 100).toFixed(1);
            const wgt = (m.weight * 100).toFixed(0);
            mhtml += `
                <div class="d-flex align-items-center mb-1 small">
                    <div style="width:120px" class="text-end pe-2 fw-bold">${m.model}</div>
                    <div class="flex-grow-1">
                        <div class="progress" style="height:18px">
                            <div class="progress-bar ${m.accuracy > 0.8 ? 'bg-success' : m.accuracy > 0.6 ? 'bg-warning' : 'bg-danger'}"
                                 style="width:${pct}%">${pct}%</div>
                        </div>
                    </div>
                    <div style="width:50px" class="text-end text-muted">${wgt}%w</div>
                </div>`;
        });
        modelsEl.innerHTML = mhtml;

        // Uncertain samples table
        if (d.uncertain_samples && d.uncertain_samples.length > 0) {
            el('ens-uncertain-table').classList.remove('d-none');
            const tbody = el('ens-uncertain-tbody');
            tbody.innerHTML = '';
            d.uncertain_samples.slice(0, 5).forEach(s => {
                const votes = Object.entries(s.model_predictions || {})
                    .map(([m, p]) => `<span class="badge bg-secondary bg-opacity-50 me-1">${m.replace(/_/g,' ')}: ${p}</span>`)
                    .join('');
                tbody.innerHTML += `
                    <tr>
                        <td>${s.depth || '-'}</td>
                        <td>${s.azimuth}</td>
                        <td>${s.dip}</td>
                        <td><strong>${s.true_type || '-'}</strong></td>
                        <td><span class="badge ${s.agreement < 0.5 ? 'bg-danger' : s.agreement < 0.8 ? 'bg-warning' : 'bg-success'}">${(s.agreement * 100).toFixed(0)}%</span></td>
                        <td class="small">${votes}</td>
                    </tr>`;
            });
        }

        if (d.errors && d.errors.length > 0) {
            modelsEl.innerHTML += `<div class="alert alert-warning py-1 small mt-2">${d.errors.length} model(s) failed: ${d.errors.join(', ')}</div>`;
        }
    } catch (e) {
        hideLoading();
        alert('Ensemble prediction failed: ' + e.message);
    }
}


// ── Adversarial Robustness Test ──────────────────────

async function runAugmentedClassify() {
    const well = document.getElementById('well-select')?.value || '3P';
    const noise = parseFloat(document.getElementById('aug-noise')?.value || '5');
    showLoading('Running adversarial robustness test...');
    try {
        const res = await fetch('/api/analysis/augmented-classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ well, noise_std: noise, source: 'demo' }),
        });
        const d = await res.json();
        hideLoading();

        const el = id => document.getElementById(id);
        el('aug-results').classList.remove('d-none');

        el('aug-orig-acc').textContent = (d.original.accuracy * 100).toFixed(1) + '%';
        el('aug-orig-n').textContent = d.original.n_samples + ' samples';
        el('aug-new-acc').textContent = (d.augmented.accuracy * 100).toFixed(1) + '%';
        el('aug-new-n').textContent = d.augmented.n_samples + ' samples (+' + d.augmented.n_added + ')';

        const rob = d.comparison.robustness;
        el('aug-robustness').textContent = rob;
        el('aug-change').textContent = (d.comparison.accuracy_change >= 0 ? '+' : '') +
            (d.comparison.accuracy_change * 100).toFixed(1) + '%';

        const verdictEl = el('aug-verdict');
        verdictEl.style.borderLeft = rob === 'ROBUST' ? '4px solid #28a745' :
            rob === 'IMPROVED' ? '4px solid #007bff' : '4px solid #dc3545';

        el('aug-interpretation').innerHTML = '<i class="bi bi-info-circle"></i> ' + d.comparison.interpretation;
        el('aug-interpretation').className = 'alert small ' + (
            rob === 'ROBUST' ? 'alert-success' :
            rob === 'IMPROVED' ? 'alert-info' : 'alert-danger'
        );
    } catch (e) {
        hideLoading();
        alert('Robustness test failed: ' + e.message);
    }
}


// ── Glossary ────────────────────────────────────────

async function loadGlossary() {
    try {
        const res = await fetch('/api/help/glossary');
        const d = await res.json();
        const body = document.getElementById('glossary-body');
        if (!body || !d.terms) return;

        let html = '<div class="input-group mb-3"><span class="input-group-text"><i class="bi bi-search"></i></span>' +
            '<input type="text" class="form-control" id="glossary-search" placeholder="Search terms..." oninput="filterGlossary(this.value)"></div>';
        html += '<div id="glossary-items">';

        for (const [key, term] of Object.entries(d.terms)) {
            html += `
                <div class="glossary-item card mb-2" data-term="${key}">
                    <div class="card-body py-2">
                        <h6 class="mb-1">
                            <i class="bi bi-${term.icon || 'circle'} text-primary me-1"></i>
                            ${term.term}
                        </h6>
                        <p class="mb-1 fw-semibold" style="color:#2c5f2d">${term.plain}</p>
                        <p class="mb-1 small text-muted">${term.detail}</p>
                        <div class="alert alert-warning py-1 px-2 mb-0 small">
                            <strong>Why it matters:</strong> ${term.why_it_matters}
                        </div>
                    </div>
                </div>`;
        }
        html += '</div>';
        body.innerHTML = html;
    } catch (e) {
        console.warn('Glossary load failed:', e);
    }
}

function filterGlossary(query) {
    const items = document.querySelectorAll('.glossary-item');
    const q = query.toLowerCase();
    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(q) ? '' : 'none';
    });
}

// Load glossary when modal opens
document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('glossaryModal');
    if (modal) {
        modal.addEventListener('shown.bs.modal', function() {
            if (document.getElementById('glossary-body')?.textContent?.includes('Loading')) {
                loadGlossary();
            }
        });
    }
});


// ── PDF Report Download ─────────────────────────────

async function downloadPdfReport() {
    const well = document.getElementById('well-select')?.value || '3P';
    const depth = document.getElementById('depth-input')?.value || '3000';
    showLoading('Generating PDF report...');
    try {
        const res = await fetch('/api/report/pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ well, depth: parseFloat(depth), source: 'demo' }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.message || 'PDF generation failed');
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `geostress_report_${well}_${new Date().toISOString().slice(0,10)}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        hideLoading();
    } catch (e) {
        hideLoading();
        alert('PDF generation failed: ' + e.message);
    }
}


// ── Negative Scenario Check ──────────────────────────

async function runScenarioCheck() {
    const well = document.getElementById('well-select')?.value || '3P';
    showLoading('Checking failure scenarios...');
    try {
        const res = await fetch('/api/analysis/scenario-check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ well, source: 'demo' }),
        });
        const d = await res.json();
        hideLoading();

        const el = id => document.getElementById(id);
        el('sc-results').classList.remove('d-none');

        // Overall status banner
        const statusEl = el('sc-overall');
        const statusColors = {
            'SAFE': 'alert-success',
            'MINOR_ISSUES': 'alert-info',
            'CAUTION': 'alert-warning',
            'CRITICAL_ISSUES': 'alert-danger',
        };
        statusEl.className = `alert mb-3 text-center ${statusColors[d.overall_status] || 'alert-secondary'}`;
        el('sc-status').textContent = d.overall_status.replace(/_/g, ' ');
        el('sc-summary').textContent =
            `${d.n_triggered} of ${d.n_total_scenarios} scenarios triggered | Well: ${d.well} | ${d.n_fractures} fractures`;

        // Triggered scenarios
        const triggeredEl = el('sc-triggered');
        triggeredEl.innerHTML = '';
        if (d.triggered && d.triggered.length > 0) {
            d.triggered.forEach(s => {
                const sevColor = s.severity === 'CRITICAL' ? 'danger' :
                                 s.severity === 'HIGH' ? 'warning' : 'info';
                triggeredEl.innerHTML += `
                    <div class="card mb-2 border-${sevColor}">
                        <div class="card-header bg-${sevColor} bg-opacity-10 d-flex justify-content-between">
                            <strong>${s.id}: ${s.name}</strong>
                            <span class="badge bg-${sevColor}">${s.severity}</span>
                        </div>
                        <div class="card-body small">
                            <p>${s.description}</p>
                            <div class="alert alert-${sevColor} py-1 px-2 mb-2">
                                <strong>Evidence:</strong> ${s.evidence}
                            </div>
                            <div class="row g-2">
                                <div class="col-md-6">
                                    <strong>Consequence:</strong> ${s.consequence}
                                </div>
                                <div class="col-md-6">
                                    <strong>Mitigation:</strong> ${s.mitigation}
                                </div>
                            </div>
                        </div>
                    </div>`;
            });
        } else {
            triggeredEl.innerHTML = '<div class="text-success text-center py-3"><i class="bi bi-check-circle fs-3"></i><p>No failure scenarios triggered</p></div>';
        }

        // Safe checks
        if (d.not_triggered && d.not_triggered.length > 0) {
            el('sc-safe').classList.remove('d-none');
            el('sc-safe-list').innerHTML = d.not_triggered.map(s =>
                `<span class="badge bg-success bg-opacity-25 text-success me-1 mb-1">${s.id}: ${s.reason}</span>`
            ).join('');
        }
    } catch (e) {
        hideLoading();
        alert('Scenario check failed: ' + e.message);
    }
}

async function viewScenarioLibrary() {
    showLoading('Loading scenario library...');
    try {
        const res = await fetch('/api/analysis/negative-scenarios');
        const d = await res.json();
        hideLoading();

        const el = document.getElementById('sc-results');
        el.classList.remove('d-none');

        const triggered = document.getElementById('sc-triggered');
        triggered.innerHTML = '<h6><i class="bi bi-book"></i> Complete Failure Scenario Library</h6>';
        d.scenarios.forEach(s => {
            const sevColor = s.severity === 'CRITICAL' ? 'danger' :
                             s.severity === 'HIGH' ? 'warning' : 'info';
            triggered.innerHTML += `
                <div class="card mb-2 border-${sevColor}">
                    <div class="card-header bg-${sevColor} bg-opacity-10 d-flex justify-content-between">
                        <strong>${s.id}: ${s.name}</strong>
                        <div>
                            <span class="badge bg-secondary me-1">${s.category}</span>
                            <span class="badge bg-${sevColor}">${s.severity}</span>
                        </div>
                    </div>
                    <div class="card-body small">
                        <p>${s.description}</p>
                        <div class="row g-2">
                            <div class="col-md-4"><strong>Trigger:</strong> ${s.trigger}</div>
                            <div class="col-md-4"><strong>Consequence:</strong> ${s.consequence}</div>
                            <div class="col-md-4"><strong>Mitigation:</strong> ${s.mitigation}</div>
                        </div>
                    </div>
                </div>`;
        });
    } catch (e) {
        hideLoading();
        alert('Failed to load library: ' + e.message);
    }
}


// ── Database Management ──────────────────────────────

async function loadDbStats() {
    try {
        const res = await fetch('/api/db/stats');
        const d = await res.json();
        const el = id => document.getElementById(id);
        if (el('db-audit-count')) el('db-audit-count').textContent = d.audit_count || 0;
        if (el('db-model-count')) el('db-model-count').textContent = d.model_count || 0;
        if (el('db-pref-count')) el('db-pref-count').textContent = d.preference_count || 0;
        if (el('db-size')) el('db-size').textContent = `${d.db_size_kb || 0} KB`;
    } catch (e) {
        console.warn('DB stats unavailable:', e);
    }
}

async function exportDatabase() {
    try {
        showLoading('Exporting database backup...');
        const res = await fetch('/api/db/export', { method: 'POST' });
        const data = await res.json();
        // Download as JSON file
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `geostress_backup_${new Date().toISOString().slice(0,10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        hideLoading();
        loadDbStats();
    } catch (e) {
        hideLoading();
        alert('Export failed: ' + e.message);
    }
}

async function importDatabase(event) {
    const file = event.target.files[0];
    if (!file) return;
    if (!confirm('Import backup data? This will ADD records to the existing database.')) {
        event.target.value = '';
        return;
    }
    try {
        showLoading('Importing database backup...');
        const text = await file.text();
        const data = JSON.parse(text);
        const res = await fetch('/api/db/import', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        const result = await res.json();
        hideLoading();
        alert(`Imported: ${result.counts.audit} audit, ${result.counts.models} models, ${result.counts.preferences} preferences`);
        loadDbStats();
    } catch (e) {
        hideLoading();
        alert('Import failed: ' + e.message);
    }
    event.target.value = '';
}


// ── Instant Startup Snapshot ─────────────────────────

async function loadStartupSnapshot() {
    try {
        const res = await fetch('/api/snapshot');
        const d = await res.json();
        if (d.status === 'warming') {
            // Caches still building — retry in 5s
            setTimeout(loadStartupSnapshot, 5000);
            return;
        }

        // Build instant executive summary from snapshot
        const container = document.getElementById('exec-sections');
        if (!container) return;

        let html = '';

        // Quick status cards
        html += '<div class="row g-3 mb-4">';
        html += `<div class="col-md-3"><div class="metric-card"><div class="metric-label">Total Fractures</div><div class="metric-value">${d.total_fractures || 0}</div><small class="text-muted">${d.n_wells} wells</small></div></div>`;

        const allAlerts = [];
        for (const [w, ws] of Object.entries(d.wells || {})) {
            (ws.alerts || []).forEach(a => allAlerts.push({...a, well: w}));
        }
        const critCount = allAlerts.filter(a => a.severity === 'CRITICAL').length;
        const highCount = allAlerts.filter(a => a.severity === 'HIGH').length;
        const alertColor = critCount > 0 ? 'text-danger' : highCount > 0 ? 'text-warning' : 'text-success';
        const alertText = critCount + highCount === 0 ? 'No Issues' : `${critCount + highCount} Alerts`;
        html += `<div class="col-md-3"><div class="metric-card"><div class="metric-label">Data Alerts</div><div class="metric-value ${alertColor}">${alertText}</div><small class="text-muted">${critCount} critical, ${highCount} high</small></div></div>`;

        html += `<div class="col-md-3"><div class="metric-card"><div class="metric-label">Expert Feedback</div><div class="metric-value">${d.expert_consensus?.status || 'NONE'}</div><small class="text-muted">${d.expert_consensus?.n_selections || 0} selections</small></div></div>`;

        html += `<div class="col-md-3"><div class="metric-card"><div class="metric-label">Audit Trail</div><div class="metric-value">${d.db?.audit_records || 0}</div><small class="text-muted">${d.db?.model_runs || 0} model runs</small></div></div>`;
        html += '</div>';

        // Per-well cards
        html += '<div class="row g-3 mb-4">';
        for (const [wellName, ws] of Object.entries(d.wells || {})) {
            const confColor = ws.regime_confidence === 'HIGH' ? 'success' :
                              ws.regime_confidence === 'MODERATE' ? 'warning' : 'danger';
            html += `
                <div class="col-md-6">
                    <div class="card border-${confColor}">
                        <div class="card-header bg-${confColor} bg-opacity-10 d-flex justify-content-between">
                            <strong><i class="bi bi-geo-alt"></i> Well ${wellName}</strong>
                            <span class="badge bg-${confColor}">${ws.regime_confidence} Confidence</span>
                        </div>
                        <div class="card-body">
                            <div class="row g-2 text-center mb-2">
                                <div class="col-3">
                                    <div class="fw-bold">${ws.n_fractures}</div>
                                    <small class="text-muted">Fractures</small>
                                </div>
                                <div class="col-3">
                                    <div class="fw-bold text-primary">${ws.regime || '?'}</div>
                                    <small class="text-muted">Regime</small>
                                </div>
                                <div class="col-3">
                                    <div class="fw-bold">${ws.shmax_azimuth || 0}&deg;</div>
                                    <small class="text-muted">SHmax</small>
                                </div>
                                <div class="col-3">
                                    <div class="fw-bold">${ws.accuracy ? (ws.accuracy * 100).toFixed(0) + '%' : 'N/A'}</div>
                                    <small class="text-muted">ML Acc.</small>
                                </div>
                            </div>
                            <div class="small text-muted mb-1">
                                &sigma;1 = ${ws.sigma1} MPa | &sigma;3 = ${ws.sigma3} MPa
                                ${ws.depth_range ? ` | Depth: ${ws.depth_range[0]}-${ws.depth_range[1]}m` : ''}
                            </div>`;

            // Fracture type badges
            if (ws.type_distribution) {
                html += '<div class="mt-1">';
                for (const [ft, count] of Object.entries(ws.type_distribution)) {
                    html += `<span class="badge bg-secondary bg-opacity-50 me-1">${ft}: ${count}</span>`;
                }
                html += '</div>';
            }

            // Alerts
            if (ws.alerts && ws.alerts.length > 0) {
                html += '<div class="mt-2">';
                ws.alerts.forEach(a => {
                    const ac = a.severity === 'CRITICAL' ? 'danger' : 'warning';
                    html += `<div class="alert alert-${ac} py-1 px-2 mb-1 small"><i class="bi bi-exclamation-triangle"></i> ${a.msg}</div>`;
                });
                html += '</div>';
            }

            html += '</div></div></div>';
        }
        html += '</div>';

        // Quick action prompts
        html += `
            <div class="alert alert-light border small">
                <strong><i class="bi bi-lightbulb"></i> Quick Actions:</strong>
                Click <em>Generate Summary</em> above for detailed analysis, or use the
                <button class="btn btn-sm btn-outline-info py-0 px-1" onclick="document.querySelector('[data-bs-target=\\'#glossaryModal\\']').click()">
                <i class="bi bi-question-circle"></i> Help</button> button for term explanations.
                ${critCount > 0 ? '<span class="text-danger fw-bold ms-2">CRITICAL alerts require attention before proceeding.</span>' : ''}
            </div>`;

        container.innerHTML = html;
    } catch (e) {
        console.warn('Startup snapshot unavailable:', e);
    }
}


// ── v3.3.0: Production MLOps Functions ─────────────

async function loadSystemHealth() {
    const el = document.getElementById('system-health-result');
    if (!el) return;
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Checking system health...</div>';
    try {
        const r = await fetch('/api/system/health');
        const d = await r.json();
        const statusColor = d.status === 'HEALTHY' ? 'success' : d.status === 'DEGRADED' ? 'warning' : 'danger';
        let html = `
            <div class="row g-3">
                <div class="col-md-3">
                    <div class="card border-${statusColor} h-100">
                        <div class="card-body text-center">
                            <h3 class="text-${statusColor}">${d.health_score}</h3>
                            <span class="badge bg-${statusColor}">${d.status}</span>
                            <div class="small text-muted mt-1">Health Score</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card h-100">
                        <div class="card-body text-center">
                            <h3>${d.total_cached_items || 0}</h3>
                            <div class="small text-muted">Cached Items</div>
                            <div class="small">${d.snapshot_ready ? '<span class="text-success">Snapshot Ready</span>' : '<span class="text-warning">Warming Up</span>'}</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card h-100">
                        <div class="card-body text-center">
                            <h3>${d.failure_rate || 0}%</h3>
                            <div class="small text-muted">Failure Rate</div>
                            <div class="small">${d.unresolved_failures || 0} unresolved</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card h-100">
                        <div class="card-body text-center">
                            <h3>${(d.active_models || []).length}</h3>
                            <div class="small text-muted">Active Models</div>
                            <div class="small">v${d.app_version || '?'}</div>
                        </div>
                    </div>
                </div>
            </div>`;

        // Drift status
        if (d.drift_status) {
            html += '<div class="mt-3"><strong>Drift Status:</strong> ';
            for (const [w, s] of Object.entries(d.drift_status)) {
                const dColor = s === 'NO_BASELINE' ? 'secondary' : 'success';
                html += `<span class="badge bg-${dColor} me-2">${w}: ${s}</span>`;
            }
            html += '</div>';
        }

        // Database stats
        if (d.database) {
            html += `<div class="mt-2 small text-muted">DB: ${d.database.audit_records} audit, ${d.database.model_versions} versions, ${d.database.failure_cases} failures (${d.database.db_size_kb}KB)</div>`;
        }

        // Recommendations
        if (d.recommendations && d.recommendations.length) {
            html += '<div class="alert alert-info mt-3 mb-0 small"><strong>Recommendations:</strong><ul class="mb-0">';
            d.recommendations.forEach(r => { html += `<li>${r}</li>`; });
            html += '</ul></div>';
        }
        el.innerHTML = html;
    } catch(e) {
        el.innerHTML = `<div class="text-danger">Error: ${e.message}</div>`;
    }
}

async function runDriftDetection() {
    const el = document.getElementById('drift-result');
    const well = document.getElementById('well-select')?.value || '3P';
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Analyzing feature distributions...</div>';
    try {
        const r = await fetch('/api/analysis/drift-detection', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({well, source: window._source || 'demo'})
        });
        const d = await r.json();
        if (d.status === 'BASELINE_SET') {
            el.innerHTML = `<div class="alert alert-success"><i class="bi bi-check-circle"></i> ${d.message}</div>`;
            return;
        }
        const statusColor = d.status === 'OK' ? 'success' : d.status === 'WARNING' ? 'warning' : 'danger';
        let html = `
            <div class="alert alert-${statusColor}">
                <strong>${d.status}</strong>: ${d.recommendation}
            </div>
            <div class="row g-2 mb-3">
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h4>${d.avg_psi}</h4><small>Avg PSI</small></div></div></div>
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h4>${d.pct_drifted}%</h4><small>Features Drifted</small></div></div></div>
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h4>${d.n_features_checked}</h4><small>Features Checked</small></div></div></div>
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h4>${d.current_samples}</h4><small>Current Samples</small></div></div></div>
            </div>`;
        if (d.features && d.features.length) {
            html += '<table class="table table-sm table-hover"><thead><tr><th>Feature</th><th>PSI</th><th>KS p-value</th><th>Mean Shift (σ)</th><th>Severity</th></tr></thead><tbody>';
            d.features.forEach(f => {
                const sc = f.severity === 'OK' ? 'success' : f.severity === 'WARNING' ? 'warning' : 'danger';
                html += `<tr><td><code>${f.feature}</code></td><td>${f.psi}</td><td>${f.ks_pvalue}</td><td>${f.mean_shift_sigma}</td><td><span class="badge bg-${sc}">${f.severity}</span></td></tr>`;
            });
            html += '</tbody></table>';
        }
        el.innerHTML = html;
    } catch(e) {
        el.innerHTML = `<div class="text-danger">Error: ${e.message}</div>`;
    }
}

async function resetDriftBaseline() {
    if (!confirm('Reset drift baseline for this well? This cannot be undone.')) return;
    const well = document.getElementById('well-select')?.value || '3P';
    try {
        const r = await fetch('/api/analysis/drift-reset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({well, source: window._source || 'demo'})
        });
        const d = await r.json();
        document.getElementById('drift-result').innerHTML = `<div class="alert alert-success"><i class="bi bi-check-circle"></i> ${d.message}</div>`;
    } catch(e) {
        document.getElementById('drift-result').innerHTML = `<div class="text-danger">Error: ${e.message}</div>`;
    }
}

async function loadModelRegistry() {
    const el = document.getElementById('model-registry-result');
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Loading model versions...</div>';
    try {
        const r = await fetch('/api/models/registry');
        const d = await r.json();
        if (!d.versions || d.versions.length === 0) {
            el.innerHTML = '<div class="alert alert-info">No model versions registered yet. Click "Register" to register the current model.</div>';
            return;
        }
        let html = '<table class="table table-sm table-hover"><thead><tr><th>Ver</th><th>Model</th><th>Well</th><th>Accuracy</th><th>F1</th><th>Samples</th><th>Active</th><th>Date</th><th>Action</th></tr></thead><tbody>';
        d.versions.forEach(v => {
            html += `<tr class="${v.is_active ? 'table-success' : ''}">
                <td><strong>v${v.version}</strong></td>
                <td>${v.model_type}</td>
                <td>${v.well || 'All'}</td>
                <td>${v.accuracy ? (v.accuracy * 100).toFixed(1) + '%' : '-'}</td>
                <td>${v.f1 ? (v.f1 * 100).toFixed(1) + '%' : '-'}</td>
                <td>${v.n_samples || '-'}</td>
                <td>${v.is_active ? '<span class="badge bg-success">ACTIVE</span>' : '<span class="badge bg-secondary">old</span>'}</td>
                <td><small>${v.timestamp ? v.timestamp.substring(0, 10) : '-'}</small></td>
                <td>${!v.is_active ? `<button class="btn btn-sm btn-outline-warning py-0" onclick="rollbackModel('${v.model_type}', ${v.version}, '${v.well || ''}')">Rollback</button>` : '<span class="text-success">current</span>'}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        el.innerHTML = html;
    } catch(e) {
        el.innerHTML = `<div class="text-danger">Error: ${e.message}</div>`;
    }
}

async function registerModel() {
    const el = document.getElementById('model-registry-result');
    const well = document.getElementById('well-select')?.value || '3P';
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Registering model version...</div>';
    try {
        const r = await fetch('/api/models/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({well, source: window._source || 'demo'})
        });
        const d = await r.json();
        el.innerHTML = `<div class="alert alert-success"><i class="bi bi-check-circle"></i> ${d.message}</div>`;
        setTimeout(() => loadModelRegistry(), 500);
    } catch(e) {
        el.innerHTML = `<div class="text-danger">Error: ${e.message}</div>`;
    }
}

async function compareModelVersions() {
    const el = document.getElementById('model-registry-result');
    const well = document.getElementById('well-select')?.value || '3P';
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Comparing versions...</div>';
    try {
        const r = await fetch('/api/models/compare-versions', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({well})
        });
        const d = await r.json();
        if (d.message && !d.verdict) {
            el.innerHTML = `<div class="alert alert-info">${d.message}</div>`;
            return;
        }
        const vc = d.verdict === 'IMPROVED' ? 'success' : d.verdict === 'DEGRADED' ? 'danger' : 'info';
        let html = `
            <div class="alert alert-${vc}"><strong>${d.verdict}</strong>: ${d.recommendation}</div>
            <div class="row g-3">
                <div class="col-md-6">
                    <div class="card border-success"><div class="card-header bg-success text-white">Latest (v${d.latest?.version})</div>
                    <div class="card-body">
                        <div>Accuracy: <strong>${d.latest?.accuracy ? (d.latest.accuracy * 100).toFixed(1) + '%' : '-'}</strong></div>
                        <div>F1: <strong>${d.latest?.f1 ? (d.latest.f1 * 100).toFixed(1) + '%' : '-'}</strong></div>
                        <div>Samples: ${d.latest?.n_samples || '-'}</div>
                    </div></div>
                </div>
                <div class="col-md-6">
                    <div class="card border-secondary"><div class="card-header">Previous (v${d.previous?.version})</div>
                    <div class="card-body">
                        <div>Accuracy: <strong>${d.previous?.accuracy ? (d.previous.accuracy * 100).toFixed(1) + '%' : '-'}</strong></div>
                        <div>F1: <strong>${d.previous?.f1 ? (d.previous.f1 * 100).toFixed(1) + '%' : '-'}</strong></div>
                        <div>Samples: ${d.previous?.n_samples || '-'}</div>
                    </div></div>
                </div>
            </div>
            <div class="mt-2 small text-muted">Accuracy delta: ${d.deltas?.accuracy > 0 ? '+' : ''}${d.deltas?.accuracy ? (d.deltas.accuracy * 100).toFixed(1) : 0}%</div>`;
        el.innerHTML = html;
        if (d.stakeholder_brief) {
            renderStakeholderBrief('ab-test-brief', d.stakeholder_brief, 'ver-compare-detail');
        }
    } catch(e) {
        el.innerHTML = `<div class="text-danger">Error: ${e.message}</div>`;
    }
}

async function runAbTest() {
    var el = document.getElementById('model-registry-result');
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Running A/B test (comparing two models on same data)...</div>';
    try {
        var r = await fetch('/api/models/ab-test', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({source: currentSource, model_a: 'gradient_boosting', model_b: 'random_forest'})
        });
        var d = await r.json();
        if (d.message && !d.verdict) {
            el.innerHTML = '<div class="alert alert-info">' + d.message + '</div>';
            return;
        }
        var agr = d.agreement || {};
        var vc = agr.agreement_pct >= 90 ? 'success' : agr.agreement_pct >= 75 ? 'warning' : 'danger';
        var html = '<h6>A/B Test: ' + d.model_a.name + ' vs ' + d.model_b.name + '</h6>';
        html += '<div class="alert alert-' + vc + '">';
        html += '<strong>' + d.verdict + '</strong>: Agreement ' + agr.agreement_pct + '% (' + agr.agree + '/' + agr.total + ' fractures match)';
        if (d.winner && d.winner !== 'neither (both are comparable)') html += ' | Winner: <strong>' + d.winner + '</strong>';
        html += '</div>';
        html += '<div class="row g-3">';
        html += '<div class="col-md-6"><div class="card ' + (d.verdict === 'MODEL_A_BETTER' ? 'border-success' : '') + '"><div class="card-header">' + d.model_a.name + '</div><div class="card-body">';
        html += '<div>Accuracy: <strong>' + (d.model_a.accuracy * 100).toFixed(1) + '%</strong></div>';
        html += '<div>F1: <strong>' + (d.model_a.f1 * 100).toFixed(1) + '%</strong></div>';
        html += '</div></div></div>';
        html += '<div class="col-md-6"><div class="card ' + (d.verdict === 'MODEL_B_BETTER' ? 'border-success' : '') + '"><div class="card-header">' + d.model_b.name + '</div><div class="card-body">';
        html += '<div>Accuracy: <strong>' + (d.model_b.accuracy * 100).toFixed(1) + '%</strong></div>';
        html += '<div>F1: <strong>' + (d.model_b.f1 * 100).toFixed(1) + '%</strong></div>';
        html += '</div></div></div></div>';
        // Disagreement table
        if (d.disagreements && d.disagreements.length > 0) {
            html += '<div class="mt-3"><h6>Disagreements (' + d.disagreements.length + ' fractures)</h6>';
            html += '<div class="table-responsive"><table class="table table-sm table-striped"><thead><tr><th>Depth (m)</th><th>Azimuth</th><th>' + d.model_a.name + '</th><th>' + d.model_b.name + '</th></tr></thead><tbody>';
            d.disagreements.slice(0, 20).forEach(function(dis) {
                html += '<tr><td>' + dis.depth + '</td><td>' + dis.azimuth + '</td><td>' + dis.model_a_pred + '</td><td class="text-danger">' + dis.model_b_pred + '</td></tr>';
            });
            html += '</tbody></table>';
            if (d.disagreements.length > 20) html += '<div class="small text-muted">Showing 20 of ' + d.disagreements.length + ' disagreements</div>';
            html += '</div>';
        }
        el.innerHTML = html;
        if (d.stakeholder_brief) {
            renderStakeholderBrief('ab-test-brief', d.stakeholder_brief, 'ab-test-detail');
        }
    } catch(e) {
        el.innerHTML = '<div class="text-danger">A/B test failed: ' + e.message + '</div>';
    }
}

async function runEnsembleVote() {
    var el = document.getElementById('model-registry-result');
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Running ensemble vote across all models...</div>';
    try {
        var r = await fetch('/api/models/ensemble-vote', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({source: currentSource})
        });
        var d = await r.json();
        if (d.message && !d.ensemble) {
            el.innerHTML = '<div class="alert alert-info">' + d.message + '</div>';
            return;
        }
        var ens = d.ensemble || {};
        var vc = ens.mean_agreement_pct >= 90 ? 'success' : ens.mean_agreement_pct >= 75 ? 'warning' : 'danger';
        var html = '<h6>Ensemble Vote: ' + d.n_models + ' Models</h6>';
        html += '<div class="alert alert-' + vc + '">';
        html += '<strong>' + ens.mean_agreement_pct + '% agreement</strong> | ';
        html += 'Unanimous: ' + ens.unanimous_count + '/' + d.n_fractures + ' | ';
        html += 'Contested: <span class="text-danger">' + ens.contested_count + '</span>';
        html += '</div>';
        // Model metrics
        html += '<div class="row g-2 mb-3">';
        for (var m in d.models) {
            html += '<div class="col-auto"><span class="badge bg-secondary">' + m + ': ' + (d.models[m].accuracy * 100).toFixed(1) + '%</span></div>';
        }
        html += '</div>';
        // Contested fractures table
        if (d.contested_fractures && d.contested_fractures.length > 0) {
            html += '<h6 class="text-danger">Contested Fractures (need expert review)</h6>';
            html += '<div class="table-responsive"><table class="table table-sm table-striped"><thead><tr><th>Depth</th><th>Azimuth</th><th>Dip</th><th>Vote</th><th>Agreement</th></tr></thead><tbody>';
            d.contested_fractures.forEach(function(c) {
                html += '<tr><td>' + c.depth + 'm</td><td>' + c.azimuth + '</td><td>' + c.dip + '</td><td>' + c.majority_vote + '</td><td class="text-danger">' + c.agreement_pct + '%</td></tr>';
            });
            html += '</tbody></table></div>';
        }
        el.innerHTML = html;
        if (d.stakeholder_brief) {
            renderStakeholderBrief('ab-test-brief', d.stakeholder_brief, 'ensemble-vote-detail');
        }
    } catch(e) {
        el.innerHTML = '<div class="text-danger">Ensemble vote failed: ' + e.message + '</div>';
    }
}

async function rollbackModel(modelType, version, well) {
    if (!confirm(`Rollback ${modelType} to version ${version}?`)) return;
    try {
        const r = await fetch('/api/models/rollback', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_type: modelType, target_version: version, well: well || null})
        });
        const d = await r.json();
        alert(d.message || 'Rollback complete');
        loadModelRegistry();
    } catch(e) {
        alert('Rollback failed: ' + e.message);
    }
}

async function runFieldStressModel() {
    const el = document.getElementById('field-stress-result');
    const depth = document.getElementById('depth-input')?.value || 3000;
    const friction = document.getElementById('friction-input')?.value || 0.6;
    const pp = document.getElementById('pp-input')?.value || 30;
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Integrating multi-well stress field... (may take 10-30s)</div>';
    try {
        const r = await fetch('/api/analysis/field-stress-model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({source: window._source || 'demo', depth_m: +depth, friction: +friction, pp_mpa: +pp})
        });
        const d = await r.json();
        if (d.status === 'INSUFFICIENT') {
            el.innerHTML = `<div class="alert alert-warning">${d.message}</div>`;
            return;
        }
        const consColor = d.consistency === 'HIGH' ? 'success' : d.consistency === 'MODERATE' ? 'warning' : 'danger';
        let html = `
            <div class="alert alert-${consColor}">
                <strong>Field SHmax: ${d.field_shmax_deg}° (${d.field_shmax_direction})</strong> — ${d.consistency} consistency (R=${d.resultant_length})
            </div>
            <p>${d.interpretation}</p>
            <div class="row g-2 mb-3">
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h4>${d.field_shmax_deg}°</h4><small>Field SHmax</small></div></div></div>
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h4>${d.dominant_regime}</h4><small>Dominant Regime</small></div></div></div>
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h4>${d.regime_agreement}%</h4><small>Regime Agreement</small></div></div></div>
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h4>${d.n_wells}</h4><small>Wells Integrated</small></div></div></div>
            </div>`;

        // Per-well results table
        if (d.well_results && d.well_results.length) {
            html += '<table class="table table-sm"><thead><tr><th>Well</th><th>SHmax (°)</th><th>Regime</th><th>Misfit</th><th>σ1</th><th>σ3</th><th>R</th><th>Fractures</th></tr></thead><tbody>';
            d.well_results.forEach(w => {
                html += `<tr><td><strong>${w.well}</strong></td><td>${w.shmax_deg || '-'}</td><td>${w.regime || w.error || '-'}</td><td>${w.misfit || '-'}</td><td>${w.sigma1 || '-'}</td><td>${w.sigma3 || '-'}</td><td>${w.r_ratio || '-'}</td><td>${w.n_fractures}</td></tr>`;
            });
            html += '</tbody></table>';
        }

        // Domain boundary warning
        if (d.domain_boundary) {
            html += `<div class="alert alert-warning"><strong>Domain Boundary Detected:</strong> ${d.domain_boundary.interpretation} (${d.domain_boundary.well_a} vs ${d.domain_boundary.well_b}: ${d.domain_boundary.shmax_difference_deg}° difference)</div>`;
        }

        // Recommendations
        if (d.recommendations) {
            html += '<div class="card border-info"><div class="card-header"><i class="bi bi-lightbulb"></i> Recommendations</div><div class="card-body"><ul class="mb-0">';
            d.recommendations.forEach(r => { html += `<li>${r}</li>`; });
            html += '</ul></div></div>';
        }
        el.innerHTML = html;
    } catch(e) {
        el.innerHTML = `<div class="text-danger">Error: ${e.message}</div>`;
    }
}

async function viewFailureAnalysis() {
    const el = document.getElementById('failure-analysis-result');
    const well = document.getElementById('well-select')?.value || '';
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Analyzing failure patterns...</div>';
    try {
        const r = await fetch(`/api/feedback/failure-analysis?well=${well}`);
        const d = await r.json();
        if (d.n_cases === 0) {
            el.innerHTML = `<div class="alert alert-info">${d.message}</div>`;
            return;
        }
        let html = `
            <div class="row g-2 mb-3">
                <div class="col-md-3"><div class="card border-danger"><div class="card-body text-center"><h4>${d.n_cases}</h4><small>Total Failures</small></div></div></div>
                <div class="col-md-3"><div class="card border-success"><div class="card-body text-center"><h4>${d.n_resolved}</h4><small>Resolved</small></div></div></div>
                <div class="col-md-3"><div class="card border-warning"><div class="card-body text-center"><h4>${d.n_unresolved}</h4><small>Unresolved</small></div></div></div>
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h4>${d.confidence_analysis ? d.confidence_analysis.avg_confidence.toFixed(2) : '-'}</h4><small>Avg Confidence</small></div></div></div>
            </div>`;

        // Confidence analysis
        if (d.confidence_analysis) {
            const ca = d.confidence_analysis;
            const caColor = ca.high_confidence_failures > 3 ? 'danger' : 'info';
            html += `<div class="alert alert-${caColor} small">${ca.interpretation}</div>`;
        }

        // Failure patterns
        if (d.patterns && d.patterns.length) {
            html += '<h6>Failure Patterns</h6><table class="table table-sm"><thead><tr><th>Type</th><th>Count</th><th>Resolved</th></tr></thead><tbody>';
            d.patterns.forEach(p => {
                html += `<tr><td><code>${p.type}</code></td><td>${p.count}</td><td>${p.resolved}/${p.count}</td></tr>`;
            });
            html += '</tbody></table>';
        }

        // Top confusions
        if (d.top_confusions && d.top_confusions.length) {
            html += '<h6>Top Confusion Pairs</h6><ul>';
            d.top_confusions.forEach(c => { html += `<li><code>${c.pair}</code>: ${c.count} cases</li>`; });
            html += '</ul>';
        }

        // Depth pattern
        if (d.depth_pattern) {
            html += `<div class="small text-muted">${d.depth_pattern.interpretation}</div>`;
        }

        // Recommendations
        if (d.recommendations) {
            html += '<div class="alert alert-info mt-2 small"><strong>Actions:</strong><ul class="mb-0">';
            d.recommendations.forEach(r => { html += `<li>${r}</li>`; });
            html += '</ul></div>';
        }
        el.innerHTML = html;
    } catch(e) {
        el.innerHTML = `<div class="text-danger">Error: ${e.message}</div>`;
    }
}

async function reportFailure() {
    const well = document.getElementById('well-select')?.value || '3P';
    const body = {
        failure_type: document.getElementById('failure-type')?.value || 'wrong_prediction',
        well: well,
        predicted: document.getElementById('failure-predicted')?.value || null,
        actual: document.getElementById('failure-actual')?.value || null,
        depth_m: document.getElementById('failure-depth')?.value ? +document.getElementById('failure-depth').value : null,
        description: document.getElementById('failure-description')?.value || null,
        source: window._source || 'demo',
    };
    try {
        const r = await fetch('/api/feedback/failure-case', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });
        const d = await r.json();
        showToast('Failure Recorded', `Case #${d.case_id} recorded. Use "Analyze Patterns" to see trends.`, 'warning');
        // Clear form
        ['failure-predicted', 'failure-actual', 'failure-depth', 'failure-description'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = '';
        });
    } catch(e) {
        showToast('Error', e.message, 'danger');
    }
}

async function retrainWithFailures() {
    const el = document.getElementById('failure-analysis-result');
    const well = document.getElementById('well-select')?.value || '3P';
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Retraining with failure-aware weights... (may take 10-20s)</div>';
    try {
        const r = await fetch('/api/feedback/retrain-with-failures', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({well, source: window._source || 'demo'})
        });
        const d = await r.json();
        if (d.message && !d.version) {
            el.innerHTML = `<div class="alert alert-info">${d.message}</div>`;
            return;
        }
        el.innerHTML = `
            <div class="alert alert-success">
                <strong>Retrained: Version ${d.version}</strong><br>
                ${d.message}
            </div>
            <div class="row g-2">
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h5>${d.accuracy ? (d.accuracy * 100).toFixed(1) + '%' : '-'}</h5><small>Accuracy</small></div></div></div>
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h5>${d.n_failures_used}</h5><small>Failures Used</small></div></div></div>
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h5>${d.n_depths_weighted}</h5><small>Depth-Weighted</small></div></div></div>
                <div class="col-md-3"><div class="card"><div class="card-body text-center"><h5>v${d.version}</h5><small>New Version</small></div></div></div>
            </div>`;
    } catch(e) {
        el.innerHTML = `<div class="text-danger">Error: ${e.message}</div>`;
    }
}


// ── v3.3.1: RLHF + Batch Functions ─────────────────

async function loadRlhfQueue() {
    const el = document.getElementById('rlhf-queue-result');
    const well = document.getElementById('well-select')?.value || '3P';
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Building expert review queue...</div>';
    try {
        const r = await fetch('/api/rlhf/review-queue', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({well, source: window._source || 'demo', n_samples: 15})
        });
        const d = await r.json();
        let html = '';
        // Stakeholder brief for RLHF queue
        if (d.stakeholder_brief) {
            var sb = d.stakeholder_brief;
            html += '<div class="card border-info shadow-sm mb-3"><div class="card-body py-2">';
            html += '<h6 class="card-title mb-1"><i class="bi bi-people-fill"></i> Expert Review Guide</h6>';
            if (sb.why_these_samples) html += '<p class="small mb-1">' + sb.why_these_samples + '</p>';
            if (sb.what_to_look_for) html += '<p class="small mb-1 text-muted"><strong>Look for:</strong> ' + sb.what_to_look_for + '</p>';
            if (sb.progress) html += '<p class="small mb-0"><strong>Progress:</strong> ' + sb.progress + '</p>';
            html += '</div></div>';
        }
        html += `<p class="text-muted small">${d.interpretation || ''}</p>`;
        if (d.review_stats) {
            const rs = d.review_stats;
            html += `<div class="mb-2 small">Reviews: <span class="badge bg-success">${rs.accepted || 0} accepted</span> <span class="badge bg-danger">${rs.rejected || 0} rejected</span> <span class="badge bg-info">${rs.corrected || 0} corrected</span></div>`;
        }
        if (d.queue && d.queue.length) {
            html += '<table class="table table-sm table-hover"><thead><tr><th>#</th><th>Depth</th><th>Az/Dip</th><th>Predicted</th><th>True</th><th>Conf</th><th>Priority</th><th>Actions</th></tr></thead><tbody>';
            d.queue.forEach(s => {
                html += `<tr class="${s.already_reviewed ? 'table-light' : ''}">
                    <td>${s.index}</td><td>${s.depth_m || '-'}</td><td>${s.azimuth}\u00B0/${s.dip}\u00B0</td>
                    <td><strong>${s.predicted_type}</strong></td><td>${s.true_type || '?'}</td>
                    <td>${(s.confidence * 100).toFixed(0)}%</td><td>${s.priority_score.toFixed(3)}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-success py-0 px-1" onclick="rlhfVerdict('${well}',${s.index},'accept','${s.predicted_type}','${s.true_type||''}',${s.depth_m||'null'},${s.azimuth},${s.dip},${s.confidence})">&#10003;</button>
                        <button class="btn btn-sm btn-outline-danger py-0 px-1" onclick="rlhfVerdict('${well}',${s.index},'reject','${s.predicted_type}','${s.true_type||''}',${s.depth_m||'null'},${s.azimuth},${s.dip},${s.confidence})">&#10007;</button>
                    </td></tr>`;
            });
            html += '</tbody></table>';
        }
        el.innerHTML = html;
    } catch(e) { el.innerHTML = `<div class="text-danger">Error: ${e.message}</div>`; }
}

async function rlhfVerdict(well, index, verdict, predicted, trueType, depth, az, dip, conf) {
    let correctType = null;
    if (verdict === 'reject') {
        correctType = prompt('What is the correct fracture type? (leave blank to just reject)');
        if (correctType) verdict = 'correct';
    }
    try {
        await fetch('/api/rlhf/accept-reject', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({well, sample_index: index, verdict, predicted_type: predicted, true_type: correctType || trueType, depth_m: depth, azimuth: az, dip: dip, confidence: conf})
        });
        showToast('RLHF', 'Expert ' + verdict + ' recorded for sample #' + index, verdict === 'accept' ? 'success' : 'warning');
        setTimeout(loadRlhfQueue, 300);
    } catch(e) { showToast('Error', e.message, 'danger'); }
}

async function viewRlhfImpact() {
    const el = document.getElementById('rlhf-queue-result');
    const well = document.getElementById('well-select')?.value || '';
    el.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Analyzing RLHF impact...</div>';
    try {
        const r = await fetch('/api/rlhf/impact?well=' + well);
        const d = await r.json();
        if (d.total_reviews === 0) { el.innerHTML = '<div class="alert alert-info">' + d.message + '</div>'; return; }
        const rc = d.acceptance_rate > 0.8 ? 'success' : d.acceptance_rate > 0.6 ? 'warning' : 'danger';
        let html = '<div class="row g-2 mb-3">' +
            '<div class="col-md-3"><div class="card border-' + rc + '"><div class="card-body text-center"><h4>' + (d.acceptance_rate*100).toFixed(0) + '%</h4><small>Accept Rate</small></div></div></div>' +
            '<div class="col-md-3"><div class="card"><div class="card-body text-center"><h4>' + d.total_reviews + '</h4><small>Total Reviews</small></div></div></div>' +
            '<div class="col-md-3"><div class="card border-success"><div class="card-body text-center"><h4>' + d.accepted + '</h4><small>Accepted</small></div></div></div>' +
            '<div class="col-md-3"><div class="card border-danger"><div class="card-body text-center"><h4>' + (d.rejected+d.corrected) + '</h4><small>Rejected</small></div></div></div></div>';
        if (d.confidence_analysis) html += '<div class="alert alert-' + (d.confidence_analysis.calibration_gap > 0.1 ? 'success' : 'warning') + ' small">' + d.confidence_analysis.interpretation + '</div>';
        if (d.top_corrections && d.top_corrections.length) {
            html += '<h6>Top Corrections</h6><ul class="small">';
            d.top_corrections.forEach(function(c) { html += '<li><code>' + c.pair + '</code>: ' + c.count + '</li>'; });
            html += '</ul>';
        }
        if (d.recommendations) { html += '<div class="alert alert-info small"><ul class="mb-0">'; d.recommendations.forEach(function(r) { html += '<li>' + r + '</li>'; }); html += '</ul></div>'; }
        el.innerHTML = html;
    } catch(e) { el.innerHTML = '<div class="text-danger">Error: ' + e.message + '</div>'; }
}

async function rlhfRetrain() {
    const el = document.getElementById('rlhf-queue-result');
    const well = document.getElementById('well-select')?.value || '3P';
    showLoading("Retraining model with RLHF feedback...");
    try {
        const r = await fetch('/api/rlhf/retrain', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({well: well, source: window._source || 'demo'})
        });
        const d = await r.json();
        hideLoading();
        if (d.error) { el.innerHTML = '<div class="alert alert-warning">' + d.error + '</div>'; return; }
        const impColor = d.improvement > 0 ? 'success' : d.improvement < 0 ? 'danger' : 'secondary';
        let html = '<div class="alert alert-' + impColor + '">' +
            '<h5><i class="bi bi-lightning"></i> RLHF Retrain Complete</h5>' +
            '<div class="row g-2">' +
            '<div class="col-md-3"><strong>Before:</strong> ' + (d.accuracy_before*100).toFixed(1) + '%</div>' +
            '<div class="col-md-3"><strong>After:</strong> ' + (d.accuracy_after*100).toFixed(1) + '%</div>' +
            '<div class="col-md-3"><strong>Improvement:</strong> ' + (d.improvement>0?'+':'') + (d.improvement*100).toFixed(1) + '%</div>' +
            '<div class="col-md-3"><strong>Reviews:</strong> ' + d.reviews_used + ' (' + d.corrections_applied + ' corrections)</div>' +
            '</div>' +
            '<p class="mt-2 mb-0">' + d.message + '</p>' +
            '</div>';
        el.innerHTML = html;
        showToast(d.message);
        refreshMlopsStatus();
    } catch(e) { hideLoading(); el.innerHTML = '<div class="text-danger">Error: ' + e.message + '</div>'; }
}

async function runBatchAnalysis() {
    const el = document.getElementById('batch-result');
    const depth = document.getElementById('depth-input')?.value || 3000;
    const pp = document.getElementById('pp-input')?.value || 30;
    var taskId = generateTaskId();
    showLoadingWithProgress("Running batch analysis on all wells...", taskId);
    try {
        const r = await fetch('/api/batch/analyze-all', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({source: window._source || 'demo', depth_m: +depth, pp_mpa: +pp, task_id: taskId})
        });
        const d = await r.json();
        let html = '';
        if (d.field_summary) {
            const fs = d.field_summary;
            html += '<div class="alert alert-info"><strong>Field:</strong> ' + fs.n_wells_analyzed + ' wells, SHmax ' + (fs.shmax_range||[])[0] + '\u00B0\u2013' + (fs.shmax_range||[])[1] + '\u00B0, Avg acc: ' + (fs.avg_accuracy ? (fs.avg_accuracy*100).toFixed(1)+'%' : '-') + '</div>';
        }
        // Safety alerts
        if (d.alerts && d.alerts.length > 0) {
            d.alerts.forEach(function(a) {
                var alertColor = a.severity === 'CRITICAL' ? 'danger' : a.severity === 'HIGH' ? 'warning' : 'info';
                html += '<div class="alert alert-' + alertColor + ' py-2"><i class="bi bi-exclamation-triangle me-1"></i><strong>' + a.type.replace(/_/g, ' ') + ':</strong> ' + a.message + '</div>';
            });
        }
        // Consistency
        if (d.field_summary && d.field_summary.consistency) {
            var c = d.field_summary.consistency;
            var cColor = c.assessment === 'CONSISTENT' ? 'success' : c.assessment === 'MINOR_VARIATION' ? 'info' : 'warning';
            html += '<div class="alert alert-' + cColor + ' py-2"><i class="bi bi-arrows-angle-expand me-1"></i><strong>Well Consistency:</strong> ' + c.assessment.replace(/_/g, ' ') + ' — SHmax spread: ' + c.shmax_spread_deg + '\u00B0, Regimes: ' + c.regimes.join(', ') + '</div>';
        }
        if (d.wells && d.wells.length) {
            html += '<table class="table table-sm"><thead><tr><th>Well</th><th>N</th><th>Regime</th><th>SHmax</th><th>Accuracy</th><th>Quality</th><th>CS%</th></tr></thead><tbody>';
            d.wells.forEach(function(w) {
                var csClass = (w.critically_stressed_pct > 10) ? ' class="table-danger"' : (w.critically_stressed_pct > 5 ? ' class="table-warning"' : '');
                html += '<tr' + csClass + '><td><strong>' + w.well + '</strong></td><td>' + w.n_fractures + '</td><td>' + (w.regime||'-') + '</td><td>' + (w.shmax_deg||'-') + '\u00B0</td><td>' + (w.accuracy?(w.accuracy*100).toFixed(1)+'%':'-') + '</td><td>' + (w.quality_grade||'-') + '</td><td>' + (w.critically_stressed_pct!=null?w.critically_stressed_pct+'%':'-') + '</td></tr>';
            });
            html += '</tbody></table>';
        }
        html += '<div class="text-muted small">' + d.elapsed_s + 's | ' + (d.alerts ? d.alerts.length : 0) + ' alert(s)</div>';
        el.innerHTML = html;
        hideLoading();
    } catch(e) { el.innerHTML = '<div class="text-danger">Error: ' + e.message + '</div>'; hideLoading(); }
}


// ── MLOps Auto-Refresh ────────────────────────────

var _mlopsRefreshTimer = null;

async function refreshMlopsStatus() {
    try {
        var r = await fetch('/api/system/health');
        var d = await r.json();
        var dot = document.getElementById('mlops-health-dot');
        var score = d.health_score || 0;
        if (dot) {
            dot.textContent = d.status || '?';
            dot.className = 'badge rounded-pill ' + (score >= 80 ? 'bg-success' : score >= 50 ? 'bg-warning' : 'bg-danger');
        }
        var el = document.getElementById('mlops-health-score');
        if (el) el.textContent = score;
        el = document.getElementById('mlops-model-count');
        if (el) el.textContent = Array.isArray(d.active_models) ? d.active_models.length : (d.active_models || 0);
        el = document.getElementById('mlops-failure-count');
        if (el) el.textContent = (d.unresolved_failures || 0);
        el = document.getElementById('mlops-rlhf-count');
        if (el) el.textContent = (d.total_reviews || d.rlhf_reviews || 0);
        el = document.getElementById('mlops-last-refresh');
        if (el) el.textContent = 'Updated ' + new Date().toLocaleTimeString();
    } catch(e) {}
}

function startMlopsRefresh() {
    refreshMlopsStatus();
    if (!_mlopsRefreshTimer) {
        _mlopsRefreshTimer = setInterval(refreshMlopsStatus, 30000);
    }
}

function stopMlopsRefresh() {
    if (_mlopsRefreshTimer) {
        clearInterval(_mlopsRefreshTimer);
        _mlopsRefreshTimer = null;
    }
}

// ── Init ──────────────────────────────────────────

document.addEventListener("DOMContentLoaded", function() {
    loadSummary();
    loadFeedbackSummary();
    loadDbStats();
    loadStartupSnapshot();
    // Auto-run overview after a short delay (let summary load first)
    setTimeout(function() { runOverview(); }, 500);
    initTooltips();
    // Watch for dynamically added content
    _tooltipObserver.observe(document.body, { childList: true, subtree: true });

    // Auto-load MLOps status when tab is activated
    document.querySelectorAll('[data-tab]').forEach(function(link) {
        link.addEventListener('click', function() {
            if (this.getAttribute('data-tab') === 'mlops') {
                startMlopsRefresh();
                loadSystemHealth();
            } else {
                stopMlopsRefresh();
            }
        });
    });
});

// ── 1D Stress Profile ─────────────────────────────────────────
async function loadStressProfile() {
    var btn = document.getElementById("btn-load-stress-profile");
    var contentEl = document.getElementById("stress-profile-content");
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Computing...';
    }
    if (contentEl) contentEl.innerHTML = '<div class="spinner-border text-primary"></div><p class="text-muted mt-2">Running inversion at multiple depths...</p>';
    try {
        var r = await apiPost("/api/analysis/stress-profile", {
            well: getWell(),
            source: currentSource,
            regime: getRegime(),
            depth_min: 500,
            depth_max: 6000,
            n_points: 25
        });
        if (!r || r.error) {
            contentEl.innerHTML = '<div class="alert alert-danger">Error: ' + (r ? r.message : "No response") + '</div>';
            return;
        }
        var html = '';
        // Plot image
        if (r.plot_img) {
            html += '<img src="' + r.plot_img + '" class="img-fluid mb-3 border rounded" alt="1D Stress Profile" style="max-height:600px;">';
        }
        // Summary info
        html += '<div class="row text-center mb-3">';
        html += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Regime</div><div class="metric-value">' + (r.regime || "?") + '</div></div></div>';
        html += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">SHmax Azimuth</div><div class="metric-value">' + (r.shmax_azimuth_deg || "?") + '&deg;</div></div></div>';
        html += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">R Ratio</div><div class="metric-value">' + (r.R || "?") + '</div></div></div>';
        html += '<div class="col-md-3"><div class="metric-card"><div class="metric-label">Ref Depth</div><div class="metric-value">' + (r.reference_depth_m || "?") + ' m</div></div></div>';
        html += '</div>';
        // Profile table
        if (r.profile && r.profile.length > 0) {
            html += '<div class="table-responsive"><table class="table table-sm table-striped">';
            html += '<thead><tr><th>Depth (m)</th><th>Sv (MPa)</th><th>SHmax (MPa)</th><th>Shmin (MPa)</th><th>Pp (MPa)</th></tr></thead><tbody>';
            for (var i = 0; i < r.profile.length; i++) {
                var p = r.profile[i];
                html += '<tr><td>' + p.depth_m + '</td><td>' + p.sv_mpa + '</td><td>' + p.shmax_mpa + '</td><td>' + p.shmin_mpa + '</td><td>' + p.pp_mpa + '</td></tr>';
            }
            html += '</tbody></table></div>';
        }
        // Note
        if (r.note) {
            html += '<div class="alert alert-info small mt-2"><i class="bi bi-info-circle me-1"></i>' + r.note + '</div>';
        }
        contentEl.innerHTML = html;
    } catch (e) {
        contentEl.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Generate Profile';
        }
    }
}

// ── Geological Context ─────────────────────────────────────────────
async function runGeologicalContext() {
    var results = document.getElementById('gc-results');
    results.classList.add('d-none');
    showLoading('Analyzing geological context...');
    try {
        var r = await apiPost('/api/analysis/geological-context', {source: currentSource()});
        results.classList.remove('d-none');
        document.getElementById('gc-n-wells').textContent = r.n_wells;
        var regime = r.wells && r.wells.length > 0 ? r.wells[0].inferred_regime : '-';
        document.getElementById('gc-regime').textContent = regime.replace(/\s*\(.*\)/, '');
        if (r.stakeholder_brief) {
            document.getElementById('gc-risk').innerHTML = '<span class="badge bg-' + (r.stakeholder_brief.risk_level === 'GREEN' ? 'success' : r.stakeholder_brief.risk_level === 'AMBER' ? 'warning' : 'danger') + '">' + r.stakeholder_brief.risk_level + '</span>';
            document.getElementById('gc-crosswell').textContent = r.cross_well_comparison ? 'Available' : 'Single Well';
            document.getElementById('gc-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        // Fracture sets table
        var setsBody = document.getElementById('gc-sets-body');
        setsBody.innerHTML = '';
        if (r.wells) {
            for (var i = 0; i < r.wells.length; i++) {
                var w = r.wells[i];
                if (w.fracture_sets) {
                    for (var j = 0; j < w.fracture_sets.length; j++) {
                        var s = w.fracture_sets[j];
                        setsBody.innerHTML += '<tr><td>' + w.well + '</td><td>Set ' + s.set_id + '</td><td>' + s.count + '</td><td>' + (s.mean_azimuth || '-') + '</td><td>' + (s.mean_dip || '-') + '</td><td>' + s.interpretation + '</td></tr>';
                    }
                }
            }
        }
        // Depth zones table
        var zonesBody = document.getElementById('gc-zones-body');
        zonesBody.innerHTML = '';
        if (r.wells) {
            for (var i = 0; i < r.wells.length; i++) {
                var w = r.wells[i];
                if (w.depth_zones) {
                    for (var j = 0; j < w.depth_zones.length; j++) {
                        var z = w.depth_zones[j];
                        zonesBody.innerHTML += '<tr><td>' + w.well + '</td><td>' + z.zone + '</td><td>' + z.depth_range + '</td><td>' + z.count + '</td><td>' + (z.mean_azimuth || '-') + '</td><td>' + (z.mean_dip || '-') + '</td></tr>';
                    }
                }
            }
        }
        // Type distribution table
        var typesBody = document.getElementById('gc-types-body');
        typesBody.innerHTML = '';
        if (r.wells) {
            for (var i = 0; i < r.wells.length; i++) {
                var w = r.wells[i];
                if (w.type_distribution) {
                    var types = Object.keys(w.type_distribution);
                    for (var j = 0; j < types.length; j++) {
                        typesBody.innerHTML += '<tr><td>' + w.well + '</td><td>' + types[j] + '</td><td>' + w.type_distribution[types[j]] + '</td></tr>';
                    }
                }
            }
        }
        // Cross-well comparison
        var compDiv = document.getElementById('gc-comparison');
        if (r.cross_well_comparison) {
            compDiv.classList.remove('d-none');
            document.getElementById('gc-comp-text').textContent = r.cross_well_comparison.interpretation;
        } else {
            compDiv.classList.add('d-none');
        }
        // Plot
        if (r.plot) document.getElementById('gc-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Decision Dashboard ─────────────────────────────────────────────
async function runDecisionDashboard() {
    var results = document.getElementById('dd-results');
    results.classList.add('d-none');
    showLoading('Evaluating decision confidence...');
    try {
        var r = await apiPost('/api/report/decision-dashboard', {well: currentWell(), source: currentSource()});
        results.classList.remove('d-none');
        var decColor = r.overall_color === 'GREEN' ? 'success' : r.overall_color === 'AMBER' ? 'warning' : 'danger';
        document.getElementById('dd-decision').innerHTML = '<span class="badge bg-' + decColor + ' fs-5">' + r.overall_decision + '</span>';
        document.getElementById('dd-accuracy').textContent = (r.accuracy * 100).toFixed(1) + '%';
        document.getElementById('dd-bal-acc').textContent = (r.balanced_accuracy * 100).toFixed(1) + '%';
        document.getElementById('dd-samples').textContent = r.n_samples;
        // Signals
        var sigRow = document.getElementById('dd-signals-row');
        sigRow.innerHTML = '';
        if (r.signals) {
            var sigs = Object.keys(r.signals);
            for (var i = 0; i < sigs.length; i++) {
                var sig = r.signals[sigs[i]];
                var sc = sig.status === 'GREEN' ? 'success' : sig.status === 'AMBER' ? 'warning' : 'danger';
                sigRow.innerHTML += '<div class="col-md-2 col-4"><div class="card border-' + sc + '"><div class="card-body text-center p-2"><span class="badge bg-' + sc + '">' + sig.status + '</span><br><small class="text-muted">' + sigs[i].replace(/_/g, ' ') + '</small><br><strong>' + sig.value + '</strong></div></div></div>';
            }
        }
        // Class decisions table
        var classBody = document.getElementById('dd-class-body');
        classBody.innerHTML = '';
        if (r.class_decisions) {
            for (var i = 0; i < r.class_decisions.length; i++) {
                var c = r.class_decisions[i];
                var cd = c.decision === 'GO' ? 'success' : c.decision === 'CONDITIONAL' ? 'warning' : 'danger';
                classBody.innerHTML += '<tr><td>' + c['class'] + '</td><td>' + (c.recall * 100).toFixed(0) + '%</td><td>' + (c.precision * 100).toFixed(0) + '%</td><td>' + c.support + '</td><td><span class="badge bg-' + cd + '">' + c.decision + '</span></td><td class="small">' + c.reason + '</td></tr>';
            }
        }
        // Scenarios
        var scenRow = document.getElementById('dd-scenarios-row');
        scenRow.innerHTML = '';
        if (r.scenarios) {
            var scens = ['best_case', 'expected', 'worst_case'];
            var scenColors = ['success', 'primary', 'danger'];
            var scenLabels = ['Best Case', 'Expected', 'Worst Case'];
            for (var i = 0; i < scens.length; i++) {
                var s = r.scenarios[scens[i]];
                if (s) {
                    scenRow.innerHTML += '<div class="col-md-4"><div class="card border-' + scenColors[i] + '"><div class="card-header bg-' + scenColors[i] + ' bg-opacity-10 py-1 small"><strong>' + scenLabels[i] + '</strong></div><div class="card-body p-2 small"><div class="fs-5 fw-bold">' + (s.accuracy * 100).toFixed(1) + '%</div><p class="mb-1">' + s.description + '</p><em>' + s.risk + '</em></div></div></div>';
                }
            }
        }
        // Actions
        var actionsDiv = document.getElementById('dd-actions');
        if (r.recommended_actions && r.recommended_actions.length > 0) {
            actionsDiv.classList.remove('d-none');
            var actList = document.getElementById('dd-action-list');
            actList.innerHTML = '';
            for (var i = 0; i < r.recommended_actions.length; i++) {
                actList.innerHTML += '<li>' + r.recommended_actions[i] + '</li>';
            }
        } else {
            actionsDiv.classList.add('d-none');
        }
        // Plot
        if (r.plot) document.getElementById('dd-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Model Significance Testing ─────────────────────────────────────
async function runModelSignificance() {
    var results = document.getElementById('msig-results');
    results.classList.add('d-none');
    showLoading('Running statistical significance tests on all models...');
    try {
        var r = await apiPost('/api/analysis/model-significance', {well: currentWell(), source: currentSource()});
        results.classList.remove('d-none');
        var rec = r.recommendation || {};
        document.getElementById('msig-best').textContent = rec.best_model || '-';
        document.getElementById('msig-acc').textContent = (rec.accuracy * 100).toFixed(1) + '%';
        document.getElementById('msig-sig-count').textContent = rec.significantly_better_than + '/' + rec.total_compared;
        document.getElementById('msig-n-models').textContent = r.n_models;
        document.getElementById('msig-verdict').innerHTML = '<strong>Verdict:</strong> ' + rec.verdict;
        // Model table
        var tbody = document.getElementById('msig-table-body');
        tbody.innerHTML = '';
        if (r.models) {
            for (var i = 0; i < r.models.length; i++) {
                var m = r.models[i];
                var isBest = m.model === rec.best_model;
                tbody.innerHTML += '<tr' + (isBest ? ' class="table-success"' : '') + '><td>' + (i+1) + '</td><td>' + m.model + '</td><td>' + (m.accuracy * 100).toFixed(1) + '%</td><td>' + (m.f1 * 100).toFixed(1) + '%</td><td>' + (m.balanced_accuracy * 100).toFixed(1) + '%</td><td>' + m.time_s + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('msig-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Data Collection Planner ────────────────────────────────────────
async function runCollectionPlanner() {
    var results = document.getElementById('cp-results');
    results.classList.add('d-none');
    showLoading('Analyzing data gaps and planning collection...');
    try {
        var r = await apiPost('/api/data/collection-planner', {source: currentSource()});
        results.classList.remove('d-none');
        document.getElementById('cp-n-actions').textContent = r.n_priorities;
        if (r.wells && r.wells.length > 0) {
            document.getElementById('cp-current-acc').textContent = (r.wells[0].current_accuracy * 100).toFixed(1) + '%';
            document.getElementById('cp-projected-acc').textContent = (r.wells[0].projected_accuracy_2x * 100).toFixed(1) + '%';
        }
        if (r.stakeholder_brief) {
            var sc = r.stakeholder_brief.risk_level === 'GREEN' ? 'success' : r.stakeholder_brief.risk_level === 'AMBER' ? 'warning' : 'danger';
            document.getElementById('cp-risk').innerHTML = '<span class="badge bg-' + sc + '">' + r.stakeholder_brief.risk_level + '</span>';
            document.getElementById('cp-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence;
        }
        // Class gaps table
        var classBody = document.getElementById('cp-class-body');
        classBody.innerHTML = '';
        if (r.wells) {
            for (var i = 0; i < r.wells.length; i++) {
                var w = r.wells[i];
                if (w.class_gaps) {
                    for (var j = 0; j < w.class_gaps.length; j++) {
                        var g = w.class_gaps[j];
                        var pc = g.priority === 'HIGH' ? 'danger' : g.priority === 'MEDIUM' ? 'warning' : 'success';
                        classBody.innerHTML += '<tr><td>' + w.well + '</td><td>' + g['class'] + '</td><td>' + g.current_count + '</td><td>' + g.ideal_count + '</td><td>' + g.gap + '</td><td><span class="badge bg-' + pc + '">' + g.priority + '</span></td><td class="small">' + g.action + '</td></tr>';
                    }
                }
            }
        }
        // Depth gaps table
        var depthBody = document.getElementById('cp-depth-body');
        depthBody.innerHTML = '';
        if (r.wells) {
            for (var i = 0; i < r.wells.length; i++) {
                var w = r.wells[i];
                if (w.depth_gaps) {
                    for (var j = 0; j < w.depth_gaps.length; j++) {
                        var d = w.depth_gaps[j];
                        var dc = d.status === 'SPARSE' ? 'warning' : '';
                        depthBody.innerHTML += '<tr class="' + dc + '"><td>' + w.well + '</td><td>' + d.range + '</td><td>' + d.count + '</td><td>' + d.density + '</td><td>' + d.status + '</td></tr>';
                    }
                }
            }
        }
        // Priorities
        var priDiv = document.getElementById('cp-priorities');
        if (r.priorities && r.priorities.length > 0) {
            priDiv.classList.remove('d-none');
            var priList = document.getElementById('cp-priority-list');
            priList.innerHTML = '';
            for (var i = 0; i < Math.min(r.priorities.length, 10); i++) {
                var p = r.priorities[i];
                var pi = p.priority === 'HIGH' ? 'text-danger fw-bold' : p.priority === 'MEDIUM' ? 'text-warning' : '';
                priList.innerHTML += '<li class="' + pi + '">[' + p.priority + '] ' + p.action + '</li>';
            }
        }
        if (r.plot) document.getElementById('cp-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Conformal Prediction ───────────────────────────────────────────
async function runConformalPredict() {
    var results = document.getElementById('cf-results');
    results.classList.add('d-none');
    showLoading('Running conformal prediction analysis...');
    try {
        var r = await apiPost('/api/analysis/conformal-predict', {well: currentWell(), source: currentSource()});
        results.classList.remove('d-none');
        document.getElementById('cf-coverage').textContent = r.actual_coverage + '%';
        document.getElementById('cf-avg-size').textContent = r.avg_set_size;
        document.getElementById('cf-singleton').textContent = r.singleton_pct + '%';
        if (r.stakeholder_brief) {
            var rc = r.stakeholder_brief.risk_level === 'GREEN' ? 'success' : r.stakeholder_brief.risk_level === 'AMBER' ? 'warning' : 'danger';
            document.getElementById('cf-risk').innerHTML = '<span class="badge bg-' + rc + '">' + r.stakeholder_brief.risk_level + '</span>';
            document.getElementById('cf-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence;
        }
        // Per-class table
        var classBody = document.getElementById('cf-class-body');
        classBody.innerHTML = '';
        if (r.class_analysis) {
            for (var i = 0; i < r.class_analysis.length; i++) {
                var c = r.class_analysis[i];
                var cc = c.confidence === 'HIGH' ? 'success' : c.confidence === 'MEDIUM' ? 'warning' : 'danger';
                classBody.innerHTML += '<tr><td>' + c['class'] + '</td><td>' + c.count + '</td><td>' + c.avg_set_size + '</td><td>' + c.singleton_pct + '%</td><td>' + c.coverage + '%</td><td><span class="badge bg-' + cc + '">' + c.confidence + '</span></td></tr>';
            }
        }
        // Uncertain samples
        var uncBody = document.getElementById('cf-uncertain-body');
        uncBody.innerHTML = '';
        if (r.uncertain_samples) {
            for (var i = 0; i < r.uncertain_samples.length; i++) {
                var u = r.uncertain_samples[i];
                uncBody.innerHTML += '<tr><td>' + (u.depth_m || '-') + '</td><td>' + u.true_class + '</td><td>' + u.prediction_set.join(', ') + '</td><td>' + u.set_size + '</td><td>' + u.max_probability + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('cf-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Cross-Well Generalization Test ─────────────────────────────────
async function runCrossWellTest() {
    var results = document.getElementById('cw-results');
    results.classList.add('d-none');
    showLoading('Testing cross-well generalization...');
    try {
        var r = await apiPost('/api/analysis/cross-well-test', {source: currentSource()});
        results.classList.remove('d-none');
        if (r.status === 'INSUFFICIENT_WELLS') {
            results.innerHTML = '<div class="alert alert-warning">' + r.message + '</div>';
            return;
        }
        var gc = r.transfer_grade === 'A' || r.transfer_grade === 'B' ? 'success' : r.transfer_grade === 'C' ? 'warning' : 'danger';
        document.getElementById('cw-grade').innerHTML = '<span class="badge bg-' + gc + ' fs-4">' + r.transfer_grade + '</span>';
        document.getElementById('cw-within').textContent = (r.avg_within_accuracy * 100).toFixed(1) + '%';
        document.getElementById('cw-cross').textContent = (r.avg_cross_accuracy * 100).toFixed(1) + '%';
        document.getElementById('cw-degrad').textContent = (r.degradation * 100).toFixed(1) + '%';
        if (r.stakeholder_brief) {
            document.getElementById('cw-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence;
        }
        // Cross-well table
        var tbody = document.getElementById('cw-table-body');
        tbody.innerHTML = '';
        if (r.cross_results) {
            for (var i = 0; i < r.cross_results.length; i++) {
                var cr = r.cross_results[i];
                var ac = cr.accuracy >= 0.7 ? 'table-success' : cr.accuracy >= 0.5 ? 'table-warning' : 'table-danger';
                tbody.innerHTML += '<tr class="' + ac + '"><td>' + cr.train_well + '</td><td>' + cr.test_well + '</td><td>' + cr.train_samples + '</td><td>' + cr.test_samples + '</td><td>' + (cr.accuracy * 100).toFixed(1) + '%</td><td>' + (cr.f1 * 100).toFixed(1) + '%</td><td>' + (cr.balanced_accuracy * 100).toFixed(1) + '%</td></tr>';
            }
        }
        if (r.plot) document.getElementById('cw-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Drift Detection ────────────────────────────────────────────────
async function runDriftDetection() {
    var results = document.getElementById('drift-results');
    results.classList.add('d-none');
    showLoading('Detecting data drift between wells...');
    try {
        var r = await apiPost('/api/analysis/cross-well-drift', {source: currentSource()});
        results.classList.remove('d-none');
        if (r.status === 'INSUFFICIENT_WELLS') {
            results.innerHTML = '<div class="alert alert-warning">' + r.message + '</div>';
            return;
        }
        var ac = r.overall_alert === 'HIGH' ? 'danger' : r.overall_alert === 'MEDIUM' ? 'warning' : 'success';
        document.getElementById('drift-alert').innerHTML = '<span class="badge bg-' + ac + '">' + r.overall_alert + '</span>';
        document.getElementById('drift-max').textContent = r.max_drift_pct + '%';
        document.getElementById('drift-retrain').textContent = r.retrain_needed ? 'YES' : 'No';
        document.getElementById('drift-wells').textContent = r.n_wells;
        if (r.stakeholder_brief) {
            document.getElementById('drift-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence;
        }
        var tbody = document.getElementById('drift-table-body');
        tbody.innerHTML = '';
        if (r.comparisons) {
            for (var i = 0; i < r.comparisons.length; i++) {
                var comp = r.comparisons[i];
                var td = comp.top_drifted || [];
                for (var j = 0; j < Math.min(td.length, 5); j++) {
                    var f = td[j];
                    var sc = f.severity === 'HIGH' ? 'danger' : f.severity === 'MEDIUM' ? 'warning' : 'success';
                    tbody.innerHTML += '<tr><td>' + comp.well_a + ' vs ' + comp.well_b + '</td><td>' + f.feature + '</td><td>' + f.ks_statistic + '</td><td>' + f.p_value + '</td><td><span class="badge bg-' + sc + '">' + f.severity + '</span></td></tr>';
                }
            }
        }
        if (r.plot) document.getElementById('drift-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Domain Adaptation ──────────────────────────────────────────────
async function runDomainAdapt() {
    var results = document.getElementById('da-results');
    results.classList.add('d-none');
    showLoading('Running domain adaptation between wells...');
    try {
        var r = await apiPost('/api/analysis/domain-adapt-wells', {source: currentSource()});
        results.classList.remove('d-none');
        document.getElementById('da-best').textContent = r.best_method || '-';
        var imp = r.improvement || 0;
        document.getElementById('da-improve').textContent = (imp >= 0 ? '+' : '') + (imp * 100).toFixed(1) + '%';
        document.getElementById('da-train').textContent = r.train_well;
        document.getElementById('da-test').textContent = r.test_well;
        if (r.stakeholder_brief) {
            document.getElementById('da-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence;
        }
        // Methods table
        var mBody = document.getElementById('da-methods-body');
        mBody.innerHTML = '';
        if (r.methods) {
            for (var i = 0; i < r.methods.length; i++) {
                var m = r.methods[i];
                var isBest = m.method === r.best_method;
                mBody.innerHTML += '<tr' + (isBest ? ' class="table-success"' : '') + '><td>' + m.method + '</td><td>' + (m.accuracy * 100).toFixed(1) + '%</td><td>' + (m.f1 * 100).toFixed(1) + '%</td><td class="small">' + m.description + '</td></tr>';
            }
        }
        // Per-class
        var cBody = document.getElementById('da-class-body');
        cBody.innerHTML = '';
        if (r.per_class) {
            for (var i = 0; i < r.per_class.length; i++) {
                var c = r.per_class[i];
                cBody.innerHTML += '<tr><td>' + c['class'] + '</td><td>' + (c.precision * 100).toFixed(0) + '%</td><td>' + (c.recall * 100).toFixed(0) + '%</td><td>' + c.support + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('da-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Depth-Stratified Cross-Validation ─────────────────────────────────
async function runDepthStratCV() {
    showLoading('Running depth-stratified validation...');
    var results = document.getElementById('dscv-results');
    try {
        var r = await apiPost('/api/analysis/depth-stratified-cv', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        if (r.error) { results.innerHTML = '<div class="alert alert-warning">' + r.error + '</div>'; return; }
        document.getElementById('dscv-acc').textContent = (r.overall_accuracy * 100).toFixed(1) + '%';
        document.getElementById('dscv-baseline').textContent = (r.random_baseline_avg * 100).toFixed(1) + '%';
        var riskBadge = r.deployment_risk === 'LOW' ? 'success' : (r.deployment_risk === 'MEDIUM' ? 'warning' : 'danger');
        document.getElementById('dscv-risk').innerHTML = '<span class="badge bg-' + riskBadge + '">' + r.deployment_risk + '</span>';
        document.getElementById('dscv-worst').textContent = 'Z' + r.worst_zone.zone_id + ' (' + (r.worst_zone.accuracy * 100).toFixed(0) + '%)';
        if (r.stakeholder_brief) {
            document.getElementById('dscv-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var body = document.getElementById('dscv-zone-body');
        body.innerHTML = '';
        if (r.zones) {
            for (var i = 0; i < r.zones.length; i++) {
                var z = r.zones[i];
                var gc = z.grade === 'A' ? 'success' : (z.grade === 'B' ? 'info' : (z.grade === 'C' ? 'warning' : 'danger'));
                body.innerHTML += '<tr><td>Z' + z.zone_id + '</td><td>' + z.depth_range_m[0].toFixed(0) + '-' + z.depth_range_m[1].toFixed(0) + 'm</td><td>' + (z.accuracy * 100).toFixed(1) + '%</td><td>' + (z.f1_weighted * 100).toFixed(1) + '%</td><td>' + (z.random_baseline * 100).toFixed(1) + '%</td><td>' + (z.degradation_vs_random * 100).toFixed(1) + '%</td><td><span class="badge bg-' + gc + '">' + z.grade + '</span></td></tr>';
            }
        }
        if (r.plot) document.getElementById('dscv-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Probability Calibration ───────────────────────────────────────────
async function runCalibProb() {
    showLoading('Running probability calibration...');
    var results = document.getElementById('tcal-results');
    try {
        var r = await apiPost('/api/analysis/calibrate-probabilities', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('tcal-temp').textContent = 'T=' + r.temperature.toFixed(2);
        document.getElementById('tcal-before').textContent = r.ece_before.toFixed(4);
        document.getElementById('tcal-after').textContent = r.ece_after.toFixed(4);
        var gc = (r.grade === 'A' || r.grade === 'B') ? 'success' : (r.grade === 'C' ? 'warning' : 'danger');
        document.getElementById('tcal-grade').innerHTML = '<span class="badge bg-' + gc + '">' + r.grade + '</span>';
        if (r.stakeholder_brief) {
            document.getElementById('tcal-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var body = document.getElementById('tcal-class-body');
        body.innerHTML = '';
        if (r.per_class) {
            for (var i = 0; i < r.per_class.length; i++) {
                var c = r.per_class[i];
                body.innerHTML += '<tr><td>' + c['class'] + '</td><td>' + c.before_avg_confidence.toFixed(3) + '</td><td>' + c.after_avg_confidence.toFixed(3) + '</td><td>' + c.actual_frequency.toFixed(3) + '</td><td>' + c.before_gap.toFixed(3) + '</td><td>' + c.after_gap.toFixed(3) + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('tcal-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Feature Interaction Discovery ─────────────────────────────────────
async function runFeatInteract() {
    showLoading('Discovering feature interactions...');
    var results = document.getElementById('fi-results');
    try {
        var r = await apiPost('/api/analysis/feature-interactions', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('fi-syn').textContent = r.n_synergistic;
        document.getElementById('fi-red').textContent = r.n_redundant;
        document.getElementById('fi-ind').textContent = r.n_independent;
        if (r.strongest_interaction) {
            document.getElementById('fi-strong').textContent = r.strongest_interaction.feature_a.substring(0,8) + ' x ' + r.strongest_interaction.feature_b.substring(0,8);
        }
        if (r.stakeholder_brief) {
            document.getElementById('fi-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence;
        }
        var body = document.getElementById('fi-table-body');
        body.innerHTML = '';
        if (r.interactions) {
            for (var i = 0; i < r.interactions.length; i++) {
                var inter = r.interactions[i];
                var tc = inter.type === 'synergistic' ? 'success' : (inter.type === 'redundant' ? 'danger' : 'secondary');
                body.innerHTML += '<tr><td>' + inter.feature_a + '</td><td>' + inter.feature_b + '</td><td>' + inter.interaction_strength.toFixed(4) + '</td><td><span class="badge bg-' + tc + '">' + inter.type + '</span></td><td>' + inter.joint_drop.toFixed(4) + '</td></tr>';
            }
        }
        var notesDiv = document.getElementById('fi-notes');
        notesDiv.innerHTML = '';
        if (r.physical_notes) {
            for (var i = 0; i < r.physical_notes.length; i++) {
                notesDiv.innerHTML += '<div><i class="bi bi-lightbulb"></i> ' + r.physical_notes[i] + '</div>';
            }
        }
        if (r.plot) document.getElementById('fi-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Data Augmentation Analysis ────────────────────────────────────────
async function runAugmentation() {
    showLoading('Testing augmentation strategies...');
    var results = document.getElementById('aug-results');
    try {
        var r = await apiPost('/api/analysis/augmentation-analysis', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('aug-best').textContent = r.best_strategy || '-';
        var imp = r.improvement_over_baseline || 0;
        document.getElementById('aug-improve').textContent = (imp >= 0 ? '+' : '') + (imp * 100).toFixed(1) + '%';
        document.getElementById('aug-ratio').textContent = (r.imbalance_ratio || 0).toFixed(1) + ':1';
        document.getElementById('aug-minority').textContent = r.minority_class || '-';
        if (r.stakeholder_brief) {
            document.getElementById('aug-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var sb = document.getElementById('aug-strat-body');
        sb.innerHTML = '';
        if (r.strategies) {
            for (var i = 0; i < r.strategies.length; i++) {
                var s = r.strategies[i];
                var isBest = s.strategy === r.best_strategy;
                sb.innerHTML += '<tr' + (isBest ? ' class="table-success"' : '') + '><td>' + s.strategy + '</td><td>' + (s.accuracy * 100).toFixed(1) + '%</td><td>' + (s.f1_weighted * 100).toFixed(1) + '%</td><td>' + (s.balanced_accuracy * 100).toFixed(1) + '%</td></tr>';
            }
        }
        var cb = document.getElementById('aug-class-body');
        cb.innerHTML = '';
        if (r.minority_improvements) {
            for (var i = 0; i < r.minority_improvements.length; i++) {
                var m = r.minority_improvements[i];
                var gc = m.improvement > 0.05 ? 'success' : (m.improvement > 0 ? 'warning' : 'danger');
                cb.innerHTML += '<tr><td>' + m['class'] + '</td><td>' + m.count + '</td><td>' + m.baseline_f1.toFixed(3) + '</td><td>' + m.best_f1.toFixed(3) + '</td><td><span class="text-' + gc + '">' + (m.improvement >= 0 ? '+' : '') + m.improvement.toFixed(3) + '</span></td><td>' + m.best_strategy + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('aug-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Multi-Objective Optimization ──────────────────────────────────────
async function runMultiObj() {
    showLoading('Computing Pareto frontier...');
    var results = document.getElementById('mo-results');
    try {
        var r = await apiPost('/api/analysis/multi-objective', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('mo-pareto').textContent = r.n_pareto || 0;
        if (r.recommended) {
            document.getElementById('mo-acc').textContent = (r.recommended.accuracy * 100).toFixed(1) + '%';
            document.getElementById('mo-cov').textContent = (r.recommended.coverage * 100).toFixed(1) + '%';
            document.getElementById('mo-err').textContent = (r.recommended.error_rate * 100).toFixed(1) + '%';
        }
        if (r.stakeholder_brief) {
            document.getElementById('mo-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var sb = document.getElementById('mo-scen-body');
        sb.innerHTML = '';
        if (r.scenarios) {
            for (var i = 0; i < r.scenarios.length; i++) {
                var s = r.scenarios[i];
                sb.innerHTML += '<tr><td>' + s.name + '</td><td>' + (s.threshold * 100).toFixed(0) + '%</td><td>' + (s.accuracy * 100).toFixed(1) + '%</td><td>' + (s.coverage * 100).toFixed(1) + '%</td><td>' + (s.error_rate * 100).toFixed(1) + '%</td><td>' + s.n_classified + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('mo-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Explainability Report ─────────────────────────────────────────────
async function runExplainReport() {
    showLoading('Generating explanations...');
    var results = document.getElementById('expl-results');
    try {
        var r = await apiPost('/api/analysis/explainability-report', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('expl-count').textContent = r.n_samples_explained || 0;
        document.getElementById('expl-correct').textContent = r.n_correct || 0;
        document.getElementById('expl-wrong').textContent = r.n_misclassified || 0;
        document.getElementById('expl-conf').textContent = ((r.avg_confidence || 0) * 100).toFixed(0) + '%';
        if (r.stakeholder_brief) {
            document.getElementById('expl-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence;
        }
        // Render narratives
        var nd = document.getElementById('expl-narratives');
        nd.innerHTML = '';
        if (r.explanations) {
            for (var i = 0; i < Math.min(r.explanations.length, 10); i++) {
                var ex = r.explanations[i];
                var bc = ex.correct ? 'success' : 'danger';
                var icon = ex.correct ? 'check-circle' : 'x-circle';
                nd.innerHTML += '<div class="alert alert-' + bc + ' py-1 px-2 mb-1 small"><i class="bi bi-' + icon + '"></i> ' + (ex.depth_m ? '<strong>[' + ex.depth_m + 'm]</strong> ' : '') + ex.narrative + '</div>';
            }
        }
        // Feature ranking
        var fb = document.getElementById('expl-feat-body');
        fb.innerHTML = '';
        if (r.global_feature_ranking) {
            for (var i = 0; i < r.global_feature_ranking.length; i++) {
                var f = r.global_feature_ranking[i];
                fb.innerHTML += '<tr><td>' + f.feature + '</td><td>' + f.importance.toFixed(4) + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('expl-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── RLHF Reward Model Training ────────────────────────────────────────
async function runRewardTrain() {
    showLoading('Training RLHF reward model...');
    var results = document.getElementById('rw-results');
    try {
        var r = await apiPost('/api/rlhf/reward-model-train', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        if (r.error) { results.innerHTML = '<div class="alert alert-warning">' + r.error + '</div>'; return; }
        document.getElementById('rw-acc').textContent = (r.rlhf_accuracy * 100).toFixed(1) + '%';
        var imp = r.improvement || 0;
        document.getElementById('rw-improve').textContent = (imp >= 0 ? '+' : '') + (imp * 100).toFixed(1) + '%';
        document.getElementById('rw-pair').textContent = (r.pair_accuracy * 100).toFixed(1) + '%';
        document.getElementById('rw-sep').textContent = (r.reward_separation || 0).toFixed(3);
        if (r.stakeholder_brief) {
            document.getElementById('rw-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var fb = document.getElementById('rw-feat-body');
        fb.innerHTML = '';
        if (r.reward_features) {
            for (var i = 0; i < r.reward_features.length; i++) {
                var f = r.reward_features[i];
                fb.innerHTML += '<tr><td>' + f.feature + '</td><td>' + f.weight.toFixed(4) + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('rw-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Negative Outcome Learning ─────────────────────────────────────────
async function runNegLearn() {
    showLoading('Learning from negative outcomes...');
    var results = document.getElementById('nl-results');
    try {
        var r = await apiPost('/api/analysis/negative-learning', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('nl-hard').textContent = r.n_hard_examples + ' (' + r.hard_pct.toFixed(0) + '%)';
        var imp = r.improvement_accuracy || 0;
        document.getElementById('nl-improve').textContent = (imp >= 0 ? '+' : '') + (imp * 100).toFixed(1) + '%';
        document.getElementById('nl-fixed').textContent = r.n_fixed || 0;
        document.getElementById('nl-still').textContent = r.n_still_wrong || 0;
        if (r.stakeholder_brief) {
            document.getElementById('nl-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var cb = document.getElementById('nl-class-body');
        cb.innerHTML = '';
        if (r.per_class) {
            for (var i = 0; i < r.per_class.length; i++) {
                var c = r.per_class[i];
                var gc = c.f1_change > 0.02 ? 'success' : (c.f1_change > -0.02 ? 'secondary' : 'danger');
                cb.innerHTML += '<tr><td>' + c['class'] + '</td><td>' + c.count + '</td><td>' + c.n_hard + '</td><td>' + c.hard_pct.toFixed(0) + '%</td><td>' + c.base_f1.toFixed(3) + '</td><td>' + c.neg_f1.toFixed(3) + '</td><td><span class="text-' + gc + '">' + (c.f1_change >= 0 ? '+' : '') + c.f1_change.toFixed(3) + '</span></td></tr>';
            }
        }
        if (r.plot) document.getElementById('nl-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Production Monitoring Simulation ──────────────────────────────────
async function runMonitorSim() {
    showLoading('Simulating production monitoring...');
    var results = document.getElementById('ms-results');
    try {
        var r = await apiPost('/api/analysis/monitoring-simulation', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('ms-acc').textContent = (r.monitoring_accuracy * 100).toFixed(1) + '%';
        var tc = r.trend === 'STABLE' ? 'success' : (r.trend === 'IMPROVING' ? 'primary' : 'danger');
        document.getElementById('ms-trend').innerHTML = '<span class="badge bg-' + tc + '">' + r.trend + '</span>';
        document.getElementById('ms-alerts').textContent = (r.alerts || []).length;
        document.getElementById('ms-retrain').innerHTML = r.retrain_needed ? '<span class="badge bg-danger">YES</span>' : '<span class="badge bg-success">NO</span>';
        if (r.stakeholder_brief) {
            document.getElementById('ms-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var bb = document.getElementById('ms-batch-body');
        bb.innerHTML = '';
        if (r.batches) {
            for (var i = 0; i < r.batches.length; i++) {
                var b = r.batches[i];
                var sc = b.status === 'GREEN' ? 'success' : (b.status === 'AMBER' ? 'warning' : 'danger');
                bb.innerHTML += '<tr><td>' + b.batch_id + '</td><td>' + b.depth_range_m[0].toFixed(0) + '-' + b.depth_range_m[1].toFixed(0) + 'm</td><td>' + b.n_samples + '</td><td>' + (b.accuracy * 100).toFixed(1) + '%</td><td>' + (b.cumulative_accuracy * 100).toFixed(1) + '%</td><td><span class="badge bg-' + sc + '">' + b.status + '</span></td></tr>';
            }
        }
        var al = document.getElementById('ms-alert-list');
        al.innerHTML = '';
        if (r.alerts && r.alerts.length > 0) {
            for (var i = 0; i < r.alerts.length; i++) {
                var a = r.alerts[i];
                var ac = a.severity === 'CRITICAL' ? 'danger' : 'warning';
                al.innerHTML += '<div class="alert alert-' + ac + ' py-1 px-2 mb-1 small"><i class="bi bi-exclamation-triangle"></i> ' + a.message + '</div>';
            }
        }
        if (r.plot) document.getElementById('ms-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Per-Sample Data Quality ──────────────────────────────────────────
async function runSampleQuality() {
    showLoading('Scoring sample quality...');
    var results = document.getElementById('sq-results');
    try {
        var r = await apiPost('/api/analysis/sample-quality', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('sq-clean').textContent = r.n_clean;
        document.getElementById('sq-minor').textContent = r.n_minor;
        document.getElementById('sq-warning').textContent = r.n_warning;
        document.getElementById('sq-critical').textContent = r.n_critical;
        document.getElementById('sq-quality').textContent = r.overall_quality_pct + '%';
        document.getElementById('sq-n').textContent = r.n_samples;
        if (r.stakeholder_brief) {
            document.getElementById('sq-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var fb = document.getElementById('sq-flag-body');
        fb.innerHTML = '';
        if (r.flag_types) {
            for (var k in r.flag_types) {
                fb.innerHTML += '<tr><td><code>' + k + '</code></td><td>' + r.flag_types[k] + '</td></tr>';
            }
        }
        var sb = document.getElementById('sq-sample-body');
        sb.innerHTML = '';
        if (r.flagged_samples) {
            for (var i = 0; i < r.flagged_samples.length; i++) {
                var s = r.flagged_samples[i];
                var gc = s.grade === 'CRITICAL' ? 'danger' : (s.grade === 'WARNING' ? 'warning' : 'secondary');
                sb.innerHTML += '<tr><td>' + s.index + '</td><td>' + s.depth_m + '</td><td>' + s.azimuth_deg + '</td><td>' + s.dip_deg + '</td><td>' + s.quality_score + '</td><td><span class="badge bg-' + gc + '">' + s.grade + '</span></td><td class="small">' + (s.flags || []).join('; ') + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('sq-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Learning Curve Projection ────────────────────────────────────────
async function runLearningProj() {
    showLoading('Projecting learning curve...');
    var results = document.getElementById('lcp-results');
    try {
        var r = await apiPost('/api/analysis/learning-curve-projection', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('lcp-acc').textContent = (r.current_accuracy * 100).toFixed(1) + '%';
        document.getElementById('lcp-asymptote').textContent = (r.asymptote * 100).toFixed(1) + '%';
        document.getElementById('lcp-gap').textContent = (r.remaining_gap * 100).toFixed(1) + '%';
        var rc = r.roi_grade === 'HIGH' ? 'success' : (r.roi_grade === 'MEDIUM' ? 'warning' : 'secondary');
        document.getElementById('lcp-roi').innerHTML = '<span class="badge bg-' + rc + '">' + r.roi_grade + '</span>';
        document.getElementById('lcp-fit').innerHTML = r.fit_success ? '<span class="badge bg-success">YES</span>' : '<span class="badge bg-danger">NO</span>';
        document.getElementById('lcp-target').textContent = (r.n_for_90pct_asymptote || '-').toLocaleString();
        if (r.stakeholder_brief) {
            document.getElementById('lcp-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var pb = document.getElementById('lcp-proj-body');
        pb.innerHTML = '';
        if (r.projections) {
            for (var i = 0; i < r.projections.length; i++) {
                var p = r.projections[i];
                pb.innerHTML += '<tr><td>' + p.multiplier + 'x</td><td>' + p.n_samples.toLocaleString() + '</td><td>' + (p.projected_accuracy * 100).toFixed(1) + '%</td><td>' + (p.gain_vs_current >= 0 ? '+' : '') + (p.gain_vs_current * 100).toFixed(2) + '%</td></tr>';
            }
        }
        if (r.plot) document.getElementById('lcp-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Consensus Ensemble ───────────────────────────────────────────────
async function runConsensusEnsemble() {
    showLoading('Running consensus ensemble...');
    var results = document.getElementById('ce-results');
    try {
        var r = await apiPost('/api/analysis/consensus-ensemble', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('ce-models').textContent = r.n_models;
        document.getElementById('ce-accepted').textContent = r.n_accepted;
        document.getElementById('ce-rejected').textContent = r.n_rejected;
        document.getElementById('ce-rate').textContent = (r.consensus_rate * 100).toFixed(1) + '%';
        document.getElementById('ce-acc').textContent = (r.accepted_accuracy * 100).toFixed(1) + '%';
        document.getElementById('ce-thresh').textContent = (r.min_agreement * 100).toFixed(0) + '%';
        if (r.stakeholder_brief) {
            document.getElementById('ce-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var mb = document.getElementById('ce-model-body');
        mb.innerHTML = '';
        if (r.model_ranking) {
            for (var i = 0; i < r.model_ranking.length; i++) {
                var m = r.model_ranking[i];
                mb.innerHTML += '<tr><td>' + m.model + '</td><td>' + (m.accuracy * 100).toFixed(1) + '%</td></tr>';
            }
        }
        var cb = document.getElementById('ce-class-body');
        cb.innerHTML = '';
        if (r.per_class) {
            for (var i = 0; i < r.per_class.length; i++) {
                var c = r.per_class[i];
                cb.innerHTML += '<tr><td>' + c['class'] + '</td><td>' + c.count + '</td><td>' + (c.consensus_rate * 100).toFixed(1) + '%</td><td>' + (c.accuracy_when_accepted * 100).toFixed(1) + '%</td><td>' + (c.avg_agreement * 100).toFixed(1) + '%</td></tr>';
            }
        }
        var rb = document.getElementById('ce-reject-body');
        rb.innerHTML = '';
        if (r.rejected_samples) {
            for (var i = 0; i < r.rejected_samples.length; i++) {
                var s = r.rejected_samples[i];
                var votes = '';
                for (var k in (s.vote_distribution || {})) { votes += k + ':' + s.vote_distribution[k] + ' '; }
                rb.innerHTML += '<tr><td>' + s.index + '</td><td>' + (s.depth_m || '-') + '</td><td>' + s.true_class + '</td><td>' + (s.max_agreement * 100).toFixed(0) + '%</td><td class="small">' + votes + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('ce-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Batch Prediction ─────────────────────────────────────────────────
async function runBatchPredict() {
    showLoading('Running batch prediction...');
    var results = document.getElementById('bp-results');
    try {
        var r = await apiPost('/api/analysis/batch-predict', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('bp-models').textContent = r.n_models;
        document.getElementById('bp-n').textContent = r.n_predicted;
        document.getElementById('bp-acc').textContent = (r.batch_accuracy * 100).toFixed(1) + '%';
        document.getElementById('bp-high').textContent = r.high_confidence_count;
        document.getElementById('bp-low').textContent = r.low_confidence_count;
        document.getElementById('bp-time').textContent = r.elapsed_s + 's';
        if (r.stakeholder_brief) {
            document.getElementById('bp-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var mb = document.getElementById('bp-model-body');
        mb.innerHTML = '';
        if (r.model_summary) {
            for (var i = 0; i < r.model_summary.length; i++) {
                var m = r.model_summary[i];
                mb.innerHTML += '<tr><td>' + m.model + '</td><td>' + (m.accuracy * 100).toFixed(1) + '%</td><td>' + m.time_s + '</td></tr>';
            }
        }
        var pb = document.getElementById('bp-pred-body');
        pb.innerHTML = '';
        if (r.predictions) {
            for (var i = 0; i < Math.min(r.predictions.length, 30); i++) {
                var p = r.predictions[i];
                var rc = p.correct ? 'table-success' : 'table-danger';
                var votes = '';
                for (var k in (p.model_votes || {})) { votes += k + ':' + p.model_votes[k] + ' '; }
                pb.innerHTML += '<tr class="' + rc + '"><td>' + p.index + '</td><td>' + p.depth_m + '</td><td>' + p.true_class + '</td><td>' + p.predicted_class + '</td><td>' + (p.agreement * 100).toFixed(0) + '%</td><td class="small">' + votes + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('bp-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Model Selection Advisor ──────────────────────────────────────────
async function runModelAdvisor() {
    showLoading('Evaluating all models...');
    var results = document.getElementById('ma-results');
    try {
        var r = await apiPost('/api/analysis/model-advisor', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('ma-best').textContent = r.recommended_model;
        var be = r.evaluations ? r.evaluations[0] : null;
        document.getElementById('ma-bal').textContent = be ? (be.balanced_accuracy * 100).toFixed(1) + '%' : '-';
        document.getElementById('ma-n').textContent = r.n_models;
        document.getElementById('ma-classes').textContent = r.n_classes;
        if (r.recommendation_rationale) {
            document.getElementById('ma-rationale').innerHTML = '<i class="bi bi-lightbulb"></i> <strong>Recommendation:</strong> ' + r.recommendation_rationale.join(' ');
        }
        if (r.stakeholder_brief) {
            document.getElementById('ma-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var eb = document.getElementById('ma-eval-body');
        eb.innerHTML = '';
        if (r.evaluations) {
            for (var i = 0; i < r.evaluations.length; i++) {
                var e = r.evaluations[i];
                var sc = e.stability === 'STABLE' ? 'success' : (e.stability === 'MODERATE' ? 'warning' : 'danger');
                var oc = e.overfit_risk === 'LOW' ? 'success' : (e.overfit_risk === 'MEDIUM' ? 'warning' : 'danger');
                var bold = e.model === r.recommended_model ? 'fw-bold' : '';
                eb.innerHTML += '<tr class="' + bold + '"><td>' + e.model + '</td><td>' + (e.accuracy * 100).toFixed(1) + '%</td><td>' + (e.balanced_accuracy * 100).toFixed(1) + '%</td><td>' + (e.f1_weighted * 100).toFixed(1) + '%</td><td>' + e.train_time_s + 's</td><td><span class="badge bg-' + sc + '">' + e.stability + '</span></td><td><span class="badge bg-' + oc + '">' + e.overfit_risk + '</span></td></tr>';
            }
        }
        if (r.plot) document.getElementById('ma-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Operational Readiness ────────────────────────────────────────────
async function runOperationalReadiness() {
    showLoading('Running operational readiness assessment...');
    var results = document.getElementById('or-results');
    try {
        var r = await apiPost('/api/analysis/operational-readiness', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        var sc = r.overall_status === 'READY' ? 'success' : (r.overall_status === 'CONDITIONAL' ? 'warning' : 'danger');
        document.getElementById('or-status').innerHTML = '<span class="badge bg-' + sc + '">' + r.overall_status + '</span>';
        document.getElementById('or-score').textContent = r.readiness_score + '%';
        document.getElementById('or-pass').textContent = r.n_pass;
        document.getElementById('or-warn').textContent = r.n_warn;
        document.getElementById('or-fail').textContent = r.n_fail;
        document.getElementById('or-model').textContent = r.best_model;
        if (r.stakeholder_brief) {
            document.getElementById('or-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var cb = document.getElementById('or-check-body');
        cb.innerHTML = '';
        if (r.checks) {
            for (var i = 0; i < r.checks.length; i++) {
                var c = r.checks[i];
                var gc = c.grade === 'PASS' ? 'success' : (c.grade === 'WARN' ? 'warning' : 'danger');
                cb.innerHTML += '<tr><td>' + c.check + '</td><td><span class="badge bg-' + gc + '">' + c.grade + '</span></td><td>' + c.detail + '</td><td class="small text-muted">' + c.threshold + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('or-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Geomechanical Feature Enrichment ─────────────────────────────────
async function runGeomechFeatures() {
    showLoading('Computing geomechanical features...');
    var results = document.getElementById('gf-results');
    try {
        var r = await apiPost('/api/analysis/geomech-features', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('gf-n').textContent = r.n_geomech_features;
        document.getElementById('gf-delta').textContent = (r.avg_accuracy_delta >= 0 ? '+' : '') + (r.avg_accuracy_delta * 100).toFixed(2) + '%';
        document.getElementById('gf-improved').textContent = r.n_models_improved + '/' + r.comparisons.length;
        document.getElementById('gf-crit').textContent = r.n_critically_stressed;
        document.getElementById('gf-shmax').textContent = r.shmax_azimuth + '°';
        document.getElementById('gf-ratio').textContent = r.stress_ratio;
        if (r.stakeholder_brief) {
            document.getElementById('gf-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var cb = document.getElementById('gf-comp-body');
        cb.innerHTML = '';
        if (r.comparisons) {
            for (var i = 0; i < r.comparisons.length; i++) {
                var c = r.comparisons[i];
                var dc = c.improved ? 'text-success' : 'text-danger';
                cb.innerHTML += '<tr><td>' + c.model + '</td><td>' + (c.baseline_accuracy * 100).toFixed(1) + '%</td><td>' + (c.enriched_accuracy * 100).toFixed(1) + '%</td><td class="' + dc + '">' + (c.accuracy_delta >= 0 ? '+' : '') + (c.accuracy_delta * 100).toFixed(2) + '%</td><td>' + (c.improved ? '<i class="bi bi-check-circle text-success"></i>' : '<i class="bi bi-x-circle text-danger"></i>') + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('gf-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── RLHF Iterative Loop ─────────────────────────────────────────────
async function runRlhfIterate() {
    showLoading('Running RLHF iterations...');
    var results = document.getElementById('ri-results');
    try {
        var r = await apiPost('/api/analysis/rlhf-iterate', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('ri-base').textContent = (r.baseline_accuracy * 100).toFixed(1) + '%';
        document.getElementById('ri-final').textContent = (r.final_accuracy * 100).toFixed(1) + '%';
        var gc = r.total_improvement >= 0 ? 'text-success' : 'text-danger';
        document.getElementById('ri-gain').innerHTML = '<span class="' + gc + '">' + (r.total_improvement >= 0 ? '+' : '') + (r.total_improvement * 100).toFixed(2) + '%</span>';
        document.getElementById('ri-n').textContent = r.n_iterations;
        document.getElementById('ri-conv').innerHTML = r.converged ? '<span class="badge bg-success">YES</span>' : '<span class="badge bg-warning">NO</span>';
        document.getElementById('ri-citer').textContent = r.convergence_iteration || '-';
        if (r.stakeholder_brief) {
            document.getElementById('ri-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var ib = document.getElementById('ri-iter-body');
        ib.innerHTML = '';
        if (r.iterations) {
            for (var i = 0; i < r.iterations.length; i++) {
                var it = r.iterations[i];
                var ic = it.improvement_vs_prev >= 0 ? 'text-success' : 'text-danger';
                ib.innerHTML += '<tr><td>' + it.iteration + '</td><td>' + (it.accuracy * 100).toFixed(1) + '%</td><td class="' + ic + '">' + (it.improvement_vs_prev >= 0 ? '+' : '') + (it.improvement_vs_prev * 100).toFixed(2) + '%</td><td>' + (it.total_improvement >= 0 ? '+' : '') + (it.total_improvement * 100).toFixed(2) + '%</td><td>' + it.n_errors + '</td><td>' + it.n_pairs_trained + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('ri-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Domain Shift Robustness ──────────────────────────────────────────
async function runDomainShift() {
    showLoading('Testing domain shift robustness...');
    var results = document.getElementById('ds-results');
    try {
        var r = await apiPost('/api/analysis/domain-shift', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('ds-zones').textContent = r.n_zones;
        document.getElementById('ds-same').textContent = (r.avg_same_domain * 100).toFixed(1) + '%';
        document.getElementById('ds-cross').textContent = (r.avg_cross_domain * 100).toFixed(1) + '%';
        var gc = r.domain_gap < 0.1 ? 'text-success' : (r.domain_gap < 0.2 ? 'text-warning' : 'text-danger');
        document.getElementById('ds-gap').innerHTML = '<span class="' + gc + '">' + (r.domain_gap * 100).toFixed(1) + '%</span>';
        document.getElementById('ds-n').textContent = r.n_samples;
        if (r.stakeholder_brief) {
            document.getElementById('ds-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var zb = document.getElementById('ds-zone-body');
        zb.innerHTML = '';
        if (r.zone_summary) {
            for (var i = 0; i < r.zone_summary.length; i++) {
                var z = r.zone_summary[i];
                var zc = z.gap < 0.1 ? '' : (z.gap < 0.2 ? 'table-warning' : 'table-danger');
                zb.innerHTML += '<tr class="' + zc + '"><td>Zone ' + z.zone + '</td><td>' + z.depth_range[0] + '-' + z.depth_range[1] + 'm</td><td>' + z.n_samples + '</td><td>' + (z.self_accuracy * 100).toFixed(1) + '%</td><td>' + (z.transfer_accuracy * 100).toFixed(1) + '%</td><td>' + (z.gap * 100).toFixed(1) + '%</td></tr>';
            }
        }
        if (r.plot) document.getElementById('ds-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Decision Support Matrix ──────────────────────────────────────────
async function runDecisionSupport() {
    showLoading('Computing decision matrix...');
    var results = document.getElementById('dsx-results');
    try {
        var r = await apiPost('/api/analysis/decision-support', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        var dc = r.decision === 'GO' ? 'success' : (r.decision === 'CAUTION' ? 'warning' : 'danger');
        document.getElementById('dsx-decision').innerHTML = '<span class="badge bg-' + dc + ' fs-5">' + r.decision + '</span>';
        document.getElementById('dsx-score').textContent = r.overall_score + '%';
        document.getElementById('dsx-model').textContent = r.best_model;
        document.getElementById('dsx-acc').textContent = (r.best_accuracy * 100).toFixed(1) + '%';
        if (r.stakeholder_brief) {
            document.getElementById('dsx-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var cb = document.getElementById('dsx-crit-body');
        cb.innerHTML = '';
        if (r.criteria) {
            for (var i = 0; i < r.criteria.length; i++) {
                var c = r.criteria[i];
                var sc = c.score >= 70 ? 'success' : (c.score >= 45 ? 'warning' : 'danger');
                cb.innerHTML += '<tr><td>' + c.criterion + '</td><td><span class="badge bg-' + sc + '">' + c.score + '%</span></td><td>' + c.weight + '</td><td>' + c.detail + '</td></tr>';
            }
        }
        var rl = document.getElementById('dsx-recs');
        rl.innerHTML = '';
        if (r.recommendations) {
            for (var i = 0; i < r.recommendations.length; i++) {
                rl.innerHTML += '<li>' + r.recommendations[i] + '</li>';
            }
        }
        if (r.plot) document.getElementById('dsx-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Risk Communication Report ────────────────────────────────────────
async function runRiskReport() {
    showLoading('Generating risk report...');
    var results = document.getElementById('rr-results');
    try {
        var r = await apiPost('/api/analysis/risk-report', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        var rc = r.overall_risk === 'LOW' ? 'success' : (r.overall_risk === 'MEDIUM' ? 'warning' : 'danger');
        document.getElementById('rr-risk').innerHTML = '<span class="badge bg-' + rc + '">' + r.overall_risk + '</span>';
        document.getElementById('rr-high').textContent = r.n_high_risks;
        document.getElementById('rr-med').textContent = r.n_medium_risks;
        document.getElementById('rr-low').textContent = r.n_low_risks;
        document.getElementById('rr-exec').innerHTML = '<i class="bi bi-info-circle"></i> ' + r.executive_summary;
        var rd = document.getElementById('rr-risks');
        rd.innerHTML = '';
        if (r.risks) {
            for (var i = 0; i < r.risks.length; i++) {
                var rk = r.risks[i];
                var rkc = rk.risk_level === 'LOW' ? 'success' : (rk.risk_level === 'MEDIUM' ? 'warning' : 'danger');
                rd.innerHTML += '<div class="card mb-2 border-' + rkc + '"><div class="card-body py-2"><h6><span class="badge bg-' + rkc + '">' + rk.risk_level + '</span> ' + rk.category + '</h6><p class="small mb-1">' + rk.plain_english + '</p><p class="small mb-1"><strong>Impact:</strong> ' + rk.impact + '</p><p class="small mb-0"><strong>Mitigation:</strong> ' + rk.mitigation + '</p></div></div>';
            }
        }
        if (r.plot) document.getElementById('rr-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}

// ── Model Transparency Audit ─────────────────────────────────────────
async function runTransparencyAudit() {
    showLoading('Running transparency audit...');
    var results = document.getElementById('ta-results');
    try {
        var r = await apiPost('/api/analysis/transparency-audit', {source: currentSource(), well: currentWell()});
        results.classList.remove('d-none');
        document.getElementById('ta-n').textContent = r.n_audited;
        document.getElementById('ta-correct').textContent = r.n_correct;
        document.getElementById('ta-wrong').textContent = r.n_wrong;
        document.getElementById('ta-acc').textContent = (r.audit_accuracy * 100).toFixed(1) + '%';
        document.getElementById('ta-models').textContent = r.n_models;
        if (r.stakeholder_brief) {
            document.getElementById('ta-brief').innerHTML = '<strong>' + r.stakeholder_brief.headline + '</strong><br>' + r.stakeholder_brief.confidence_sentence + '<br><em>' + r.stakeholder_brief.action + '</em>';
        }
        var fb = document.getElementById('ta-feat-body');
        fb.innerHTML = '';
        if (r.global_feature_importances) {
            for (var i = 0; i < r.global_feature_importances.length; i++) {
                var f = r.global_feature_importances[i];
                fb.innerHTML += '<tr><td>' + f.feature + '</td><td>' + (f.importance * 100).toFixed(2) + '%</td></tr>';
            }
        }
        var cb = document.getElementById('ta-card-body');
        cb.innerHTML = '';
        if (r.transparency_cards) {
            for (var i = 0; i < Math.min(r.transparency_cards.length, 20); i++) {
                var c = r.transparency_cards[i];
                var cc = c.correct ? 'table-success' : 'table-danger';
                cb.innerHTML += '<tr class="' + cc + '"><td>' + c.index + '</td><td>' + c.depth_m + '</td><td>' + c.true_class + '</td><td>' + c.consensus_class + '</td><td>' + (c.agreement * 100).toFixed(0) + '%</td><td class="small">' + c.geology_note + '</td></tr>';
            }
        }
        if (r.plot) document.getElementById('ta-plot').src = 'data:image/png;base64,' + r.plot;
    } catch (e) {
        results.classList.remove('d-none');
        results.innerHTML = '<div class="alert alert-danger">Error: ' + e.message + '</div>';
    } finally {
        hideLoading();
    }
}
