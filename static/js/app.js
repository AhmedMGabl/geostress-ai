/* GeoStress AI - Frontend Logic v2.0 (Industrial Grade) */

var currentSource = "demo";
var currentWell = "3P";
var feedbackRating = 3;

// ── Helpers ───────────────────────────────────────

var _loadingTimer = null;
var _loadingStart = 0;

function showLoading(text) {
    var el = document.getElementById("loading-text");
    el.textContent = text || "Processing";
    document.getElementById("loading-overlay").classList.remove("d-none");
    _loadingStart = Date.now();
    if (_loadingTimer) clearInterval(_loadingTimer);
    _loadingTimer = setInterval(function() {
        var elapsed = Math.round((Date.now() - _loadingStart) / 1000);
        el.textContent = (text || "Processing") + " (" + elapsed + "s)";
    }, 1000);
}

function hideLoading() {
    document.getElementById("loading-overlay").classList.add("d-none");
    if (_loadingTimer) { clearInterval(_loadingTimer); _loadingTimer = null; }
}

function showToast(msg, title) {
    document.getElementById("toast-title").textContent = title || "GeoStress AI";
    document.getElementById("toast-body").textContent = msg;
    var toast = new bootstrap.Toast(document.getElementById("toast"), { delay: 4000 });
    toast.show();
}

async function api(url, options) {
    var resp = await fetch(url, options || {});
    if (!resp.ok) {
        var text = await resp.text();
        throw new Error(text || resp.statusText);
    }
    return resp.json();
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

function getPorePresure() {
    var ppEl = document.getElementById("pp-input");
    var ppVal = ppEl ? ppEl.value : "";
    return ppVal === "" ? null : parseFloat(ppVal);
}

// ── Tab Switching ─────────────────────────────────

var tabNames = {
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
    audit: "Audit Trail"
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

// ── Well selector sync ────────────────────────────

document.getElementById("well-select").addEventListener("change", function() {
    currentWell = this.value;
});

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

        // Show OOD warning if detected
        var oodMsg = "";
        if (result.ood_check) {
            var ood = result.ood_check;
            if (ood.ood_detected) {
                oodMsg = " | WARNING: " + ood.severity + " distribution shift detected. " + ood.message;
            }
        }
        showToast("Loaded " + result.rows + " fractures from " + result.filename + oodMsg);
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
        var body = {
            well: currentWell,
            regime: selectedRegime,
            depth_m: parseFloat(document.getElementById("depth-input").value),
            cohesion: 0,
            source: currentSource,
            pore_pressure: getPorePresure()
        };

        var r = await api("/api/analysis/inversion", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });

        document.getElementById("inversion-results").classList.remove("d-none");
        val("inv-sigma1", r.sigma1.toFixed(1));
        val("inv-sigma2", r.sigma2.toFixed(1));
        val("inv-sigma3", r.sigma3.toFixed(1));
        val("inv-R", r.R.toFixed(3));
        val("inv-shmax", r.shmax_azimuth_deg.toFixed(1) + "\u00b0");
        val("inv-pp", (r.pore_pressure_mpa || 0).toFixed(1) + " MPa");

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
        val("clf-accuracy", (r.cv_mean_accuracy * 100).toFixed(1) + "%");
        val("clf-std", "\u00b1" + (r.cv_std_accuracy * 100).toFixed(1) + "%");
        val("clf-f1", r.cv_f1_mean ? (r.cv_f1_mean * 100).toFixed(1) + "%" : "--");
        val("clf-type", classifier.replace("_", " "));

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

        showToast("Classification: " + (r.cv_mean_accuracy * 100).toFixed(1) + "% accuracy (" + classifier + ")");
    } catch (err) {
        showToast("Classification error: " + err.message, "Error");
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

        await api("/api/feedback/submit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });

        showToast("Feedback submitted. Thank you!");
        document.getElementById("fb-comment").value = "";

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
    } catch (err) {
        showToast("SHAP error: " + err.message, "Error");
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


// ── Init ──────────────────────────────────────────

document.addEventListener("DOMContentLoaded", function() {
    loadSummary();
    loadFeedbackSummary();
    // Auto-run overview after a short delay (let summary load first)
    setTimeout(function() { runOverview(); }, 500);

    // Enable Bootstrap tooltips for all info icons
    var tooltipEls = document.querySelectorAll('[title]');
    tooltipEls.forEach(function(el) {
        new bootstrap.Tooltip(el, { trigger: 'hover', placement: 'top' });
    });
});
