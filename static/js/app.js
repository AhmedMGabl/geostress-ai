/* GeoStress AI - Frontend Logic v2.0 (Industrial Grade) */

var currentSource = "demo";
var currentWell = "3P";
var feedbackRating = 3;

// ── Helpers ───────────────────────────────────────

function showLoading(text) {
    document.getElementById("loading-text").textContent = text || "Processing...";
    document.getElementById("loading-overlay").classList.remove("d-none");
}

function hideLoading() {
    document.getElementById("loading-overlay").classList.add("d-none");
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
    feedback: "Expert Feedback"
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

        showToast("Loaded " + result.rows + " fractures from " + result.filename);
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
    showLoading("Running stress inversion (with pore pressure correction)...");
    try {
        var body = {
            well: currentWell,
            regime: document.getElementById("regime-select").value,
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

        showToast("Inversion complete: SHmax=" + r.shmax_azimuth_deg + "\u00b0, Pp=" + (r.pore_pressure_mpa || 0).toFixed(1) + " MPa");
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
    } catch (err) {
        // Silently fail - feedback summary is not critical
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

// ── Init ──────────────────────────────────────────

document.addEventListener("DOMContentLoaded", function() {
    loadSummary();
    loadFeedbackSummary();
});
