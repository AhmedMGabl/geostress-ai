/* GeoStress AI - Frontend Logic */

let currentSource = "demo";
let currentWell = "3P";

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

// ── Tab Switching ─────────────────────────────────

var tabNames = {
    data: "Data Overview",
    viz: "Visualizations",
    inversion: "Stress Inversion",
    classify: "ML Classification",
    cluster: "Fracture Clustering"
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

        // Populate table using DOM methods
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

        // Populate well selector
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
    } catch (err) {
        showToast("Error loading data: " + err.message, "Error");
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

// ── Inversion ─────────────────────────────────────

async function runInversion() {
    showLoading("Running stress inversion...");
    try {
        var body = {
            well: currentWell,
            regime: document.getElementById("regime-select").value,
            depth_m: parseFloat(document.getElementById("depth-input").value),
            cohesion: 0,
            source: currentSource
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
        val("inv-cs", r.critically_stressed_count + "/" + r.critically_stressed_total + " (" + r.critically_stressed_pct + "%)");

        setImg("mohr-img", r.mohr_circle_img);
        setImg("slip-img", r.slip_tendency_img);
        setImg("dilation-img", r.dilation_tendency_img);
        setImg("dashboard-img", r.dashboard_img);

        showToast("Inversion complete: SHmax=" + r.shmax_azimuth_deg + "\u00b0, regime=" + r.regime);
    } catch (err) {
        showToast("Inversion error: " + err.message, "Error");
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
                    source: currentSource
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
        showToast("Regime comparison complete");
    } catch (err) {
        showToast("Comparison error: " + err.message, "Error");
    } finally {
        hideLoading();
    }
}

// ── Classification ────────────────────────────────

async function runClassification() {
    showLoading("Running ML classification...");
    try {
        var classifier = document.getElementById("classifier-select").value;
        var r = await api("/api/analysis/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ classifier: classifier, source: currentSource })
        });

        document.getElementById("classify-results").classList.remove("d-none");
        val("clf-accuracy", (r.cv_mean_accuracy * 100).toFixed(1) + "%");
        val("clf-std", "\u00b1" + (r.cv_std_accuracy * 100).toFixed(1) + "%");
        val("clf-type", classifier.replace("_", " "));

        // Feature importances using DOM methods
        var container = document.getElementById("feat-imp-container");
        clearChildren(container);
        var sorted = Object.entries(r.feature_importances).sort(function(a, b) { return b[1] - a[1]; });
        var maxVal = sorted[0][1];
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

        // Confusion matrix using DOM methods
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

        showToast("Classification: " + (r.cv_mean_accuracy * 100).toFixed(1) + "% accuracy");
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

// ── Init ──────────────────────────────────────────

document.addEventListener("DOMContentLoaded", function() {
    loadSummary();
});
