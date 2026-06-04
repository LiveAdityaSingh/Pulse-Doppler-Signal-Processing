document.addEventListener("DOMContentLoaded", () => {
    const analyzeBtn = document.getElementById("analyzeBtn");
    const fileInput = document.getElementById("fileInput");
    const loader = document.getElementById("loader");
    const btnText = document.getElementById("btnText");
    const chartsSection = document.getElementById("chartsSection");

    let chart1 = null;
    let chart2 = null;
    let chart3 = null;

    analyzeBtn.addEventListener("click", async () => {
        if (!fileInput.files.length) {
            alert("Please select a wave file first.");
            return;
        }

        const fmin = document.getElementById("fmin").value;
        const fmax = document.getElementById("fmax").value;
        const tmin = document.getElementById("tmin").value;
        const tmax = document.getElementById("tmax").value;
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append("file", file);
        formData.append("fmin", fmin);
        formData.append("fmax", fmax);
        formData.append("tmin", tmin);
        formData.append("tmax", tmax);

        // UI loading state
        analyzeBtn.disabled = true;
        btnText.style.display = "none";
        loader.style.display = "inline-block";

        try {
            const res = await fetch("/analyze", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const text = await res.text();
                throw new Error("Analysis failed: " + text);
            }

            const data = await res.json();
            renderCharts(data);
            
            const downloadBtn = document.getElementById("downloadPdfBtn");
            if (downloadBtn) {
                downloadBtn.style.display = "inline-block";
                if (!downloadBtn.hasAttribute('data-bound')) {
                    downloadBtn.setAttribute('data-bound', 'true');
                    downloadBtn.addEventListener("click", (e) => {
                        e.preventDefault();
                        const element = document.getElementById('chartsSection');
                        const opt = {
                          margin:       [0.5, 0.5, 0.5, 0.5],
                          filename:     'PerDeCT_Clinical_Report.pdf',
                          image:        { type: 'jpeg', quality: 0.98 },
                          html2canvas:  { scale: 2, useCORS: true, backgroundColor: '#0f172a' },
                          jsPDF:        { unit: 'in', format: 'a4', orientation: 'portrait' },
                          pagebreak:    { mode: ['avoid-all', 'css', 'legacy'] }
                        };
                        const oldText = downloadBtn.innerText;
                        downloadBtn.innerText = "Generating PDF...";
                        html2pdf().set(opt).from(element).save().then(() => {
                            downloadBtn.innerText = oldText;
                        });
                    });
                }
            }

            const specImg = document.getElementById("spectrogramImg");
            if (specImg) {
                specImg.src = data.spec_url;
                specImg.style.display = "block";
            }
            
            const reportSection = document.getElementById("reportSection");
            const reportContent = document.getElementById("reportContent");
            if (reportSection && data.report) {
                let formattedReport = data.report
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n/g, '<br>');
                reportContent.innerHTML = formattedReport;
                reportSection.style.display = "block";
            }
            
            chartsSection.style.display = "flex";
            
        } catch(e) {
            alert(e.message);
        } finally {
            analyzeBtn.disabled = false;
            btnText.style.display = "inline";
            loader.style.display = "none";
        }
    });

    function renderCharts(data) {
        const timeLabels = data.time.map(t => t.toFixed(3) + "s");

        // Destroy previous charts if exist
        if (chart1) chart1.destroy();
        if (chart2) chart2.destroy();
        if (chart3) chart3.destroy();

        const ctx1 = document.getElementById('chart1').getContext('2d');
        chart1 = new Chart(ctx1, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [
                    {
                        label: 'Original Wave (White)',
                        data: data.original,
                        borderColor: 'rgba(255, 255, 255, 0.9)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1,
                        fill: false,
                        order: 2 // Draws underneath
                    },
                    {
                        label: 'Noise (Red)',
                        data: data.noise,
                        borderColor: 'rgba(239, 68, 68, 1.0)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1,
                        fill: false,
                        order: 1 // Drawn on TOP of White graph
                    }
                ]
            },
            options: getChartOptions()
        });

        const ctx2 = document.getElementById('chart2').getContext('2d');
        chart2 = new Chart(ctx2, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [
                    {
                        label: 'Clean Wave (Green)',
                        data: data.clean,
                        borderColor: 'rgba(34, 197, 94, 0.9)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1,
                        fill: false
                    }
                ]
            },
            options: getChartOptions()
        });

        const ctx3 = document.getElementById('chart3').getContext('2d');
        chart3 = new Chart(ctx3, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [
                    {
                        label: 'Pulse Envelope (Line Graph)',
                        data: data.pulse,
                        borderColor: 'rgba(56, 189, 248, 1.0)',
                        borderWidth: 2.0,
                        pointRadius: 0,
                        tension: 0.3, // Adds smoothing for the beautiful physiological line
                        fill: false
                    }
                ]
            },
            options: getChartOptions()
        });
    }

    function getChartOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: { color: '#f8fafc' }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 10 }
                },
                y: {
                    grid: { color: 'rgba(148, 163, 184, 0.1)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        };
    }
});
