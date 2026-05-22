async function postJson(url) {
    const response = await fetch(url, { method: "POST" });
    return await response.json();
}

async function getJson(url) {
    const response = await fetch(url);
    return await response.json();
}

function setText(id, value) {
    const el = document.getElementById(id);

    if (el) {
        el.textContent = value;
    }
}

function isWebcamPage() {
    return document.getElementById("metric-live-fps") !== null;
}

function stopWebcamWorkerSync() {
    if (!isWebcamPage()) {
        return;
    }

    try {
        navigator.sendBeacon("/webcam/stop_worker");
    } catch (error) {
        fetch("/webcam/stop_worker", {
            method: "POST",
            keepalive: true
        }).catch(() => {});
    }
}

async function startWebcamWorker() {
    if (!isWebcamPage()) {
        return;
    }

    try {
        await postJson("/webcam/start_worker");
    } catch (error) {
        console.warn("Cannot start webcam worker:", error);
    }
}

function renderRecordingResult(data) {
    const box = document.getElementById("recording-result");

    if (!box) {
        return;
    }

    if (!data.success) {
        box.innerHTML = `
            <div class="alert alert-error">${data.message || "Recording failed."}</div>
        `;
        return;
    }

    let links = "";

    if (data.video_url) {
        links += `<a class="btn btn-secondary" href="${data.video_url}" target="_blank">Open Video</a>`;
    }

    if (data.video_download_url) {
        links += `<a class="btn btn-secondary" href="${data.video_download_url}">Download Video</a>`;
    }

    if (data.txt_download_url) {
        links += `<a class="btn btn-secondary" href="${data.txt_download_url}">Download TXT</a>`;
    }

    if (data.metrics_download_url) {
        links += `<a class="btn btn-secondary" href="${data.metrics_download_url}">Download Metrics</a>`;
    }

    box.innerHTML = `
        <div class="record-card">
            <h4>${data.message || "Recording completed."}</h4>

            <div class="stats-list">
                <div><span>Run ID</span><strong>${data.run_id || "-"}</strong></div>
                <div><span>Frames</span><strong>${data.frame_count || 0}</strong></div>
                <div><span>Duration</span><strong>${data.duration_sec || 0}s</strong></div>
                <div><span>Average FPS</span><strong>${data.fps || 0}</strong></div>
                <div><span>Unique Tracks</span><strong>${data.unique_tracks || 0}</strong></div>
                <div><span>Max Active Tracks</span><strong>${data.max_active_tracks || 0}</strong></div>
            </div>

            <div class="download-group">${links}</div>
        </div>
    `;
}

const startBtn = document.getElementById("start-record");
const stopBtn = document.getElementById("stop-record");

if (startBtn) {
    startBtn.addEventListener("click", async () => {
        startBtn.disabled = true;

        const data = await postJson("/webcam/start_record");
        renderRecordingResult(data);

        startBtn.disabled = false;
    });
}

if (stopBtn) {
    stopBtn.addEventListener("click", async () => {
        stopBtn.disabled = true;

        const data = await postJson("/webcam/stop_record");
        renderRecordingResult(data);

        stopBtn.disabled = false;
    });
}

async function refreshWebcamMetrics() {
    const liveFpsEl = document.getElementById("metric-live-fps");

    if (!liveFpsEl) {
        return;
    }

    try {
        const data = await getJson("/webcam/metrics");

        setText("metric-live-fps", data.live_fps ?? 0);
        setText("metric-average-fps", data.average_fps ?? 0);
        setText("metric-current-vehicles", data.current_active_tracks ?? 0);
        setText("metric-total-ids", data.total_unique_tracks ?? 0);
        setText("metric-max-active", data.max_active_tracks ?? 0);
        setText("metric-session-frames", data.session_frames ?? 0);
        setText("metric-duration", `${data.session_duration_sec ?? 0}s`);

        const recordingText = data.is_recording
            ? `ON (${data.record_frame_count ?? 0} frames)`
            : "OFF";

        setText("metric-recording", recordingText);
        setText("metric-camera", data.camera_opened ? "ON" : "OFF");
        setText("metric-worker", data.worker_running ? "ON" : "OFF");

        const box = document.getElementById("recording-result");

        if (box && data.latest_error) {
            box.innerHTML = `<div class="alert alert-error">${data.latest_error}</div>`;
        }
    } catch (error) {
        setText("metric-live-fps", "ERR");
        setText("metric-worker", "ERR");
    }
}

/*
    Webcam lifecycle:
    - Vào /webcam: app.py đã start worker.
    - Rời trang / đóng tab / reload: stop worker để tắt camera laptop.
    - Chuyển sang browser tab khác: stop worker để tắt camera.
    - Quay lại tab webcam: start worker lại.
*/
if (isWebcamPage()) {
    window.addEventListener("pagehide", () => {
        stopWebcamWorkerSync();
    });

    window.addEventListener("beforeunload", () => {
        stopWebcamWorkerSync();
    });

    document.addEventListener("visibilitychange", async () => {
        if (document.hidden) {
            stopWebcamWorkerSync();
        } else {
            await startWebcamWorker();
        }
    });

    setInterval(refreshWebcamMetrics, 1000);
    refreshWebcamMetrics();
}
