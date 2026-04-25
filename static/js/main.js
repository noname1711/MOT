document.addEventListener("DOMContentLoaded", () => {
    initWebcamRecording();
});


function initWebcamRecording() {
    const startRecordBtn = document.getElementById("start-record-btn");
    const stopRecordBtn = document.getElementById("stop-record-btn");
    const recordStatus = document.getElementById("record-status");
    const recordResult = document.getElementById("record-result");
    const recordedVideo = document.getElementById("recorded-video");
    const downloadRecordLink = document.getElementById("download-record-link");
    const openRecordLink = document.getElementById("open-record-link");

    if (!startRecordBtn || !stopRecordBtn || !recordStatus) {
        return;
    }

    const startUrl = startRecordBtn.dataset.startUrl;
    const stopUrl = stopRecordBtn.dataset.stopUrl;

    startRecordBtn.addEventListener("click", () => {
        startWebcamRecording({
            startUrl,
            startRecordBtn,
            stopRecordBtn,
            recordStatus,
            recordResult
        });
    });

    stopRecordBtn.addEventListener("click", () => {
        stopWebcamRecording({
            stopUrl,
            startRecordBtn,
            stopRecordBtn,
            recordStatus,
            recordResult,
            recordedVideo,
            downloadRecordLink,
            openRecordLink
        });
    });
}


async function startWebcamRecording({
    startUrl,
    startRecordBtn,
    stopRecordBtn,
    recordStatus,
    recordResult
}) {
    setButtonState(startRecordBtn, true);
    setButtonState(stopRecordBtn, false);
    setStatus(recordStatus, "Trạng thái: đang bắt đầu ghi hình...");

    hideElement(recordResult);

    try {
        const data = await postJson(startUrl);

        if (data.success) {
            setStatus(recordStatus, `Trạng thái: đang ghi hình (${data.filename}).`);
            return;
        }

        setStatus(recordStatus, `Lỗi: ${data.message}`);
        setButtonState(startRecordBtn, false);
        setButtonState(stopRecordBtn, true);
    } catch (error) {
        setStatus(recordStatus, "Lỗi: không thể bắt đầu ghi hình.");
        setButtonState(startRecordBtn, false);
        setButtonState(stopRecordBtn, true);
    }
}


async function stopWebcamRecording({
    stopUrl,
    startRecordBtn,
    stopRecordBtn,
    recordStatus,
    recordResult,
    recordedVideo,
    downloadRecordLink,
    openRecordLink
}) {
    setButtonState(stopRecordBtn, true);
    setStatus(recordStatus, "Trạng thái: đang dừng và xử lý video...");

    try {
        const data = await postJson(stopUrl);

        if (data.success) {
            setStatus(recordStatus, `Trạng thái: đã ghi xong ${data.frame_count} frame.`);

            updateRecordedVideo(recordedVideo, data.video_url);
            updateLink(downloadRecordLink, data.download_url);
            updateLink(openRecordLink, data.video_url);
            showElement(recordResult);
        } else {
            setStatus(recordStatus, `Lỗi: ${data.message}`);
        }
    } catch (error) {
        setStatus(recordStatus, "Lỗi: không thể dừng ghi hình.");
    }

    setButtonState(startRecordBtn, false);
}


async function postJson(url) {
    const response = await fetch(url, {
        method: "POST"
    });

    if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
    }

    return response.json();
}


function updateRecordedVideo(videoElement, videoUrl) {
    if (!videoElement || !videoUrl) {
        return;
    }

    videoElement.innerHTML = "";

    const source = document.createElement("source");
    source.src = videoUrl;
    source.type = "video/mp4";

    videoElement.appendChild(source);
    videoElement.load();
}


function updateLink(linkElement, url) {
    if (!linkElement || !url) {
        return;
    }

    linkElement.href = url;
}


function setButtonState(button, isDisabled) {
    if (!button) {
        return;
    }

    button.disabled = isDisabled;
}


function setStatus(statusElement, message) {
    if (!statusElement) {
        return;
    }

    statusElement.textContent = message;
}


function showElement(element) {
    if (!element) {
        return;
    }

    element.classList.remove("is-hidden");
}


function hideElement(element) {
    if (!element) {
        return;
    }

    element.classList.add("is-hidden");
}