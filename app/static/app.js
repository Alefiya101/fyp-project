/**
 * ABDM Scribe Studio - Frontend Client Logic
 * 
 * Handles microphone audio capture, API dispatch, stepper state transitions,
 * clinical EMR visualization, and ABDM FHIR R4 exploration.
 * Strictly guarantees ZERO transcripts are displayed in the UI.
 */

// Global State
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let timerInterval = null;
let elapsedSeconds = 0;
let currentFhirBundle = null;

// DOM Elements
const btnRecordToggle = document.getElementById("btn-record-toggle");
const micPulseRing = document.getElementById("mic-pulse-ring");
const micIcon = document.getElementById("mic-icon");
const recordTimer = document.getElementById("record-timer");
const recordInstruction = document.getElementById("record-instruction");
const waveformVisualizer = document.getElementById("waveform-visualizer");
const btnTimedRecord = document.getElementById("btn-timed-record");
const recordDurationSelect = document.getElementById("record-duration-select");
const btnUploadFile = document.getElementById("btn-upload-file");
const audioFileInput = document.getElementById("audio-file-input");
const btnLoadLatest = document.getElementById("btn-load-latest");

// Stepper Elements
const stepVad = document.getElementById("step-vad");
const stepAsr = document.getElementById("step-asr");
const stepLlm = document.getElementById("step-llm");
const stepFhir = document.getElementById("step-fhir");

// Meta & Alert Elements
const metaSessionId = document.getElementById("meta-session-id");
const metaPatientName = document.getElementById("meta-patient-name");
const metaPractitionerName = document.getElementById("meta-practitioner-name");
const metaAbdmStatus = document.getElementById("meta-abdm-status");
const metaGroundingStatus = document.getElementById("meta-grounding-status");
const groundingAlertsCard = document.getElementById("grounding-alerts-card");
const groundingAlertsList = document.getElementById("grounding-alerts-list");
const alertCountPill = document.getElementById("alert-count-pill");

// EMR Containers
const containerSymptoms = document.getElementById("container-symptoms");
const containerMedications = document.getElementById("container-medications");
const containerVitals = document.getElementById("container-vitals");
const containerDiagnoses = document.getElementById("container-diagnoses");
const countSymptoms = document.getElementById("count-symptoms");
const countMedications = document.getElementById("count-medications");
const countVitals = document.getElementById("count-vitals");
const countDiagnoses = document.getElementById("count-diagnoses");

// FHIR Explorer Elements
const fhirJsonCode = document.getElementById("fhir-json-code");
const resourcePills = document.getElementById("resource-pills");
const btnCopyFhir = document.getElementById("btn-copy-fhir");
const btnDownloadFhir = document.getElementById("btn-download-fhir");
const toast = document.getElementById("toast");


// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadLatestConsultation();
});


function setupEventListeners() {
  btnRecordToggle.addEventListener("click", toggleRecording);
  btnTimedRecord.addEventListener("click", startTimedServerRecording);
  btnUploadFile.addEventListener("click", () => audioFileInput.click());
  audioFileInput.addEventListener("change", handleFileUpload);
  btnLoadLatest.addEventListener("click", loadLatestConsultation);
  btnCopyFhir.addEventListener("click", copyFhirToClipboard);
  btnDownloadFhir.addEventListener("click", downloadFhirBundle);
}


/* ==========================================================================
   Microphone Recording via Browser Web Audio API (16kHz WAV format)
   ========================================================================== */

async function toggleRecording() {
  if (!isRecording) {
    await startBrowserRecording();
  } else {
    stopBrowserRecording();
  }
}

async function startBrowserRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    // Check supported MIME type
    let mimeType = "audio/webm";
    if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
      mimeType = "audio/webm;codecs=opus";
    } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
      mimeType = "audio/ogg;codecs=opus";
    }

    mediaRecorder = new MediaRecorder(stream, { mimeType });
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(track => track.stop());
      const audioBlob = new Blob(audioChunks, { type: mimeType });
      await sendAudioBlobForProcessing(audioBlob, "consultation_recording.webm");
    };

    mediaRecorder.start(250);
    setRecordingUIState(true);
    startTimer();
    showToast("Live consultation recording started...");
  } catch (err) {
    console.warn("Browser mic access error, falling back to server-side microphone:", err);
    startTimedServerRecording();
  }
}

function stopBrowserRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  setRecordingUIState(false);
  stopTimer();
  showToast("Audio captured. Processing through ABDM pipeline...");
}

function setRecordingUIState(recording) {
  isRecording = recording;
  if (recording) {
    micPulseRing.classList.add("recording");
    btnRecordToggle.classList.add("is-recording");
    micIcon.className = "ph-fill ph-stop";
    waveformVisualizer.classList.add("active");
    recordInstruction.textContent = "Recording in progress... Click STOP when consultation concludes";
  } else {
    micPulseRing.classList.remove("recording");
    btnRecordToggle.classList.remove("is-recording");
    micIcon.className = "ph-fill ph-microphone";
    waveformVisualizer.classList.remove("active");
    recordInstruction.textContent = "Processing audio through Whisper ASR and Qwen2.5 Refiner...";
  }
}

function startTimer() {
  elapsedSeconds = 0;
  recordTimer.textContent = "00:00";
  timerInterval = setInterval(() => {
    elapsedSeconds++;
    const mins = String(Math.floor(elapsedSeconds / 60)).padStart(2, "0");
    const secs = String(elapsedSeconds % 60).padStart(2, "0");
    recordTimer.textContent = `${mins}:${secs}`;
  }, 1000);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}


/* ==========================================================================
   Server-side Timed Recording (Intel Smart Sound Mic Array)
   ========================================================================== */

async function startTimedServerRecording() {
  const duration = parseFloat(recordDurationSelect.value) || 15;
  setRecordingUIState(true);
  recordInstruction.textContent = `Server recording ${duration}s from laptop microphone array... Speak now!`;
  
  // Local countdown
  let remaining = duration;
  recordTimer.textContent = `00:${String(remaining).padStart(2, "0")}`;
  const countdownInterval = setInterval(() => {
    remaining--;
    if (remaining >= 0) {
      recordTimer.textContent = `00:${String(remaining).padStart(2, "0")}`;
    } else {
      clearInterval(countdownInterval);
    }
  }, 1000);

  try {
    updateStepperProgress(1); // VAD
    const resp = await fetch("/api/record-mic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ duration }),
    });

    clearInterval(countdownInterval);
    setRecordingUIState(false);

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Server recording failed");
    }

    const data = await resp.json();
    completeStepperProgress();
    renderConsultation(data);
    showToast(`Consultation scribed successfully! (${duration}s recording)`);
  } catch (err) {
    clearInterval(countdownInterval);
    setRecordingUIState(false);
    resetStepper();
    showToast(`Error: ${err.message}`, true);
  }
}


/* ==========================================================================
   File Upload Handling
   ========================================================================== */

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  showToast(`Uploading ${file.name}...`);
  await sendAudioBlobForProcessing(file, file.name);
  audioFileInput.value = "";
}

async function sendAudioBlobForProcessing(blobOrFile, filename) {
  try {
    simulateStepperSequence();
    const formData = new FormData();
    formData.append("file", blobOrFile, filename);

    const resp = await fetch("/api/process-audio", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Audio processing failed");
    }

    const data = await resp.json();
    completeStepperProgress();
    renderConsultation(data);
    showToast("Audio successfully scribed & mapped to ABDM FHIR R4!");
  } catch (err) {
    resetStepper();
    recordInstruction.textContent = "Click microphone to start live consultation capture";
    showToast(`Processing error: ${err.message}`, true);
  }
}


/* ==========================================================================
   Load Latest Consultation from output/live_fhir_bundle.json
   ========================================================================== */

async function loadLatestConsultation() {
  try {
    const resp = await fetch("/api/latest");
    if (!resp.ok) {
      console.log("No previous consultation found yet.");
      return;
    }
    const data = await resp.json();
    renderConsultation(data);
    showToast("Latest consultation loaded.");
  } catch (err) {
    console.error("Error loading latest consultation:", err);
  }
}


/* ==========================================================================
   Rendering Consultation & ABDM FHIR EMR Views (ZERO Transcript)
   ========================================================================== */

function renderConsultation(data) {
  currentFhirBundle = data.fhir_bundle;
  const summary = data.summary || {};
  const validation = data.validation || {};
  const groundingFlags = data.grounding_flags || [];

  // 1. Meta Banner
  metaSessionId.textContent = data.session_id || "LIVE_SESSION_01";
  if (summary.patient) {
    metaPatientName.textContent = `${summary.patient.name} (${summary.patient.id})`;
  }
  if (summary.practitioner) {
    metaPractitionerName.textContent = `${summary.practitioner.name} (General OPD)`;
  }
  
  if (validation.schema_valid) {
    metaAbdmStatus.className = "meta-status status-valid";
    metaAbdmStatus.innerHTML = '<i class="ph-bold ph-check-circle"></i> Schema Valid (NRCES)';
  } else {
    metaAbdmStatus.className = "meta-status";
    metaAbdmStatus.style.color = "#F87171";
    metaAbdmStatus.innerHTML = `<i class="ph-bold ph-x-circle"></i> ${validation.errors.length} Errors`;
  }

  if (groundingFlags.length === 0) {
    metaGroundingStatus.className = "meta-status status-valid";
    metaGroundingStatus.innerHTML = '<i class="ph-bold ph-shield-check"></i> 100% Grounded';
  } else {
    metaGroundingStatus.className = "meta-status";
    metaGroundingStatus.style.color = "#FBBF24";
    metaGroundingStatus.innerHTML = `<i class="ph-bold ph-warning"></i> ${groundingFlags.length} Safety Flag${groundingFlags.length > 1 ? 's' : ''}`;
  }

  // 2. Grounding Safety Warnings Card
  renderGroundingAlerts(groundingFlags);

  // 3. Symptoms Card
  renderSymptoms(summary.symptoms || []);

  // 4. Medications Card
  renderMedications(summary.medications || []);

  // 5. Vitals Card
  renderVitals(summary.vitals || []);

  // 6. Diagnoses Card
  renderDiagnoses(summary.diagnoses || []);

  // 7. FHIR Resource Pills & JSON Viewer
  renderFhirExplorer(data.fhir_bundle, summary.resource_counts || {});
  
  recordInstruction.textContent = "Consultation ready. Click microphone to record next patient.";
}

function renderGroundingAlerts(flags) {
  if (!flags || flags.length === 0) {
    groundingAlertsCard.classList.add("hidden");
    return;
  }

  groundingAlertsCard.classList.remove("hidden");
  alertCountPill.textContent = `${flags.length} Flag${flags.length > 1 ? 's' : ''}`;
  groundingAlertsList.innerHTML = "";

  flags.forEach(flag => {
    const item = document.createElement("div");
    item.className = "alert-item";
    item.innerHTML = `
      <div class="alert-item-icon">
        <i class="ph-fill ph-warning"></i>
      </div>
      <div class="alert-item-content">
        <div class="alert-item-title">${escapeHtml(flag.entity)}: ${escapeHtml(flag.offending_value || '')}</div>
        <div class="alert-item-desc">${escapeHtml(flag.issue)}</div>
      </div>
    `;
    groundingAlertsList.appendChild(item);
  });
}

function renderSymptoms(symptoms) {
  countSymptoms.textContent = symptoms.length;
  if (symptoms.length === 0) {
    containerSymptoms.innerHTML = '<div class="empty-state">No symptoms recorded in this session.</div>';
    return;
  }

  containerSymptoms.innerHTML = "";
  symptoms.forEach(sym => {
    const row = document.createElement("div");
    row.className = "clinical-item-row";
    row.innerHTML = `
      <div class="item-main">
        <span class="item-name">${escapeHtml(sym.name)}</span>
        <span class="item-subtext">SNOMED CT: <span class="code-pill">${escapeHtml(sym.snomed_code || '386661006')}</span></span>
      </div>
      <span class="item-badge badge-active">${escapeHtml(sym.clinical_status || 'active')}</span>
    `;
    containerSymptoms.appendChild(row);
  });
}

function renderMedications(medications) {
  countMedications.textContent = medications.length;
  if (medications.length === 0) {
    containerMedications.innerHTML = '<div class="empty-state">No medications prescribed in this session.</div>';
    return;
  }

  containerMedications.innerHTML = "";
  medications.forEach(med => {
    const row = document.createElement("div");
    row.className = "clinical-item-row";
    row.innerHTML = `
      <div class="item-main">
        <span class="item-name">${escapeHtml(med.name)}</span>
        <span class="item-subtext">
          RxNorm: <span class="code-pill">${escapeHtml(med.rxnorm_code || 'RxNorm')}</span>
          ${med.note ? `&bull; ${escapeHtml(med.note)}` : ''}
        </span>
      </div>
      <span class="item-badge badge-new">${escapeHtml(med.status || 'new')}</span>
    `;
    containerMedications.appendChild(row);
  });
}

function renderVitals(vitals) {
  countVitals.textContent = vitals.length;
  if (vitals.length === 0) {
    containerVitals.innerHTML = '<div class="empty-state">No vitals observed in this session.</div>';
    return;
  }

  containerVitals.innerHTML = "";
  vitals.forEach(vit => {
    const row = document.createElement("div");
    row.className = "clinical-item-row";
    row.innerHTML = `
      <div class="item-main">
        <span class="item-name">${escapeHtml(vit.name)}</span>
        <span class="item-subtext">LOINC: <span class="code-pill">${escapeHtml(vit.loinc_code || 'LOINC')}</span></span>
      </div>
      <span class="item-badge badge-active">${escapeHtml(vit.value)}</span>
    `;
    containerVitals.appendChild(row);
  });
}

function renderDiagnoses(diagnoses) {
  countDiagnoses.textContent = diagnoses.length;
  if (diagnoses.length === 0) {
    containerDiagnoses.innerHTML = '<div class="empty-state">No provisional diagnoses entered.</div>';
    return;
  }

  containerDiagnoses.innerHTML = "";
  diagnoses.forEach(diag => {
    const row = document.createElement("div");
    row.className = "clinical-item-row";
    row.innerHTML = `
      <div class="item-main">
        <span class="item-name">${escapeHtml(diag.name)}</span>
        <span class="item-subtext">SNOMED CT: <span class="code-pill">${escapeHtml(diag.snomed_code || 'SNOMED')}</span></span>
      </div>
      <span class="item-badge badge-active">${escapeHtml(diag.clinical_status || 'active')}</span>
    `;
    containerDiagnoses.appendChild(row);
  });
}

function renderFhirExplorer(bundle, counts) {
  if (!bundle) return;

  // Resource Pills
  resourcePills.innerHTML = "";
  for (const [rtype, count] of Object.entries(counts)) {
    const pill = document.createElement("span");
    pill.className = "res-pill";
    pill.textContent = `${rtype}: ${count}`;
    resourcePills.appendChild(pill);
  }

  // Pretty JSON syntax highlighting
  const jsonStr = JSON.stringify(bundle, null, 2);
  fhirJsonCode.innerHTML = syntaxHighlightJson(jsonStr);
}


/* ==========================================================================
   Stepper Progression
   ========================================================================== */

function updateStepperProgress(stepNumber) {
  resetStepper();
  if (stepNumber >= 1) stepVad.className = "step-item active";
  if (stepNumber >= 2) { stepVad.className = "step-item done"; stepAsr.className = "step-item active"; }
  if (stepNumber >= 3) { stepAsr.className = "step-item done"; stepLlm.className = "step-item active"; }
  if (stepNumber >= 4) { stepLlm.className = "step-item done"; stepFhir.className = "step-item active"; }
}

function completeStepperProgress() {
  stepVad.className = "step-item done";
  stepAsr.className = "step-item done";
  stepLlm.className = "step-item done";
  stepFhir.className = "step-item done";
}

function resetStepper() {
  stepVad.className = "step-item";
  stepAsr.className = "step-item";
  stepLlm.className = "step-item";
  stepFhir.className = "step-item";
}

function simulateStepperSequence() {
  updateStepperProgress(1);
  setTimeout(() => updateStepperProgress(2), 2000);
  setTimeout(() => updateStepperProgress(3), 5000);
  setTimeout(() => updateStepperProgress(4), 8000);
}


/* ==========================================================================
   Utilities (Copy, Download, JSON Syntax Highlighting, Toast)
   ========================================================================== */

function copyFhirToClipboard() {
  if (!currentFhirBundle) {
    showToast("No FHIR bundle to copy.", true);
    return;
  }
  navigator.clipboard.writeText(JSON.stringify(currentFhirBundle, null, 2));
  showToast("ABDM FHIR R4 Bundle copied to clipboard!");
}

function downloadFhirBundle() {
  if (!currentFhirBundle) {
    showToast("No FHIR bundle available to download.", true);
    return;
  }
  const blob = new Blob([JSON.stringify(currentFhirBundle, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `abdm_fhir_bundle_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("Downloaded ABDM OPConsultRecord Bundle JSON.");
}

function syntaxHighlightJson(json) {
  json = json.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
    let cls = "json-number";
    if (/^"/.test(match)) {
      if (/:$/.test(match)) {
        cls = "json-key";
      } else {
        cls = "json-string";
      }
    } else if (/true|false/.test(match)) {
      cls = "json-boolean";
    } else if (/null/.test(match)) {
      cls = "json-null";
    }
    return '<span class="' + cls + '">' + match + "</span>";
  });
}

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.style.borderColor = isError ? "rgba(239, 68, 68, 0.4)" : "rgba(37, 99, 235, 0.4)";
  toast.style.color = isError ? "#FCA5A5" : "#F8FAFC";
  toast.classList.add("show");
  setTimeout(() => {
    toast.classList.remove("show");
  }, 3500);
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
