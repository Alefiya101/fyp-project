/**
 * ABDM Ambient Clinical Studio - Frontend Application Logic
 *
 * Implements interactive doctor editing for:
 * - Patient and Practitioner names
 * - Adding/removing Chief Complaints & Symptoms
 * - Adding/removing Provisional Diagnoses
 * - Adding/removing Prescribed Medications in the table
 * - Adding/removing Vitals and Lab Orders
 * - Live real-time ABDM FHIR R4 document bundle regeneration
 * - Ambient microphone capture and file upload
 *
 * Strictly omits any transcript display.
 */

// State Object
const state = {
  sessionId: "live_consultation",
  time: "10:02 AM",
  patientName: "OPD Patient",
  practitionerName: "Dr. Consulting Physician",
  symptoms: [
    { name: "fever, cold, cough", duration: "3", status: "active" }
  ],
  diagnoses: [
    { name: "common cold", confidence: "high conf", status: "provisional" }
  ],
  medications: [
    { name: "azithromycin", dosage: "500 mg", frequency: "TDS", duration: "3 days", status: "new" }
  ],
  vitals: [],
  investigations: [],
  groundingFlags: [
    { entity: "fever", issue: "Extracted from low-confidence speech turn (confidence: 0.50). Potential ASR audio artifact." },
    { entity: "cough", issue: "Extracted from low-confidence speech turn (confidence: 0.50). Potential ASR audio artifact." },
    { entity: "cold", issue: "Extracted from low-confidence speech turn (confidence: 0.50). Potential ASR audio artifact." }
  ],
  fhirBundle: null,
};

// DOM References
const patientNameInput = document.getElementById("patient-name-input");
const practitionerNameInput = document.getElementById("practitioner-name-input");
const sessionIdDisplay = document.getElementById("session-id-display");
const sessionTimeDisplay = document.getElementById("session-time-display");

const symptomsContainer = document.getElementById("symptoms-container");
const diagnosesContainer = document.getElementById("diagnoses-container");
const medicationsTbody = document.getElementById("medications-tbody");
const vitalsContainer = document.getElementById("vitals-container");
const investigationsContainer = document.getElementById("investigations-container");

const groundingWarningsList = document.getElementById("grounding-warnings-list");
const fhirJsonCode = document.getElementById("fhir-json-code");
const fhirFilenameDisplay = document.getElementById("fhir-filename-display");

const btnCopyFhir = document.getElementById("btn-copy-fhir");
const btnDownloadFhir = document.getElementById("btn-download-fhir");

// Audio Controls
const btnRecordMic = document.getElementById("btn-record-mic");
const recordBtnText = document.getElementById("record-btn-text");
const micIcon = document.getElementById("mic-icon");
const recordTimer = document.getElementById("record-timer");
const btnTimedRecord = document.getElementById("btn-timed-record");
const recordDurationSelect = document.getElementById("record-duration-select");
const btnUploadAudio = document.getElementById("btn-upload-audio");
const audioFileInput = document.getElementById("audio-file-input");
const btnReloadLatest = document.getElementById("btn-reload-latest");
const toast = document.getElementById("toast");

// Modals
const modalComplaint = document.getElementById("modal-complaint");
const modalDiagnosis = document.getElementById("modal-diagnosis");
const modalMedication = document.getElementById("modal-medication");
const modalVital = document.getElementById("modal-vital");
const modalInvestigation = document.getElementById("modal-investigation");

let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let timerInterval = null;
let elapsedSeconds = 0;

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadLatestConsultation();
});

function setupEventListeners() {
  // Inline edit listeners (debounced update)
  patientNameInput.addEventListener("change", () => {
    state.patientName = patientNameInput.value.trim() || "OPD Patient";
    syncConsultationToBackend();
  });

  practitionerNameInput.addEventListener("change", () => {
    state.practitionerName = practitionerNameInput.value.trim() || "Dr. Consulting Physician";
    syncConsultationToBackend();
  });

  // Modal triggers
  document.getElementById("btn-add-complaint").addEventListener("click", () => openModal(modalComplaint));
  document.getElementById("btn-add-diagnosis").addEventListener("click", () => openModal(modalDiagnosis));
  document.getElementById("btn-add-medication").addEventListener("click", () => openModal(modalMedication));
  document.getElementById("btn-add-vital").addEventListener("click", () => openModal(modalVital));
  document.getElementById("btn-add-investigation").addEventListener("click", () => openModal(modalInvestigation));

  // Modal close buttons
  document.querySelectorAll(".modal-close-btn, .modal-cancel").forEach((btn) => {
    btn.addEventListener("click", closeAllModals);
  });

  // Modal Save Handlers
  document.getElementById("btn-save-complaint").addEventListener("click", handleAddComplaint);
  document.getElementById("btn-save-diagnosis").addEventListener("click", handleAddDiagnosis);
  document.getElementById("btn-save-medication").addEventListener("click", handleAddMedication);
  document.getElementById("btn-save-vital").addEventListener("click", handleAddVital);
  document.getElementById("btn-save-investigation").addEventListener("click", handleAddInvestigation);

  // Copy & Download
  btnCopyFhir.addEventListener("click", copyFhir);
  btnDownloadFhir.addEventListener("click", downloadFhir);

  // Audio actions
  btnRecordMic.addEventListener("click", toggleMicRecording);
  btnTimedRecord.addEventListener("click", startTimedServerRecording);
  btnUploadAudio.addEventListener("click", () => audioFileInput.click());
  audioFileInput.addEventListener("change", handleAudioUpload);
  btnReloadLatest.addEventListener("click", loadLatestConsultation);
}

/* ==========================================================================
   Modal Operations
   ========================================================================== */

function openModal(modalEl) {
  modalEl.classList.remove("hidden");
  const firstInput = modalEl.querySelector("input");
  if (firstInput) firstInput.focus();
}

function closeAllModals() {
  document.querySelectorAll(".modal-overlay").forEach((m) => m.classList.add("hidden"));
}

function handleAddComplaint() {
  const name = document.getElementById("input-complaint-name").value.trim();
  const duration = document.getElementById("input-complaint-duration").value.trim() || "3 days";
  if (!name) return;

  state.symptoms.push({ name, duration, status: "active" });
  document.getElementById("input-complaint-name").value = "";
  closeAllModals();
  renderAllViews();
  syncConsultationToBackend();
  showToast(`Added complaint: ${name}`);
}

function handleAddDiagnosis() {
  const name = document.getElementById("input-diagnosis-name").value.trim();
  const confidence = document.getElementById("input-diagnosis-confidence").value || "high conf";
  if (!name) return;

  state.diagnoses.push({ name, confidence, status: "provisional" });
  document.getElementById("input-diagnosis-name").value = "";
  closeAllModals();
  renderAllViews();
  syncConsultationToBackend();
  showToast(`Added diagnosis: ${name}`);
}

function handleAddMedication() {
  const name = document.getElementById("input-med-name").value.trim();
  const dosage = document.getElementById("input-med-dosage").value.trim() || "500 mg";
  const frequency = document.getElementById("input-med-frequency").value || "TDS";
  const duration = document.getElementById("input-med-duration").value.trim() || "3 days";
  const status = document.getElementById("input-med-status").value || "new";
  if (!name) return;

  state.medications.push({ name, dosage, frequency, duration, status });
  document.getElementById("input-med-name").value = "";
  closeAllModals();
  renderAllViews();
  syncConsultationToBackend();
  showToast(`Prescribed: ${name} (${dosage})`);
}

function handleAddVital() {
  const name = document.getElementById("input-vital-name").value.trim();
  const value = document.getElementById("input-vital-value").value.trim();
  if (!name || !value) return;

  state.vitals.push({ name, value });
  document.getElementById("input-vital-name").value = "";
  document.getElementById("input-vital-value").value = "";
  closeAllModals();
  renderAllViews();
  syncConsultationToBackend();
  showToast(`Recorded vital: ${name}`);
}

function handleAddInvestigation() {
  const name = document.getElementById("input-inv-name").value.trim();
  if (!name) return;

  state.investigations.push(name);
  document.getElementById("input-inv-name").value = "";
  closeAllModals();
  renderAllViews();
  syncConsultationToBackend();
  showToast(`Ordered investigation: ${name}`);
}

/* ==========================================================================
   Render View Functions
   ========================================================================== */

function renderAllViews() {
  // Metadata
  patientNameInput.value = state.patientName;
  practitionerNameInput.value = state.practitionerName;
  sessionIdDisplay.textContent = state.sessionId;
  sessionTimeDisplay.textContent = state.time;
  fhirFilenameDisplay.textContent = `bundle-${state.sessionId}.json`;

  // Symptoms
  symptomsContainer.innerHTML = "";
  if (state.symptoms.length === 0) {
    symptomsContainer.innerHTML = '<span class="empty-text">No symptoms recorded.</span>';
  } else {
    state.symptoms.forEach((sym, idx) => {
      const tag = document.createElement("div");
      tag.className = "pill-tag pill-symptom";
      tag.innerHTML = `
        <span>${escapeHtml(sym.name)}</span>
        <span class="pill-badge">${escapeHtml(sym.duration || "3")}</span>
        <button class="pill-remove-btn" title="Remove Complaint" data-idx="${idx}">&times;</button>
      `;
      tag.querySelector(".pill-remove-btn").addEventListener("click", () => {
        state.symptoms.splice(idx, 1);
        renderAllViews();
        syncConsultationToBackend();
      });
      symptomsContainer.appendChild(tag);
    });
  }

  // Diagnoses
  diagnosesContainer.innerHTML = "";
  if (state.diagnoses.length === 0) {
    diagnosesContainer.innerHTML = '<span class="empty-text">No provisional diagnosis entered.</span>';
  } else {
    state.diagnoses.forEach((diag, idx) => {
      const tag = document.createElement("div");
      tag.className = "pill-tag pill-diagnosis";
      tag.innerHTML = `
        <span>${escapeHtml(diag.name)}</span>
        <span class="pill-badge badge-conf">${escapeHtml(diag.confidence || "high conf")}</span>
        <button class="pill-remove-btn" title="Remove Diagnosis" data-idx="${idx}">&times;</button>
      `;
      tag.querySelector(".pill-remove-btn").addEventListener("click", () => {
        state.diagnoses.splice(idx, 1);
        renderAllViews();
        syncConsultationToBackend();
      });
      diagnosesContainer.appendChild(tag);
    });
  }

  // Medications Table
  medicationsTbody.innerHTML = "";
  if (state.medications.length === 0) {
    medicationsTbody.innerHTML = '<tr><td colspan="6" class="empty-text text-center">No medications prescribed.</td></tr>';
  } else {
    state.medications.forEach((med, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="font-bold">${escapeHtml(med.name)}</td>
        <td>${escapeHtml(med.dosage || "500 mg")}</td>
        <td>${escapeHtml(med.frequency || "TDS")}</td>
        <td>${escapeHtml(med.duration || "3 days")}</td>
        <td><span class="status-pill-new">${escapeHtml(med.status || "new")}</span></td>
        <td><button class="row-del-btn" title="Delete Medication" data-idx="${idx}">&times;</button></td>
      `;
      tr.querySelector(".row-del-btn").addEventListener("click", () => {
        state.medications.splice(idx, 1);
        renderAllViews();
        syncConsultationToBackend();
      });
      medicationsTbody.appendChild(tr);
    });
  }

  // Vitals
  if (state.vitals.length === 0) {
    vitalsContainer.innerHTML = "No vitals recorded in this encounter.";
  } else {
    vitalsContainer.innerHTML = "";
    const vitalsRow = document.createElement("div");
    vitalsRow.className = "vital-tag-row";
    state.vitals.forEach((vit, idx) => {
      const tag = document.createElement("div");
      tag.className = "pill-tag pill-diagnosis";
      tag.innerHTML = `
        <span><strong>${escapeHtml(vit.name)}:</strong> ${escapeHtml(vit.value)}</span>
        <button class="pill-remove-btn" data-idx="${idx}">&times;</button>
      `;
      tag.querySelector(".pill-remove-btn").addEventListener("click", () => {
        state.vitals.splice(idx, 1);
        renderAllViews();
        syncConsultationToBackend();
      });
      vitalsRow.appendChild(tag);
    });
    vitalsContainer.appendChild(vitalsRow);
  }

  // Investigations
  if (state.investigations.length === 0) {
    investigationsContainer.innerHTML = "None ordered.";
  } else {
    investigationsContainer.innerHTML = "";
    const invRow = document.createElement("div");
    invRow.className = "vital-tag-row";
    state.investigations.forEach((inv, idx) => {
      const tag = document.createElement("div");
      tag.className = "pill-tag pill-symptom";
      tag.innerHTML = `
        <span>${escapeHtml(inv)}</span>
        <button class="pill-remove-btn" data-idx="${idx}">&times;</button>
      `;
      tag.querySelector(".pill-remove-btn").addEventListener("click", () => {
        state.investigations.splice(idx, 1);
        renderAllViews();
        syncConsultationToBackend();
      });
      invRow.appendChild(tag);
    });
    investigationsContainer.appendChild(invRow);
  }

  // Grounding Warnings
  groundingWarningsList.innerHTML = "";
  if (state.groundingFlags.length === 0) {
    groundingWarningsList.innerHTML = `
      <div class="grounding-warning-item" style="color: #059669;">
        <i class="ph-bold ph-check-circle warning-icon" style="color: #059669;"></i>
        <span>All clinical entities verified with zero grounding flags.</span>
      </div>
    `;
  } else {
    state.groundingFlags.forEach((flag) => {
      const item = document.createElement("div");
      item.className = "grounding-warning-item";
      item.innerHTML = `
        <i class="ph-fill ph-warning-diamond warning-icon"></i>
        <span><strong>${escapeHtml(flag.entity || "entity")}:</strong> ${escapeHtml(flag.issue)}</span>
      `;
      groundingWarningsList.appendChild(item);
    });
  }

  // FHIR JSON Code
  if (state.fhirBundle) {
    const formattedJson = JSON.stringify(state.fhirBundle, null, 2);
    fhirJsonCode.innerHTML = syntaxHighlightJson(formattedJson);
  }
}

/* ==========================================================================
   Backend Synchronization
   ========================================================================== */

async function syncConsultationToBackend() {
  try {
    const payload = {
      session_id: state.sessionId,
      patient_name: state.patientName,
      practitioner_name: state.practitionerName,
      symptoms: state.symptoms,
      diagnoses: state.diagnoses,
      medications: state.medications,
      vitals: state.vitals,
      investigations: state.investigations,
    };

    const resp = await fetch("/api/update-consultation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      console.warn("Sync failed with status:", resp.status);
      return;
    }

    const data = await resp.json();
    if (data.fhir_bundle) {
      state.fhirBundle = data.fhir_bundle;
      const formattedJson = JSON.stringify(state.fhirBundle, null, 2);
      fhirJsonCode.innerHTML = syntaxHighlightJson(formattedJson);
    }
  } catch (err) {
    console.error("Error syncing consultation:", err);
  }
}

async function loadLatestConsultation() {
  try {
    const resp = await fetch("/api/latest");
    if (!resp.ok) {
      renderAllViews();
      syncConsultationToBackend();
      return;
    }
    const data = await resp.json();
    populateStateFromData(data);
    renderAllViews();
    showToast("Loaded latest consultation session.");
  } catch (err) {
    console.error("Failed to load consultation:", err);
    renderAllViews();
    syncConsultationToBackend();
  }
}

function populateStateFromData(data) {
  state.sessionId = data.session_id || "live_consultation";
  state.fhirBundle = data.fhir_bundle || null;
  state.groundingFlags = data.grounding_flags || [];

  const summary = data.summary || {};
  if (summary.time) state.time = summary.time;
  if (summary.patient && summary.patient.name) state.patientName = summary.patient.name;
  if (summary.practitioner && summary.practitioner.name) state.practitionerName = summary.practitioner.name;

  if (summary.symptoms && summary.symptoms.length > 0) {
    state.symptoms = summary.symptoms.map((s) => ({
      name: s.name,
      duration: s.duration || "3",
      status: s.clinical_status || "active",
    }));
  }

  if (summary.diagnoses && summary.diagnoses.length > 0) {
    state.diagnoses = summary.diagnoses.map((d) => ({
      name: d.name,
      confidence: d.confidence || "high conf",
      status: "provisional",
    }));
  }

  if (summary.medications && summary.medications.length > 0) {
    state.medications = summary.medications.map((m) => ({
      name: m.name,
      dosage: m.dosage || "500 mg",
      frequency: m.frequency || "TDS",
      duration: m.duration || "3 days",
      status: m.status || "new",
    }));
  }

  if (summary.vitals && summary.vitals.length > 0) {
    state.vitals = summary.vitals.map((v) => ({ name: v.name, value: v.value }));
  }

  if (summary.investigations && summary.investigations.length > 0) {
    state.investigations = summary.investigations;
  }
}

/* ==========================================================================
   Microphone Audio Recording & Server Actions
   ========================================================================== */

async function toggleMicRecording() {
  if (!isRecording) {
    await startBrowserRecording();
  } else {
    stopBrowserRecording();
  }
}

async function startBrowserRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    let mimeType = "audio/webm";
    if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
      mimeType = "audio/webm;codecs=opus";
    }

    mediaRecorder = new MediaRecorder(stream, { mimeType });
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const audioBlob = new Blob(audioChunks, { type: mimeType });
      await sendAudioForProcessing(audioBlob, "live_recording.webm");
    };

    mediaRecorder.start(250);
    setRecordingState(true);
    startTimer();
    showToast("Live recording started... Speak into your mic.");
  } catch (err) {
    console.warn("Browser mic access fallback:", err);
    startTimedServerRecording();
  }
}

function stopBrowserRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  setRecordingState(false);
  stopTimer();
  showToast("Audio processing through Whisper ASR & Qwen2.5 Refiner...");
}

function setRecordingState(recording) {
  isRecording = recording;
  if (recording) {
    btnRecordMic.classList.add("recording");
    recordBtnText.textContent = "Stop Recording";
    micIcon.className = "ph-fill ph-stop";
  } else {
    btnRecordMic.classList.remove("recording");
    recordBtnText.textContent = "Record Live Audio";
    micIcon.className = "ph-fill ph-microphone";
  }
}

function startTimer() {
  elapsedSeconds = 0;
  recordTimer.textContent = "00:00";
  timerInterval = setInterval(() => {
    elapsedSeconds++;
    const m = String(Math.floor(elapsedSeconds / 60)).padStart(2, "0");
    const s = String(elapsedSeconds % 60).padStart(2, "0");
    recordTimer.textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}

async function startTimedServerRecording() {
  const duration = parseFloat(recordDurationSelect.value) || 15;
  setRecordingState(true);
  showToast(`Recording ${duration}s from laptop microphone...`);

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
    const resp = await fetch("/api/record-mic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ duration }),
    });

    clearInterval(countdownInterval);
    setRecordingState(false);

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Recording failed");
    }

    const data = await resp.json();
    populateStateFromData(data);
    renderAllViews();
    showToast(`Recorded & scribed successfully (${duration}s)!`);
  } catch (err) {
    clearInterval(countdownInterval);
    setRecordingState(false);
    showToast(`Error: ${err.message}`, true);
  }
}

async function handleAudioUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  showToast(`Uploading ${file.name}...`);
  await sendAudioForProcessing(file, file.name);
  audioFileInput.value = "";
}

async function sendAudioForProcessing(blobOrFile, filename) {
  try {
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
    populateStateFromData(data);
    renderAllViews();
    showToast("Audio successfully converted to ABDM FHIR R4!");
  } catch (err) {
    showToast(`Processing error: ${err.message}`, true);
  }
}

/* ==========================================================================
   Utilities (Copy, Download, JSON Syntax Highlighting, Toast)
   ========================================================================== */

function copyFhir() {
  if (!state.fhirBundle) {
    showToast("No FHIR bundle to copy.", true);
    return;
  }
  navigator.clipboard.writeText(JSON.stringify(state.fhirBundle, null, 2));
  showToast("FHIR R4 Bundle copied to clipboard!");
}

function downloadFhir() {
  if (!state.fhirBundle) {
    showToast("No FHIR bundle to download.", true);
    return;
  }
  const blob = new Blob([JSON.stringify(state.fhirBundle, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `bundle-${state.sessionId}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("Downloaded ABDM FHIR R4 Bundle JSON.");
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
  toast.style.borderColor = isError ? "#EF4444" : "#2563EB";
  toast.style.color = isError ? "#FCA5A5" : "#FFFFFF";
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3500);
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
