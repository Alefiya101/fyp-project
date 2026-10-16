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
  approval: {
    status: "DRAFT",
    doctor_name: "Dr. Consulting Physician",
    doctor_id: "practitioner-001",
    approved_at: null,
    signature: null,
    notes: null,
  },
  abdmSync: {
    status: "PENDING_APPROVAL",
    mode: "MOCK_SANDBOX",
    request_id: null,
    care_context: "OPD-live_consultation",
    synced_at: null,
  },
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
const btnValidateProfile = document.getElementById("btn-validate-profile");

// Doctor Review & Decision Controls
const btnApproveConsultation = document.getElementById("btn-approve-consultation");
const btnRejectConsultation = document.getElementById("btn-reject-consultation");
const modalReject = document.getElementById("modal-reject");
const btnConfirmReject = document.getElementById("btn-confirm-reject");
const selectRejectReason = document.getElementById("select-reject-reason");
const inputRejectNotes = document.getElementById("input-reject-notes");
const modalProfileAudit = document.getElementById("modal-profile-audit");
const profileAuditContent = document.getElementById("profile-audit-content");

// Audio Controls
const btnRecordToggle = document.getElementById("btn-record-toggle");
const recordBtnText = document.getElementById("record-btn-text");
const micIcon = document.getElementById("mic-icon");
const recordTimer = document.getElementById("record-timer");
const btnUploadAudio = document.getElementById("btn-upload-audio");
const audioFileInput = document.getElementById("audio-file-input");
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
  checkAbdmStatus();
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
  if (btnValidateProfile) btnValidateProfile.addEventListener("click", handleAuditProfiles);

  // Doctor Review & Decision Handlers
  if (btnApproveConsultation) btnApproveConsultation.addEventListener("click", handleApproveConsultation);
  if (btnRejectConsultation) btnRejectConsultation.addEventListener("click", () => openModal(modalReject));
  if (btnConfirmReject) btnConfirmReject.addEventListener("click", handleConfirmReject);

  // Audio actions (Option 1: Start/Stop Recording, Option 2: Upload Audio)
  btnRecordToggle.addEventListener("click", toggleRecording);
  btnUploadAudio.addEventListener("click", () => audioFileInput.click());
  audioFileInput.addEventListener("change", handleAudioUpload);

  // Option 3: Quick Demo Case Select
  const selectDemoCase = document.getElementById("select-demo-case");
  if (selectDemoCase) {
    selectDemoCase.addEventListener("change", async (e) => {
      const caseId = e.target.value;
      if (!caseId) return;
      showToast(`Loading ${caseId}...`);
      try {
        const resp = await fetch(`/api/demo-cases/${caseId}`, { method: "POST" });
        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || "Failed to load demo case");
        }
        const data = await resp.json();
        populateStateFromData(data);
        renderAllViews();
        showToast("Demo consultation loaded & ABDM FHIR bundle generated!");
      } catch (err) {
        showToast(`Demo error: ${err.message}`, true);
      } finally {
        selectDemoCase.value = "";
      }
    });
  }
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

  // Doctor Review Decision Bar
  renderDecisionBar();
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

  state.symptoms = (summary.symptoms || []).map((s) => ({
    name: s.name,
    duration: s.duration || "3",
    status: s.clinical_status || "active",
  }));

  state.diagnoses = (summary.diagnoses || []).map((d) => ({
    name: d.name,
    confidence: d.confidence || "high conf",
    status: "provisional",
  }));

  state.medications = (summary.medications || []).map((m) => ({
    name: m.name,
    dosage: m.dosage || "500 mg",
    frequency: m.frequency || "TDS",
    duration: m.duration || "3 days",
    status: m.status || "new",
  }));

  state.vitals = (summary.vitals || []).map((v) => ({ name: v.name, value: v.value }));

  state.investigations = summary.investigations || [];

  if (data.approval) {
    state.approval = data.approval;
  } else {
    state.approval = {
      status: "DRAFT",
      doctor_name: state.practitionerName,
      doctor_id: "practitioner-001",
      approved_at: null,
      signature: null,
      notes: null,
    };
  }

  if (data.abdm_sync) {
    state.abdmSync = data.abdm_sync;
  } else {
    state.abdmSync = {
      status: "PENDING_APPROVAL",
      mode: "MOCK_SANDBOX",
      request_id: null,
      care_context: `OPD-${state.sessionId}`,
      synced_at: null,
    };
  }
}

/* ==========================================================================
   Microphone Audio Recording & Server Actions
   ========================================================================== */

async function toggleRecording() {
  if (!isRecording) {
    await startRecording();
  } else {
    await stopRecording();
  }
}

async function startRecording() {
  try {
    const resp = await fetch("/api/record/start", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Failed to start recording");
    }
    setRecordingState(true);
    startTimer();
    showToast("Live consultation recording started. Speak freely...");
  } catch (err) {
    showToast(`Error starting recording: ${err.message}`, true);
  }
}

async function stopRecording() {
  setRecordingState(false);
  stopTimer();
  showToast("Stopping recording & processing with Whisper & Qwen2.5 LLM Refiner...");

  btnRecordToggle.disabled = true;
  recordBtnText.textContent = "Processing...";
  micIcon.className = "ph-bold ph-spinner ph-spin";

  try {
    const resp = await fetch("/api/record/stop", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Audio processing failed");
    }

    const data = await resp.json();
    populateStateFromData(data);
    renderAllViews();
    showToast("Consultation processed & ABDM FHIR bundle updated!");
  } catch (err) {
    showToast(`Recording error: ${err.message}`, true);
  } finally {
    btnRecordToggle.disabled = false;
    recordBtnText.textContent = "Start Recording";
    micIcon.className = "ph-fill ph-microphone";
    recordTimer.textContent = "00:00";
  }
}

function setRecordingState(recording) {
  isRecording = recording;
  if (recording) {
    btnRecordToggle.classList.add("recording");
    recordBtnText.textContent = "Stop Recording";
    micIcon.className = "ph-fill ph-stop";
  } else {
    btnRecordToggle.classList.remove("recording");
    recordBtnText.textContent = "Start Recording";
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

async function handleAudioUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  showToast(`Uploading ${file.name}...`);
  await sendAudioForProcessing(file, file.name);
  audioFileInput.value = "";
}

async function sendAudioForProcessing(blobOrFile, filename) {
  const uploadBtn = document.getElementById("btn-upload-audio");
  const origHtml = uploadBtn ? uploadBtn.innerHTML : "";
  if (uploadBtn) {
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="ph-bold ph-spinner ph-spin"></i> Processing...';
  }
  showToast(`Processing ${filename} with Whisper ASR & Clinical NLP...`);

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
  } finally {
    if (uploadBtn) {
      uploadBtn.disabled = false;
      uploadBtn.innerHTML = origHtml;
    }
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

/* ==========================================================================
   Doctor Review, Approval, Rejection & ABDM Integration Logic
   ========================================================================== */

function renderDecisionBar() {
  const doctorDecisionBar = document.getElementById("doctor-decision-bar");
  const decisionStatusText = document.getElementById("decision-status-text");
  const decisionMainIcon = document.getElementById("decision-main-icon");
  const approvalSealDetails = document.getElementById("approval-seal-details");
  const sealSignatureDisplay = document.getElementById("seal-signature-display");
  const sealDoctorDisplay = document.getElementById("seal-doctor-display");
  const sealCareContextDisplay = document.getElementById("seal-care-context-display");
  const sealSyncStatusDisplay = document.getElementById("seal-sync-status-display");
  const rejectionDetails = document.getElementById("rejection-details");
  const rejectionReasonText = document.getElementById("rejection-reason-text");
  const btnApprove = document.getElementById("btn-approve-consultation");
  const btnReject = document.getElementById("btn-reject-consultation");

  if (!doctorDecisionBar) return;

  const status = (state.approval && state.approval.status) ? state.approval.status.toUpperCase() : "DRAFT";

  if (status === "APPROVED") {
    doctorDecisionBar.className = "doctor-decision-bar approved";
    decisionStatusText.innerHTML = `<span class="status-indicator status-approved"></span> Approved by ${escapeHtml(state.approval.doctor_name || state.practitionerName)} &bull; ABDM Synced`;
    if (decisionMainIcon) decisionMainIcon.className = "ph-duotone ph-seal-check";
    if (approvalSealDetails) approvalSealDetails.classList.remove("hidden");
    if (rejectionDetails) rejectionDetails.classList.add("hidden");

    if (sealSignatureDisplay) {
      const sig = state.approval.signature || "HMAC-SHA256";
      sealSignatureDisplay.textContent = sig.length > 28 ? sig.substring(0, 24) + "..." : sig;
      sealSignatureDisplay.title = sig;
    }
    if (sealDoctorDisplay) sealDoctorDisplay.textContent = state.approval.doctor_name || state.practitionerName;
    if (sealCareContextDisplay) sealCareContextDisplay.textContent = (state.abdmSync && state.abdmSync.care_context) || `OPD-${state.sessionId}`;
    if (sealSyncStatusDisplay) {
      const mode = (state.abdmSync && state.abdmSync.mode) || "SANDBOX";
      const reqId = (state.abdmSync && state.abdmSync.request_id) ? state.abdmSync.request_id.substring(0, 8) + "..." : "Active";
      sealSyncStatusDisplay.textContent = `Synced to ABDM Gateway (${mode} &bull; ${reqId})`;
    }

    if (btnApprove) {
      btnApprove.innerHTML = '<i class="ph-bold ph-check"></i> Approved &amp; Synced';
      btnApprove.disabled = true;
      btnApprove.style.opacity = "0.75";
    }
    if (btnReject) {
      btnReject.disabled = false;
      btnReject.style.opacity = "1";
    }
  } else if (status === "REJECTED") {
    doctorDecisionBar.className = "doctor-decision-bar rejected";
    decisionStatusText.innerHTML = `<span class="status-indicator status-rejected"></span> Rejected by Doctor`;
    if (decisionMainIcon) decisionMainIcon.className = "ph-duotone ph-prohibit";
    if (approvalSealDetails) approvalSealDetails.classList.add("hidden");
    if (rejectionDetails) rejectionDetails.classList.remove("hidden");
    if (rejectionReasonText) rejectionReasonText.textContent = state.approval.reason || "Draft rejected by clinician";

    if (btnApprove) {
      btnApprove.innerHTML = '<i class="ph-bold ph-check-circle"></i> Re-Approve &amp; Push';
      btnApprove.disabled = false;
      btnApprove.style.opacity = "1";
    }
    if (btnReject) {
      btnReject.disabled = true;
      btnReject.style.opacity = "0.5";
    }
  } else {
    // DRAFT
    doctorDecisionBar.className = "doctor-decision-bar";
    decisionStatusText.innerHTML = `<span class="status-indicator status-draft"></span> Draft Ready for Clinician Review`;
    if (decisionMainIcon) decisionMainIcon.className = "ph-duotone ph-signature";
    if (approvalSealDetails) approvalSealDetails.classList.add("hidden");
    if (rejectionDetails) rejectionDetails.classList.add("hidden");

    if (btnApprove) {
      btnApprove.innerHTML = '<i class="ph-bold ph-check-circle"></i> Approve &amp; Push to ABDM';
      btnApprove.disabled = false;
      btnApprove.style.opacity = "1";
    }
    if (btnReject) {
      btnReject.disabled = false;
      btnReject.style.opacity = "1";
    }
  }
}

async function handleApproveConsultation() {
  const btnApprove = document.getElementById("btn-approve-consultation");
  const origHtml = btnApprove.innerHTML;
  btnApprove.disabled = true;
  btnApprove.innerHTML = '<i class="ph-bold ph-spinner ph-spin"></i> Validating &amp; Syncing...';

  try {
    const payload = {
      session_id: state.sessionId,
      doctor_name: state.practitionerName,
      doctor_id: "practitioner-001",
    };

    const resp = await fetch("/api/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Approval failed");
    }

    const data = await resp.json();
    populateStateFromData(data);
    renderAllViews();
    showToast("✓ Record approved by doctor, digitally sealed, and synced to ABDM Sandbox!");
  } catch (err) {
    showToast(`Approval error: ${err.message}`, true);
    btnApprove.disabled = false;
    btnApprove.innerHTML = origHtml;
  }
}

async function handleConfirmReject() {
  const selectReason = document.getElementById("select-reject-reason");
  const inputNotes = document.getElementById("input-reject-notes");
  let reason = selectReason.value;
  if (inputNotes && inputNotes.value.trim()) {
    reason += ` - ${inputNotes.value.trim()}`;
  }

  closeAllModals();

  try {
    const payload = {
      session_id: state.sessionId,
      doctor_name: state.practitionerName,
      doctor_id: "practitioner-001",
      reason: reason,
    };

    const resp = await fetch("/api/reject", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Rejection failed");
    }

    const data = await resp.json();
    populateStateFromData(data);
    renderAllViews();
    showToast("Record marked as REJECTED. Sync to ABDM Sandbox halted.");
  } catch (err) {
    showToast(`Rejection error: ${err.message}`, true);
  }
}

async function handleAuditProfiles() {
  const modalProfileAudit = document.getElementById("modal-profile-audit");
  const contentEl = document.getElementById("profile-audit-content");
  openModal(modalProfileAudit);
  contentEl.innerHTML = '<div style="text-align: center; padding: 1.5rem;"><i class="ph-bold ph-spinner ph-spin" style="font-size: 1.5rem; color: #2563EB;"></i><p style="margin-top: 0.5rem; color: #64748B;">Validating all resources against ABDM OPConsultRecord Profiles...</p></div>';

  try {
    const resp = await fetch("/api/validate-bundle");
    if (!resp.ok) {
      throw new Error("Failed to validate profiles");
    }
    const data = await resp.json();
    const conf = data.abdm_profile_conformance || {};
    const conforms = conf.conforms;

    let html = `
      <div style="background: ${conforms ? '#F0FDF4' : '#FEF2F2'}; border: 1px solid ${conforms ? '#BBF7D0' : '#FECACA'}; border-radius: 8px; padding: 0.85rem; margin-bottom: 1rem;">
        <div style="font-weight: 700; color: ${conforms ? '#166534' : '#991B1B'}; display: flex; align-items: center; gap: 0.4rem;">
          <i class="ph-bold ${conforms ? 'ph-check-circle' : 'ph-warning'}" style="font-size: 1.2rem;"></i>
          ${conforms ? 'ABDM OPConsultRecord Profile Conformance: 100% VALID' : 'Profile Conformance Warnings'}
        </div>
        <div style="font-size: 0.78rem; color: #475569; margin-top: 0.25rem;">
          Target Profile: <code>${escapeHtml(conf.target_profile || "https://nrces.in/ndhm/fhir/r4/StructureDefinition/OPConsultRecord")}</code>
        </div>
      </div>

      <h4 style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 0.5rem;">Resource Profile Conformance Breakdown</h4>
      <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-bottom: 1rem;">
        <thead>
          <tr style="border-bottom: 2px solid #E2E8F0; text-align: left; color: #475569;">
            <th style="padding: 0.4rem 0.5rem;">RESOURCE</th>
            <th style="padding: 0.4rem 0.5rem;">COUNT</th>
            <th style="padding: 0.4rem 0.5rem;">ABDM PROFILE URI</th>
            <th style="padding: 0.4rem 0.5rem;">STATUS</th>
          </tr>
        </thead>
        <tbody>
    `;

    const resources = conf.profile_conformance || {};
    for (const [rtype, rinfo] of Object.entries(resources)) {
      html += `
        <tr style="border-bottom: 1px solid #F1F5F9;">
          <td style="padding: 0.4rem 0.5rem; font-weight: 600;">${escapeHtml(rtype)}</td>
          <td style="padding: 0.4rem 0.5rem;">${rinfo.count || 1}</td>
          <td style="padding: 0.4rem 0.5rem; font-family: monospace; font-size: 0.72rem; color: #475569;">${escapeHtml(rinfo.profile || "")}</td>
          <td style="padding: 0.4rem 0.5rem;">
            <span style="display: inline-flex; align-items: center; gap: 0.25rem; font-weight: 600; color: ${rinfo.present ? '#16A34A' : '#CA8A04'};">
              <i class="ph-bold ${rinfo.present ? 'ph-check' : 'ph-warning'}"></i> ${rinfo.present ? 'Conforms' : 'Partial'}
            </span>
          </td>
        </tr>
      `;
    }

    html += `
        </tbody>
      </table>

      <h4 style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 0.4rem;">Standard Coding Systems Verified</h4>
      <div style="display: flex; flex-wrap: wrap; gap: 0.4rem;">
        ${(conf.coding_systems || ["http://snomed.info/sct", "http://loinc.org", "http://www.nlm.nih.gov/research/umls/rxnorm"]).map(cs =>
          `<span style="background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; border-radius: 4px; padding: 0.2rem 0.5rem; font-size: 0.75rem; font-family: monospace;">${escapeHtml(cs)}</span>`
        ).join('')}
      </div>
    `;

    contentEl.innerHTML = html;
  } catch (err) {
    contentEl.innerHTML = `<div style="color: #DC2626; padding: 1rem;">Failed to audit profiles: ${escapeHtml(err.message)}</div>`;
  }
}

async function checkAbdmStatus() {
  try {
    const resp = await fetch("/api/abdm/status");
    if (!resp.ok) return;
    const data = await resp.json();
    const sandboxBadge = document.getElementById("abdm-sandbox-badge");
    const statusText = document.getElementById("sandbox-status-text");
    if (sandboxBadge && statusText) {
      if (data.status === "online") {
        sandboxBadge.classList.add("online");
        statusText.textContent = `ABDM Sandbox: ${data.mode === "MOCK_SANDBOX" ? "Active (Sandbox Mode)" : "Live"}`;
      } else {
        sandboxBadge.classList.remove("online");
        statusText.textContent = "ABDM Sandbox: Standby";
      }
    }
  } catch (e) {
    console.debug("ABDM status check skipped:", e);
  }
}
