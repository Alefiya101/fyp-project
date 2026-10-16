// Clinician Review Screen Client Logic
let currentSessionId = "";
let currentTranscript = null;
let currentDraft = null;

async function initReview(sessionId) {
  currentSessionId = sessionId;
  await loadConsultationData();
}

async function loadConsultationData() {
  try {
    const [transcriptRes, draftRes] = await Promise.all([
      fetch(`/api/sessions/${currentSessionId}/transcript`),
      fetch(`/api/sessions/${currentSessionId}/draft`),
    ]);

    if (!transcriptRes.ok || !draftRes.ok) {
      document.getElementById("content-loading").innerHTML = `
        <div style="padding: 2rem; color: #dc2626;">
          <h3>Consultation Data Not Ready</h3>
          <p>The consultation draft has not been generated or is still processing. Check status on dashboard.</p>
        </div>`;
      return;
    }

    currentTranscript = await transcriptRes.json();
    currentDraft = await draftRes.json();

    renderTranscript(currentTranscript);
    renderDraft(currentDraft);
    document.getElementById("content-loading").style.display = "none";
    document.getElementById("review-workspace").style.display = "grid";
  } catch (err) {
    console.error("Failed to load review data", err);
  }
}

function renderTranscript(transcript) {
  const container = document.getElementById("transcript-list");
  container.innerHTML = "";

  (transcript.segments || []).forEach((seg, idx) => {
    const segId = seg.segment_id !== null && seg.segment_id !== undefined ? seg.segment_id : idx;
    const isDoctor = (seg.speaker || "").toLowerCase() === "doctor";
    const speakerClass = isDoctor ? "speaker-doctor" : "speaker-patient";
    const turnClass = isDoctor ? "turn-doctor" : "turn-patient";
    const conf = seg.confidence !== null && seg.confidence !== undefined ? seg.confidence : 1.0;
    const lowConfBadge = conf < 0.55 ? `<span class="badge" style="background:#fef3c7; color:#b45309;">Audio Low Conf: ${(conf * 100).toFixed(0)}%</span>` : "";

    const card = document.createElement("div");
    card.id = `transcript-seg-${segId}`;
    card.className = `transcript-turn ${turnClass}`;
    card.innerHTML = `
      <div class="turn-meta">
        <span class="speaker-tag ${speakerClass}">[seg ${segId}] ${seg.speaker || "SPEAKER"}</span>
        <span>${formatTime(seg.start)} - ${formatTime(seg.end)} ${lowConfBadge}</span>
      </div>
      <div class="turn-text">${escapeHtml(seg.text)}</div>
    `;
    container.appendChild(card);
  });
}

function renderDraft(draft) {
  const ext = draft.extraction.entities || {};
  renderGroundingFlags(draft.grounding_flags || []);
  renderSymptoms(ext.symptoms || []);
  renderDiagnosis(ext.diagnosis || []);
  renderMedications(ext.medications || []);
  renderVitals(ext.vitals || []);
  renderInvestigations(ext.investigations || []);
}

function renderGroundingFlags(flags) {
  const banner = document.getElementById("grounding-banner");
  if (!flags || flags.length === 0) {
    banner.style.display = "none";
    return;
  }

  banner.style.display = "block";
  banner.innerHTML = `
    <div class="grounding-alert">
      <div class="grounding-alert-title">⚠️ ${flags.length} Transcript-Grounding Warnings Flagged</div>
      <ul style="margin-left: 1.25rem; font-size: 0.82rem;">
        ${flags.map(f => `<li><strong>${escapeHtml(f.entity)}:</strong> ${escapeHtml(f.issue)} ${f.offending_value ? `(Value: <em>${escapeHtml(f.offending_value)}</em>)` : ""}</li>`).join("")}
      </ul>
    </div>
  `;
}

function renderSymptoms(symptoms) {
  const tbody = document.getElementById("symptoms-tbody");
  tbody.innerHTML = symptoms.map(s => {
    const negatedPill = s.negated
      ? `<span class="entity-tag tag-negated">Absent / Ruled Out</span>`
      : `<span class="entity-tag tag-confirmed">Present</span>`;
    return `
      <tr>
        <td><strong>${escapeHtml(s.text)}</strong></td>
        <td>${escapeHtml(s.duration || "-")}</td>
        <td>${negatedPill}</td>
        <td>${renderSourceTags(s.source_segments)}</td>
      </tr>
    `;
  }).join("");
}

function renderDiagnosis(diagnosis) {
  const tbody = document.getElementById("diagnosis-tbody");
  tbody.innerHTML = diagnosis.map(d => `
    <tr>
      <td><strong>${escapeHtml(d.text)}</strong></td>
      <td><span class="entity-tag tag-new">${escapeHtml(d.status || "provisional")}</span></td>
      <td>${escapeHtml(d.confidence || "medium")}</td>
      <td>${renderSourceTags(d.source_segments)}</td>
    </tr>
  `).join("");
}

function renderMedications(medications) {
  const tbody = document.getElementById("medications-tbody");
  tbody.innerHTML = medications.map((m, idx) => {
    const statusPill = m.status === "new"
      ? `<span class="entity-tag tag-new">NEW</span>`
      : `<span class="entity-tag tag-existing">${escapeHtml(m.status || "EXISTING")}</span>`;
    return `
      <tr>
        <td><strong>${escapeHtml(m.name)}</strong></td>
        <td><input type="text" class="edit-input" id="med-dose-${idx}" value="${escapeHtml(m.dose || "")}" style="width: 80px; padding: 2px 4px;"></td>
        <td><input type="text" class="edit-input" id="med-freq-${idx}" value="${escapeHtml(m.frequency || "")}" style="width: 60px; padding: 2px 4px;"></td>
        <td><input type="text" class="edit-input" id="med-dur-${idx}" value="${escapeHtml(m.duration || "")}" style="width: 70px; padding: 2px 4px;"></td>
        <td>${statusPill}</td>
        <td>${renderSourceTags(m.source_segments)}</td>
      </tr>
    `;
  }).join("");
}

function renderVitals(vitals) {
  const tbody = document.getElementById("vitals-tbody");
  tbody.innerHTML = vitals.map(v => `
    <tr>
      <td>${escapeHtml(v.type.replace(/_/g, " ").toUpperCase())}</td>
      <td><strong>${escapeHtml(String(v.value))} ${escapeHtml(v.unit || "")}</strong></td>
      <td>${renderSourceTags(v.source_segments)}</td>
    </tr>
  `).join("");
}

function renderInvestigations(investigations) {
  const tbody = document.getElementById("investigations-tbody");
  tbody.innerHTML = investigations.map(i => `
    <tr>
      <td><strong>${escapeHtml(i.name || i.text || "-")}</strong></td>
      <td><span class="entity-tag tag-new">${escapeHtml(i.status || "ordered")}</span></td>
      <td>${escapeHtml(i.result || "Pending")}</td>
      <td>${renderSourceTags(i.source_segments)}</td>
    </tr>
  `).join("");
}

function renderSourceTags(sources) {
  if (!sources || sources.length === 0) return `<span style="color:#94a3b8; font-size:0.75rem;">None</span>`;
  return sources.map(s => `
    <span class="entity-tag tag-source" onclick="highlightTranscriptSegment(${s})" title="Click to locate in transcript">
      seg [${s}]
    </span>
  `).join(" ");
}

function highlightTranscriptSegment(segIndex) {
  // Clear previous highlights
  document.querySelectorAll(".transcript-turn").forEach(el => el.classList.remove("highlighted"));
  const target = document.getElementById(`transcript-seg-${segIndex}`);
  if (target) {
    target.classList.add("highlighted");
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

async function saveDraftEdits() {
  if (!currentDraft) return;
  const ext = currentDraft.extraction.entities;

  // Collect updated medication inputs
  (ext.medications || []).forEach((m, idx) => {
    const doseEl = document.getElementById(`med-dose-${idx}`);
    const freqEl = document.getElementById(`med-freq-${idx}`);
    const durEl = document.getElementById(`med-dur-${idx}`);
    if (doseEl) m.dose = doseEl.value.trim() || null;
    if (freqEl) m.frequency = freqEl.value.trim() || null;
    if (durEl) m.duration = durEl.value.trim() || null;
  });

  try {
    const res = await fetch(`/api/sessions/${currentSessionId}/draft?actor_id=dr.sharma`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentDraft.extraction),
    });
    if (!res.ok) throw new Error("Failed to save draft edits");
    alert("Draft successfully re-mapped to FHIR and re-validated!");
    await loadConsultationData();
  } catch (err) {
    alert("Error updating draft: " + err.message);
  }
}

async function approveSession() {
  const clinicianId = prompt("Enter Clinician ID or PIN to cryptographically sign and approve:", "dr.sharma@opd");
  if (!clinicianId) return;

  try {
    const res = await fetch(`/api/sessions/${currentSessionId}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clinician_id: clinicianId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Approval failed");

    alert(`Consultation approved successfully!\nDigital Signature: ${data.signature.slice(0, 16)}...\nSynced to ABDM: ${data.synced_immediately ? "Yes (Instant)" : "Enqueued in Outbox"}`);
    window.location.href = "/";
  } catch (err) {
    alert("Approval error: " + err.message);
  }
}

async function rejectSession() {
  const reason = prompt("Enter reason for rejection:", "Incorrect clinical attribution");
  if (!reason) return;

  try {
    const res = await fetch(`/api/sessions/${currentSessionId}/reject`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clinician_id: "dr.sharma@opd", reason: reason }),
    });
    if (!res.ok) throw new Error("Failed to reject draft");
    alert("Session marked as rejected.");
    window.location.href = "/";
  } catch (err) {
    alert("Error: " + err.message);
  }
}

function openFhirModal() {
  if (!currentDraft) return;
  document.getElementById("modal-title").innerText = "FHIR R4 ABDM OPConsultRecord Document Bundle";
  document.getElementById("modal-json").innerText = JSON.stringify(currentDraft.fhir_bundle, null, 2);
  document.getElementById("json-modal").classList.add("open");
}

async function openAuditModal() {
  try {
    const res = await fetch(`/api/sessions/${currentSessionId}/audit`);
    const data = await res.json();
    document.getElementById("modal-title").innerText = `Tamper-Evident Audit Trail (${data.audit_trail.length} events)`;
    document.getElementById("modal-json").innerText = JSON.stringify(data.audit_trail, null, 2);
    document.getElementById("json-modal").classList.add("open");
  } catch (err) {
    alert("Could not load audit trail");
  }
}

function closeModal() {
  document.getElementById("json-modal").classList.remove("open");
}

function formatTime(seconds) {
  if (seconds === null || seconds === undefined) return "00:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
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
