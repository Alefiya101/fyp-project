"""llm_refiner.py

Local LLM Post-Correction module for noisy speech-to-text transcripts.
Uses Qwen2.5-1.5B-Instruct to restore phonetic corruptions in Indian OPD dialogues
before clinical entity extraction and FHIR bundling.

Implements Person B's SLM refinement layer (Plan Section 5 & Section 10 Fallbacks).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = """You are an expert Indian OPD Clinical Transcription Refiner.
You receive noisy speech-to-text output from a microphone in an Indian outpatient doctor consultation.
The transcript contains phonetic distortions and Hinglish/English speech errors.

Your Task:
1. Reconstruct the dialogue into natural, clean Indian OPD dialogue (Hinglish/English).
2. Fix phonetic corruptions based on acoustic sound-alike medical terms:
   - "cello rocker sar" / "hello doctor Sam" -> "Hello doctor sir"
   - "jay khaase sardi or evar uwah hai" -> "Mujhe khansi, sardi aur fever hua hai"
   - "doctor, mujhee, easev, sath dinos hai hai" -> "Doctor, mujhe fever 7 din se hai"
   - "asthma asthma cough liquor" -> "Azithromycin aur cough syrup"
   - "door timer" -> "Do time (twice daily)"
   - "afternoon meals" -> "After meals"
   - "needle" -> "Neend" / "Need"
   - "bujie" / "mooji" -> "Mujhe"
3. Preserve the speaker turns (Doctor and Patient).
4. Do NOT hallucinate or fabricate diseases or symptoms that were not phonetically suggested.
5. Output MUST be strictly valid JSON in this exact format:
{
  "refined_turns": [
    {"index": 0, "speaker": "patient", "text": "cleaned text here"}
  ]
}"""


class ClinicalLLMRefiner:
    """Post-corrects noisy ASR transcripts using a lightweight local instruction model."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        local_weights_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.local_weights_dir = Path(local_weights_dir) if local_weights_dir else None
        self._model = None
        self._tokenizer = None

    def _resolve_model_path(self) -> str | None:
        """Checks for local offline checkpoints first before attempting download."""
        candidates = [
            self.local_weights_dir,
            Path("clinical_service/models/qwen1.5b"),
            Path(__file__).resolve().parent.parent / "models" / "qwen1.5b",
            Path(r"C:\Users\Ashtha\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct"),
        ]
        for c in candidates:
            if c and c.is_dir() and (c / "config.json").is_file():
                has_weights = any(c.glob("*.safetensors")) or any(c.glob("pytorch_model*.bin"))
                if has_weights:
                    logger.info("Auto-detected local Qwen2.5 checkpoint at '%s'", c)
                    return str(c)
        return None

    def _load_model(self):
        """Lazy load the model and tokenizer onto CPU on first usage."""
        if self._model is None:
            resolved = self._resolve_model_path()
            if resolved is None:
                # No local model weights downloaded yet; fallback refiner is active
                return None
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Loading Qwen2.5-1.5B refiner model from '%s' on %s…", resolved, self.device)
                self._tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    resolved,
                    torch_dtype=torch.float32,
                    local_files_only=True,
                )
                self._model.to(torch.device(self.device))
                self._model.eval()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load Qwen2.5-1.5B model (%s). Fallback refiner active.", exc)
                self._model = None
        return self._model

    def build_refinement_prompt(self, segments: list[dict[str, Any]]) -> str:
        """Formats transcript segments into a clean prompt."""
        turn_lines = []
        for idx, seg in enumerate(segments):
            spk = seg.get("speaker", "unknown").upper()
            txt = seg.get("text", "").strip()
            turn_lines.append(f"Turn {idx} [{spk}]: {txt}")
        raw_dialogue = "\n".join(turn_lines)

        user_content = (
            "Here is the raw speech-to-text transcript from an outpatient consultation:\n\n"
            f"{raw_dialogue}\n\n"
            "Return the corrected clinical dialogue turns as valid JSON matching the schema."
        )

        tokenizer = self._tokenizer
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return f"{SYSTEM_PROMPT}\n\nUser: {user_content}\nAssistant: {{\"refined_turns\": ["

    def refine_transcript(self, transcript_input: Any) -> Any:
        """Refines the text of every segment in an Interface 1 Transcript."""
        is_dict = isinstance(transcript_input, dict)
        segments_raw = (
            transcript_input.get("segments", []) if is_dict else getattr(transcript_input, "segments", [])
        )
        if not segments_raw:
            return transcript_input

        # Normalize segment dict representation
        seg_dicts = []
        for s in segments_raw:
            if isinstance(s, dict):
                seg_dicts.append(dict(s))
            elif hasattr(s, "model_dump"):
                seg_dicts.append(s.model_dump())
            else:
                seg_dicts.append({
                    "start": getattr(s, "start", 0.0),
                    "end": getattr(s, "end", 0.0),
                    "speaker": getattr(s, "speaker", "unknown"),
                    "text": getattr(s, "text", ""),
                    "confidence": getattr(s, "confidence", 1.0),
                })

        # Check if model can be loaded
        model = self._load_model()
        if model is None or self._tokenizer is None:
            # Fallback heuristic cleanup if weights not yet loaded
            return self._heuristic_cleanup(transcript_input, seg_dicts, is_dict)

        try:
            import torch

            prompt = self.build_refinement_prompt(seg_dicts)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    repetition_penalty=1.15,
                    do_sample=False,
                )

            # Slice out input tokens
            input_len = inputs["input_ids"].shape[-1]
            generated_tokens = output_ids[0, input_len:]
            response_text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            parsed_turns = self._parse_json_turns(response_text)
            if parsed_turns:
                for idx, turn in enumerate(parsed_turns):
                    if idx < len(seg_dicts):
                        cleaned_txt = turn.get("text", "").strip()
                        if cleaned_txt:
                            seg_dicts[idx]["text"] = cleaned_txt
                            if "speaker" in turn and turn["speaker"] in ("doctor", "patient"):
                                seg_dicts[idx]["speaker"] = turn["speaker"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qwen LLM transcript refinement failed (%s); using heuristic cleanup.", exc)
            return self._heuristic_cleanup(transcript_input, seg_dicts, is_dict)

        if is_dict:
            res = dict(transcript_input)
            res["segments"] = seg_dicts
            return res

        # Pydantic Transcript object
        updated_model_segs = []
        for orig, d in zip(transcript_input.segments, seg_dicts):
            updated_model_segs.append(orig.model_copy(update={"text": d["text"], "speaker": d["speaker"]}))
        return transcript_input.model_copy(update={"segments": updated_model_segs})

    def _parse_json_turns(self, text: str) -> list[dict[str, Any]] | None:
        """Safely extracts JSON turns from LLM output."""
        try:
            # Try finding JSON block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if "refined_turns" in data and isinstance(data["refined_turns"], list):
                    return data["refined_turns"]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not parse LLM JSON response: %s", exc)
        return None

    def _heuristic_cleanup(
        self,
        transcript_input: Any,
        seg_dicts: list[dict[str, Any]],
        is_dict: bool,
    ) -> Any:
        """Rule-based clinical phonetic restoration fallback."""
        replacements = [
            (r"\bcello rocker sar\b", "Hello doctor sir"),
            (r"\bhello doctor Sam\b", "Hello doctor sir"),
            (r"\bjay khaase sardi or evar uwah hai\b", "Mujhe khansi, sardi aur fever hua hai"),
            (r"\bevar\b", "fever"),
            (r"\bfeeber\b", "fever"),
            (r"\bphyahua hai appu\b", "Kya hua hai aapko"),
            (r"\bdoctor eye health coal then curves\b", "Doctor mujhe cold and cough hai"),
            (r"\basthma asthma cough liquor\b", "Azithromycin aur cough syrup"),
            (r"\bdoor timer\b", "Do time (twice daily)"),
            (r"\bafternoon meals\b", "After meals"),
            (r"\bsath dinos\b", "7 dino se"),
            (r"\bfrom past seven days\b", "7 dino se"),
        ]

        for s in seg_dicts:
            txt = s.get("text", "")
            for pattern, repl in replacements:
                txt = re.sub(pattern, repl, txt, flags=re.IGNORECASE)
            s["text"] = txt

        if is_dict:
            res = dict(transcript_input)
            res["segments"] = seg_dicts
            return res

        updated_model_segs = []
        for orig, d in zip(transcript_input.segments, seg_dicts):
            updated_model_segs.append(orig.model_copy(update={"text": d["text"]}))
        return transcript_input.model_copy(update={"segments": updated_model_segs})


_GLOBAL_REFINER: ClinicalLLMRefiner | None = None


def get_llm_refiner() -> ClinicalLLMRefiner:
    """Returns singleton instance of ClinicalLLMRefiner."""
    global _GLOBAL_REFINER
    if _GLOBAL_REFINER is None:
        _GLOBAL_REFINER = ClinicalLLMRefiner()
    return _GLOBAL_REFINER


def refine_transcript(transcript: Any) -> Any:
    """Convenience functional interface for transcript refinement."""
    return get_llm_refiner().refine_transcript(transcript)
