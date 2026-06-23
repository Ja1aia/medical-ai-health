import re
import logging
import os
import re
import logging
import os

os.makedirs("logs", exist_ok=True)

# Pakai named logger, bukan basicConfig
guardrails_logger = logging.getLogger("guardrails")
guardrails_logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/guardrails_audit.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
guardrails_logger.addHandler(handler)

HARMFUL_PATTERNS = [
    r"overdose",
    r"how to kill",
    r"suicide",
    r"self.harm",
    r"lethal dose",
    r"how much.*to die",
    r"poison",
    r"drug.*abuse",
    r"ignore.*instructions",
    r"ignore.*system.*prompt",
    r"bypass.*guardrail",
    r"pretend.*doctor",
    r"you are now dan",
    r"as a.*professional.*lethal",
    r"for research.*overdose",
    r"not wake up",
    r"drug.*combinations.*kill",
    r"make.*poison",
]


URGENT_PATTERNS = [
    r"chest pain",
    r"can't breathe",
    r"tidak bisa napas",
    r"sesak napas",
    r"heart attack",
    r"stroke",
    r"unconscious",
    r"tidak sadarkan diri",
    r"bleeding heavily",
    r"perdarahan",
]
def check_input(text):
    text_lower = text.lower()

    for pattern in URGENT_PATTERNS:
        if re.search(pattern, text_lower):
            guardrails_logger.info(
                f"URGENT DETECTED | pattern: {pattern} | input: {text[:100]}"
            )
            return {
                "allowed": False,
                "type": "urgent",
                "message": (
                    "Gejala yang Anda sebutkan memerlukan penanganan segera. "
                    "Hubungi 119 (Indonesia) atau pergi ke UGD rumah sakit terdekat sekarang. "
                    "Jangan menunda untuk mencari bantuan medis."
                ),
            }

    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, text_lower):
            guardrails_logger.info(
                f"HARMFUL DETECTED | pattern: {pattern} | input: {text[:100]}"
            )
            return {
                "allowed": False,
                "type": "harmful",
                "message": (
                    "Saya tidak dapat membantu dengan permintaan tersebut. "
                    "Jika Anda membutuhkan bantuan, hubungi 119."
                ),
            }

    guardrails_logger.info(f"INPUT ALLOWED | input: {text[:100]}")
    return {"allowed": True, "type": None, "message": None}


def check_output(text):
    dose_pattern = r"\b\d+\s*(mg|ml|mcg|gram|tablet|pill)\b"
    if re.search(dose_pattern, text.lower()):
        guardrails_logger.info(f"OUTPUT WARNING | dose information detected")
        return {
            "allowed": True,
            "warning": "Response mengandung informasi dosis. Tambahkan disclaimer.",
            "text": text
            + "\n\n*Disclaimer: Informasi dosis di atas hanya untuk referensi umum. Selalu konsultasikan dengan dokter atau apoteker sebelum mengonsumsi obat apapun.*",
        }
    return {"allowed": True, "warning": None, "text": text}
