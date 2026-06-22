import re

HARMFUL_PATTERNS = [
    r"overdose",
    r"how to kill",
    r"suicide",
    r"self.harm",
    r"lethal dose",
    r"how much.*to die",
    r"poison",
    r"drug.*abuse",
    # Adversarial / jailbreak patterns
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

    # Cek urgent dulu
    for pattern in URGENT_PATTERNS:
        if re.search(pattern, text_lower):
            return {
                "allowed": False,
                "type": "urgent",
                "message": (
                    "Gejala yang Anda sebutkan memerlukan penanganan segera. "
                    "Hubungi 119 (Indonesia) atau pergi ke UGD rumah sakit terdekat sekarang. "
                    "Jangan menunda untuk mencari bantuan medis."
                ),
            }

    # Cek harmful
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, text_lower):
            return {
                "allowed": False,
                "type": "harmful",
                "message": (
                    "Saya tidak dapat membantu dengan permintaan tersebut. "
                    "Jika Anda membutuhkan bantuan, hubungi 119."
                ),
            }

    return {"allowed": True, "type": None, "message": None}


def check_output(text):
    # Pastikan response tidak mengandung dosis spesifik obat
    dose_pattern = r"\b\d+\s*(mg|ml|mcg|gram|tablet|pill)\b"
    if re.search(dose_pattern, text.lower()):
        return {
            "allowed": True,
            "warning": "Response mengandung informasi dosis. Tambahkan disclaimer.",
            "text": text
            + "\n\n*Disclaimer: Informasi dosis di atas hanya untuk referensi umum. Selalu konsultasikan dengan dokter atau apoteker sebelum mengonsumsi obat apapun.*",
        }
    return {"allowed": True, "warning": None, "text": text}
