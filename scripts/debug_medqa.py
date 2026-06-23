import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import requests
import re

dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

for i, row in enumerate(dataset):
    if i >= 5:
        break

    question = row.get("question", "")
    opa = row.get("opa", "")
    opb = row.get("opb", "")
    opc = row.get("opc", "")
    opd = row.get("opd", "")
    cop = row.get("cop", 0)
    answer_letter = idx_to_letter.get(cop, "A")

    response = requests.post(
        "http://localhost:8000/eval_mcq",
        json={
            "question": question,
            "option_a": opa,
            "option_b": opb,
            "option_c": opc,
            "option_d": opd,
        },
        timeout=30,
    )
    data = response.json()
    ai_answer = data.get("answer", "")
    match = re.search(r"\b([A-D])\b", ai_answer[:20])
    predicted = match.group(1).upper() if match else "?"

    print(f"Q: {question[:100]}")
    print(f"A) {opa} | B) {opb} | C) {opc} | D) {opd}")
    print(f"Correct: {answer_letter} | Predicted: {predicted} | Raw: {ai_answer[:30]}")
    print("---")
