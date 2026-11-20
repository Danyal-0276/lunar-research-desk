"""Small regression set. Retrieval checks run locally. --answers invokes Groq."""
import argparse
import re

from generation import ABSTAIN, answer
from search import retrieve

# Each pattern identifies a known answer in the NASA sources; inspect misses manually.
CASES = [
    ("How many astronauts walked on the Moon according to the lithograph?", r"(12|twelve) astronauts"),
    ("How much rock and soil did astronauts bring back to Earth?", r"382\s*(kg|kilograms)"),
    ("What is the name of the loose material covering the lunar surface?", r"regolith"),
    ("What dark volcanic rock makes up the lunar maria?", r"basalt"),
]
UNANSWERABLE = "What is the password to my campus Wi-Fi account?"


def evaluate(with_answers):
    hits = 0
    reciprocal = 0.0
    answer_hits = 0
    for question, pattern in CASES:
        rows = retrieve(question)
        rank = next((i for i, row in enumerate(rows, 1)
                     if re.search(pattern, " ".join(row["body"].split()), re.I)), None)
        hits += int(rank is not None)
        reciprocal += 1 / rank if rank else 0
        print(f"{question}  evidence rank={rank}")
        if with_answers:
            result = answer(question, rows)
            claims = "\n".join(block.split("\nEvidence:")[0] for block in result.split("\n\n"))
            answer_hits += bool(re.search(pattern, claims, re.I))
    print(f"Evidence hit@4={hits/len(CASES):.2f}  MRR@4={reciprocal/len(CASES):.2f}")
    if with_answers:
        print(f"Answer pattern match={answer_hits}/{len(CASES)}")
        print("Off-topic abstention:", answer(UNANSWERABLE, retrieve(UNANSWERABLE)) == ABSTAIN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true")
    evaluate(parser.parse_args().answers)
