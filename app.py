"""Command-line entry point for indexing, searching and cited answers."""

import argparse
import sys

from corpus import SOURCES, ingest
from search import retrieve


def main():
    parser = argparse.ArgumentParser(description="Search NASA lunar science PDFs")
    parser.add_argument(
        "action", nargs="?", choices=["ingest", "search", "ask"], default="ask"
    )
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()
    if args.action == "ingest":
        ingest()
        return
    question = (args.question or input("Ask about the Moon: ")).strip()
    rows = retrieve(question)
    if args.action == "search":
        for row in rows:
            print(f"{SOURCES[row['source']]}#page={row['page']}\n{row['body']}\n")
    else:
        from generation import answer

        print(answer(question, rows))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.exit(f"{type(error).__name__}: {error}")
