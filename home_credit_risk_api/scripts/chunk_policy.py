import json
import re
from pathlib import Path
from typing import Any

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = (
    PROJECT_ROOT
    / "knowledge_base"
    / "Loan_Underwriting_Policy_RAG_Demo_Edition.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "knowledge_base"
    / "processed"
    / "loan_policy_chunks.json"
)


DOCUMENT_NAME = "Loan Underwriting Policy - RAG Demo Edition"
DOCUMENT_VERSION = "1.0"

# Character-based limits.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


# Repeated page elements that should not be embedded.
REPEATED_LINES = {
    "LOAN UNDERWRITING POLICY - RAG DEMO EDITION",
    "Not an official RRFL policy | Synthetic demo rules are clearly marked",
}

PAGE_NUMBER_PATTERN = re.compile(
    r"^Page\s+\d+$",
    flags=re.IGNORECASE,
)

# Matches headings such as:
# 1. Objective and Scope
# 13. Policy Status, Definitions and Decision Hierarchy
TOP_LEVEL_SECTION_PATTERN = re.compile(
    r"^(?P<number>\d{1,2})\.\s+(?P<title>.+)$"
)

APPENDIX_PATTERN = re.compile(
    r"^(?P<title>Appendix(?:\s+[A-Z0-9]+)?.*)$",
    flags=re.IGNORECASE,
)


def normalize_line(line: str) -> str:
    """
    Normalize whitespace and PDF-specific bullet characters.
    """

    line = line.replace("\u00a0", " ")
    line = line.replace("", "-")
    line = line.replace("\uf0b7", "-")

    return re.sub(r"[ \t]+", " ", line).strip()


def clean_page_text(text: str) -> list[str]:
    """
    Remove repeated headers, footers and empty lines.
    """

    cleaned_lines: list[str] = []

    for raw_line in text.splitlines():
        line = normalize_line(raw_line)

        if not line:
            continue

        if line in REPEATED_LINES:
            continue

        if PAGE_NUMBER_PATTERN.fullmatch(line):
            continue

        cleaned_lines.append(line)

    return cleaned_lines


def classify_policy_type(section_number: str) -> str:
    """
    Mark whether content came from the source policy,
    synthetic demo rules or general document notices.
    """

    if section_number.isdigit():
        number = int(section_number)

        if 1 <= number <= 12:
            return "source_derived"

        if number >= 13:
            return "synthetic_demo"

    if section_number.startswith("appendix"):
        return "synthetic_demo"

    return "document_notice"


def extract_sections(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Extract PDF text and group it under numbered policy sections.
    """

    sections: list[dict[str, Any]] = []

    current_section: dict[str, Any] = {
        "section_number": "front_matter",
        "section_title": "Front Matter",
        "page_start": 1,
        "page_end": 1,
        "lines": [],
    }

    seen_section_numbers: set[int] = set()
    last_section_number = 0

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            # sort=True requests top-left to bottom-right reading order.
            page_text = page.get_text(
                "text",
                sort=True,
            )

            lines = clean_page_text(page_text)

            for line in lines:
                section_match = TOP_LEVEL_SECTION_PATTERN.match(
                    line
                )

                appendix_match = APPENDIX_PATTERN.match(line)

                if section_match:
                    candidate_number = int(
                        section_match.group("number")
                    )

                    # Accept only unique, sequentially increasing
                    # top-level policy section numbers.
                    is_top_level_section = (
                        candidate_number not in seen_section_numbers
                        and candidate_number > last_section_number
                    )

                    if is_top_level_section:
                        save_section_if_not_empty(
                            sections,
                            current_section,
                        )

                        current_section = {
                            "section_number": str(candidate_number),
                            "section_title": section_match.group(
                                "title"
                            ),
                            "page_start": page_number,
                            "page_end": page_number,
                            "lines": [],
                        }

                        seen_section_numbers.add(candidate_number)
                        last_section_number = candidate_number

                        continue

                if appendix_match:
                    save_section_if_not_empty(
                        sections,
                        current_section,
                    )

                    appendix_title = appendix_match.group(
                        "title"
                    )

                    current_section = {
                        "section_number": (
                            f"appendix-{len(sections) + 1}"
                        ),
                        "section_title": appendix_title,
                        "page_start": page_number,
                        "page_end": page_number,
                        "lines": [],
                    }

                    continue

                current_section["lines"].append(line)
                current_section["page_end"] = page_number

    save_section_if_not_empty(
        sections,
        current_section,
    )

    return sections


def save_section_if_not_empty(
    sections: list[dict[str, Any]],
    section: dict[str, Any],
) -> None:
    """
    Add a completed section only when it contains useful text.
    """

    body = "\n".join(section["lines"]).strip()

    if not body:
        return

    sections.append({
        "section_number": section["section_number"],
        "section_title": section["section_title"],
        "page_start": section["page_start"],
        "page_end": section["page_end"],
        "body": body,
    })

def clean_front_matter(body: str) -> str:
    """
    Remove the table of contents from the front matter while
    retaining the document-status disclaimer and usage guidance.
    """

    cleaned_lines: list[str] = []
    inside_contents = False

    for raw_line in body.splitlines():
        line = raw_line.strip()

        if line.lower() == "contents":
            inside_contents = True
            continue

        if (
            inside_contents
            and line.lower().startswith(
                "how to read this document"
            )
        ):
            inside_contents = False
            cleaned_lines.append(line)
            continue

        if inside_contents:
            continue

        cleaned_lines.append(raw_line)

    return "\n".join(cleaned_lines).strip()

def create_chunks(
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    chunks: list[dict[str, Any]] = []

    

    for section in sections:
        section_number = str(section["section_number"])

        section_body = section["body"]

        if section_number == "front_matter":
            section_body = clean_front_matter(
                section_body
            )

        if section_number == "front_matter":
            section_heading = "Front Matter"
        else:
            section_heading = (
                f"{section_number}. "
                f"{section['section_title']}"
            )

        # Reserve space for the heading.
        body_chunk_size = max(
            300,
            CHUNK_SIZE - len(section_heading) - 2,
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_chunk_size,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
            length_function=len,
            keep_separator=True,
        )

        body_chunks = splitter.split_text(section_body)

        for chunk_index, body_chunk in enumerate(
            body_chunks,
            start=1,
        ):
            chunk_text = (
                f"{section_heading}\n\n"
                f"{body_chunk.strip()}"
            )

            chunk_id = (
                f"loan-policy-v{DOCUMENT_VERSION}"
                f"-section-{section_number}"
                f"-chunk-{chunk_index:03d}"
            )

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "document_name": DOCUMENT_NAME,
                    "document_version": DOCUMENT_VERSION,
                    "source_file": PDF_PATH.name,
                    "section_number": section_number,
                    "section_title": section[
                        "section_title"
                    ],
                    "policy_type": classify_policy_type(
                        section_number
                    ),
                    "page_start": int(
                        section["page_start"]
                    ),
                    "page_end": int(
                        section["page_end"]
                    ),
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_text),
                },
            })

    return chunks

def validate_chunks(
    chunks: list[dict[str, Any]],
) -> None:
    """
    Perform basic integrity checks before embedding.
    """

    if not chunks:
        raise ValueError("No policy chunks were generated.")

    ids = [chunk["id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    empty_chunks = [
        chunk["id"]
        for chunk in chunks
        if not chunk["text"].strip()
    ]

    if empty_chunks:
        raise ValueError(
            f"Empty chunks detected: {empty_chunks}"
        )
    
    for chunk in chunks:
        text = chunk["text"]

        if "Section Topic" in text:
            raise ValueError(
                f"Table of contents leaked into: {chunk['id']}"
            )

        if (
            "1-12 Objective, business scope" in text
            or "13 Policy status, definitions" in text
        ):
            raise ValueError(
                f"Table of contents leaked into: {chunk['id']}"
            )


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Policy PDF not found: {PDF_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sections = extract_sections(PDF_PATH)

    chunks = create_chunks(sections)

    validate_chunks(chunks)

    output = {
        "document": {
            "name": DOCUMENT_NAME,
            "version": DOCUMENT_VERSION,
            "source_file": PDF_PATH.name,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "section_count": len(sections),
            "chunk_count": len(chunks),
        },
        "chunks": chunks,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    chunk_lengths = [
        len(chunk["text"])
        for chunk in chunks
    ]

    print(f"Sections found: {len(sections)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Minimum chunk size: {min(chunk_lengths)}")
    print(f"Maximum chunk size: {max(chunk_lengths)}")
    print(
        "Average chunk size:",
        round(sum(chunk_lengths) / len(chunk_lengths), 1),
    )
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFirst chunk metadata:")
    print(
        json.dumps(
            chunks[0]["metadata"],
            indent=2,
        )
    )

    print("\nFirst chunk text:")
    print(chunks[0]["text"][:700])


if __name__ == "__main__":
    main()