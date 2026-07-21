from app.ingestion.chunk import chunk_text
from app.ingestion.clean import clean_text


def test_clean_rejoins_hyphenated_linebreaks():
    assert "reimbursement" in clean_text("reimburse-\nment")


def test_clean_strips_page_furniture():
    out = clean_text("Real content here.\nPage 3 of 12\nMore content.")
    assert "Page 3 of 12" not in out
    assert "Real content here." in out


def test_chunk_attaches_metadata():
    chunks = chunk_text("Alpha para.\n\nBeta para.", {"source": "x.pdf", "page": 2})
    assert chunks
    assert all(c["metadata"]["source"] == "x.pdf" for c in chunks)
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_chunk_keeps_tables_atomic():
    text = "Intro.\n\n[TABLE]\na | b\nc | d\n[/TABLE]\n\nOutro."
    chunks = chunk_text(text, {"source": "t.pdf", "page": 1}, chunk_size=2000)
    joined = " ".join(c["text"] for c in chunks)
    assert "[TABLE]" in joined and "[/TABLE]" in joined


def test_empty_text_produces_no_chunks():
    assert chunk_text("   ", {"source": "e.pdf"}) == []
