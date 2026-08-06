from types import SimpleNamespace

import rag.ingestion.pipeline as pipeline


def make_page(text, section, doc_id="doc"):
    return SimpleNamespace(text=text, metadata={"section": section, "doc_id": doc_id})


def test_group_pages_by_section():
    pages = [make_page("a", "s1"), make_page("b", "s2"), make_page("c", "s1")]
    grouped = pipeline.group_pages_by_section(pages)
    assert set(grouped.keys()) == {"s1", "s2"}
    assert len(grouped["s1"]) == 2


def test_chunk_single_document_uses_chunk_text_and_assigns_metadata(monkeypatch):
    pages = [make_page("p1", "sec"), make_page("p2", "sec")]

    # monkeypatch read_pdf_file to return our fake pages
    monkeypatch.setattr(pipeline, "read_pdf_file", lambda pdf_file, fitz_module=None, document_factory=None: pages)

    called = {}

    def fake_chunk_text(pages_arg, settings_arg=None):
        # record that we received the pages and settings
        called["pages"] = list(pages_arg)
        called["settings"] = settings_arg
        return ["node1", "node2"]

    opts = pipeline.IngestOptions(chunk_text_fn=fake_chunk_text)
    chunks = pipeline.chunk_single_document("dummy.pdf", options=opts, chunk_id_offset=10)

    assert len(chunks) == 2
    assert chunks[0].metadata["chunk_id"] == 10
    assert chunks[1].metadata["chunk_id"] == 11
    assert chunks[0].metadata["section"] == "sec"
    # ensure fake_chunk_text was called with the page list
    assert called.get("pages") is not None


def test_chunk_multiple_documents_concatenates_results(monkeypatch):
    def fake_chunk_single(pdf_file, options=None, chunk_id_offset=0):
        # produce a variable number of chunks depending on filename
        if "a" in pdf_file:
            return [pipeline.Chunk(text="x", metadata={"chunk_id": chunk_id_offset}) for _ in range(2)]
        return [pipeline.Chunk(text="y", metadata={"chunk_id": chunk_id_offset + i}) for i in range(3)]

    monkeypatch.setattr(pipeline, "chunk_single_document", fake_chunk_single)

    files = ["a.pdf", "b.pdf"]
    all_chunks = pipeline.chunk_multiple_documents(files)
    assert len(all_chunks) == 5
