from suicide_risk_detection.spans import evidence_candidates, sentence_spans


def test_sentence_spans_are_verbatim_and_offset_aligned():
    text = "First sentence.  Second sentence!\nThird."
    spans = sentence_spans(text)
    assert [span.text for span in spans] == ["First sentence.", "Second sentence!", "Third."]
    assert all(text[span.start : span.end] == span.text for span in spans)


def test_evidence_candidates_are_unique_and_verbatim():
    text = "Alpha beta gamma, delta epsilon; zeta eta theta."
    spans = evidence_candidates(text, window_sizes=[3, 4], max_tokens=10)
    offsets = {(span.start, span.end) for span in spans}
    assert len(offsets) == len(spans)
    assert all(text[span.start : span.end] == span.text for span in spans)
    assert all(";" not in span.text for span in spans)
