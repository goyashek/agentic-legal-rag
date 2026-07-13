"""Tests for the RAGAS + MCQ harness plumbing (src/eval/ragas_eval.py, mcq_eval.py).

All keyless: agent runs, retrieval, ragas, and the gated dataset are injected as fakes,
so this pins the pure logic — reference building, sample extraction, RPM pacing, metric
aggregation (incl. NaN-skip), MCQ index clamping, and the honest bridge breakdown —
without a key, an index, or ragas installed. The live scoring itself runs out-of-band.
"""

from __future__ import annotations

from src.eval import mcq_eval, ragas_eval


# --- fakes -------------------------------------------------------------------
class _FakeChunk:
    def __init__(self, act: str, section_id: str, text: str = "", heading: str = "") -> None:
        self.act = act
        self.section_id = section_id
        self.text = text
        self.heading = heading


class _FakeRetrieved:
    def __init__(self, act: str, section_id: str, text: str = "", heading: str = "") -> None:
        self.chunk = _FakeChunk(act, section_id, text, heading)


class _FakeAnswer:
    def __init__(self, answer: str) -> None:
        self.answer = answer


# =============================== RAGAS =======================================
class TestBuildReference:
    def test_joins_gold_section_texts(self) -> None:
        tbs = {"BNS::103": ["murder punishment text"], "BNS::101": ["murder def"]}
        ref = ragas_eval.build_reference(["BNS::101", "BNS::103"], tbs)
        assert "murder def" in ref and "murder punishment text" in ref

    def test_missing_section_is_skipped_not_error(self) -> None:
        assert ragas_eval.build_reference(["BNS::999"], {"BNS::103": ["x"]}) == ""


class TestExtract:
    def test_prefers_graded_chunks_and_answer(self) -> None:
        state = {
            "answer": _FakeAnswer("cited advice"),
            "relevant_chunks": [_FakeRetrieved("BNS", "103", "murder text")],
            "retrieved": [_FakeRetrieved("BNS", "999", "noise")],
        }
        resp, ctx = ragas_eval._extract(state)
        assert resp == "cited advice"
        assert ctx == ["murder text"]  # graded, not raw retrieved

    def test_falls_back_to_retrieved_and_fast_path(self) -> None:
        state = {
            "fast_path_answer": _FakeAnswer("fast"),
            "retrieved": [_FakeRetrieved("BNS", "103", "t")],
        }
        resp, ctx = ragas_eval._extract(state)
        assert resp == "fast" and ctx == ["t"]

    def test_empty_state_is_safe(self) -> None:
        assert ragas_eval._extract({}) == ("", [])


class TestCollectSamples:
    def test_paces_between_but_not_before_first(self) -> None:
        sleeps: list[float] = []
        scenarios = [
            {"query": "q1", "relevant_sections": ["BNS::103"], "difficulty": "easy"},
            {"query": "q2", "relevant_sections": ["BNS::303"], "difficulty": "hard"},
        ]
        corpus = [_FakeChunk("BNS", "103", "murder"), _FakeChunk("BNS", "303", "theft")]

        def fake_answer(q: str):
            return {"answer": _FakeAnswer(f"ans {q}"), "relevant_chunks": []}

        rows = ragas_eval.collect_samples(
            scenarios, answer_fn=fake_answer, corpus=corpus,
            pace_seconds=4.0, sleep=sleeps.append,
        )
        assert len(rows) == 2
        assert sleeps == [4.0]  # paced once (between), not before the first
        assert rows[0]["reference"] == "murder" and rows[1]["difficulty"] == "hard"

    def test_no_pace_when_zero(self) -> None:
        sleeps: list[float] = []
        scenarios = [{"query": "q", "relevant_sections": [], "difficulty": "easy"}] * 3
        ragas_eval.collect_samples(
            scenarios, answer_fn=lambda q: {"answer": _FakeAnswer("a")},
            corpus=[], pace_seconds=0, sleep=sleeps.append,
        )
        assert sleeps == []


class TestAggregate:
    def test_means_and_per_difficulty(self) -> None:
        rows = [
            {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0,
             "context_recall": 1.0, "difficulty": "easy"},
            {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0,
             "context_recall": 0.0, "difficulty": "hard"},
        ]
        s = ragas_eval.aggregate(rows)
        assert s.faithfulness == 0.5 and s.n_scenarios == 2
        assert s.per_difficulty["easy"]["faithfulness"] == 1.0
        assert s.per_difficulty["hard"]["faithfulness"] == 0.0

    def test_nan_is_skipped_not_counted_as_zero(self) -> None:
        nan = float("nan")
        rows = [
            {"faithfulness": 1.0, "answer_relevancy": 1.0, "context_precision": 1.0,
             "context_recall": 1.0, "difficulty": "easy"},
            {"faithfulness": nan, "answer_relevancy": nan, "context_precision": nan,
             "context_recall": nan, "difficulty": "easy"},
        ]
        # NaN dropped -> mean over the one real value = 1.0, not 0.5
        assert ragas_eval.aggregate(rows).faithfulness == 1.0

    def test_all_nan_is_zero_not_crash(self) -> None:
        nan = float("nan")
        rows = [{m: nan for m in ragas_eval.METRIC_NAMES} | {"difficulty": "easy"}]
        assert ragas_eval.aggregate(rows).faithfulness == 0.0


class TestLegacyEmbeddingAdapter:
    def test_exposes_ragas_legacy_embedding_methods(self) -> None:
        class ModernEmbeddings:
            def embed_text(self, text: str) -> list[float]:
                return [float(len(text))]

            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return [[float(len(text))] for text in texts]

        embeddings = ragas_eval._LegacyEmbeddingAdapter(ModernEmbeddings())

        assert embeddings.embed_query("law") == [3.0]
        assert embeddings.embed_documents(["BNS", "BNSS"]) == [[3.0], [4.0]]


# ============================ MCQ (BhashaBench) ==============================
class TestScore:
    def test_all_correct(self) -> None:
        assert mcq_eval.score([0, 1, 2], [0, 1, 2]) == 1.0

    def test_half(self) -> None:
        assert mcq_eval.score([0, 9], [0, 1]) == 0.5

    def test_empty_is_zero(self) -> None:
        assert mcq_eval.score([], []) == 0.0


class TestExtractIpcRefs:
    def test_section_then_ipc(self) -> None:
        assert mcq_eval._extract_ipc_refs("under Section 302 of the IPC") == ["302"]

    def test_ipc_then_section_and_letter(self) -> None:
        # reverse order + trailing letter + whitespace squeezed
        assert mcq_eval._extract_ipc_refs("u/s 124A IPC was invoked") == ["124A"]

    def test_hyphenated_suffix_survives(self) -> None:
        # BhashaBench writes "498-A" / "124-A"; the suffix is significant (498A != 498),
        # so it must be kept, not stripped. This is the real-data bug the loader caught.
        assert mcq_eval._extract_ipc_refs("Section 498-A of the Indian Penal Code") == ["498A"]
        assert mcq_eval._extract_ipc_refs("Section 124-A of the Indian Penal Code") == ["124A"]

    def test_ignores_non_ipc_section(self) -> None:
        # CrPC section must NOT be picked up — only IPC is bridged
        assert mcq_eval._extract_ipc_refs("Section 200 of Cr.P.C.") == []

    def test_no_citation(self) -> None:
        assert mcq_eval._extract_ipc_refs("Modus Operandi stands for") == []


class TestAnswerMcq:
    def test_clamps_out_of_range_index(self) -> None:
        class _FakeClient:
            def create(self, **_):
                return mcq_eval._MCQChoice(answer_idx=99)  # model hallucinates OOB

        idx = mcq_eval.answer_mcq(
            "q", ["a", "b", "c"],
            retriever=_FakeRetriever(), client=_FakeClient(),
        )
        assert idx == 2  # clamped to last valid

    def test_negative_index_clamped_to_zero(self) -> None:
        class _FakeClient:
            def create(self, **_):
                return mcq_eval._MCQChoice(answer_idx=-5)

        assert mcq_eval.answer_mcq(
            "q", ["a", "b"], retriever=_FakeRetriever(), client=_FakeClient()
        ) == 0


class _FakeRetriever:
    def retrieve(self, query: str, *, top_k: int = 20):
        return [_FakeRetrieved("BNS", "103", "murder text", "Punishment for murder")]


class TestComputeResult:
    def _slice(self):
        return [
            {"question": "q1", "options": ["a", "b"], "answer_idx": 0, "ipc_refs": ["302"]},
            {"question": "q2", "options": ["a", "b"], "answer_idx": 1, "ipc_refs": ["999"]},
            {"question": "q3", "options": ["a", "b"], "answer_idx": 0, "ipc_refs": []},
        ]

    def test_overall_and_bridge_subset(self) -> None:
        # 302 maps -> bridge-dependent; 999 unmapped; [] none. Only q1 is on the bridge.
        mapping = {"302": "103"}
        preds = [0, 0, 0]  # q1 right, q2 wrong, q3 right -> 2/3 overall
        res = mcq_eval.compute_result(self._slice(), preds, mapping)
        assert res.total == 3 and res.correct == 2
        assert abs(res.accuracy - 2 / 3) < 1e-9
        assert res.bridge_resolved == 1          # only q1
        assert res.bridge_accuracy == 1.0        # q1 predicted correctly
        assert res.baseline_accuracy is None

    def test_baseline_included_when_given(self) -> None:
        res = mcq_eval.compute_result(
            self._slice(), [0, 1, 0], {"302": "103"}, baseline_predictions=[1, 1, 1]
        )
        assert res.accuracy == 1.0
        assert abs(res.baseline_accuracy - 1 / 3) < 1e-9  # baseline: only q2 right
        # bridge subset is q1 (ans 0); baseline predicted 1 -> wrong -> 0.0
        assert res.baseline_bridge_accuracy == 0.0

    def test_baseline_bridge_none_without_baseline(self) -> None:
        res = mcq_eval.compute_result(self._slice(), [0, 1, 0], {"302": "103"})
        assert res.baseline_bridge_accuracy is None


class TestRunAibeOrchestration:
    def test_paces_and_delegates(self) -> None:
        sleeps: list[float] = []
        slice_ = [
            {"question": "q1", "options": ["a", "b"], "answer_idx": 0, "ipc_refs": ["302"]},
            {"question": "q2", "options": ["a", "b"], "answer_idx": 1, "ipc_refs": []},
        ]
        res = mcq_eval.run_mcq_eval(
            slice_=slice_,
            mcq_fn=lambda q, opts: 0,          # always picks index 0
            ipc_bns_mapping={"302": "103"},
            pace_seconds=4.0,
            sleep=sleeps.append,
            with_baseline=False,
        )
        assert sleeps == [4.0]                  # paced once between the two
        assert res.correct == 1                 # q1 (ans 0) right, q2 (ans 1) wrong
        assert res.bridge_resolved == 1
