from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from chatbot.models import ChatbotConfig
from chatbot.services.gemini_manager import get_gemini_manager
from rag_benchmark.models import BenchmarkCase
from rag_benchmark.services.constants import (
    DEFAULT_ANSWER_RELEVANCY_STRICTNESS,
    DEFAULT_RAGAS_BATCH_SIZE,
    DEFAULT_RAGAS_CONTEXT_CHAR_LIMIT,
    DEFAULT_RAGAS_CONTEXT_TOP_K,
    DEFAULT_RAGAS_MAX_TOKENS,
    DEFAULT_RAGAS_MAX_WORKERS,
    DEFAULT_RAGAS_METRICS,
    DEFAULT_RAGAS_RETRY_MAX_TOKENS,
    DEFAULT_RAGAS_TIMEOUT_SECONDS,
)
from vector_store.services.constants import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_API_KEY_ENV_NAME,
    OPENROUTER_BASE_URL_ENV_NAME,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ContextCandidate:
    text: str
    source: str
    section: str
    title: str
    score: float


class RagasJudgeEvaluator:
    """Ragas-based LLM judge used as the primary benchmark scorer."""

    def __init__(self, *, metrics: list[str] | None = None) -> None:
        self.metric_names = metrics or list(DEFAULT_RAGAS_METRICS)

    def evaluate_case(
        self,
        *,
        case: BenchmarkCase,
        runtime_output: dict[str, Any],
    ) -> dict[str, Any]:
        batch_results = self.evaluate_batch(
            cases=[case],
            runtime_outputs=[runtime_output],
        )
        if batch_results:
            return batch_results[0]
        return {
            "enabled": True,
            "available": False,
            "error": "ragas_evaluation_failed: empty_batch_result",
        }

    def evaluate_batch(
        self,
        *,
        cases: list[BenchmarkCase],
        runtime_outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not cases or not runtime_outputs:
            return []
        if len(cases) != len(runtime_outputs):
            raise ValueError("cases and runtime_outputs must have the same length")

        dependencies, import_error = self._import_ragas_dependencies()
        if import_error is not None:
            return [
                {
                    "enabled": True,
                    "available": False,
                    "error": f"ragas_unavailable: {import_error}",
                }
                for _ in cases
            ]
        Dataset = dependencies["Dataset"]
        RunConfig = dependencies["RunConfig"]
        aevaluate = dependencies["aevaluate"]

        answer_relevancy_strictness = self._get_int_config(
            key="RAGAS_ANSWER_RELEVANCY_STRICTNESS",
            default=DEFAULT_ANSWER_RELEVANCY_STRICTNESS,
            minimum=1,
            maximum=5,
        )
        metric_registry = self._build_metric_registry(
            faithfulness=dependencies["faithfulness"],
            answer_relevancy=dependencies["answer_relevancy"],
            context_precision=dependencies["context_precision"],
            context_recall=dependencies["context_recall"],
            answer_relevancy_strictness=answer_relevancy_strictness,
        )
        selected_metrics = [
            metric_registry[name]
            for name in self.metric_names
            if name in metric_registry
        ]
        selected_metric_names = [metric.name for metric in selected_metrics]
        if not selected_metrics:
            return self._no_metric_results(cases)

        ragas_context_top_k = self._get_int_config(
            key="RAGAS_CONTEXT_TOP_K",
            default=DEFAULT_RAGAS_CONTEXT_TOP_K,
            minimum=1,
            maximum=32,
        )
        ragas_context_char_limit = self._get_int_config(
            key="RAGAS_CONTEXT_CHAR_LIMIT",
            default=DEFAULT_RAGAS_CONTEXT_CHAR_LIMIT,
            minimum=120,
            maximum=4000,
        )
        dataset_payloads = self._build_ragas_dataset_payloads(
            cases=cases,
            runtime_outputs=runtime_outputs,
            ragas_context_top_k=ragas_context_top_k,
            ragas_context_char_limit=ragas_context_char_limit,
        )

        runtime_config = self._ragas_runtime_config(case_count=len(cases))

        try:
            llm, embeddings = self._create_ragas_models(
                max_tokens=runtime_config["max_tokens"]
            )
        except Exception as exc:
            return self._runtime_unavailable_results(
                cases=cases,
                selected_metric_names=selected_metric_names,
                error=exc,
            )

        dataset = Dataset.from_dict(self._ragas_dataset_dict(dataset_payloads))
        evaluation = self._evaluate_ragas_dataset_with_retry(
            aevaluate=aevaluate,
            run_config_cls=RunConfig,
            dataset=dataset,
            metrics=selected_metrics,
            llm=llm,
            embeddings=embeddings,
            runtime_config=runtime_config,
        )
        if evaluation["error"] is not None:
            return self._evaluation_error_results(
                cases=cases,
                selected_metric_names=selected_metric_names,
                answer_relevancy_strictness=answer_relevancy_strictness,
                max_tokens=evaluation["max_tokens"],
                retry_used=evaluation["retry_used"],
                error=evaluation["error"],
            )

        max_tokens = evaluation["max_tokens"]
        retry_used = evaluation["retry_used"]
        result = evaluation["result"]
        score_payload = result.to_pandas().to_dict(orient="records")

        if not score_payload:
            return self._evaluation_error_results(
                cases=cases,
                selected_metric_names=selected_metric_names,
                answer_relevancy_strictness=answer_relevancy_strictness,
                max_tokens=max_tokens,
                retry_used=retry_used,
                error="empty_result",
            )

        if len(score_payload) < len(cases):
            score_payload.extend([{} for _ in range(len(cases) - len(score_payload))])

        parsed_scores = [
            self._extract_numeric_scores(row) for row in score_payload[: len(cases)]
        ]
        fallback_retry_used = self._fill_missing_ragas_scores(
            dataset_cls=Dataset,
            aevaluate=aevaluate,
            run_config_cls=RunConfig,
            selected_metrics=selected_metrics,
            selected_metric_names=selected_metric_names,
            llm=llm,
            embeddings=embeddings,
            runtime_config={**runtime_config, "max_tokens": max_tokens},
            dataset_payloads=dataset_payloads,
            parsed_scores=parsed_scores,
        )
        return self._ragas_success_results(
            parsed_scores=parsed_scores,
            selected_metric_names=selected_metric_names,
            max_tokens=max_tokens,
            answer_relevancy_strictness=answer_relevancy_strictness,
            retry_used=retry_used or fallback_retry_used,
        )

    def _import_ragas_dependencies(self) -> tuple[dict[str, Any], Exception | None]:
        try:
            from datasets import Dataset
            from ragas import aevaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            from ragas.run_config import RunConfig
        except Exception as exc:
            return {}, exc

        return {
            "Dataset": Dataset,
            "RunConfig": RunConfig,
            "aevaluate": aevaluate,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }, None

    def _no_metric_results(self, cases: list[BenchmarkCase]) -> list[dict[str, Any]]:
        return [
            {
                "enabled": True,
                "available": True,
                "selected_metrics": [],
                "scores": {},
                "warning": "No valid ragas metrics configured",
            }
            for _ in cases
        ]

    def _build_ragas_dataset_payloads(
        self,
        *,
        cases: list[BenchmarkCase],
        runtime_outputs: list[dict[str, Any]],
        ragas_context_top_k: int,
        ragas_context_char_limit: int,
    ) -> dict[str, list[Any]]:
        question_texts: list[str] = []
        answer_texts: list[str] = []
        reference_texts: list[str] = []
        contexts_batch: list[list[str]] = []

        for case, runtime_output in zip(cases, runtime_outputs):
            generation_output = runtime_output.get("generation_output", {}) or {}
            question_text = str(case.question or "").strip()
            contexts = self._contexts_for_ragas_case(
                runtime_output=runtime_output,
                question_text=question_text,
                default_top_k=ragas_context_top_k,
                char_limit=ragas_context_char_limit,
            )
            question_texts.append(question_text)
            answer_texts.append(str(generation_output.get("final_answer", "")).strip())
            reference_texts.append(str(case.reference_answer or "").strip())
            contexts_batch.append(contexts)

        return {
            "question_texts": question_texts,
            "answer_texts": answer_texts,
            "reference_texts": reference_texts,
            "contexts_batch": contexts_batch,
        }

    def _contexts_for_ragas_case(
        self,
        *,
        runtime_output: dict[str, Any],
        question_text: str,
        default_top_k: int,
        char_limit: int,
    ) -> list[str]:
        generation_output = runtime_output.get("generation_output", {}) or {}
        retrieval_output = runtime_output.get("retrieval_output", {}) or {}
        try:
            contexts = self._build_judge_contexts(
                retrieval_output=retrieval_output,
                generation_output=generation_output,
                question_text=question_text,
                default_top_k=default_top_k,
                char_limit=char_limit,
            )
        except Exception as exc:
            logger.warning(
                "Ragas context collection fallback to legacy ordering: %s", exc
            )
            contexts = []

        if contexts:
            return contexts
        raw_contexts = self._collect_runtime_contexts(
            runtime_output,
            question_text=question_text,
        )
        effective_top_k = self._resolve_dynamic_context_top_k(
            question_text=question_text,
            default_top_k=default_top_k,
        )
        return self._prepare_ragas_contexts(
            raw_contexts=raw_contexts,
            top_k=effective_top_k,
            char_limit=char_limit,
        )

    def _ragas_runtime_config(self, *, case_count: int) -> dict[str, int]:
        max_tokens = self._get_int_config(
            key="RAGAS_LLM_MAX_TOKENS",
            default=DEFAULT_RAGAS_MAX_TOKENS,
            minimum=512,
            maximum=32768,
        )
        return {
            "timeout_seconds": self._get_int_config(
                key="RAGAS_TIMEOUT_SECONDS",
                default=DEFAULT_RAGAS_TIMEOUT_SECONDS,
                minimum=30,
                maximum=900,
            ),
            "max_tokens": max_tokens,
            "retry_max_tokens": self._get_int_config(
                key="RAGAS_LLM_RETRY_MAX_TOKENS",
                default=max(DEFAULT_RAGAS_RETRY_MAX_TOKENS, max_tokens),
                minimum=max_tokens,
                maximum=65536,
            ),
            "max_workers": self._get_int_config(
                key="RAGAS_MAX_WORKERS",
                default=DEFAULT_RAGAS_MAX_WORKERS,
                minimum=1,
                maximum=16,
            ),
            "batch_size": self._get_int_config(
                key="RAGAS_BATCH_SIZE",
                default=min(DEFAULT_RAGAS_BATCH_SIZE, case_count),
                minimum=1,
                maximum=128,
            ),
        }

    def _runtime_unavailable_results(
        self,
        *,
        cases: list[BenchmarkCase],
        selected_metric_names: list[str],
        error: Exception,
    ) -> list[dict[str, Any]]:
        return [
            {
                "enabled": True,
                "available": False,
                "selected_metrics": selected_metric_names,
                "error": f"ragas_runtime_unavailable: {error}",
            }
            for _ in cases
        ]

    def _ragas_dataset_dict(
        self,
        dataset_payloads: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:
        return {
            "user_input": dataset_payloads["question_texts"],
            "response": dataset_payloads["answer_texts"],
            "retrieved_contexts": dataset_payloads["contexts_batch"],
            "reference": dataset_payloads["reference_texts"],
        }

    def _evaluate_ragas_dataset_with_retry(
        self,
        *,
        aevaluate: Any,
        run_config_cls: Any,
        dataset: Any,
        metrics: list[Any],
        llm: Any,
        embeddings: Any,
        runtime_config: dict[str, int],
    ) -> dict[str, Any]:
        try:
            result = self._run_configured_ragas_evaluate(
                aevaluate=aevaluate,
                run_config_cls=run_config_cls,
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                runtime_config=runtime_config,
            )
        except Exception as exc:
            return self._retry_ragas_after_error(
                exc=exc,
                aevaluate=aevaluate,
                run_config_cls=run_config_cls,
                dataset=dataset,
                metrics=metrics,
                runtime_config=runtime_config,
            )
        return {
            "result": result,
            "error": None,
            "retry_used": False,
            "max_tokens": runtime_config["max_tokens"],
        }

    def _retry_ragas_after_error(
        self,
        *,
        exc: Exception,
        aevaluate: Any,
        run_config_cls: Any,
        dataset: Any,
        metrics: list[Any],
        runtime_config: dict[str, int],
    ) -> dict[str, Any]:
        max_tokens = runtime_config["max_tokens"]
        retry_max_tokens = runtime_config["retry_max_tokens"]
        if (
            not self._is_incomplete_generation_error(exc)
            or retry_max_tokens <= max_tokens
        ):
            return {
                "result": None,
                "error": exc,
                "retry_used": False,
                "max_tokens": max_tokens,
            }

        try:
            retry_llm, retry_embeddings = self._create_ragas_models(
                max_tokens=retry_max_tokens
            )
            retry_config = {**runtime_config, "max_tokens": retry_max_tokens}
            result = self._run_configured_ragas_evaluate(
                aevaluate=aevaluate,
                run_config_cls=run_config_cls,
                dataset=dataset,
                metrics=metrics,
                llm=retry_llm,
                embeddings=retry_embeddings,
                runtime_config=retry_config,
            )
        except Exception as retry_exc:
            return {
                "result": None,
                "error": retry_exc,
                "retry_used": True,
                "max_tokens": retry_max_tokens,
            }
        return {
            "result": result,
            "error": None,
            "retry_used": True,
            "max_tokens": retry_max_tokens,
        }

    def _run_configured_ragas_evaluate(
        self,
        *,
        aevaluate: Any,
        run_config_cls: Any,
        dataset: Any,
        metrics: list[Any],
        llm: Any,
        embeddings: Any,
        runtime_config: dict[str, int],
    ) -> Any:
        return self._run_ragas_evaluate(
            aevaluate=aevaluate,
            run_config_cls=run_config_cls,
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            timeout_seconds=runtime_config["timeout_seconds"],
            max_workers=runtime_config["max_workers"],
            batch_size=runtime_config["batch_size"],
        )

    def _evaluation_error_results(
        self,
        *,
        cases: list[BenchmarkCase],
        selected_metric_names: list[str],
        answer_relevancy_strictness: int,
        max_tokens: int,
        retry_used: bool,
        error: Exception | str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "enabled": True,
                "available": True,
                "selected_metrics": selected_metric_names,
                "llm_max_tokens": max_tokens,
                "answer_relevancy_strictness": answer_relevancy_strictness,
                "retry_used": retry_used,
                "error": f"ragas_evaluation_failed: {error}",
            }
            for _ in cases
        ]

    def _fill_missing_ragas_scores(
        self,
        *,
        dataset_cls: Any,
        aevaluate: Any,
        run_config_cls: Any,
        selected_metrics: list[Any],
        selected_metric_names: list[str],
        llm: Any,
        embeddings: Any,
        runtime_config: dict[str, int],
        dataset_payloads: dict[str, list[Any]],
        parsed_scores: list[dict[str, float]],
    ) -> bool:
        fallback_retry_used = False
        fallback_single_case_count = 0
        for idx, scores in enumerate(parsed_scores):
            missing_metric_names = self._missing_metric_names(
                selected_metric_names=selected_metric_names,
                scores=scores,
            )
            if not missing_metric_names:
                continue

            fallback_single_case_count += 1
            fallback_retry_used = (
                self._fill_single_case_scores(
                    idx=idx,
                    missing_metric_names=missing_metric_names,
                    scores=scores,
                    dataset_cls=dataset_cls,
                    aevaluate=aevaluate,
                    run_config_cls=run_config_cls,
                    selected_metrics=selected_metrics,
                    llm=llm,
                    embeddings=embeddings,
                    runtime_config=runtime_config,
                    dataset_payloads=dataset_payloads,
                )
                or fallback_retry_used
            )

        self._log_single_case_fallback_count(
            fallback_single_case_count=fallback_single_case_count,
            total_cases=len(parsed_scores),
        )
        return fallback_retry_used

    def _missing_metric_names(
        self,
        *,
        selected_metric_names: list[str],
        scores: dict[str, float],
    ) -> list[str]:
        return [
            metric_name
            for metric_name in selected_metric_names
            if metric_name not in scores
        ]

    def _fill_single_case_scores(
        self,
        *,
        idx: int,
        missing_metric_names: list[str],
        scores: dict[str, float],
        dataset_cls: Any,
        aevaluate: Any,
        run_config_cls: Any,
        selected_metrics: list[Any],
        llm: Any,
        embeddings: Any,
        runtime_config: dict[str, int],
        dataset_payloads: dict[str, list[Any]],
    ) -> bool:
        try:
            fallback_scores, used_retry = self._evaluate_single_case_fallback(
                dataset_cls=dataset_cls,
                aevaluate=aevaluate,
                run_config_cls=run_config_cls,
                metrics=selected_metrics,
                llm=llm,
                embeddings=embeddings,
                timeout_seconds=runtime_config["timeout_seconds"],
                max_workers=runtime_config["max_workers"],
                max_tokens=runtime_config["max_tokens"],
                retry_max_tokens=runtime_config["retry_max_tokens"],
                user_input=dataset_payloads["question_texts"][idx],
                response=dataset_payloads["answer_texts"][idx],
                retrieved_contexts=dataset_payloads["contexts_batch"][idx],
                reference=dataset_payloads["reference_texts"][idx],
            )
        except Exception as exc:
            logger.warning(
                "Ragas single-case fallback failed for case index %s "
                "(missing=%s): %s",
                idx,
                ",".join(missing_metric_names),
                exc,
            )
            return False

        for metric_name in missing_metric_names:
            metric_value = fallback_scores.get(metric_name)
            if metric_value is not None:
                scores[metric_name] = metric_value
        return used_retry

    def _log_single_case_fallback_count(
        self,
        *,
        fallback_single_case_count: int,
        total_cases: int,
    ) -> None:
        if fallback_single_case_count:
            logger.warning(
                "Ragas single-case fallback applied for %s/%s cases with missing metrics.",
                fallback_single_case_count,
                total_cases,
            )

    def _ragas_success_results(
        self,
        *,
        parsed_scores: list[dict[str, float]],
        selected_metric_names: list[str],
        max_tokens: int,
        answer_relevancy_strictness: int,
        retry_used: bool,
    ) -> list[dict[str, Any]]:
        return [
            {
                "enabled": True,
                "available": True,
                "selected_metrics": selected_metric_names,
                "llm_max_tokens": max_tokens,
                "answer_relevancy_strictness": answer_relevancy_strictness,
                "retry_used": retry_used,
                "scores": scores,
            }
            for scores in parsed_scores
        ]

    def _run_ragas_evaluate(
        self,
        *,
        aevaluate: Any,
        run_config_cls: Any,
        dataset: Any,
        metrics: list[Any],
        llm: Any,
        embeddings: Any,
        timeout_seconds: int,
        max_workers: int,
        batch_size: int,
    ) -> Any:
        return self._run_in_isolated_loop(
            aevaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config_cls(
                    timeout=timeout_seconds,
                    max_retries=3,
                    max_wait=30,
                    max_workers=max_workers,
                ),
                raise_exceptions=False,
                show_progress=False,
                batch_size=batch_size,
            )
        )

    def _evaluate_single_case_fallback(
        self,
        *,
        dataset_cls: Any,
        aevaluate: Any,
        run_config_cls: Any,
        metrics: list[Any],
        llm: Any,
        embeddings: Any,
        timeout_seconds: int,
        max_workers: int,
        max_tokens: int,
        retry_max_tokens: int,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
    ) -> tuple[dict[str, float], bool]:
        dataset = dataset_cls.from_dict(
            {
                "user_input": [user_input],
                "response": [response],
                "retrieved_contexts": [retrieved_contexts],
                "reference": [reference],
            }
        )
        try:
            result = self._run_ragas_evaluate(
                aevaluate=aevaluate,
                run_config_cls=run_config_cls,
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                timeout_seconds=timeout_seconds,
                max_workers=max_workers,
                batch_size=1,
            )
            return self._extract_numeric_scores_from_result(result), False
        except Exception as exc:
            if (
                self._is_incomplete_generation_error(exc)
                and retry_max_tokens > max_tokens
            ):
                retry_llm, retry_embeddings = self._create_ragas_models(
                    max_tokens=retry_max_tokens
                )
                retry_result = self._run_ragas_evaluate(
                    aevaluate=aevaluate,
                    run_config_cls=run_config_cls,
                    dataset=dataset,
                    metrics=metrics,
                    llm=retry_llm,
                    embeddings=retry_embeddings,
                    timeout_seconds=timeout_seconds,
                    max_workers=max_workers,
                    batch_size=1,
                )
                return self._extract_numeric_scores_from_result(retry_result), True
            raise

    def _extract_numeric_scores_from_result(self, result: Any) -> dict[str, float]:
        rows = result.to_pandas().to_dict(orient="records")
        if not rows:
            return {}
        return self._extract_numeric_scores(rows[0])

    def _extract_numeric_scores(self, row: dict[str, Any]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for key, value in (row or {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scores[str(key)] = round(float(value), 6)
        return scores

    def _run_in_isolated_loop(self, coro: Any) -> Any:
        loop = asyncio.new_event_loop()

        def _loop_exception_handler(
            loop_obj: asyncio.AbstractEventLoop, context: dict[str, Any]
        ) -> None:
            exc = context.get("exception")
            message = str(context.get("message", "")).strip().lower()
            if self._is_ignorable_asyncio_shutdown_error(exc=exc, message=message):
                logger.debug(
                    "Suppressed asyncio shutdown noise during ragas evaluation: %s",
                    exc or message,
                )
                return
            loop_obj.default_exception_handler(context)

        loop.set_exception_handler(_loop_exception_handler)
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                pending = list(asyncio.all_tasks(loop))
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
                if hasattr(loop, "shutdown_default_executor"):
                    loop.run_until_complete(loop.shutdown_default_executor())
            except Exception as cleanup_exc:
                logger.debug(
                    "Async loop cleanup warning in ragas evaluator: %s", cleanup_exc
                )
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    def _build_metric_registry(
        self,
        *,
        faithfulness: Any,
        answer_relevancy: Any,
        context_precision: Any,
        context_recall: Any,
        answer_relevancy_strictness: int,
    ) -> dict[str, Any]:
        metrics = {
            "faithfulness": copy.deepcopy(faithfulness),
            "answer_relevancy": copy.deepcopy(answer_relevancy),
            "context_precision": copy.deepcopy(context_precision),
            "context_recall": copy.deepcopy(context_recall),
        }
        answer_relevancy_metric = metrics.get("answer_relevancy")
        if answer_relevancy_metric is not None:
            try:
                answer_relevancy_metric.strictness = answer_relevancy_strictness
            except Exception:
                pass
        return metrics

    def _create_ragas_models(self, *, max_tokens: int):
        manager = get_gemini_manager()
        provider = str(ChatbotConfig.get_config("LLM_PROVIDER", "gemini")).strip()
        if provider.lower() == "openrouter":
            return (
                self._create_openrouter_llm(max_tokens=max_tokens),
                self._create_openrouter_embeddings(),
            )
        llm_model = str(ChatbotConfig.get_config("LLM_MODEL", "gemini-2.5-flash"))
        llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=manager.get_current_key(),
            temperature=0.0,
            max_output_tokens=max_tokens,
            streaming=False,
        )
        return llm, manager.create_embeddings(normalized=True)

    def _create_openrouter_llm(self, *, max_tokens: int) -> ChatOpenAI:
        api_key = self._resolve_openrouter_api_key()
        base_url = self._resolve_openrouter_base_url()
        llm_model = str(ChatbotConfig.get_config("LLM_MODEL", "openai/gpt-4.1-mini"))
        return ChatOpenAI(
            model=llm_model,
            api_key=SecretStr(api_key),
            base_url=base_url,
            temperature=0.0,
            model_kwargs={"max_tokens": max_tokens},
            streaming=False,
        )

    def _create_openrouter_embeddings(self) -> OpenAIEmbeddings:
        api_key = self._resolve_openrouter_api_key()
        base_url = self._resolve_openrouter_base_url()
        model = str(
            ChatbotConfig.get_config("EMBEDDING_MODEL", DEFAULT_OPENROUTER_MODEL)
        )
        return OpenAIEmbeddings(
            model=model,
            api_key=SecretStr(api_key),
            base_url=base_url,
        )

    def _resolve_openrouter_api_key(self) -> str:
        api_key_raw = ChatbotConfig.get_config(OPENROUTER_API_KEY_ENV_NAME, None)
        api_key = (
            api_key_raw.strip()
            if isinstance(api_key_raw, str) and api_key_raw.strip()
            else os.getenv(OPENROUTER_API_KEY_ENV_NAME)
        )
        if not api_key:
            raise ValueError(
                f"{OPENROUTER_API_KEY_ENV_NAME} is required for LLM_PROVIDER=openrouter"
            )
        return api_key

    def _resolve_openrouter_base_url(self) -> str:
        db_base_url = ChatbotConfig.get_config("OPENROUTER_BASE_URL", None)
        return (
            db_base_url.strip()
            if isinstance(db_base_url, str) and db_base_url.strip()
            else os.getenv(OPENROUTER_BASE_URL_ENV_NAME) or DEFAULT_OPENROUTER_BASE_URL
        )

    def _is_incomplete_generation_error(self, exc: Exception) -> bool:
        terms = (
            "llmdidnotfinishexception",
            "generation was not completed",
            "increase the max_tokens",
            "max tokens",
        )
        cursor: BaseException | None = exc
        seen: set[int] = set()
        while cursor is not None and id(cursor) not in seen:
            seen.add(id(cursor))
            message = str(cursor).lower()
            if any(term in message for term in terms):
                return True
            cursor = cursor.__cause__ or cursor.__context__
        return False

    def _is_ignorable_asyncio_shutdown_error(
        self, *, exc: BaseException | None, message: str
    ) -> bool:
        if isinstance(exc, RuntimeError) and "event loop is closed" in str(exc).lower():
            return True
        if "task exception was never retrieved" in message:
            return True
        if "event loop is closed" in message:
            return True
        return False

    def _get_int_config(
        self,
        *,
        key: str,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        raw = ChatbotConfig.get_config(key, default)
        try:
            value = int(str(raw).strip()) if raw is not None else default
        except (TypeError, ValueError):
            value = default
        if value < minimum:
            return minimum
        if value > maximum:
            return maximum
        return value

    def _prepare_ragas_contexts(
        self,
        *,
        raw_contexts: list[str],
        top_k: int,
        char_limit: int,
    ) -> list[str]:
        contexts: list[str] = []
        seen: set[str] = set()
        for raw in raw_contexts:
            normalized = re.sub(r"\s+", " ", str(raw or "")).strip()
            if not normalized:
                continue
            dedupe_key = self._normalize_context_dedupe_key(normalized)
            if not dedupe_key or self._has_duplicate_context_key(dedupe_key, seen):
                continue
            seen.add(dedupe_key)
            contexts.append(normalized[:char_limit])
            if len(contexts) >= top_k:
                break
        return contexts

    def _normalize_context_dedupe_key(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        if not normalized:
            return ""
        normalized = re.sub(
            r"^(main symptoms include|risk factors include|cause:|prevention guidance:)\s+",
            "",
            normalized,
        )
        normalized = re.sub(r"[\s\.,;:!?]+$", "", normalized)
        return normalized

    def _has_duplicate_context_key(self, key: str, seen: set[str]) -> bool:
        if key in seen:
            return True
        for existing in seen:
            if self._is_near_duplicate_context_key(existing, key):
                return True
        return False

    def _is_near_duplicate_context_key(self, left: str, right: str) -> bool:
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        if len(shorter) < 80:
            return False
        if longer.startswith(shorter):
            return True
        if shorter in longer and len(shorter) / max(1, len(longer)) >= 0.85:
            return True
        return False

    def _rank_context_candidates(
        self,
        candidates: list[_ContextCandidate],
        *,
        target_sections: set[str] | None = None,
    ) -> list[str]:
        if not candidates:
            return []

        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        contexts: list[str] = []
        seen: set[str] = set()
        used_non_summary_sections: set[str] = set()
        section_targets = {
            str(section).strip().lower()
            for section in (target_sections or set())
            if str(section).strip()
        }

        for candidate in ranked:
            normalized = re.sub(r"\s+", " ", candidate.text).strip()
            if not normalized or normalized in seen:
                continue
            candidate_section = str(candidate.section).strip().lower()
            if (
                candidate_section
                and candidate_section in section_targets
                and candidate_section in used_non_summary_sections
                and candidate.source != "summary"
            ):
                continue
            seen.add(normalized)
            contexts.append(normalized)
            if candidate_section and candidate.source != "summary":
                used_non_summary_sections.add(candidate_section)

        return contexts

    def _resolve_dynamic_context_top_k(
        self,
        *,
        question_text: str,
        default_top_k: int,
    ) -> int:
        q_tokens = self._tokenize_context_text(question_text)
        section_count = 0
        if any(token in q_tokens for token in {"risk", "nguy", "cơ"}):
            section_count += 1
        if any(
            token in q_tokens for token in {"symptom", "symptoms", "triệu", "chứng"}
        ):
            section_count += 1
        if any(
            token in q_tokens
            for token in {"cause", "causes", "nguyên", "nhân", "aetiology", "etiology"}
        ):
            section_count += 1
        if any(
            token in q_tokens
            for token in {"prevent", "prevention", "vaccine", "phòng", "ngừa"}
        ):
            section_count += 1

        if section_count <= 1:
            return max(1, min(default_top_k, 3))
        if section_count == 2:
            has_symptom = any(
                token in q_tokens for token in {"symptom", "symptoms", "triệu", "chứng"}
            )
            has_aetiology = any(
                token in q_tokens
                for token in {
                    "cause",
                    "causes",
                    "nguyên",
                    "nhân",
                    "aetiology",
                    "etiology",
                }
            )
            if has_symptom and has_aetiology:
                return max(2, min(default_top_k, 4))
            return max(2, min(default_top_k, 3))
        return max(2, default_top_k)

    def _estimate_section_count_from_targets(self, target_sections: set[str]) -> int:
        return len(
            {
                section
                for section in target_sections
                if section
                in {
                    "risk",
                    "symptom",
                    "aetiologies",
                    "living_and_preventive",
                    "general",
                }
            }
        )

    def _append_fallback_contexts(
        self,
        *,
        contexts: list[str],
        summaries: dict[str, Any],
        context_snapshot: str,
    ) -> None:
        for summary in summaries.values():
            text = str(summary).strip()
            if text:
                contexts.append(text)
        if context_snapshot:
            contexts.append(context_snapshot)

    def _should_include_context_snapshot(
        self,
        *,
        current_contexts: list[str],
        target_sections: set[str],
    ) -> bool:
        section_count = self._estimate_section_count_from_targets(target_sections)
        if section_count <= 1 and len(current_contexts) >= 1:
            return False
        if section_count == 2 and len(current_contexts) >= 3:
            return False
        return True

    def _should_include_summaries(
        self,
        *,
        current_contexts: list[str],
        target_sections: set[str],
    ) -> bool:
        section_count = self._estimate_section_count_from_targets(target_sections)
        if section_count <= 1 and len(current_contexts) >= 2:
            return False
        if section_count == 2 and len(current_contexts) >= 3:
            return False
        return True

    def _has_sufficient_context_candidates(
        self,
        *,
        candidates: list[_ContextCandidate],
        target_sections: set[str],
    ) -> bool:
        if not candidates:
            return False
        if not target_sections:
            return len(candidates) >= 2

        covered = {
            str(candidate.section).strip().lower()
            for candidate in candidates
            if str(candidate.section).strip().lower() in target_sections
        }
        if covered:
            return True
        return len(candidates) >= 2

    def _dedupe_context_list(self, contexts: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for raw in contexts:
            normalized = re.sub(r"\s+", " ", str(raw or "")).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _build_context_from_candidates(
        self,
        *,
        candidates: list[_ContextCandidate],
        target_sections: set[str],
        summaries: dict[str, Any],
        context_snapshot: str,
    ) -> list[str]:
        ranked_contexts = self._rank_context_candidates(
            candidates,
            target_sections=target_sections,
        )
        contexts = list(ranked_contexts)

        if self._should_include_context_snapshot(
            current_contexts=contexts,
            target_sections=target_sections,
        ):
            self._append_fallback_contexts(
                contexts=contexts,
                summaries={},
                context_snapshot=context_snapshot,
            )

        deduped = self._dedupe_context_list(contexts)
        return self._remove_redundant_general_contexts(
            contexts=deduped,
            target_sections=target_sections,
        )

    def _extract_context_entries(
        self,
        *,
        contexts: list[str],
        target_sections: set[str],
    ) -> list[tuple[str, set[str]]]:
        entries: list[tuple[str, set[str]]] = []
        for raw in contexts:
            normalized = re.sub(r"\s+", " ", str(raw or "")).strip()
            if not normalized:
                continue
            sections = self._detect_context_sections(
                text=normalized,
                target_sections=target_sections,
            )
            entries.append((normalized, sections))
        return entries

    def _detect_context_sections(
        self,
        *,
        text: str,
        target_sections: set[str],
    ) -> set[str]:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return set()

        detected = self._detect_keyword_sections(lowered)
        if "covid-19 is an infectious disease caused by" in lowered:
            detected.update(self._detect_general_covid_sections(target_sections))

        if target_sections:
            return detected.intersection(target_sections)
        return detected

    def _detect_keyword_sections(self, lowered: str) -> set[str]:
        section_keywords = {
            "symptom": ("symptom", "triệu chứng"),
            "aetiologies": ("cause", "nguyên nhân", "aetiolog"),
            "risk": ("risk", "yếu tố nguy cơ"),
            "living_and_preventive": ("prevention", "phòng ngừa"),
        }
        return {
            section
            for section, keywords in section_keywords.items()
            if any(keyword in lowered for keyword in keywords)
        }

    def _detect_general_covid_sections(self, target_sections: set[str]) -> set[str]:
        if not target_sections:
            return {"aetiologies", "symptom"}
        return target_sections.intersection({"risk", "aetiologies", "symptom"})

    def _ensure_target_section_coverage(
        self,
        *,
        contexts: list[str],
        target_sections: set[str],
    ) -> list[str]:
        if not contexts or not target_sections:
            return contexts

        entries = self._extract_context_entries(
            contexts=contexts,
            target_sections=target_sections,
        )
        covered_sections: set[str] = set()
        final_contexts: list[str] = []

        for text, sections in entries:
            final_contexts.append(text)
            covered_sections.update(sections)

        missing = set(target_sections).difference(covered_sections)
        if not missing:
            return final_contexts

        for section in list(missing):
            replacement_index = None
            for idx, (_, sections) in enumerate(entries):
                if sections:
                    continue
                replacement_index = idx
                break
            if replacement_index is None:
                break
            candidate_text, candidate_sections = entries[replacement_index]
            inferred = self._detect_context_sections(
                text=candidate_text,
                target_sections={section},
            )
            if not inferred:
                continue
            final_contexts[replacement_index] = candidate_text
            covered_sections.update(inferred)
            missing = set(target_sections).difference(covered_sections)
            if not missing:
                break

        return final_contexts

    def _filter_contexts_for_expected_sections(
        self,
        *,
        contexts: list[str],
        target_sections: set[str],
        top_k: int,
    ) -> list[str]:
        if not contexts:
            return []

        if not target_sections:
            return contexts[:top_k]

        entries = self._extract_context_entries(
            contexts=contexts,
            target_sections=target_sections,
        )
        selected: list[str] = []
        used_indices: set[int] = set()

        for section in target_sections:
            for idx, (text, sections) in enumerate(entries):
                if idx in used_indices:
                    continue
                if section in sections:
                    selected.append(text)
                    used_indices.add(idx)
                    break

        for idx, (text, _) in enumerate(entries):
            if len(selected) >= top_k:
                break
            if idx in used_indices:
                continue
            selected.append(text)
            used_indices.add(idx)

        return selected[:top_k]

    def _finalize_contexts_for_judge(
        self,
        *,
        contexts: list[str],
        target_sections: set[str],
        top_k: int,
        char_limit: int,
    ) -> list[str]:
        filtered = self._filter_contexts_for_expected_sections(
            contexts=contexts,
            target_sections=target_sections,
            top_k=top_k,
        )
        ensured = self._ensure_target_section_coverage(
            contexts=filtered,
            target_sections=target_sections,
        )
        return self._prepare_ragas_contexts(
            raw_contexts=ensured,
            top_k=top_k,
            char_limit=char_limit,
        )

    def _resolve_context_top_k(
        self,
        *,
        target_sections: set[str],
        default_top_k: int,
    ) -> int:
        section_count = self._estimate_section_count_from_targets(target_sections)
        if section_count <= 1:
            return max(1, min(default_top_k, 3))
        if section_count == 2:
            has_symptom = "symptom" in target_sections
            has_aetiology = "aetiologies" in target_sections
            if has_symptom and has_aetiology:
                return max(3, min(default_top_k, 4))
            return max(2, min(default_top_k, 3))
        return max(2, default_top_k)

    def _build_judge_contexts(
        self,
        *,
        retrieval_output: dict[str, Any],
        generation_output: dict[str, Any],
        question_text: str,
        default_top_k: int,
        char_limit: int,
    ) -> list[str]:
        target_sections = self._extract_target_sections(
            retrieval_output=retrieval_output,
            question_text=question_text,
        )
        question_tokens = self._tokenize_context_text(question_text)
        candidates = self._build_context_candidates(
            retrieval_output=retrieval_output,
            question_tokens=question_tokens,
            target_sections=target_sections,
        )

        summaries = retrieval_output.get("summaries", {})
        summaries_map = summaries if isinstance(summaries, dict) else {}
        context_snapshot = str(generation_output.get("context_snapshot", "")).strip()
        top_k = self._resolve_context_top_k(
            target_sections=target_sections,
            default_top_k=default_top_k,
        )

        if self._has_sufficient_context_candidates(
            candidates=candidates,
            target_sections=target_sections,
        ):
            contexts = self._build_context_from_candidates(
                candidates=candidates,
                target_sections=target_sections,
                summaries=summaries_map,
                context_snapshot=context_snapshot,
            )
            return self._finalize_contexts_for_judge(
                contexts=contexts,
                target_sections=target_sections,
                top_k=top_k,
                char_limit=char_limit,
            )

        legacy = self._append_legacy_fallback_contexts(
            contexts=self._rank_context_candidates(
                candidates, target_sections=target_sections
            ),
            summaries=summaries_map,
            context_snapshot=context_snapshot,
        )
        return self._finalize_contexts_for_judge(
            contexts=legacy,
            target_sections=target_sections,
            top_k=top_k,
            char_limit=char_limit,
        )

    def _append_legacy_fallback_contexts(
        self,
        *,
        contexts: list[str],
        summaries: dict[str, Any],
        context_snapshot: str,
    ) -> list[str]:
        legacy = list(contexts)
        self._append_fallback_contexts(
            contexts=legacy,
            summaries=summaries,
            context_snapshot=context_snapshot,
        )
        return self._dedupe_context_list(legacy)

    def _remove_redundant_general_contexts(
        self,
        *,
        contexts: list[str],
        target_sections: set[str],
    ) -> list[str]:
        if not contexts:
            return contexts

        if not target_sections:
            return contexts

        has_risk_target = "risk" in target_sections
        has_symptom_target = "symptom" in target_sections
        has_aetiology_target = "aetiologies" in target_sections

        if not has_risk_target and not (has_symptom_target and has_aetiology_target):
            return contexts

        filtered: list[str] = []
        for text in contexts:
            lowered = str(text or "").strip().lower()
            if not lowered:
                continue
            if lowered.startswith("coronavirus disease (covid-19) "):
                continue
            filtered.append(text)

        return filtered or contexts

    def _build_context_candidates(
        self,
        *,
        retrieval_output: dict[str, Any],
        question_tokens: set[str],
        target_sections: set[str],
    ) -> list[_ContextCandidate]:
        candidates: list[_ContextCandidate] = []
        if target_sections:
            candidates.extend(
                self._build_summary_candidates(
                    retrieval_output=retrieval_output,
                    question_tokens=question_tokens,
                    target_sections=target_sections,
                )
            )
        candidates.extend(
            self._build_retrieved_item_candidates(
                retrieval_output=retrieval_output,
                question_tokens=question_tokens,
                target_sections=target_sections,
            )
        )
        candidates.extend(
            self._build_retrieved_text_candidates(
                retrieval_output=retrieval_output,
                question_tokens=question_tokens,
                target_sections=target_sections,
            )
        )
        return candidates

    def _resolve_context_pipeline(
        self,
        *,
        retrieval_output: dict[str, Any],
        generation_output: dict[str, Any],
        question_text: str,
    ) -> list[str]:
        target_sections = self._extract_target_sections(
            retrieval_output=retrieval_output,
            question_text=question_text,
        )
        question_tokens = self._tokenize_context_text(question_text)
        candidates = self._build_context_candidates(
            retrieval_output=retrieval_output,
            question_tokens=question_tokens,
            target_sections=target_sections,
        )

        summaries = retrieval_output.get("summaries", {})
        summaries_map = summaries if isinstance(summaries, dict) else {}
        context_snapshot = str(generation_output.get("context_snapshot", "")).strip()

        if self._has_sufficient_context_candidates(
            candidates=candidates,
            target_sections=target_sections,
        ):
            return self._build_context_from_candidates(
                candidates=candidates,
                target_sections=target_sections,
                summaries=summaries_map,
                context_snapshot=context_snapshot,
            )

        return self._append_legacy_fallback_contexts(
            contexts=self._rank_context_candidates(
                candidates, target_sections=target_sections
            ),
            summaries=summaries_map,
            context_snapshot=context_snapshot,
        )

    def _collect_runtime_contexts(
        self,
        runtime_output: dict[str, Any],
        *,
        question_text: str = "",
    ) -> list[str]:
        """
        Build judge contexts with evidence-first priority and legacy fallback.

        Preferred order:
        1) ranked retrieval evidence snippets
        2) retrieval summaries
        3) generation context snapshot
        """
        retrieval_output = runtime_output.get("retrieval_output", {}) or {}
        generation_output = runtime_output.get("generation_output", {}) or {}
        raw_contexts: list[str] = []

        try:
            raw_contexts = self._resolve_context_pipeline(
                retrieval_output=retrieval_output,
                generation_output=generation_output,
                question_text=question_text,
            )
            if raw_contexts:
                return raw_contexts
        except Exception as exc:
            logger.warning(
                "Ragas context collection fallback to legacy ordering: %s", exc
            )

        return self._collect_runtime_contexts_legacy(runtime_output)

    def _collect_runtime_contexts_legacy(
        self, runtime_output: dict[str, Any]
    ) -> list[str]:
        retrieval_output = runtime_output.get("retrieval_output", {}) or {}
        generation_output = runtime_output.get("generation_output", {}) or {}
        raw_contexts: list[str] = []

        context_snapshot = str(generation_output.get("context_snapshot", "")).strip()
        if context_snapshot:
            raw_contexts.append(context_snapshot)

        summaries = retrieval_output.get("summaries", {})
        if isinstance(summaries, dict):
            for summary in summaries.values():
                text = str(summary).strip()
                if text:
                    raw_contexts.append(text)

        raw_contexts.extend(
            str(item).strip()
            for item in (retrieval_output.get("retrieved_context_texts", []) or [])
            if str(item).strip()
        )
        return raw_contexts

    def _build_retrieved_item_candidates(
        self,
        *,
        retrieval_output: dict[str, Any],
        question_tokens: set[str],
        target_sections: set[str],
    ) -> list[_ContextCandidate]:
        items = retrieval_output.get("retrieved_items", []) or []
        if not isinstance(items, list):
            return []

        candidates: list[_ContextCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("content_preview", "")).strip()
            if not text:
                continue
            section = str(item.get("section", "")).strip().lower()
            title = str(item.get("title", "")).strip().lower()
            formatted_text = self._format_section_context_text(
                text=text,
                section=section,
            )
            score = self._score_context_candidate(
                text=formatted_text,
                section=section,
                title=title,
                question_tokens=question_tokens,
                target_sections=target_sections,
                source="retrieved_items",
            )
            candidates.append(
                _ContextCandidate(
                    text=formatted_text,
                    source="retrieved_items",
                    section=section,
                    title=title,
                    score=score,
                )
            )
        return candidates

    def _build_summary_candidates(
        self,
        *,
        retrieval_output: dict[str, Any],
        question_tokens: set[str],
        target_sections: set[str],
    ) -> list[_ContextCandidate]:
        summaries = retrieval_output.get("summaries", {})
        if not isinstance(summaries, dict):
            return []

        candidates: list[_ContextCandidate] = []
        for title, raw_text in summaries.items():
            text = str(raw_text).strip()
            if not text:
                continue
            lowered = text.lower()
            section = ""
            if target_sections:
                for target in target_sections:
                    marker = str(target).strip().lower()
                    if marker and marker in lowered:
                        section = marker
                        break
                if not section:
                    retrieved_items = retrieval_output.get("retrieved_items")
                    if (
                        len(target_sections) == 1
                        and isinstance(retrieved_items, list)
                        and len(retrieved_items) > 0
                    ):
                        section = next(iter(target_sections))
                    else:
                        continue
            score = self._score_context_candidate(
                text=text,
                section=section,
                title=str(title).strip().lower(),
                question_tokens=question_tokens,
                target_sections=target_sections,
                source="summary",
            )
            candidates.append(
                _ContextCandidate(
                    text=text,
                    source="summary",
                    section=section,
                    title=str(title).strip().lower(),
                    score=score,
                )
            )
        return candidates

    def _build_retrieved_text_candidates(
        self,
        *,
        retrieval_output: dict[str, Any],
        question_tokens: set[str],
        target_sections: set[str],
    ) -> list[_ContextCandidate]:
        retrieved_context_texts = (
            retrieval_output.get("retrieved_context_texts", []) or []
        )
        if not isinstance(retrieved_context_texts, list):
            return []

        candidates: list[_ContextCandidate] = []
        for raw in retrieved_context_texts:
            text = str(raw).strip()
            if not text:
                continue
            score = self._score_context_candidate(
                text=text,
                section="",
                title="",
                question_tokens=question_tokens,
                target_sections=target_sections,
                source="retrieved_context_texts",
            )
            candidates.append(
                _ContextCandidate(
                    text=text,
                    source="retrieved_context_texts",
                    section="",
                    title="",
                    score=score,
                )
            )
        return candidates

    def _extract_source_weight(self, source: str) -> float:
        if source == "summary":
            return 0.06
        if source == "retrieved_context_texts":
            return 0.03
        if source == "retrieved_items":
            return 0.04
        return 0.0

    def _apply_section_weight(
        self,
        *,
        section: str,
        target_sections: set[str],
        score: float,
    ) -> float:
        if section and section in target_sections:
            return score + 0.18
        if section and target_sections:
            return score - 0.08
        return score

    def _score_context_candidate(
        self,
        *,
        text: str,
        section: str,
        title: str,
        question_tokens: set[str],
        target_sections: set[str],
        source: str,
    ) -> float:
        target_text = " ".join(part for part in [title, section, text] if part)
        target_tokens = self._tokenize_context_text(target_text)
        if not target_tokens:
            return 0.0

        overlap = 0.0
        if question_tokens:
            overlap = len(question_tokens.intersection(target_tokens)) / max(
                1, len(question_tokens)
            )

        score = overlap
        score += self._extract_source_weight(source)
        score = self._apply_section_weight(
            section=section,
            target_sections=target_sections,
            score=score,
        )
        return max(0.0, score)

    def _extract_target_sections(
        self,
        *,
        retrieval_output: dict[str, Any],
        question_text: str,
    ) -> set[str]:
        sections: set[str] = set()
        rerank = retrieval_output.get("rerank", {})
        if isinstance(rerank, dict):
            intent = rerank.get("intent", {})
            if isinstance(intent, dict):
                targets = intent.get("target_sections", [])
                if isinstance(targets, list):
                    sections.update(
                        str(item).strip().lower()
                        for item in targets
                        if str(item).strip()
                    )

        if sections:
            return sections

        q_tokens = self._tokenize_context_text(question_text)
        if any(token in q_tokens for token in {"risk", "nguy", "cơ"}):
            sections.add("risk")
        if any(
            token in q_tokens for token in {"symptom", "symptoms", "triệu", "chứng"}
        ):
            sections.add("symptom")
        if any(
            token in q_tokens
            for token in {"cause", "causes", "nguyên", "nhân", "aetiology", "etiology"}
        ):
            sections.add("aetiologies")
        if any(
            token in q_tokens
            for token in {"prevent", "prevention", "vaccine", "phòng", "ngừa"}
        ):
            sections.add("living_and_preventive")

        return sections

    def _format_section_context_text(self, *, text: str, section: str) -> str:
        normalized_text = str(text).strip()
        if not normalized_text:
            return ""
        normalized_text = re.sub(r"[.!?]+$", "", normalized_text)
        section_key = str(section).strip().lower()
        if section_key == "risk":
            return f"Risk factors include {normalized_text}."
        if section_key == "symptom":
            return f"Main symptoms include {normalized_text}."
        if section_key == "aetiologies":
            return f"Cause: {normalized_text}."
        if section_key == "living_and_preventive":
            return f"Prevention guidance: {normalized_text}."
        return normalized_text

    def _tokenize_context_text(self, text: str) -> set[str]:
        normalized = re.sub(r"[^0-9a-zA-ZÀ-ỹ]+", " ", str(text or "").lower())
        return {token for token in normalized.split() if token}
