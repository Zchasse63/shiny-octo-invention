# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and utilities for reranking documents."""

import inspect
import json
import math
import os
import sys

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.knowledge.provider_utils import resolve_provider
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, Configured
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.template import CustomTemplate, SafeSub, RepFunc

if tp.TYPE_CHECKING:
    from voyageai import Client as VoyageClientT
else:
    VoyageClientT = "voyageai.Client"
if tp.TYPE_CHECKING:
    from cohere import ClientV2 as CohereClientT
else:
    CohereClientT = "cohere.ClientV2"
if tp.TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as CrossEncoderT
else:
    CrossEncoderT = "sentence_transformers.CrossEncoder"
if tp.TYPE_CHECKING:
    from vectorbtpro.knowledge.completions import Completions as CompletionsT
else:
    CompletionsT = "vectorbtpro.knowledge.completions.Completions"

__all__ = [
    "Reranker",
    "VoyageReranker",
    "CohereReranker",
    "JinaReranker",
    "HFCrossEncoderReranker",
    "CompletionsReranker",
    "rerank",
]


class Reranker(Configured):
    """Abstract class for reranking providers.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.reranker_config`.

    Args:
        batch_size (Optional[int]): Number of documents per batch.

            Use None to process all documents in a single batch.
        show_progress (Optional[bool]): Whether to display a progress bar.
        pbar_kwargs (KwargsLike): Keyword arguments for `vectorbtpro.utils.pbar.ProgressBar`.
        template_context (Kwargs): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _score_range: tp.ClassVar[tp.Optional[tp.Tuple[float, float]]] = None
    """Known (min, max) bounds for relevance scores, or None if unknown."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat", "knowledge.chat.reranker_config"]

    def __init__(
        self,
        batch_size: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            batch_size=batch_size,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            **kwargs,
        )

        batch_size = self.resolve_setting(batch_size, "batch_size")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._batch_size = batch_size
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs
        self._template_context = template_context

    @property
    def batch_size(self) -> tp.Optional[int]:
        """Number of documents per batch.

        Use None to process all documents in a single batch.

        Returns:
            Optional[int]: Batch size.
        """
        return self._batch_size

    @property
    def show_progress(self) -> tp.Optional[bool]:
        """Whether to display a progress bar.

        Returns:
            Optional[bool]: True if progress bar is shown, False otherwise.
        """
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `vectorbtpro.utils.pbar.ProgressBar`.

        Returns:
            Kwargs: Keyword arguments for the progress bar.
        """
        return self._pbar_kwargs

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def model(self) -> tp.Optional[str]:
        """Model identifier.

        Returns:
            Optional[str]: Model identifier; None by default.
        """
        return None

    @property
    def score_range(self) -> tp.Optional[tp.Tuple[float, float]]:
        """Known (min, max) bounds for relevance scores returned by this reranker.

        Returns:
            Optional[Tuple[float, float]]: (min, max) bounds, or None if unknown.
        """
        return self._score_range

    def rerank_batch(self, query: str, documents: tp.List[str]) -> tp.List[tp.Tuple[int, float]]:
        """Rerank a single batch of documents by relevance to a query.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            query (str): Query string.
            documents (List[str]): List of document content strings.

        Returns:
            List[Tuple[int, float]]: List of (batch-local index, relevance_score) tuples.
        """
        raise NotImplementedError

    def rerank(self, query: str, documents: tp.List[str]) -> tp.List[tp.Tuple[int, float]]:
        """Rerank documents by relevance to a query.

        Splits documents into batches, calls `rerank_batch` per batch,
        maps batch-local indices to global indices, and shows a progress bar.

        Args:
            query (str): Query string.
            documents (List[str]): List of document content strings.

        Returns:
            List[Tuple[int, float]]: List of (original_index, relevance_score) tuples,
                sorted by relevance descending.
        """
        from vectorbtpro.utils.pbar import ProgressBar

        if len(documents) == 0:
            return []
        if self.batch_size is not None:
            batches = [documents[i : i + self.batch_size] for i in range(0, len(documents), self.batch_size)]
            batch_offsets = list(range(0, len(documents), self.batch_size))
        else:
            batches = [documents]
            batch_offsets = [0]

        all_results = []
        pbar_kwargs = merge_dicts(dict(prefix="rerank"), self.pbar_kwargs)
        with ProgressBar(total=len(documents), show_progress=self.show_progress, **pbar_kwargs) as pbar:
            for batch, offset in zip(batches, batch_offsets):
                batch_results = self.rerank_batch(query, batch)
                invalid_result = False
                seen_indices = set()
                new_batch_results = []
                for result in batch_results:
                    if not isinstance(result, tuple) or len(result) != 2:
                        invalid_result = True
                        break
                    idx, score = result
                    if not checks.is_int(idx):
                        invalid_result = True
                        break
                    if idx < 0 or idx >= len(batch) or idx in seen_indices:
                        invalid_result = True
                        break
                    seen_indices.add(idx)
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        score = float("nan")
                    new_batch_results.append((idx, score))
                if invalid_result or len(new_batch_results) != len(batch):
                    batch_results = [(i, float("nan")) for i in range(len(batch))]
                else:
                    batch_results = new_batch_results
                all_results.extend([(idx + offset, score) for idx, score in batch_results])
                pbar.update(len(batch))
        all_results.sort(key=lambda x: (not math.isnan(x[1]), x[1]), reverse=True)
        return all_results


class VoyageReranker(Reranker):
    """Reranker class for Voyage.

    !!! info
        For default settings, see `chat.reranker_configs.voyage` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Voyage model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `voyageai.Client`.
        rerank_kwargs (KwargsLike): Keyword arguments for `voyageai.Client.rerank`.
        **kwargs: Keyword arguments for `Reranker` or used as `client_kwargs` or `rerank_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "voyage"

    _score_range: tp.ClassVar[tp.Optional[tp.Tuple[float, float]]] = (0.0, 1.0)

    _settings_path: tp.SettingsPath = "knowledge.chat.reranker_configs.voyage"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        rerank_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Reranker.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            rerank_kwargs=rerank_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("voyageai")
        from voyageai import Client

        voyage_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = voyage_config.pop("model", None)
        def_client_kwargs = voyage_config.pop("client_kwargs", None)
        def_rerank_kwargs = voyage_config.pop("rerank_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(voyage_config.keys()):
            if k in init_arg_names:
                voyage_config.pop(k)

        client_arg_names = set(get_func_arg_names(Client.__init__))
        _client_kwargs = {}
        _rerank_kwargs = {}
        for k, v in voyage_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _rerank_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        rerank_kwargs = merge_dicts(_rerank_kwargs, def_rerank_kwargs, rerank_kwargs)
        client = Client(**client_kwargs)

        self._model = model
        self._client = client
        self._rerank_kwargs = rerank_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> VoyageClientT:
        """Voyage client instance.

        Returns:
            Client: Voyage client instance.
        """
        return self._client

    @property
    def rerank_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `voyageai.Client.rerank`.

        Returns:
            Kwargs: Keyword arguments for reranking.
        """
        return self._rerank_kwargs

    def rerank_batch(self, query: str, documents: tp.List[str]) -> tp.List[tp.Tuple[int, float]]:
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            **self.rerank_kwargs,
        )
        return [(result.index, result.relevance_score) for result in response.results]


class CohereReranker(Reranker):
    """Reranker class for Cohere.

    !!! info
        For default settings, see `chat.reranker_configs.cohere` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Cohere model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `cohere.ClientV2`.
        rerank_kwargs (KwargsLike): Keyword arguments for `cohere.ClientV2.rerank`.
        **kwargs: Keyword arguments for `Reranker` or used as `client_kwargs` or `rerank_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "cohere"

    _score_range: tp.ClassVar[tp.Optional[tp.Tuple[float, float]]] = (0.0, 1.0)

    _settings_path: tp.SettingsPath = "knowledge.chat.reranker_configs.cohere"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        rerank_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Reranker.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            rerank_kwargs=rerank_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("cohere")
        from cohere import ClientV2

        cohere_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = cohere_config.pop("model", None)
        def_client_kwargs = cohere_config.pop("client_kwargs", None)
        def_rerank_kwargs = cohere_config.pop("rerank_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(cohere_config.keys()):
            if k in init_arg_names:
                cohere_config.pop(k)

        client_arg_names = set(get_func_arg_names(ClientV2.__init__))
        _client_kwargs = {}
        _rerank_kwargs = {}
        for k, v in cohere_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _rerank_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        rerank_kwargs = merge_dicts(_rerank_kwargs, def_rerank_kwargs, rerank_kwargs)
        client = ClientV2(**client_kwargs)

        self._model = model
        self._client = client
        self._rerank_kwargs = rerank_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> CohereClientT:
        """Cohere client instance.

        Returns:
            ClientV2: Cohere client instance.
        """
        return self._client

    @property
    def rerank_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `cohere.ClientV2.rerank`.

        Returns:
            Kwargs: Keyword arguments for reranking.
        """
        return self._rerank_kwargs

    def rerank_batch(self, query: str, documents: tp.List[str]) -> tp.List[tp.Tuple[int, float]]:
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            **self.rerank_kwargs,
        )
        return [(result.index, result.relevance_score) for result in response.results]


class JinaReranker(Reranker):
    """Reranker class for Jina.

    !!! info
        For default settings, see `chat.reranker_configs.jina` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Jina model identifier.
        api_key (Optional[str]): Jina API key.

            If None, uses the `JINA_API_KEY` environment variable.
        api_url (Optional[str]): Jina API URL.
        rerank_kwargs (KwargsLike): Additional keyword arguments for the Jina rerank API request.
        **kwargs: Keyword arguments for `Reranker`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "jina"

    _score_range: tp.ClassVar[tp.Optional[tp.Tuple[float, float]]] = (0.0, 1.0)

    _settings_path: tp.SettingsPath = "knowledge.chat.reranker_configs.jina"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        api_key: tp.Optional[str] = None,
        api_url: tp.Optional[str] = None,
        rerank_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Reranker.__init__(
            self,
            model=model,
            api_key=api_key,
            api_url=api_url,
            rerank_kwargs=rerank_kwargs,
            **kwargs,
        )

        jina_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = jina_config.pop("model", None)
        def_api_key = jina_config.pop("api_key", None)
        def_api_url = jina_config.pop("api_url", None)
        def_rerank_kwargs = jina_config.pop("rerank_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        if api_key is None:
            api_key = def_api_key
        if api_key is None:
            api_key = os.getenv("JINA_API_KEY")
        if api_key is None:
            raise ValueError("Must provide an api_key or set the JINA_API_KEY environment variable")
        if api_url is None:
            api_url = def_api_url
        if api_url is None:
            api_url = "https://api.jina.ai/v1/rerank"
        init_arg_names = self.get_init_arg_names()
        for k in list(jina_config.keys()):
            if k in init_arg_names:
                jina_config.pop(k)

        rerank_kwargs = merge_dicts(jina_config, def_rerank_kwargs, rerank_kwargs)

        self._model = model
        self._api_key = api_key
        self._api_url = api_url
        self._rerank_kwargs = rerank_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_key(self) -> str:
        """Jina API key.

        Returns:
            str: API key.
        """
        return self._api_key

    @property
    def api_url(self) -> str:
        """Jina API URL.

        Returns:
            str: API URL.
        """
        return self._api_url

    @property
    def rerank_kwargs(self) -> tp.Kwargs:
        """Additional keyword arguments for the Jina rerank API request.

        Returns:
            Kwargs: Keyword arguments for reranking.
        """
        return self._rerank_kwargs

    def rerank_batch(self, query: str, documents: tp.List[str]) -> tp.List[tp.Tuple[int, float]]:
        import httpx

        response = httpx.post(
            self.api_url,
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                **self.rerank_kwargs,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=None,
        )
        response.raise_for_status()
        results = response.json()["results"]
        return [(r["index"], r["relevance_score"]) for r in results]


class HFCrossEncoderReranker(Reranker):
    """Reranker class for HuggingFace cross-encoder models via sentence-transformers.

    !!! info
        For default settings, see `chat.reranker_configs.hf_cross_encoder` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Cross-encoder model identifier.
        cross_encoder_kwargs (KwargsLike): Keyword arguments for `sentence_transformers.CrossEncoder`.
        predict_kwargs (KwargsLike): Keyword arguments for `sentence_transformers.CrossEncoder.predict`.
        **kwargs: Keyword arguments for `Reranker`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "hf_cross_encoder"

    _settings_path: tp.SettingsPath = "knowledge.chat.reranker_configs.hf_cross_encoder"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        cross_encoder_kwargs: tp.KwargsLike = None,
        predict_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Reranker.__init__(
            self,
            model=model,
            cross_encoder_kwargs=cross_encoder_kwargs,
            predict_kwargs=predict_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sentence_transformers")
        from sentence_transformers import CrossEncoder

        hf_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = hf_config.pop("model", None)
        def_cross_encoder_kwargs = hf_config.pop("cross_encoder_kwargs", None)
        def_predict_kwargs = hf_config.pop("predict_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(hf_config.keys()):
            if k in init_arg_names:
                hf_config.pop(k)

        client_arg_names = set(get_func_arg_names(CrossEncoder.__init__))
        _cross_encoder_kwargs = {}
        _predict_kwargs = {}
        for k, v in hf_config.items():
            if k in client_arg_names:
                _cross_encoder_kwargs[k] = v
            else:
                _predict_kwargs[k] = v
        cross_encoder_kwargs = merge_dicts(_cross_encoder_kwargs, def_cross_encoder_kwargs, cross_encoder_kwargs)
        predict_kwargs = merge_dicts(_predict_kwargs, def_predict_kwargs, predict_kwargs)
        cross_encoder = CrossEncoder(model, **cross_encoder_kwargs)

        self._model = model
        self._cross_encoder = cross_encoder
        self._predict_kwargs = predict_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def cross_encoder(self) -> CrossEncoderT:
        """Cross-encoder model instance.

        Returns:
            CrossEncoder: Cross-encoder model instance.
        """
        return self._cross_encoder

    @property
    def predict_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `sentence_transformers.CrossEncoder.predict`.

        Returns:
            Kwargs: Keyword arguments for prediction.
        """
        return self._predict_kwargs

    def rerank_batch(self, query: str, documents: tp.List[str]) -> tp.List[tp.Tuple[int, float]]:
        pairs = [[query, doc] for doc in documents]
        scores = np.asarray(self.cross_encoder.predict(pairs, **self.predict_kwargs)).tolist()
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores


class CompletionsReranker(Reranker):
    """Reranker class using LLM completions.

    !!! info
        For default settings, see `chat.reranker_configs.completions` in `vectorbtpro._settings.knowledge`.

    Args:
        completions (CompletionsLike): Identifier, subclass, or instance of
            `vectorbtpro.knowledge.completions.Completions`.

            Resolved using `vectorbtpro.knowledge.completions.resolve_completions`.
        completions_kwargs (KwargsLike): Keyword arguments to initialize or update `completions`.
        strategy (Optional[str]): Reranking strategy.

            Supported strategies:

            * "pointwise": Score each document individually (one LLM call per document).
            * "listwise": Score all documents in a single LLM call.
        pointwise_template (Optional[CustomTemplateLike]): Prompt template for pointwise ranking,
            as a string, function, or custom template.

            Should include placeholders for `query_json` and `document_json`.
        listwise_template (Optional[CustomTemplateLike]): Prompt template for listwise ranking,
            as a string, function, or custom template.

            Should include placeholders for `query_json` and `documents_json`.
        **kwargs: Keyword arguments for `Reranker`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "completions"

    _score_range: tp.ClassVar[tp.Optional[tp.Tuple[float, float]]] = (0.0, 4.0)

    _settings_path: tp.SettingsPath = "knowledge.chat.reranker_configs.completions"

    def __init__(
        self,
        completions: tp.CompletionsLike = None,
        completions_kwargs: tp.KwargsLike = None,
        strategy: tp.Optional[str] = None,
        pointwise_template: tp.Optional[tp.CustomTemplateLike] = None,
        listwise_template: tp.Optional[tp.CustomTemplateLike] = None,
        **kwargs,
    ) -> None:
        Reranker.__init__(
            self,
            completions=completions,
            completions_kwargs=completions_kwargs,
            strategy=strategy,
            pointwise_template=pointwise_template,
            listwise_template=listwise_template,
            **kwargs,
        )

        from vectorbtpro.knowledge.completions import resolve_completions

        completions_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_completions = completions_config.pop("completions", None)
        def_completions_kwargs = completions_config.pop("completions_kwargs", None)
        def_strategy = completions_config.pop("strategy", None)
        def_pointwise_template = completions_config.pop("pointwise_template", None)
        def_listwise_template = completions_config.pop("listwise_template", None)

        if completions is None:
            completions = def_completions
        if strategy is None:
            strategy = def_strategy
        if pointwise_template is None:
            pointwise_template = def_pointwise_template
        if isinstance(pointwise_template, str):
            pointwise_template = SafeSub(pointwise_template)
        elif checks.is_function(pointwise_template):
            pointwise_template = RepFunc(pointwise_template)
        elif not isinstance(pointwise_template, CustomTemplate):
            raise TypeError("Pointwise template must be a string, function, or template")
        if listwise_template is None:
            listwise_template = def_listwise_template
        if isinstance(listwise_template, str):
            listwise_template = SafeSub(listwise_template)
        elif checks.is_function(listwise_template):
            listwise_template = RepFunc(listwise_template)
        elif not isinstance(listwise_template, CustomTemplate):
            raise TypeError("Listwise template must be a string, function, or template")
        init_arg_names = self.get_init_arg_names()
        for k in list(completions_config.keys()):
            if k in init_arg_names:
                completions_config.pop(k)

        completions_kwargs = merge_dicts(completions_config, def_completions_kwargs, completions_kwargs)
        completions = resolve_completions(completions)
        if isinstance(completions, type):
            completions = completions(**completions_kwargs)
        elif completions_kwargs:
            completions = completions.replace(**completions_kwargs)

        self._completions = completions
        self._strategy = strategy
        self._pointwise_template = pointwise_template
        self._listwise_template = listwise_template

    @property
    def completions(self) -> CompletionsT:
        """Completions instance used for LLM-based reranking.

        Returns:
            Completions: Completions instance.
        """
        return self._completions

    @property
    def model(self) -> tp.Optional[str]:
        return self.completions.model

    @property
    def strategy(self) -> str:
        """Reranking strategy.

        Returns:
            str: Either "pointwise" or "listwise".
        """
        return self._strategy

    @property
    def pointwise_template(self) -> CustomTemplate:
        """Prompt template for pointwise scoring.

        Should include placeholders for `query_json` and `document_json`.

        Returns:
            CustomTemplate: Template instance.
        """
        return self._pointwise_template

    @property
    def listwise_template(self) -> CustomTemplate:
        """Prompt template for listwise scoring.

        Should include placeholders for `query_json` and `documents_json`.

        Returns:
            CustomTemplate: Template instance.
        """
        return self._listwise_template

    def rerank_pointwise(self, query: str, documents: tp.List[str]) -> tp.List[tp.Tuple[int, float]]:
        """Score each document individually (one LLM call per document).

        Args:
            query (str): Query string.
            documents (List[str]): List of documents to score.

        Returns:
            List[Tuple[int, float]]: List of tuples containing document index and score.
        """
        results = []
        for i, doc in enumerate(documents):
            query_json = json.dumps({"query": query}, ensure_ascii=False)
            document_json = json.dumps({"document": doc}, ensure_ascii=False)
            _template_context = flat_merge_dicts(
                dict(
                    query_json=query_json,
                    document_json=document_json,
                ),
                self.template_context,
            )
            prompt = self.pointwise_template.substitute(_template_context, eval_id="pointwise_template")
            response = self.completions.get_completion_content(prompt)
            try:
                response = response.strip()
                if response.startswith("```"):
                    response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                parsed = json.loads(response)
                if isinstance(parsed, dict) and "score" in parsed:
                    score = float(parsed["score"])
                else:
                    score = float(parsed)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                score = float("nan")
            results.append((i, score))
        return results

    def rerank_listwise(self, query: str, documents: tp.List[str]) -> tp.List[tp.Tuple[int, float]]:
        """Score all documents in a single LLM call.

        Args:
            query (str): Query string.
            documents (List[str]): List of documents to score.

        Returns:
            List[Tuple[int, float]]: List of tuples containing document index and score.
        """
        query_json = json.dumps({"query": query}, ensure_ascii=False)
        docs_payload = [{"index": i, "document": doc} for i, doc in enumerate(documents)]
        documents_json = json.dumps(docs_payload, ensure_ascii=False, indent=2)
        _template_context = flat_merge_dicts(
            dict(
                query_json=query_json,
                documents_json=documents_json,
            ),
            self.template_context,
        )
        prompt = self.listwise_template.substitute(_template_context, eval_id="listwise_template")
        response = self.completions.get_completion_content(prompt)
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(response)
            results = [(item["index"], float(item["score"])) for item in parsed]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            results = [(i, float("nan")) for i in range(len(documents))]
        return results

    def rerank_batch(self, query: str, documents: tp.List[str]) -> tp.List[tp.Tuple[int, float]]:
        if self.strategy == "listwise":
            return self.rerank_listwise(query, documents)
        return self.rerank_pointwise(query, documents)


def resolve_reranker(reranker: tp.RerankerLike = None) -> tp.Optional[tp.MaybeType[Reranker]]:
    """Return a subclass or instance of `Reranker` based on the provided identifier or object.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.

    Args:
        reranker (RerankerLike): Identifier, subclass, or instance of `Reranker`.

            Supported identifiers:

            * "voyage" for `VoyageReranker` (remote)
            * "cohere" for `CohereReranker` (remote)
            * "jina" for `JinaReranker` (remote)
            * "hf_cross_encoder" for `HFCrossEncoderReranker` (local)
            * "completions" for `CompletionsReranker` (remote or local)
            * For other options, see `vectorbtpro.knowledge.provider_utils.resolve_provider`

            If None, configuration from `vectorbtpro._settings` is used.

    Returns:
        Optional[Reranker]: Resolved reranker subclass or instance, or None if not configured.
    """
    if reranker is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        reranker = chat_cfg.get("reranker", None)
    if reranker is None:
        return None
    if isinstance(reranker, str):
        from vectorbtpro.utils.module_ import check_installed

        def _has_completions():
            try:
                from vectorbtpro.knowledge.completions import resolve_completions

                return resolve_completions() is not None
            except Exception:
                return False

        reranker = resolve_provider(
            reranker,
            remote_candidates=(
                ("voyage", lambda: check_installed("voyageai") and os.getenv("VOYAGE_API_KEY")),
                ("cohere", lambda: check_installed("cohere") and os.getenv("CO_API_KEY")),
                ("jina", lambda: os.getenv("JINA_API_KEY")),
            ),
            local_candidates=(("hf_cross_encoder", lambda: check_installed("sentence_transformers")),),
            remote_fallback_candidates=(
                ("completions", _has_completions),
                ("cohere", lambda: check_installed("cohere")),
                ("voyage", lambda: check_installed("voyageai")),
            ),
            provider_name="reranker",
        )
        if reranker is None:
            raise ValueError(
                "No reranker available. "
                "Please install one of the supported packages: "
                "voyageai, "
                "cohere, "
                "sentence-transformers, "
                "any completions package, "
                "or set the JINA_API_KEY environment variable."
            )
        curr_module = sys.modules[__name__]
        found_reranker = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Reranker"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == reranker.lower():
                    found_reranker = cls
                    break
        if found_reranker is None:
            raise ValueError(f"Invalid reranker: {reranker!r}")
        reranker = found_reranker
    if isinstance(reranker, type):
        checks.assert_subclass_of(reranker, Reranker, arg_name="reranker")
    else:
        checks.assert_instance_of(reranker, Reranker, arg_name="reranker")
    return reranker


def rerank(
    query: str,
    documents: tp.List[str],
    reranker: tp.RerankerLike = None,
    **kwargs,
) -> tp.List[tp.Tuple[int, float]]:
    """Rerank documents by relevance to a query.

    Args:
        query (str): Query string.
        documents (List[str]): List of document content strings.
        reranker (RerankerLike): Identifier, subclass, or instance of `Reranker`.

            Resolved using `resolve_reranker`.
        **kwargs: Keyword arguments to initialize or update `reranker`.

    Returns:
        List[Tuple[int, float]]: List of (original_index, relevance_score) tuples,
            sorted by relevance descending.
    """
    reranker = resolve_reranker(reranker=reranker)
    if reranker is None:
        raise ValueError("Must provide a reranker")
    if isinstance(reranker, type):
        reranker = reranker(**kwargs)
    elif kwargs:
        reranker = reranker.replace(**kwargs)
    return reranker.rerank(query, documents)
