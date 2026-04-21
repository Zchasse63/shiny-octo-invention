# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and utilities for embeddings."""

import inspect
import os
import re
import sys
import time

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.knowledge.provider_utils import check_ollama_available, resolve_provider
from vectorbtpro.knowledge.tokenization import Tokenizer, resolve_tokenizer
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.config import merge_dicts, Configured
from vectorbtpro.utils.parsing import get_func_arg_names, get_func_kwargs
from vectorbtpro.utils.pbar import ProgressBar

if tp.TYPE_CHECKING:
    from openai import OpenAI as OpenAIT
else:
    OpenAIT = "openai.OpenAI"
if tp.TYPE_CHECKING:
    from google.genai import Client as GenAIClientT
else:
    GenAIClientT = "google.genai.Client"
if tp.TYPE_CHECKING:
    from huggingface_hub import InferenceClient as InferenceClientT
else:
    InferenceClientT = "huggingface_hub.InferenceClient"
if tp.TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as SentenceTransformerT
else:
    SentenceTransformerT = "sentence_transformers.SentenceTransformer"
if tp.TYPE_CHECKING:
    from ollama import Client as OllamaClientT
else:
    OllamaClientT = "ollama.Client"
if tp.TYPE_CHECKING:
    from voyageai import Client as VoyageClientT
else:
    VoyageClientT = "voyageai.Client"
if tp.TYPE_CHECKING:
    from cohere import ClientV2 as CohereClientT
else:
    CohereClientT = "cohere.ClientV2"
if tp.TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding as BaseEmbeddingT
else:
    BaseEmbeddingT = "llama_index.core.embeddings.BaseEmbedding"

__all__ = [
    "Embedding",
    "Embeddings",
    "OpenAIEmbeddings",
    "GeminiEmbeddings",
    "HFInferenceEmbeddings",
    "VoyageEmbeddings",
    "CohereEmbeddings",
    "JinaEmbeddings",
    "LiteLLMEmbeddings",
    "HFEmbeddings",
    "OllamaEmbeddings",
    "LlamaIndexEmbeddings",
    "embed",
]


EmbeddingT = tp.TypeVar("EmbeddingT", bound="Embedding")


@define
class Embedding(DefineMixin):
    """Wrapper for embedding vectors, with optional quantization metadata."""

    data: tp.Array = define.field(repr=lambda x: f"Array[{x.shape[0]}]")
    """Embedding vector."""

    scale: tp.Optional[float] = define.field(default=None)
    """Scale factor for dequantization."""

    offset: tp.Optional[float] = define.field(default=None)
    """Offset for dequantization."""

    orig_dtype: tp.Optional[np.dtype] = define.field(default=None)
    """Original dtype before quantization."""

    @property
    def is_quantized(self) -> bool:
        """Whether this embedding is quantized.

        Returns:
            bool: True if quantized, False otherwise.
        """
        return self.scale is not None and self.offset is not None

    @classmethod
    def quantize(cls: tp.Type[EmbeddingT], data: tp.Array, dtype: tp.DType) -> EmbeddingT:
        """Quantize an embedding vector to an integer dtype using min/max scaling.

        Args:
            data (Array): Embedding vector.
            dtype (DTypeLike): NumPy dtype (e.g., `np.int8`).

        Returns:
            Embedding: Quantized embedding instance.
        """
        info = np.iinfo(dtype)
        min_val = float(data.min())
        max_val = float(data.max())
        val_range = max_val - min_val
        if val_range == 0:
            scale = 1.0
        else:
            scale = val_range / (info.max - info.min)
        offset = min_val - info.min * scale
        quantized = np.round((data - offset) / scale).astype(dtype)
        return cls(data=quantized, scale=scale, offset=offset, orig_dtype=data.dtype)

    def dequantize(self) -> tp.Array:
        """Dequantize the embedding vector back to the original dtype.

        Returns:
            Array: Dequantized embedding vector.
        """
        if not self.is_quantized:
            return self.data
        result = self.data.astype(float_) * self.scale + self.offset
        if self.orig_dtype is not None:
            result = result.astype(self.orig_dtype)
        return result


class Embeddings(Configured):
    """Abstract class for embedding providers.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.embeddings_config`.

    Args:
        batch_size (Optional[int]): Batch size for processing queries.

            Use None to disable batching.
        dtype (Optional[DTypeLike]): NumPy dtype for embedding arrays.
        quant_dtype (Optional[DTypeLike]): NumPy integer dtype for quantizing embedding arrays.
        max_tokens (Optional[int]): Maximum number of tokens per query.

            If set, each query is truncated to at most this many tokens before embedding.
        tokenizer (TokenizerLike): Identifier, subclass, or instance of
            `vectorbtpro.knowledge.tokenization.Tokenizer`.

            Resolved using `vectorbtpro.knowledge.tokenization.resolve_tokenizer`.
        tokenizer_kwargs (KwargsLike): Keyword arguments to initialize or update `tokenizer`.
        snapshot_name (Optional[str]): Override for the auto-generated snapshot name.

            If set, used instead of deriving the name from model, dimensions, and dtypes.
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        pbar_kwargs (Kwargs): Keyword arguments for configuring the progress bar.
        template_context (Kwargs): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat", "knowledge.chat.embeddings_config"]

    def __init__(
        self,
        batch_size: tp.Optional[int] = None,
        dtype: tp.Optional[tp.DTypeLike] = None,
        quant_dtype: tp.Optional[tp.DTypeLike] = None,
        max_tokens: tp.Optional[int] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        snapshot_name: tp.Optional[str] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            batch_size=batch_size,
            dtype=dtype,
            quant_dtype=quant_dtype,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            snapshot_name=snapshot_name,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            **kwargs,
        )

        batch_size = self.resolve_setting(batch_size, "batch_size")
        dtype = self.resolve_setting(dtype, "dtype")
        quant_dtype = self.resolve_setting(quant_dtype, "quant_dtype")
        max_tokens = self.resolve_setting(max_tokens, "max_tokens")
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True)
        snapshot_name = self.resolve_setting(snapshot_name, "snapshot_name")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        if dtype is not None:
            dtype = np.dtype(dtype)
            if not np.issubdtype(dtype, np.floating):
                raise ValueError(f"Float dtype expected, got {dtype!r}")
        if quant_dtype is not None:
            quant_dtype = np.dtype(quant_dtype)
            if not np.issubdtype(quant_dtype, np.integer):
                raise ValueError(f"Integer quant_dtype expected, got {quant_dtype!r}")
        tokenizer = resolve_tokenizer(tokenizer)
        if isinstance(tokenizer, type):
            tokenizer_kwargs = dict(tokenizer_kwargs)
            tokenizer_kwargs["template_context"] = merge_dicts(
                template_context, tokenizer_kwargs.get("template_context", None)
            )
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)

        self._batch_size = batch_size
        self._dtype = dtype
        self._quant_dtype = quant_dtype
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._snapshot_name = snapshot_name
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs
        self._template_context = template_context

    @property
    def batch_size(self) -> tp.Optional[int]:
        """Batch size used for processing queries.

        Use None to disable batching.

        Returns:
            Optional[int]: Batch size.
        """
        return self._batch_size

    @property
    def dtype(self) -> tp.Optional[tp.DType]:
        """NumPy dtype for embedding arrays.

        Returns:
            Optional[DType]: NumPy dtype; None if not configured.
        """
        return self._dtype

    @property
    def quant_dtype(self) -> tp.Optional[tp.DType]:
        """NumPy integer dtype for quantizing embedding arrays.

        Returns:
            Optional[DType]: NumPy integer dtype; None if not configured.
        """
        return self._quant_dtype

    @property
    def max_tokens(self) -> tp.Optional[int]:
        """Maximum number of tokens per query.

        If set, each query is truncated to at most this many tokens before embedding.

        Returns:
            Optional[int]: Maximum token count; None if not configured.
        """
        return self._max_tokens

    @property
    def tokenizer(self) -> Tokenizer:
        """`vectorbtpro.knowledge.tokenization.Tokenizer` instance used to tokenize input text.

        Returns:
            Tokenizer: Tokenizer instance used for encoding and decoding.
        """
        return self._tokenizer

    def truncate_text(self, text: str) -> str:
        """Truncate text to `Embeddings.max_tokens` using the configured tokenizer.

        Args:
            text (str): Text to truncate.

        Returns:
            str: Truncated text, or original text if `Embeddings.max_tokens` is None.
        """
        if self.max_tokens is None:
            return text
        encoded = self.tokenizer.encode(text)
        if len(encoded) > self.max_tokens:
            print(len(encoded))
            return self.tokenizer.decode(encoded[: self.max_tokens])
        return text

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

    def cast_embedding(self, embedding: tp.Sequence[float]) -> Embedding:
        """Cast the embedding to an `Embedding` instance with the configured dtype.

        Args:
            embedding (Sequence[float]): Raw embedding vector.

        Returns:
            Embedding: Embedding instance, optionally quantized.
        """
        embedding = np.asarray(embedding)
        if self.dtype is not None:
            embedding = embedding.astype(self.dtype)
        if self.quant_dtype is not None:
            return Embedding.quantize(embedding, self.quant_dtype)
        return Embedding(data=embedding)

    @property
    def model(self) -> tp.Optional[str]:
        """Model identifier.

        Returns:
            Optional[str]: Model identifier; None by default.
        """
        return None

    @property
    def dimensions(self) -> tp.Optional[int]:
        """Embedding dimensionality, if explicitly configured or otherwise discoverable.

        Returns:
            Optional[int]: Embedding dimensionality; None by default.
        """
        return None

    @classmethod
    def sanitize_snapshot_name(cls, snapshot_name: str) -> str:
        """Sanitize a snapshot name to be safe for use as an identifier.

        Args:
            snapshot_name (str): Original snapshot name.

        Returns:
            str: Sanitized snapshot name.
        """
        snapshot_name = snapshot_name.lower()
        snapshot_name = re.sub(r"[/:\s]+", "-", snapshot_name)
        snapshot_name = re.compile(r"[^a-z0-9._-]+").sub("-", snapshot_name)
        snapshot_name = re.compile(r"[-_]{2,}").sub("-", snapshot_name)
        snapshot_name = snapshot_name.strip("-.")
        if len(snapshot_name) == 0:
            raise ValueError("Snapshot name is empty after sanitization")
        return snapshot_name

    @classmethod
    def generate_snapshot_name(
        cls,
        model: str,
        dimensions: tp.Optional[int] = None,
        dtype: tp.Optional[tp.DType] = None,
        quant_dtype: tp.Optional[tp.DType] = None,
    ) -> str:
        """Generate a snapshot name based on the model, dimensions, and quantization dtype.

        Args:
            model (str): Model identifier.
            dimensions (Optional[int]): Embedding dimensionality.
            dtype (Optional[DType]): NumPy dtype for embedding arrays.
            quant_dtype (Optional[DType]): NumPy integer dtype for quantizing embedding arrays.

        Returns:
            str: Generated snapshot name.
        """
        parts = [model]
        if dimensions is not None:
            parts.append(f"d{dimensions}")
        if dtype is not None:
            parts.append(f"{dtype.kind}{dtype.itemsize}")
        if quant_dtype is not None:
            parts.append(f"{quant_dtype.kind}{quant_dtype.itemsize}")
        return cls.sanitize_snapshot_name("-".join(parts))

    @property
    def snapshot_name(self) -> tp.Optional[str]:
        """Sanitized snapshot name based on the model, dimensions, and quantization dtype.

        Can be overridden via the `Embeddings.snapshot_name` constructor argument.

        Returns:
            Optional[str]: Sanitized snapshot name or None if model is not defined.
        """
        if self._snapshot_name is not None:
            return self.sanitize_snapshot_name(self._snapshot_name)
        return self.generate_snapshot_name(
            self.model,
            dimensions=self.dimensions,
            dtype=self.dtype,
            quant_dtype=self.quant_dtype,
        )

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        """Return the raw embedding vector for the given query.

        !!! abstract
            This method should be overridden in a subclass.

        Args:
            query (str): Query text.

        Returns:
            Sequence[float]: Raw embedding vector.
        """
        raise NotImplementedError

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        """Return a batch of raw embedding vectors for a list of queries.

        Args:
            batch (List[str]): List of query texts.

        Returns:
            List[Sequence[float]]: List containing a raw embedding vector for each query.
        """
        return [self.get_embedding_raw(query) for query in batch]

    def get_embedding(self, query: str) -> Embedding:
        """Return the embedding instance for the given query.

        Args:
            query (str): Query text.

        Returns:
            Embedding: Embedding instance.
        """
        query = self.truncate_text(query)
        return self.cast_embedding(self.get_embedding_raw(query))

    def get_embedding_batch(self, batch: tp.List[str]) -> tp.List[Embedding]:
        """Return a batch of embedding instances for a list of queries.

        Args:
            batch (List[str]): List of query texts.

        Returns:
            List[Embedding]: List containing an embedding instance for each query.
        """
        if self.max_tokens is not None:
            batch = [self.truncate_text(text) for text in batch]
        return [self.cast_embedding(embedding) for embedding in self.get_embedding_batch_raw(batch)]

    def iter_embedding_batches(self, queries: tp.List[str]) -> tp.Iterator[tp.List[Embedding]]:
        """Return an iterator over batches of embedding instances.

        Args:
            queries (List[str]): List of query texts.

        Returns:
            Iterator[List[Embedding]]: Iterator yielding batches of embedding instances.
        """
        from vectorbtpro.utils.pbar import ProgressBar

        if self.batch_size is not None:
            batches = [queries[i : i + self.batch_size] for i in range(0, len(queries), self.batch_size)]
        else:
            batches = [queries]
        pbar_kwargs = merge_dicts(dict(prefix="get_embeddings"), self.pbar_kwargs)
        with ProgressBar(total=len(queries), show_progress=self.show_progress, **pbar_kwargs) as pbar:
            for batch in batches:
                yield self.get_embedding_batch(batch)
                pbar.update(len(batch))

    def get_embeddings(self, queries: tp.List[str]) -> tp.List[Embedding]:
        """Return embeddings for multiple queries.

        Args:
            queries (List[str]): List of query texts.

        Returns:
            List[Embedding]: List containing an embedding instance for each query.
        """
        return [embedding for batch in self.iter_embedding_batches(queries) for embedding in batch]


class OpenAIEmbeddings(Embeddings):
    """Embeddings class for OpenAI.

    !!! info
        For default settings, see `chat.embeddings_configs.openai` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): OpenAI model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `openai.OpenAI`.
        embeddings_kwargs (KwargsLike): Keyword arguments for `openai.resources.embeddings.Embeddings.create`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `embeddings_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "openai"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.openai"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        embeddings_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            embeddings_kwargs=embeddings_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        openai_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = openai_config.pop("model", None)
        def_client_kwargs = openai_config.pop("client_kwargs", None)
        def_embeddings_kwargs = openai_config.pop("embeddings_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(openai_config.keys()):
            if k in init_arg_names:
                openai_config.pop(k)

        client_arg_names = set(get_func_arg_names(OpenAI.__init__))
        _client_kwargs = {}
        _embeddings_kwargs = {}
        for k, v in openai_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _embeddings_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        embeddings_kwargs = merge_dicts(_embeddings_kwargs, def_embeddings_kwargs, embeddings_kwargs)
        client = OpenAI(**client_kwargs)

        self._model = model
        self._client = client
        self._embeddings_kwargs = embeddings_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> tp.Optional[int]:
        return self.embeddings_kwargs.get("dimensions", None)

    @property
    def client(self) -> OpenAIT:
        """OpenAI client instance.

        Returns:
            OpenAI: OpenAI client instance.
        """
        return self._client

    @property
    def embeddings_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `openai.resources.embeddings.Embeddings.create`.

        Returns:
            Kwargs: Keyword arguments for creating embeddings.
        """
        return self._embeddings_kwargs

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        response = self.client.embeddings.create(input=query, model=self.model, **self.embeddings_kwargs)
        return response.data[0].embedding

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        response = self.client.embeddings.create(input=batch, model=self.model, **self.embeddings_kwargs)
        return [embedding.embedding for embedding in response.data]


class GeminiEmbeddings(Embeddings):
    """Embeddings class for Google GenAI (Gemini).

    !!! info
        For default settings, see `chat.embeddings_configs.gemini` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Gemini model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `google.genai.Client`.
        embeddings_kwargs (KwargsLike): Keyword arguments for `google.genai.Client.models.embed_content`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `embeddings_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "gemini"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.gemini"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        embeddings_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            embeddings_kwargs=embeddings_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("google.genai")
        from google.genai import Client

        gemini_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = gemini_config.pop("model", None)
        def_client_kwargs = gemini_config.pop("client_kwargs", None)
        def_embeddings_kwargs = gemini_config.pop("embeddings_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(gemini_config.keys()):
            if k in init_arg_names:
                gemini_config.pop(k)

        client_arg_names = set(get_func_arg_names(Client.__init__))
        _client_kwargs = {}
        _embeddings_kwargs = {}
        for k, v in gemini_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _embeddings_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        embeddings_kwargs = merge_dicts(_embeddings_kwargs, def_embeddings_kwargs, embeddings_kwargs)

        client = Client(**client_kwargs)

        self._model = model
        self._client = client
        self._embeddings_kwargs = embeddings_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> tp.Optional[int]:
        from google.genai.types import EmbedContentConfig

        if "output_dimensionality" in self.embeddings_kwargs:
            return self.embeddings_kwargs["output_dimensionality"]
        config = self.embeddings_kwargs.get("config", None)
        if isinstance(config, EmbedContentConfig):
            return config.output_dimensionality
        if isinstance(config, dict):
            return config.get("output_dimensionality", None)
        return None

    @property
    def client(self) -> GenAIClientT:
        """Gemini client instance.

        Returns:
            Client: Gemini client instance.
        """
        return self._client

    @property
    def embeddings_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `google.genai.Client.models.embed_content`.

        Returns:
            Kwargs: Keyword arguments for generating embeddings.
        """
        return self._embeddings_kwargs

    def cast_embedding(self, embedding: tp.MaybeArray) -> Embedding:
        embedding = np.asarray(embedding)
        norm_embedding = embedding / np.linalg.norm(embedding)
        return super().cast_embedding(norm_embedding)

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        from google.genai.errors import ClientError

        attempted = False
        while True:
            try:
                response = self.client.models.embed_content(model=self.model, contents=query, **self.embeddings_kwargs)
                return response.embeddings[0].values
            except ClientError as e:
                if e.code == 429 and not attempted:
                    time.sleep(60)
                    attempted = True
                else:
                    raise e

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        from google.genai.errors import ClientError

        attempted = False
        while True:
            try:
                response = self.client.models.embed_content(model=self.model, contents=batch, **self.embeddings_kwargs)
                return [embedding.values for embedding in response.embeddings]
            except ClientError as e:
                if e.code == 429 and not attempted:
                    time.sleep(60)
                    attempted = True
                else:
                    raise e


class HFInferenceEmbeddings(Embeddings):
    """Embeddings class for HuggingFace Inference.

    !!! info
        For default settings, see `chat.embeddings_configs.hf_inference` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): HuggingFace model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `huggingface_hub.InferenceClient`.
        feature_extraction_kwargs (KwargsLike): Keyword arguments for `huggingface_hub.InferenceClient.feature_extraction`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `feature_extraction_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "hf_inference"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.hf_inference"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        feature_extraction_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            feature_extraction_kwargs=feature_extraction_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("huggingface_hub")
        from huggingface_hub import InferenceClient

        hf_inference_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = hf_inference_config.pop("model", None)
        def_client_kwargs = hf_inference_config.pop("client_kwargs", None)
        def_feature_extraction_kwargs = hf_inference_config.pop("feature_extraction_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(hf_inference_config.keys()):
            if k in init_arg_names:
                hf_inference_config.pop(k)

        client_arg_names = set(get_func_arg_names(InferenceClient.__init__))
        _client_kwargs = {}
        _feature_extraction_kwargs = {}
        for k, v in hf_inference_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _feature_extraction_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        feature_extraction_kwargs = merge_dicts(
            _feature_extraction_kwargs,
            def_feature_extraction_kwargs,
            feature_extraction_kwargs,
        )
        client = InferenceClient(model=model, **client_kwargs)

        self._model = model
        self._client = client
        self._feature_extraction_kwargs = feature_extraction_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> tp.Optional[int]:
        return self.feature_extraction_kwargs.get("dimensions", None)

    @property
    def client(self) -> InferenceClientT:
        """HuggingFace Inference client instance.

        Returns:
            InferenceClient: HuggingFace Inference client instance.
        """
        return self._client

    @property
    def feature_extraction_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `huggingface_hub.InferenceClient.feature_extraction`.

        Returns:
            Kwargs: Keyword arguments for feature extraction.
        """
        return self._feature_extraction_kwargs

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        return self.client.feature_extraction(query, **self.feature_extraction_kwargs)[0]

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        return list(self.client.feature_extraction(batch, **self.feature_extraction_kwargs))


class VoyageEmbeddings(Embeddings):
    """Embeddings class for Voyage.

    !!! info
        For default settings, see `chat.embeddings_configs.voyage` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Voyage model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `voyageai.Client`.
        embed_kwargs (KwargsLike): Keyword arguments for `voyageai.Client.embed`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `embed_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "voyage"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.voyage"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        embed_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            embed_kwargs=embed_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("voyageai")
        from voyageai import Client

        voyage_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = voyage_config.pop("model", None)
        def_client_kwargs = voyage_config.pop("client_kwargs", None)
        def_embed_kwargs = voyage_config.pop("embed_kwargs", None)

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
        _embed_kwargs = {}
        for k, v in voyage_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _embed_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        embed_kwargs = merge_dicts(_embed_kwargs, def_embed_kwargs, embed_kwargs)
        client = Client(**client_kwargs)

        self._model = model
        self._client = client
        self._embed_kwargs = embed_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> tp.Optional[int]:
        return self.embed_kwargs.get("dimensions", None)

    @property
    def client(self) -> VoyageClientT:
        """Voyage client instance.

        Returns:
            Client: Voyage client instance.
        """
        return self._client

    @property
    def embed_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `voyageai.Client.embed`.

        Returns:
            Kwargs: Keyword arguments for generating embeddings.
        """
        return self._embed_kwargs

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        response = self.client.embed([query], model=self.model, **self.embed_kwargs)
        return response.embeddings[0]

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        response = self.client.embed(batch, model=self.model, **self.embed_kwargs)
        return list(response.embeddings)


class CohereEmbeddings(Embeddings):
    """Embeddings class for Cohere.

    !!! info
        For default settings, see `chat.embeddings_configs.cohere` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Cohere model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `cohere.ClientV2`.
        embed_kwargs (KwargsLike): Keyword arguments for `cohere.ClientV2.embed`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `embed_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "cohere"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.cohere"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        embed_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            embed_kwargs=embed_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("cohere")
        from cohere import ClientV2

        cohere_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = cohere_config.pop("model", None)
        def_client_kwargs = cohere_config.pop("client_kwargs", None)
        def_embed_kwargs = cohere_config.pop("embed_kwargs", None)

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
        _embed_kwargs = {}
        for k, v in cohere_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _embed_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        embed_kwargs = merge_dicts(_embed_kwargs, def_embed_kwargs, embed_kwargs)
        client = ClientV2(**client_kwargs)

        self._model = model
        self._client = client
        self._embed_kwargs = embed_kwargs

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
    def embed_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `cohere.ClientV2.embed`.

        Returns:
            Kwargs: Keyword arguments for generating embeddings.
        """
        return self._embed_kwargs

    def _extract_embeddings(self, response) -> list:
        embedding_types = self._embed_kwargs.get("embedding_types", ["float"])
        etype = embedding_types[0]
        attr = "float_" if etype == "float" else etype
        return getattr(response.embeddings, attr)

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        response = self.client.embed(texts=[query], model=self.model, **self.embed_kwargs)
        return self._extract_embeddings(response)[0]

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        response = self.client.embed(texts=batch, model=self.model, **self.embed_kwargs)
        return list(self._extract_embeddings(response))


class JinaEmbeddings(Embeddings):
    """Embeddings class for Jina.

    !!! info
        For default settings, see `chat.embeddings_configs.jina` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Jina model identifier.
        api_key (Optional[str]): Jina API key.

            If None, uses the `JINA_API_KEY` environment variable.
        api_url (Optional[str]): Jina API URL.
        embeddings_kwargs (KwargsLike): Additional keyword arguments for the Jina embeddings API request.
        **kwargs: Keyword arguments for `Embeddings`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "jina"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.jina"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        api_key: tp.Optional[str] = None,
        api_url: tp.Optional[str] = None,
        embeddings_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            api_key=api_key,
            api_url=api_url,
            embeddings_kwargs=embeddings_kwargs,
            **kwargs,
        )

        jina_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = jina_config.pop("model", None)
        def_api_key = jina_config.pop("api_key", None)
        def_api_url = jina_config.pop("api_url", None)
        def_embeddings_kwargs = jina_config.pop("embeddings_kwargs", None)

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
            api_url = "https://api.jina.ai/v1/embeddings"
        init_arg_names = self.get_init_arg_names()
        for k in list(jina_config.keys()):
            if k in init_arg_names:
                jina_config.pop(k)

        embeddings_kwargs = merge_dicts(jina_config, def_embeddings_kwargs, embeddings_kwargs)

        self._model = model
        self._api_key = api_key
        self._api_url = api_url
        self._embeddings_kwargs = embeddings_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> tp.Optional[int]:
        return self.embeddings_kwargs.get("dimensions", None)

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
    def embeddings_kwargs(self) -> tp.Kwargs:
        """Additional keyword arguments for the Jina embeddings API request.

        Returns:
            Kwargs: Keyword arguments for generating embeddings.
        """
        return self._embeddings_kwargs

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        import httpx

        response = httpx.post(
            self.api_url,
            json={
                "model": self.model,
                "input": [query],
                **self.embeddings_kwargs,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=None,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        import httpx

        response = httpx.post(
            self.api_url,
            json={
                "model": self.model,
                "input": batch,
                **self.embeddings_kwargs,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=None,
        )
        response.raise_for_status()
        results = response.json()["data"]
        results.sort(key=lambda x: x["index"])
        return [r["embedding"] for r in results]


class LiteLLMEmbeddings(Embeddings):
    """Embeddings class for LiteLLM.

    !!! info
        For default settings, see `chat.embeddings_configs.litellm` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): LiteLLM model identifier.
        embedding_kwargs (KwargsLike): Keyword arguments for `litellm.embedding`.
        **kwargs: Keyword arguments for `Embeddings` or used as `embedding_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "litellm"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.litellm"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        embedding_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            embedding_kwargs=embedding_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        litellm_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = litellm_config.pop("model", None)
        def_embedding_kwargs = litellm_config.pop("embedding_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(litellm_config.keys()):
            if k in init_arg_names:
                litellm_config.pop(k)
        embedding_kwargs = merge_dicts(litellm_config, def_embedding_kwargs, embedding_kwargs)

        self._model = model
        self._embedding_kwargs = embedding_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> tp.Optional[int]:
        return self.embedding_kwargs.get("dimensions", None)

    @property
    def embedding_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `litellm.embedding`.

        Returns:
            Kwargs: Keyword arguments for creating embeddings.
        """
        return self._embedding_kwargs

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        from litellm import embedding

        response = embedding(self.model, input=query, **self.embedding_kwargs)
        return response.data[0]["embedding"]

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        from litellm import embedding

        response = embedding(self.model, input=batch, **self.embedding_kwargs)
        return [embedding["embedding"] for embedding in response.data]


class HFEmbeddings(Embeddings):
    """Embeddings class for HuggingFace sentence-transformers.

    !!! info
        For default settings, see `chat.embeddings_configs.hf` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): HuggingFace model identifier.

            Downloads the model automatically if not already cached locally.
        model_kwargs (KwargsLike): Keyword arguments for `sentence_transformers.SentenceTransformer`.
        encode_kwargs (KwargsLike): Keyword arguments for `sentence_transformers.SentenceTransformer.encode`.
        **kwargs: Keyword arguments for `Embeddings` or used as `model_kwargs` or `encode_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "hf"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.hf"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        model_kwargs: tp.KwargsLike = None,
        encode_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sentence_transformers")
        from sentence_transformers import SentenceTransformer

        hf_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = hf_config.pop("model", None)
        def_model_kwargs = hf_config.pop("model_kwargs", None)
        def_encode_kwargs = hf_config.pop("encode_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(hf_config.keys()):
            if k in init_arg_names:
                hf_config.pop(k)

        st_arg_names = set(get_func_arg_names(SentenceTransformer.__init__))
        _model_kwargs = {}
        _encode_kwargs = {}
        for k, v in hf_config.items():
            if k in st_arg_names:
                _model_kwargs[k] = v
            else:
                _encode_kwargs[k] = v
        model_kwargs = merge_dicts(_model_kwargs, def_model_kwargs, model_kwargs)
        encode_kwargs = merge_dicts(_encode_kwargs, def_encode_kwargs, encode_kwargs)

        st_model = SentenceTransformer(model, **model_kwargs)

        self._model = model
        self._st_model = st_model
        self._model_kwargs = model_kwargs
        self._encode_kwargs = encode_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> tp.Optional[int]:
        if "truncate_dim" in self.model_kwargs:
            return self.model_kwargs["truncate_dim"]
        if "truncate_dim" in self.encode_kwargs:
            return self.encode_kwargs["truncate_dim"]
        return None

    @property
    def st_model(self) -> SentenceTransformerT:
        """Sentence-transformers model instance.

        Returns:
            SentenceTransformer: Sentence-transformers model instance.
        """
        return self._st_model

    @property
    def model_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `sentence_transformers.SentenceTransformer`.

        Returns:
            Kwargs: Keyword arguments for the model.
        """
        return self._model_kwargs

    @property
    def encode_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `sentence_transformers.SentenceTransformer.encode`.

        Returns:
            Kwargs: Keyword arguments for encoding.
        """
        return self._encode_kwargs

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        return self.st_model.encode(query, **self.encode_kwargs)

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        return list(self.st_model.encode(batch, **self.encode_kwargs))


class OllamaEmbeddings(Embeddings):
    """Embeddings class for Ollama.

    !!! info
        For default settings, see `chat.embeddings_configs.ollama` in `vectorbtpro._settings.knowledge`.

    Args:
        model (Optional[str]): Ollama model identifier.

            Pulls the model if not already available locally.
        client_kwargs (KwargsLike): Keyword arguments for `ollama.Client`.
        embed_kwargs (KwargsLike): Keyword arguments for `ollama.Client.embed`.
        **kwargs: Keyword arguments for `Embeddings` or used as `client_kwargs` or `embed_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "ollama"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.ollama"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        embed_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            embed_kwargs=embed_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ollama")
        from ollama import Client

        ollama_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = ollama_config.pop("model", None)
        def_client_kwargs = ollama_config.pop("client_kwargs", None)
        def_embed_kwargs = ollama_config.pop("embed_kwargs", None)

        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = self.get_init_arg_names()
        for k in list(ollama_config.keys()):
            if k in init_arg_names:
                ollama_config.pop(k)

        client_arg_names = set(get_func_arg_names(Client.__init__))
        _client_kwargs = {}
        _embed_kwargs = {}
        for k, v in ollama_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _embed_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        embed_kwargs = merge_dicts(_embed_kwargs, def_embed_kwargs, embed_kwargs)

        client = Client(**client_kwargs)
        model_installed = False
        for installed_model in client.list().models:
            if installed_model.model == model:
                model_installed = True
                break
        if not model_installed:
            pbar = None
            status = None
            for response in client.pull(model, stream=True):
                if pbar is not None and status is not None and response.status != status:
                    pbar.refresh()
                    pbar.exit()
                    pbar = None
                    status = None
                if response.completed is not None:
                    status = response.status
                    if pbar is None:
                        pbar = ProgressBar(total=response.total, show_progress=self.show_progress, **self.pbar_kwargs)
                        pbar.enter()
                    pbar.set_prefix(status)
                    pbar.update_to(response.completed)

        self._model = model
        self._client = client
        self._embed_kwargs = embed_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> tp.Optional[int]:
        return self.embed_kwargs.get("dimensions", None)

    @property
    def client(self) -> OllamaClientT:
        """Ollama client instance.

        Returns:
            Client: Ollama client instance.
        """
        return self._client

    @property
    def embed_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `ollama.Client.embed`.

        Returns:
            Kwargs: Keyword arguments for generating embeddings.
        """
        return self._embed_kwargs

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        response = self.client.embed(model=self.model, input=query, **self.embed_kwargs)
        return response["embeddings"][0]

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        response = self.client.embed(model=self.model, input=batch, **self.embed_kwargs)
        return list(response["embeddings"])


class LlamaIndexEmbeddings(Embeddings):
    """Embeddings class for LlamaIndex.

    This class initializes embeddings for LlamaIndex using a specified identifier or instance.
    It combines configuration from `vectorbtpro._settings.knowledge` with provided parameters.

    !!! info
        For default settings, see `chat.embeddings_configs.llama_index` in `vectorbtpro._settings.knowledge`.

    Args:
        embedding (Union[None, str, BaseEmbedding]): Embedding identifier or instance.

            If None, a default from settings is used.
        embedding_kwargs (KwargsLike): Keyword arguments for embedding initialization.
        **kwargs: Keyword arguments for `Embeddings` or used as `embedding_kwargs`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.llama_index"

    def __init__(
        self,
        embedding: tp.Union[None, str, tp.MaybeType[BaseEmbeddingT]] = None,
        embedding_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            embedding=embedding,
            embedding_kwargs=embedding_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.embeddings import BaseEmbedding

        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_embedding = llama_index_config.pop("embedding", None)
        def_embedding_kwargs = llama_index_config.pop("embedding_kwargs", None)

        if embedding is None:
            embedding = def_embedding
        if embedding is None:
            raise ValueError("Must provide an embedding name or path")
        init_arg_names = self.get_init_arg_names()
        for k in list(llama_index_config.keys()):
            if k in init_arg_names:
                llama_index_config.pop(k)

        if isinstance(embedding, str):
            import llama_index.embeddings
            from vectorbtpro.utils.module_ import search_package

            def _match_func(k, v):
                if isinstance(v, type) and issubclass(v, BaseEmbedding):
                    if "." in embedding:
                        if k.endswith(embedding):
                            return True
                    else:
                        if k.split(".")[-1].lower() == embedding.lower():
                            return True
                        if k.split(".")[-1].replace("Embedding", "").lower() == embedding.lower().replace("_", ""):
                            return True
                return False

            found_embedding = search_package(
                llama_index.embeddings,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_embedding is None:
                raise ValueError(f"Embedding {embedding!r} not found")
            embedding = found_embedding
        if isinstance(embedding, type):
            checks.assert_subclass_of(embedding, BaseEmbedding, arg_name="embedding")
            embedding_name = embedding.__name__.replace("Embedding", "").lower()
            module_name = embedding.__module__
        else:
            checks.assert_instance_of(embedding, BaseEmbedding, arg_name="embedding")
            embedding_name = type(embedding).__name__.replace("Embedding", "").lower()
            module_name = type(embedding).__module__
        embedding_configs = llama_index_config.pop("embedding_configs", {})
        if embedding_name in embedding_configs:
            llama_index_config = merge_dicts(llama_index_config, embedding_configs[embedding_name])
        elif module_name in embedding_configs:
            llama_index_config = merge_dicts(llama_index_config, embedding_configs[module_name])
        embedding_kwargs = merge_dicts(llama_index_config, def_embedding_kwargs, embedding_kwargs)
        model_name = embedding_kwargs.get("model_name", None)
        if model_name is None:
            func_kwargs = get_func_kwargs(type(embedding).__init__)
            model_name = func_kwargs.get("model_name", None)
        if isinstance(embedding, type):
            embedding = embedding(**embedding_kwargs)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized embedding")

        self._model = model_name
        self._embedding = embedding

    @property
    def model(self) -> tp.Optional[str]:
        return self._model

    @property
    def embedding(self) -> BaseEmbeddingT:
        """Underlying embedding instance.

        Returns:
            BaseEmbedding: Embedding instance.
        """
        return self._embedding

    def get_embedding_raw(self, query: str) -> tp.Sequence[float]:
        return self.embedding.get_text_embedding(query)

    def get_embedding_batch_raw(self, batch: tp.List[str]) -> tp.List[tp.Sequence[float]]:
        return list(self.embedding.get_text_embedding_batch(batch))


def resolve_embeddings(embeddings: tp.EmbeddingsLike = None) -> tp.MaybeType[Embeddings]:
    """Return a subclass or instance of `Embeddings` based on the provided identifier or object.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.

    Args:
        embeddings (EmbeddingsLike): Identifier, subclass, or instance of `Embeddings`.

            Supported identifiers:

            * "openai" for `OpenAIEmbeddings` (remote)
            * "gemini" for `GeminiEmbeddings` (remote)
            * "hf_inference" for `HFInferenceEmbeddings` (remote)
            * "voyage" for `VoyageEmbeddings` (remote)
            * "cohere" for `CohereEmbeddings` (remote)
            * "jina" for `JinaEmbeddings` (remote)
            * "litellm" for `LiteLLMEmbeddings` (remote)
            * "hf" for `HFEmbeddings` (local)
            * "ollama" for `OllamaEmbeddings` (local)
            * "llama_index" for `LlamaIndexEmbeddings` (remote or local)
            * For other options, see `vectorbtpro.knowledge.provider_utils.resolve_provider`

            If None, configuration from `vectorbtpro._settings` is used.

    Returns:
        Embeddings: Resolved embeddings subclass or instance.
    """
    if embeddings is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        embeddings = chat_cfg["embeddings"]
    if isinstance(embeddings, str):
        from vectorbtpro.utils.module_ import check_installed

        embeddings = resolve_provider(
            embeddings,
            remote_candidates=(
                ("openai", lambda: check_installed("openai") and os.getenv("OPENAI_API_KEY")),
                ("gemini", lambda: check_installed("google.genai") and os.getenv("GEMINI_API_KEY")),
                ("hf_inference", lambda: check_installed("huggingface_hub") and os.getenv("HF_TOKEN")),
                ("voyage", lambda: check_installed("voyageai") and os.getenv("VOYAGE_API_KEY")),
                ("cohere", lambda: check_installed("cohere") and os.getenv("CO_API_KEY")),
                ("jina", lambda: os.getenv("JINA_API_KEY")),
            ),
            local_candidates=(
                ("hf", lambda: check_installed("sentence_transformers")),
                ("ollama", check_ollama_available),
            ),
            remote_fallback_candidates=(
                ("litellm", lambda: check_installed("litellm")),
                ("openai", lambda: check_installed("openai")),
                ("gemini", lambda: check_installed("google.genai")),
                ("hf_inference", lambda: check_installed("huggingface_hub")),
                ("cohere", lambda: check_installed("cohere")),
                ("voyage", lambda: check_installed("voyageai")),
            ),
            fallback_candidates=(("llama_index", lambda: check_installed("llama_index")),),
            provider_name="embeddings",
        )
        if embeddings is None:
            raise ValueError(
                "No embeddings available. "
                "Please install one of the supported packages: "
                "sentence-transformers, "
                "openai, "
                "google-genai, "
                "litellm, "
                "huggingface-hub, "
                "voyageai, "
                "cohere, "
                "sentence-transformers, "
                "ollama, "
                "llama-index."
            )
        if embeddings.lower() == "anthropic":
            raise ValueError("Anthropic does not provide embeddings. Please use a different embeddings provider.")
        curr_module = sys.modules[__name__]
        found_embeddings = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Embeddings"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == embeddings.lower():
                    found_embeddings = cls
                    break
        if found_embeddings is None:
            raise ValueError(f"Invalid embeddings: {embeddings!r}")
        embeddings = found_embeddings
    if isinstance(embeddings, type):
        checks.assert_subclass_of(embeddings, Embeddings, arg_name="embeddings")
    else:
        checks.assert_instance_of(embeddings, Embeddings, arg_name="embeddings")
    return embeddings


def embed(query: tp.MaybeList[str], embeddings: tp.EmbeddingsLike = None, **kwargs) -> tp.MaybeList[Embedding]:
    """Return embedding(s) for one or more queries.

    Args:
        query (MaybeList[str]): Query string or a list of query strings to embed.
        embeddings (EmbeddingsLike): Identifier, subclass, or instance of `Embeddings`.

            Resolved using `resolve_embeddings`.
        **kwargs: Keyword arguments to initialize or update `embeddings`.

    Returns:
        MaybeList[Embedding]: Embedding instance(s) corresponding to the input query or queries.
    """
    embeddings = resolve_embeddings(embeddings=embeddings)
    if isinstance(embeddings, type):
        embeddings = embeddings(**kwargs)
    elif kwargs:
        embeddings = embeddings.replace(**kwargs)
    if isinstance(query, str):
        return embeddings.get_embedding(query)
    return embeddings.get_embeddings(query)
