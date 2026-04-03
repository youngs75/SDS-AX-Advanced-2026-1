from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.llm.deepeval_model import OpenRouterModel
from src.llm.openrouter import get_langchain_chat_model


class TestOpenRouterModel:
    def test_model_name(self):
        with patch("src.llm.deepeval_model.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                llm_provider="openrouter",
                openrouter_api_key="test-key",
                openrouter_model_name="openai/gpt-4.1",
                openai_api_key="",
                openai_model_name="gpt-4.1",
            )
            model = OpenRouterModel(model_name="openai/gpt-4.1", api_key="test-key")
            assert model.get_model_name() == "openai/gpt-4.1"

    def test_generate(self):
        with patch("src.llm.deepeval_model.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                llm_provider="openrouter",
                openrouter_api_key="test-key",
                openrouter_model_name="openai/gpt-4.1",
                openai_api_key="",
                openai_model_name="gpt-4.1",
            )
            model = OpenRouterModel(model_name="test-model", api_key="test-key")

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]

            with patch.object(model._client.chat.completions, "create", return_value=mock_response):
                result = model.generate("test prompt")
                assert result == "test response"

    def test_generate_with_schema(self):
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            answer: str

        with patch("src.llm.deepeval_model.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                llm_provider="openrouter",
                openrouter_api_key="test-key",
                openrouter_model_name="openai/gpt-4.1",
                openai_api_key="",
                openai_model_name="gpt-4.1",
            )
            model = OpenRouterModel(model_name="test-model", api_key="test-key")

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content='{"answer": "hello"}'))]

            with patch.object(model._client.chat.completions, "create", return_value=mock_response):
                result = model.generate("test", schema=TestSchema)
                assert isinstance(result, TestSchema)
                assert result.answer == "hello"

    def test_model_name_when_provider_openai(self):
        with patch("src.llm.deepeval_model.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                llm_provider="openai",
                openrouter_api_key="",
                openrouter_model_name="openai/gpt-4.1",
                openai_api_key="test-openai-key",
                openai_model_name="gpt-4.1",
            )
            model = OpenRouterModel(api_key="test-openai-key")
            assert model.get_model_name() == "gpt-4.1"

    def test_get_langchain_chat_model_when_provider_openai(self):
        settings = MagicMock(
            llm_provider="openai",
            openrouter_api_key="",
            openrouter_model_name="openai/gpt-4.1",
            openai_api_key="test-openai-key",
            openai_model_name="gpt-5.4-mini",
        )

        model = get_langchain_chat_model(settings=settings)

        assert model.model_name == "gpt-5.4-mini"
