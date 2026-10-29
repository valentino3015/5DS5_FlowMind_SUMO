"""Gemini transport with explicit token limits and sanitized failures."""
from __future__ import annotations

import os
import logging
from pathlib import Path


class GeminiError(RuntimeError):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


def load_environment(root: Path):
    # Only this project's file; never search parent folders for credentials.
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(root) / ".env", override=False)
    except ImportError:
        pass


class GeminiClient:
    def __init__(self):
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise GeminiError("Gemini non configuré : définissez GEMINI_API_KEY côté serveur puis redémarrez le dashboard.", 503)
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise GeminiError("SDK Gemini absent : installez requirements.txt.", 503) from None
        self.types = types
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=35000, retry_options=types.HttpRetryOptions(attempts=1)))

    def close(self):
        self.client.close()

    def _call(self, operation, _retry_count=True, **kwargs):
        try:
            return getattr(self.client.models, operation)(model=self.model, **kwargs)
        except Exception as exc:
            # A dropped connection during the read-only count can be retried
            # without duplicating a billed generation. Never retry generate.
            from httpx import TransportError
            if operation == "count_tokens" and _retry_count and isinstance(exc, TransportError):
                return self._call(operation, _retry_count=False, **kwargs)
            code = getattr(exc, "code", None)
            logging.getLogger(__name__).warning("Gemini %s failed: %s (code=%s)",
                                               operation, type(exc).__name__, code)
            if code in (401, 403):
                raise GeminiError("Clé Gemini refusée ou non autorisée. Vérifiez la configuration côté serveur.", 503) from None
            if code == 429:
                raise GeminiError("Quota Gemini atteint. Réessayez plus tard ou vérifiez le quota du projet.", 429) from None
            if code == 404:
                raise GeminiError("Modèle Gemini indisponible. Vérifiez GEMINI_MODEL côté serveur.", 503) from None
            # SDK exceptions may include request details: never echo them to users.
            raise GeminiError("Gemini indisponible ou délai dépassé. Réessayez plus tard.", 502) from None

    def count_tokens(self, system, prompt):
        # Developer API's SDK count_tokens rejects system_instruction (Vertex
        # accepts it). Count all text together and reserve framing overhead.
        response = self._call("count_tokens", contents=system + "\n" + prompt)
        if response.total_tokens is None:
            raise GeminiError("Comptage des tokens Gemini indisponible.")
        return response.total_tokens + 64

    def generate(self, system, prompt, max_output_tokens):
        config = self.types.GenerateContentConfig(
            system_instruction=system, temperature=0.2, max_output_tokens=max_output_tokens)
        # 2.5 Flash permits disabling thinking; other families support minimal.
        if self.model.startswith("gemini-2.5-flash"):
            config.thinking_config = self.types.ThinkingConfig(thinking_budget=0)
        elif self.model.startswith("gemini-3"):
            config.thinking_config = self.types.ThinkingConfig(thinking_level="minimal")
        response = self._call("generate_content", contents=prompt, config=config)
        candidates = response.candidates or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        finish = getattr(finish, "value", str(finish))
        if finish == "MAX_TOKENS":
            raise GeminiError("Réponse Gemini interrompue par la limite de tokens. Posez une question plus précise.")
        if finish != "STOP" or not response.text or not response.text.strip():
            raise GeminiError("Gemini n’a pas fourni de réponse exploitable. Reformulez la question.")
        usage = response.usage_metadata
        return {"answer": response.text.strip(), "usage": {
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "thinking_tokens": getattr(usage, "thoughts_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None)}, "model": self.model}
