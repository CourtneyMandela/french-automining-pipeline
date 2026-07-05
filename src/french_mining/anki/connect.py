"""Thin client for AnkiConnect (https://foosoft.net/projects/anki-connect/).

This is the only bridge to Anki. It must be run on the same machine as Anki,
with the AnkiConnect add-on installed and Anki open.
"""
from __future__ import annotations

import os
from typing import Any

import requests


class AnkiConnectError(RuntimeError):
    """Raised when AnkiConnect returns an error for a request."""


class AnkiConnectClient:
    def __init__(self, url: str | None = None, api_version: int | None = None, timeout: float = 30.0):
        self.url = url or os.environ.get("ANKICONNECT_URL", "http://127.0.0.1:8765")
        self.api_version = api_version or int(os.environ.get("ANKICONNECT_API_VERSION", "6"))
        self.timeout = timeout

    def invoke(self, action: str, **params: Any) -> Any:
        payload = {"action": action, "version": self.api_version}
        if params:
            payload["params"] = params

        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise AnkiConnectError(
                f"Could not reach AnkiConnect at {self.url}. Is Anki running with the "
                "AnkiConnect add-on installed?"
            ) from exc

        response.raise_for_status()
        body = response.json()

        if len(body) != 2 or "error" not in body or "result" not in body:
            raise AnkiConnectError(f"Unexpected AnkiConnect response shape: {body!r}")
        if body["error"] is not None:
            raise AnkiConnectError(body["error"])
        return body["result"]

    # -- Convenience wrappers over the actions this pipeline actually needs --

    def deck_names(self) -> list[str]:
        return self.invoke("deckNames")

    def model_names(self) -> list[str]:
        return self.invoke("modelNames")

    def model_field_names(self, model_name: str) -> list[str]:
        return self.invoke("modelFieldNames", modelName=model_name)

    def create_model(
        self,
        model_name: str,
        in_order_fields: list[str],
        card_templates: list[dict[str, str]],
        css: str = "",
        is_cloze: bool = False,
    ) -> Any:
        return self.invoke(
            "createModel",
            modelName=model_name,
            inOrderFields=in_order_fields,
            css=css,
            isCloze=is_cloze,
            cardTemplates=card_templates,
        )

    def find_notes(self, query: str) -> list[int]:
        return self.invoke("findNotes", query=query)

    def notes_info(self, note_ids: list[int]) -> list[dict[str, Any]]:
        if not note_ids:
            return []
        return self.invoke("notesInfo", notes=note_ids)

    def find_cards(self, query: str) -> list[int]:
        return self.invoke("findCards", query=query)

    def cards_info(self, card_ids: list[int]) -> list[dict[str, Any]]:
        if not card_ids:
            return []
        return self.invoke("cardsInfo", cards=card_ids)

    def add_note(self, note: dict[str, Any]) -> int:
        return self.invoke("addNote", note=note)

    def set_specific_value_of_card(
        self, card_id: int, key: str, new_value: str, warning_check: bool = True
    ) -> Any:
        """Rewrite a raw card field (e.g. `due`) for new-card queue reordering.

        This is the mechanism behind the "Claude reorders the queue" principle:
        FSRS never sees a card until its first review, so until then the queue is
        just an integer `due` ordering we are free to rewrite.
        """
        return self.invoke(
            "setSpecificValueOfCard",
            card=card_id,
            keys=[key],
            newValues=[new_value],
            warning_check=warning_check,
        )

    def store_media_file(self, filename: str, path: str) -> str:
        """Copy a local file (source audio clip, source frame, TTS output)
        into Anki's media folder. `path` is an absolute path on the same
        machine AnkiConnect runs on — since this pipeline only ever runs
        there too, there's no need to read+base64-encode the file ourselves.

        Returns the filename actually used (AnkiConnect deduplicates by
        content hash and may return a different name than requested).
        """
        return self.invoke("storeMediaFile", filename=filename, path=path)
