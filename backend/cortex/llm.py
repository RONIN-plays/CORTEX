"""LLM calls through Groq: cited answers over retrieved notes, and image transcription."""

import base64
import os
from collections.abc import AsyncIterator

import groq

from . import config

ANSWER_SYSTEM = """You are Cortex, an assistant that answers questions from the user's own saved notes.

Each request contains numbered sources retrieved from the user's notes, followed by their question. Retrieval is imperfect: some sources may be irrelevant, and you should ignore those.

Ground every claim in the sources. Cite the source a claim comes from by putting its number in square brackets right after the claim, like [2], or [1][3] when several support it. The interface turns these numbers into links to the original notes, so cite only numbers that appear in the sources. Use plain ASCII square brackets exactly as shown, never other bracket characters.

If the sources do not contain the answer, say so plainly, and mention anything related that they do contain. You may add general knowledge when it helps, but mark it clearly as not from the notes and do not attach a citation to it.

Answer directly and concisely, in Markdown. Write any maths as plain text, not LaTeX."""

IMAGE_PROMPT = (
    "This image is being saved into a personal notes app so it can be searched later. "
    "Transcribe all text in it exactly, keeping its structure. Then, under a heading "
    "'Description', describe what the image shows (diagrams, charts, UI, photos) in enough "
    "detail that someone searching for it by meaning would find it. Output only the "
    "transcription and description."
)

NO_KEY = (
    "No Groq API key is set, so Cortex can search but not {what}. "
    "Add GROQ_API_KEY to the .env file and restart the server."
)


class LLMError(Exception):
    """An LLM call failed. The message is safe to show the user."""


def has_credentials() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def _friendly(exc: Exception, model: str) -> str:
    if isinstance(exc, groq.AuthenticationError):
        return "The Groq API key was rejected. Check GROQ_API_KEY in your .env file."
    if isinstance(exc, groq.PermissionDeniedError):
        return f"This Groq key is not allowed to use the model '{model}'."
    if isinstance(exc, groq.NotFoundError):
        return f"Groq does not have a model called '{model}'. Check the model setting in .env."
    if isinstance(exc, groq.RateLimitError):
        return "Groq's free-tier rate limit was reached. Wait a minute and try again."
    if isinstance(exc, groq.APIStatusError):
        if exc.status_code == 413:
            return "The request was too large for this Groq model."
        if exc.status_code >= 500:
            return "Groq is having trouble right now. Try again shortly."
        return f"Groq rejected the request: {exc.message}"
    if isinstance(exc, groq.APIConnectionError):
        return "Could not reach Groq. Check your internet connection."
    return f"LLM call failed: {exc}"


def format_sources(hits: list[dict]) -> str:
    parts = []
    for number, hit in enumerate(hits, start=1):
        location = []
        if hit.get("heading"):
            location.append(f'section="{hit["heading"]}"')
        if hit.get("page"):
            location.append(f'page="{hit["page"]}"')
        attrs = " ".join([f'id="{number}"', f'title="{hit["title"]}"', *location])
        parts.append(f"<source {attrs}>\n{hit['text']}\n</source>")
    return "<sources>\n" + "\n".join(parts) + "\n</sources>"


def build_messages(question: str, hits: list[dict], history: list[dict]) -> list[dict]:
    """The conversation so far, then the retrieved sources and the new question."""
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history[-10:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    sources = format_sources(hits) if hits else "<sources>\n(no notes matched)\n</sources>"
    messages.append({"role": "user", "content": f"{sources}\n\nQuestion: {question}"})
    return messages


async def stream_answer(
    question: str, hits: list[dict], history: list[dict]
) -> AsyncIterator[dict]:
    """Yield {"type": "delta", "text": ...} events, then a final "done" or "error" event."""
    if not has_credentials():
        yield {"type": "error", "message": NO_KEY.format(what="write answers")}
        return
    finish_reason = None
    try:
        async with groq.AsyncGroq() as client:
            stream = await client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM},
                    *build_messages(question, hits, history),
                ],
                max_completion_tokens=8192,
                temperature=0.2,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta.content:
                    yield {"type": "delta", "text": choice.delta.content}
                finish_reason = choice.finish_reason or finish_reason
    except groq.APIError as exc:
        yield {"type": "error", "message": _friendly(exc, config.LLM_MODEL)}
        return
    yield {"type": "done", "truncated": finish_reason == "length"}


def describe_image(data: bytes, media_type: str) -> str:
    """Transcribe and describe an image so it can be indexed like any other note."""
    if not has_credentials():
        raise LLMError(NO_KEY.format(what="read images"))
    encoded = base64.b64encode(data).decode("ascii")
    try:
        with groq.Groq() as client:
            completion = client.chat.completions.create(
                model=config.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": IMAGE_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                            },
                        ],
                    }
                ],
                max_completion_tokens=4096,
            )
    except groq.APIError as exc:
        raise LLMError(_friendly(exc, config.VISION_MODEL)) from exc
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise LLMError("The vision model returned no text for this image.")
    return text
