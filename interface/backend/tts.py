import os
from pathlib import Path
from typing import List, Optional

import requests
from pydub import AudioSegment


def _split_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if current_len + len(s) + (1 if current else 0) <= max_chars:
            current.append(s)
            current_len += len(s) + (1 if current_len else 0)
        else:
            if current:
                chunks.append(" ".join(current))
            if len(s) <= max_chars:
                current = [s]
                current_len = len(s)
            else:
                # sentence longer than max; hard-split by spaces
                words = s.split()
                buf: List[str] = []
                buf_len = 0
                for w in words:
                    if buf_len + len(w) + (1 if buf else 0) <= max_chars:
                        buf.append(w)
                        buf_len += len(w) + (1 if buf_len else 0)
                    else:
                        chunks.append(" ".join(buf))
                        buf = [w]
                        buf_len = len(w)
                if buf:
                    chunks.append(" ".join(buf))
                current = []
                current_len = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def _ensure_parent_dir(output_file: str) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)


def _generate_azure_tts(text: str, output_file: str, voice: Optional[str]) -> str:
    _ensure_parent_dir(output_file)
    api_key = os.getenv("AZURE_TTS_KEY", "").strip()
    region = os.getenv("AZURE_TTS_REGION", "southeastasia").strip() or "southeastasia"
    if not api_key:
        raise RuntimeError("AZURE_TTS_KEY not set")
    endpoint = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": os.getenv(
            "AZURE_TTS_FORMAT", "audio-48khz-192kbitrate-mono-mp3"
        ),
    }
    voice_name = voice or os.getenv("AZURE_TTS_VOICE", "en-US-JennyNeural")
    # Basic escape for XML special chars
    esc = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    ssml = (
        f"<speak version='1.0' xml:lang='en-US'>"
        f"<voice xml:lang='en-US' name='{voice_name}'>"
        f"{esc}"
        f"</voice>"
        f"</speak>"
    )
    resp = requests.post(endpoint, headers=headers, data=ssml.encode("utf-8"))
    if resp.status_code != 200:
        raise RuntimeError(f"Azure TTS error: {resp.status_code} {resp.text}")
    with open(output_file, "wb") as f:
        f.write(resp.content)
    return output_file


def _generate_openai_tts(text: str, output_file: str, voice: Optional[str]) -> str:
    _ensure_parent_dir(output_file)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    voice_name = voice or os.getenv("OPENAI_TTS_VOICE", "alloy")
    endpoint = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "voice": voice_name,
        "input": text,
        "format": "mp3",
    }
    resp = requests.post(endpoint, headers=headers, json=payload, stream=True)
    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = {"message": resp.text}
        raise RuntimeError(f"OpenAI TTS error: {resp.status_code} {err}")
    with open(output_file, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return output_file


def _generate_local_tts(text: str, output_file: str, voice: Optional[str]) -> str:
    _ensure_parent_dir(output_file)
    try:
        from gtts import gTTS  # type: ignore
    except Exception as e:
        raise RuntimeError("gTTS not installed for local TTS") from e
    tts = gTTS(text=text, lang=os.getenv("LOCAL_TTS_LANG", "en"))
    tts.save(output_file)
    return output_file


def _generate_cloud_tts_chunked(text: str, output_file: str, provider: str, voice: Optional[str], max_chars: int) -> str:
    chunks = _split_text(text, max_chars)
    tmp_files: List[Path] = []
    ts = os.getpid()
    for i, chunk in enumerate(chunks):
        tmp_path = Path(output_file).with_name(f"{Path(output_file).stem}_part_{ts}_{i}.mp3")
        if provider == "azure":
            _generate_azure_tts(chunk, str(tmp_path), voice)
        elif provider == "openai":
            _generate_openai_tts(chunk, str(tmp_path), voice)
        else:
            raise ValueError(f"Unsupported cloud provider for chunking: {provider}")
        tmp_files.append(tmp_path)

    combined = AudioSegment.silent(duration=200)
    for p in tmp_files:
        seg = AudioSegment.from_file(str(p), format="mp3")
        combined += seg + AudioSegment.silent(duration=100)
    _ensure_parent_dir(output_file)
    combined.export(output_file, format="mp3")
    for p in tmp_files:
        try:
            p.unlink()
        except Exception:
            pass
    return output_file


def generate_audio(text, output_file, provider=None, voice=None):
    """
    Generate audio from text using the specified TTS provider.
    
    Args:
        text (str): Text to convert to speech
        output_file (str): Output file path
        provider (str, optional): TTS provider to use. Defaults to TTS_PROVIDER env var or "festival"
        voice (str, optional): Voice to use. Defaults to provider-specific default
    
    Returns:
        str: Path to the generated audio file
    
    Raises:
        RuntimeError: If TTS provider is not available or synthesis fails
        ValueError: If text is empty or invalid
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    provider = (provider or os.getenv("TTS_PROVIDER", "local")).lower()

    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Cloud input size limit handling via environment variable
    # TTS_CLOUD_MAX_CHARS: Maximum characters per request for cloud providers
    # Defaults to 3000 if not set. Local provider is never chunked.
    max_chars_env = os.getenv("TTS_CLOUD_MAX_CHARS", "3000")
    max_chars: Optional[int] = None
    try:
        max_chars = int(max_chars_env)
        if max_chars <= 0:
            max_chars = None
    except (TypeError, ValueError):
        max_chars = 3000

    if provider in ("azure", "openai") and max_chars and len(text) > max_chars:
        return _generate_cloud_tts_chunked(text, output_file, provider, voice, max_chars)

    if provider == "azure":
        return _generate_azure_tts(text, output_file, voice)
    elif provider == "openai":
        return _generate_openai_tts(text, output_file, voice)
    elif provider == "local":
        return _generate_local_tts(text, output_file, voice)
    else:
        raise ValueError(f"Unsupported TTS_PROVIDER: {provider}")


