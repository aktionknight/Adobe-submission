import os
import io
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse

# Load environment variables
load_dotenv()

# Import 1B logic
import sys
# BASE_DIR should be the 'interface' directory
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))  # interface/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # repo root
ONE_B_DIR = os.path.join(PROJECT_ROOT, 'ADOBE', '1B')
ONE_A_DIR = os.path.join(PROJECT_ROOT, 'ADOBE', '1A')
if ONE_B_DIR not in sys.path:
    sys.path.insert(0, ONE_B_DIR)
if ONE_A_DIR not in sys.path:
    sys.path.insert(0, ONE_A_DIR)

from direct_1a_import import extract_1a_data, extract_section_content  # type: ignore
from section_ranker import SectionRanker  # type: ignore
import google.generativeai as genai  # type: ignore
try:
    from google.cloud import texttospeech as gcloud_tts  # type: ignore
except Exception:
    gcloud_tts = None  # type: ignore
try:
    from gtts import gTTS  # type: ignore
except Exception:
    gTTS = None  # type: ignore
import time
from pathlib import Path

app = FastAPI(title="Adobe 1B Interface API", version="0.1.0")

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
INTERFACE_DIR = BASE_DIR  # interface/
UPLOADS_DIR = os.path.join(INTERFACE_DIR, 'uploads')
UPLOADS_VO_DIR = os.path.join(UPLOADS_DIR, 'VO')
FRONTEND_DIR = os.path.join(INTERFACE_DIR, 'frontend')


os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(UPLOADS_VO_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Static mounts
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app/")

@app.get('/api/ping')
def ping() -> Dict[str, Any]:
    return {"ping": "ok"}

@app.get('/api/config')
def get_config() -> Dict[str, Any]:
    """Get configuration including Adobe API key"""
    return {
        "adobe_api_key": os.getenv("ADOBE_EMBED_API_KEY", ""),
        "server_info": {
            "host": os.getenv("HOST", "localhost"),
            "port": int(os.getenv("PORT", "8080"))
        }
    }

# In-memory cache of analysis results per file hash
class AnalysisCache:
    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._background_processed: bool = False

    @staticmethod
    def _hash_path(path: str) -> str:
        try:
            stat = os.stat(path)
            sig = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
        except FileNotFoundError:
            sig = path
        return hashlib.sha256(sig.encode("utf-8")).hexdigest()

    def get(self, path: str) -> Optional[Dict[str, Any]]:
        key = self._hash_path(path)
        return self._cache.get(key)

    def set(self, path: str, value: Dict[str, Any]) -> None:
        key = self._hash_path(path)
        self._cache[key] = value

    def is_background_processed(self) -> bool:
        return self._background_processed

    def set_background_processed(self, value: bool) -> None:
        self._background_processed = value

    def get_all_sections(self) -> List[Dict[str, Any]]:
        """Get all sections from all cached PDFs for cross-document search"""
        all_sections = []
        for cached_data in self._cache.values():
            sections = cached_data.get("sections", [])
            file_path = cached_data.get("file_path", "")

            # Use actual filename from uploads directory
            file_name = os.path.basename(file_path) if file_path else "Unknown.pdf"

            for section in sections:
                all_sections.append({
                    "document": file_name,   # 👈 real file name, safe to map to /uploads/
                    "title": section["title"],
                    "content": section["content"],
                    "page": section["page"],
                    "file_path": file_path,
                    "pdf_title": cached_data.get("title", file_name)  # optional: keep extracted title
                })
        return all_sections



analysis_cache = AnalysisCache()
ranker_singleton = SectionRanker()


def process_all_pdfs_background():
    """Process all uploaded PDFs in the background to cache their sections"""
    if analysis_cache.is_background_processed():
        return

    try:
        for fname in os.listdir(UPLOADS_DIR):
            if not fname.lower().endswith('.pdf'):
                continue
            pdf_path = os.path.join(UPLOADS_DIR, fname)
            safe_name = os.path.basename(fname)
            encoded_name = quote(safe_name)
            pdf_title, headings = extract_1a_data(pdf_path)
            sections_raw = extract_section_content(pdf_path, headings)
            # PATCH: ensure each section carries safe filename and file_url
            sections = []
            for s in sections_raw:
                section = {
                    "title": s["title"],
                    "content": s["content"],
                    "page": s["page"],
                    "document": safe_name,  # actual filename
                    "file_url": f"/uploads/{encoded_name}",  # usable in frontend
                    "display_name": pdf_title  # optional pretty title
                }
                sections.append(section)
            analysis_cache.set(pdf_path, {
                "title": pdf_title,
                "headings": headings,
                "sections": sections,
                "file_path": pdf_path
            })
        analysis_cache.set_background_processed(True)
    except Exception as e:
        print(f"Background processing failed: {e}")

def get_recommendations_for_text(
    selected_text: str,
    persona: str,
    job_to_be_done: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """Get recommendations based ONLY on selected text."""
    process_all_pdfs_background()
    all_sections = analysis_cache.get_all_sections()
    if not all_sections:
        return {
            "recommendations": [],
            "message": "No documents available for analysis"
        }

    query_text = selected_text.strip()

    import torch
    with torch.no_grad():
        device = torch.device("cpu")
        query_embedding = ranker_singleton.model.encode(
            [query_text], convert_to_tensor=True, convert_to_numpy=True, device=device
        )[0]

    # Encode each section separately (title + its own content only)
    section_texts = []
    for section in all_sections:
        title = section["title"].strip()
        content = section["content"].strip()
        context = f"{title}. {content}"
        section_texts.append(context)

    section_embeddings = ranker_singleton.model.encode(
        section_texts, convert_to_tensor=True, convert_to_numpy=True, device=torch.device("cpu")
    )

    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_embedding], section_embeddings)[0]

    # Rank results
    ranked_sections = []
    for section, similarity in zip(all_sections, similarities):
        preview = section["content"].strip()
        if len(preview) > 300:
            preview = preview[:300] + "..."
        # Use actual filename for file_url
        safe_name = section.get("document", "")
        ranked_sections.append({
            "document": safe_name,
            "file_url": f"/uploads/{quote(safe_name)}",
            "section_title": section["title"],
            "page_number": section["page"],
            "similarity": float(similarity),
            "content_preview": preview
})

    ranked_sections.sort(key=lambda x: x["similarity"], reverse=True)
    top_recommendations = ranked_sections[:top_k]

    for rec in top_recommendations:
        if "document" in rec:
            safe_name = rec["document"]
            rec["file_url"] = f"/uploads/{quote(safe_name)}"

    return {
        "recommendations": top_recommendations,
        "query_text": selected_text,
        "total_sections_analyzed": len(all_sections)

    }



def analyze_pdf(
    pdf_path: str,
    persona: str,
    job_to_be_done: str,
    description: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run the 1B pipeline for a single PDF and return results."""
    cached = analysis_cache.get(pdf_path)
    if cached:
        # We still want to re-rank by persona/job, but keep extracted sections
        sections = cached.get("sections", [])
    else:
        title, headings = extract_1a_data(pdf_path)
        sections = extract_section_content(pdf_path, headings)
        cached = {
            "title": title,
            "headings": headings,
            "sections": sections,
        }
        analysis_cache.set(pdf_path, cached)

    # Rank sections
    ranked = ranker_singleton.rank_sections(
        [
            {
                "document": os.path.basename(pdf_path),
                "title": s["title"],
                "content": s["content"],
                "page": s["page"],
            }
            for s in sections
        ],
        persona,
        job_to_be_done,
        description,
        top_k=top_k,
        expected_titles=None,
    )

    extracted_sections = []
    for i, r in enumerate(ranked):
        sec = r["section"]
        extracted_sections.append(
            {
                "document": sec["document"],
                "section_title": sec["title"],
                "importance_rank": i + 1,
                "page_number": sec["page"],
                "similarity": float(r.get("similarity_score", 0.0)),
            }
        )

    query_embedding, query_text = ranker_singleton.create_query_embedding(
        persona, job_to_be_done
    )

    subsection_analysis: List[Dict[str, Any]] = []
    for r in ranked:
        sec = r["section"]
        paras = ranker_singleton.extract_subsection_content(
            sec["content"], query_embedding, query_text=query_text
        )
        if not paras:
            continue
        for para in paras[:3]:
            # Create a short snippet (1-2 sentences, <= 300 chars)
            snippet = para.strip()
            if len(snippet) > 300:
                snippet = snippet[:300].rstrip() + "..."
            subsection_analysis.append(
                {
                    "document": sec["document"],
                    "refined_text": snippet,
                    "page_number": sec["page"],
                }
            )
            if len(subsection_analysis) >= top_k:
                break
        if len(subsection_analysis) >= top_k:
            break

    return {
        "metadata": {
            "input_documents": [os.path.basename(pdf_path)],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis,
        "title": cached.get("title"),
    }


def compute_related_sections(
    current_pdf: str,
    persona: str,
    job_to_be_done: str,
    description: str,
    top_related_per_section: int = 3,
) -> List[Dict[str, Any]]:
    """Find related sections from other uploaded PDFs for the current doc's top sections."""
    # Analyze current
    current = analyze_pdf(current_pdf, persona, job_to_be_done, description, top_k=5)
    current_top = current["extracted_sections"]

    # Collect sections from other PDFs
    other_sections: List[Tuple[str, Dict[str, Any]]] = []
    for fname in os.listdir(UPLOADS_DIR):
        if not fname.lower().endswith('.pdf'):
            continue
        other_path = os.path.join(UPLOADS_DIR, fname)
        if os.path.abspath(other_path) == os.path.abspath(current_pdf):
            continue
        try:
            other = analysis_cache.get(other_path)
            if not other:
                title, headings = extract_1a_data(other_path)
                sections = extract_section_content(other_path, headings)
                analysis_cache.set(other_path, {
                    "title": title,
                    "headings": headings,
                    "sections": sections,
                })
            else:
                sections = other.get("sections", [])
            for s in sections:
                other_sections.append((fname, s))
        except Exception:
            continue

    # Build embeddings for candidate sections
    query_embedding, query_text = ranker_singleton.create_query_embedding(
        persona, job_to_be_done
    )

    # For each current top section, find most similar in others
    results: List[Dict[str, Any]] = []
    for top_sec in current_top:
        this_title = top_sec["section_title"]
        this_page = top_sec["page_number"]
        this_doc = top_sec["document"]
        this_content = None
        # Locate content in cache for embedding context
        cached = analysis_cache.get(current_pdf)
        if cached:
            for s in cached.get("sections", []):
                if s["title"] == this_title and s["page"] == this_page:
                    this_content = s["content"]
                    break
        if not this_content:
            continue
        # Build short context
        this_context = this_title + ". " + this_content[:1000]
        # Encode this section
        import torch
        with torch.no_grad():
            device = torch.device('cpu')
            this_emb = ranker_singleton.model.encode([this_context], convert_to_tensor=True, convert_to_numpy=True, device=device)[0]
        # Compare to other sections
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        if other_sections:
            other_texts = [title + ". " + s["content"][:1000] for title, s in [(s[1]["title"], s[1]) for s in other_sections]]
            with torch.no_grad():
                other_embs = ranker_singleton.model.encode(other_texts, convert_to_tensor=True, convert_to_numpy=True, device=device)
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity([this_emb], other_embs)[0]
            for (fname, s), sim in zip(other_sections, sims):
                candidates.append((float(sim), {
                    "document": fname,
                    "section_title": s["title"],
                    "page_number": s["page"],
                    "similarity": float(sim),
                }))
        top_related = sorted(candidates, key=lambda x: x[0], reverse=True)[:top_related_per_section]
        results.append({
            "source": {
                "document": this_doc,
                "section_title": this_title,
                "page_number": this_page,
            },
            "related": [c[1] for c in top_related],
        })
    return results

def _truncate(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""
    t = text.strip()
    return t if len(t) <= max_chars else t[:max_chars].rstrip() + "..."

def _safe_json_parse(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    # try extract fenced code block
    import re
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # try extract first {...}
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except Exception:
            pass
    return {
        "key_insights": [],
        "did_you_know_facts": [],
        "contradictions_or_counterpoints": [],
        "inspirations_or_connections": [],
    }

def _ensure_cached(pdf_path: str) -> Dict[str, Any]:
    cached = analysis_cache.get(pdf_path)
    if cached:
        return cached
    title, headings = extract_1a_data(pdf_path)
    sections = extract_section_content(pdf_path, headings)
    cached = {"title": title, "headings": headings, "sections": sections}
    analysis_cache.set(pdf_path, cached)
    return cached

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

@app.post("/api/insights")
async def insights(
    filename: str = Form(...),
    persona: str = Form(""),
    job_to_be_done: str = Form(""),
    text: str = Form(""),  # New parameter for selected text
):
    api_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_APPLICATION_CREDENTIALS env var not set")
    pdf_path = os.path.join(UPLOADS_DIR, os.path.basename(filename))
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found. Upload first.")

    try:
        # Ensure current document cached
        cached = _ensure_cached(pdf_path)
        sections = cached.get("sections", [])
        
        # If text is provided, focus on sections containing that text
        if text and text.strip():
            # Find sections that contain the selected text
            relevant_sections = []
            for s in sections:
                if text.lower() in s["content"].lower():
                    relevant_sections.append(s)
            
            if relevant_sections:
                sections = relevant_sections
            # If no exact matches, use semantic search
            else:
                # Use the 1B model to find semantically similar sections
                query_embedding, query_text = ranker_singleton.create_query_embedding(
                    persona, job_to_be_done
                )
                
                # Create embeddings for all sections
                section_texts = [f"{s['title']}. {s['content'][:500]}" for s in sections]
                import torch
                with torch.no_grad():
                    device = torch.device('cpu')
                    section_embeddings = ranker_singleton.model.encode(
                        section_texts, convert_to_tensor=True, convert_to_numpy=True, device=device
                    )
                
                # Calculate similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity([query_embedding], section_embeddings)[0]
                
                # Sort sections by similarity to selected text
                section_similarities = list(zip(sections, similarities))
                section_similarities.sort(key=lambda x: x[1], reverse=True)
                sections = [s[0] for s in section_similarities[:5]]  # Top 5 most similar
        
        # Focus on top sections using ranker for better context
        ranked = ranker_singleton.rank_sections(
            [
                {
                    "document": os.path.basename(pdf_path),
                    "title": s["title"],
                    "content": s["content"],
                    "page": s["page"],
                }
                for s in sections
            ],
            persona,
            job_to_be_done,
            description="",
            top_k=8,
            expected_titles=None,
        )
        current_doc_context = []
        for r in ranked:
            sec = r["section"]
            current_doc_context.append(
                f"Section: {sec['title']} (p.{sec['page']})\n" + _truncate(sec["content"], 1500)
            )

        # Build connections context from other PDFs
        related = compute_related_sections(pdf_path, persona, job_to_be_done, description="")
        connections_context: List[str] = []
        for group in related:
            for rel in group.get("related", []):
                other_path = os.path.join(UPLOADS_DIR, os.path.basename(rel["document"]))
                other_cached = _ensure_cached(other_path)
                # locate content for this section
                found = None
                for s in other_cached.get("sections", []):
                    if s["title"] == rel["section_title"] and s["page"] == rel["page_number"]:
                        found = s
                        break
                if not found and other_cached.get("sections"):
                    # fallback: match by title only
                    for s in other_cached["sections"]:
                        if s["title"] == rel["section_title"]:
                            found = s
                            break
                snippet = _truncate(found["content"], 600) if found else ""
                connections_context.append(
                    f"From {rel['document']} • {rel['section_title']} (p.{rel['page_number']}):\n{snippet}"
                )

        # Configure Gemini (use a widely supported model and JSON mime type)
        genai.configure(api_key=api_key)
        
        # System instructions: force LLM to always provide at least 2 items for each key
        system_instructions = (
            "You are an analytical assistant. Given selected text and document excerpts, "
            "produce JSON with arrays for: key_insights, did_you_know_facts, contradictions_or_counterpoints, inspirations_or_connections. "
            "You Must Provide atleast 2 items for each array, even if you need to infer or generalize."
            "If not enough relevant info, provide your best attempt with shorter inferences."
            "For contradictions_or_counterparts, look for factual inconsistencies, different interpretations, or alternative perspectives, "
            "even if they come from separate documents or sections. If no clear contradictions exist, infer possible counterpoints."
            "For each claim, suggest an alternate interpretation or scenario where it may not be true."
        )
        prompt = (
            f"Selected Text: {text.strip() or 'N/A'}\n\n"
            f"Persona: {persona or 'N/A'}\n"
            f"JobToBeDone: {job_to_be_done or 'N/A'}\n\n"
            "Current Document Context:\n" + "\n\n".join(current_doc_context[:8]) + "\n\n"
        )

        if connections_context:
            prompt += "Connections Candidates (from other docs):\n" + "\n\n".join(connections_context[:12]) + "\n\n"

        prompt += (
            "Return ONLY valid JSON with these keys exactly: \n"
            "{\n"
            "  \"key_insights\": string[],\n"
            "  \"did_you_know_facts\": string[],\n"
            "  \"contradictions_or_counterpoints\": string[],\n"
            "  \"inspirations_or_connections\": string[]\n"
            "}\n"
            "You MUST provide at least 2 items for each array, even if you need to infer or generalize. Max 6 items per list."
        )

        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_instructions,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.3,
                "max_output_tokens": 1024,
            },
        )

        response = model.generate_content(prompt)

        text = None
        if hasattr(response, "text") and response.text:
            text = response.text
        elif getattr(response, "candidates", None):
            for c in response.candidates:
                if c.content and c.content.parts:
                    for p in c.content.parts:
                        if hasattr(p, "text") and p.text:
                            text = p.text
                            break
                if text:
                    break

        # If blocked, Gemini returns no candidates or empty text
        if not text:
            # Check for safety ratings if available
            safety = getattr(response, "candidates", None)
            if safety and hasattr(safety[0], "safety_ratings"):
                ratings = safety[0].safety_ratings
                reasons = [r.category for r in ratings if getattr(r, "blocked", False)]
                reason_str = ", ".join(reasons) if reasons else "unknown"
                raise HTTPException(
                    status_code=500,
                    detail=f"Gemini blocked the response for safety reasons: {reason_str}"
                )
            # Generic fallback
            raise HTTPException(
                status_code=500,
                detail="Gemini did not return any content. The response may have been blocked for safety reasons."
            )
        data = _safe_json_parse(text or "")
        # ensure keys
        for k in [
            "key_insights",
            "did_you_know_facts",
            "contradictions_or_counterpoints",
            "inspirations_or_connections",
        ]:
            if k not in data or not isinstance(data[k], list):
                data[k] = []
        return JSONResponse(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights failed: {str(e)}")

'''
@app.post("/api/podcast")
async def podcast(
    filename: str = Form(...),
    persona: str = Form(""),
    job_to_be_done: str = Form(""),
    text: str = Form(""),  # New parameter for selected text
):
    api_key = os.getenv("GEMINI_API", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API env var not set")
    pdf_path = os.path.join(UPLOADS_DIR, os.path.basename(filename))
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found. Upload first.")

    # Step 1: get insights JSON by calling the same logic as /api/insights
    try:
        # Reuse insights generation
        fake_form = {"filename": filename, "persona": persona, "job_to_be_done": job_to_be_done, "text": text}
        # Call internal function by simulating request: easiest is to reuse blocks here
        # Ensure cache and ranked contexts
        cached = _ensure_cached(pdf_path)
        sections = cached.get("sections", [])
        
        # If text is provided, focus on sections containing that text
        if text and text.strip():
            # Find sections that contain the selected text
            relevant_sections = []
            for s in sections:
                if text.lower() in s["content"].lower():
                    relevant_sections.append(s)
            
            if relevant_sections:
                sections = relevant_sections
            # If no exact matches, use semantic search
            else:
                # Use the 1B model to find semantically similar sections
                query_embedding, query_text = ranker_singleton.create_query_embedding(
                    persona, job_to_be_done
                )
                
                # Create embeddings for all sections
                section_texts = [f"{s['title']}. {s['content'][:500]}" for s in sections]
                import torch
                with torch.no_grad():
                    section_embeddings = ranker_singleton.model.encode(
                        section_texts, convert_to_tensor=True, convert_to_numpy=True
                    )
                
                # Calculate similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity([query_embedding], section_embeddings)[0]
                
                # Sort sections by similarity to selected text
                section_similarities = list(zip(sections, similarities))
                section_similarities.sort(key=lambda x: x[1], reverse=True)
                sections = [s[0] for s in section_similarities[:5]]  # Top 5 most similar
        
        ranked = ranker_singleton.rank_sections(
            [
                {"document": os.path.basename(pdf_path), "title": s["title"], "content": s["content"], "page": s["page"]}
                for s in sections
            ],
            persona,
            job_to_be_done,
            description="",
            top_k=8,
            expected_titles=None,
        )
        current_doc_context = [
            f"Section: {r['section']['title']} (p.{r['section']['page']})\n" + _truncate(r['section']["content"], 1200)
            for r in ranked
        ]
        related = compute_related_sections(pdf_path, persona, job_to_be_done, description="")
        connections_context: List[str] = []
        for group in related:
            for rel in group.get("related", []):
                other_path = os.path.join(UPLOADS_DIR, os.path.basename(rel["document"]))
                other_cached = _ensure_cached(other_path)
                found = None
                for s in other_cached.get("sections", []):
                    if s["title"] == rel["section_title"] and s["page"] == rel["page_number"]:
                        found = s
                        break
                snippet = _truncate(found["content"], 400) if found else ""
                connections_context.append(f"{rel['document']} • {rel['section_title']}: {snippet}")

        # Configure Gemini
        genai.configure(api_key=api_key)
        # Ask for JSON insights first (same prompt as /api/insights)
        insights_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json", "temperature": 0.3},
        )
        
        # Customize prompt based on whether text is selected
        if text and text.strip():
            insights_prompt = (
                f"Selected Text: {text.strip()}\n\n"
                "Summarize the core insights from the following notes, focusing on the selected text. Return strictly JSON with keys: "
                "key_insights, did_you_know_facts, contradictions_or_counterpoints, inspirations_or_connections.\n\n"
                + "Current Doc Notes:\n" + "\n\n".join(current_doc_context[:8]) + "\n\n"
            )
        else:
            insights_prompt = (
                "Summarize the core insights from the following notes. Return strictly JSON with keys: "
                "key_insights, did_you_know_facts, contradictions_or_counterpoints, inspirations_or_connections.\n\n"
                + "Current Doc Notes:\n" + "\n\n".join(current_doc_context[:8]) + "\n\n"
            )
            
        if connections_context:
            insights_prompt += "Cross-doc Notes:\n" + "\n\n".join(connections_context[:10]) + "\n\n"
        insights_response = insights_model.generate_content(insights_prompt)
        insights_text = getattr(insights_response, "text", None) or (
            insights_response.candidates[0].content.parts[0].text if getattr(insights_response, "candidates", None) else ""
        )
        insights_data = _safe_json_parse(insights_text or "")

        # Step 2: serialize into podcast script prompt and ask Gemini to humanize
        script_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"temperature": 0.7, "max_output_tokens": 2048},
        )
        
        # Customize script prompt based on whether text is selected
        if text and text.strip():
            script_prompt = (
                "You are a podcast host duo with two speakers: HOST_A and HOST_B. Turn the following structured insights into a 2-3 minute conversational script. "
                "Focus your discussion on the selected text and how it relates to the broader context. "
                "Rules: strictly format each line as 'HOST_A: ...' or 'HOST_B: ...'. No other prefixes, no stage directions. "
                "Ensure a natural alternating conversation (but do not force rigid alternation) and keep sentences spoken by only the labeled host. "
                "Make it engaging, friendly, and informative with light transitions and a short recap at the end.\n\n"
                f"Selected Text: {text.strip()}\n\n"
                f"Persona: {persona or 'N/A'} | Job To Be Done: {job_to_be_done or 'N/A'}\n\n"
                f"Insights JSON:\n{json.dumps(insights_data, ensure_ascii=False, indent=2)}\n\n"
                "Return only the script text with lines starting by HOST_A: or HOST_B: ."
            )
        else:
            script_prompt = (
                "You are a podcast host duo with two speakers: HOST_A and HOST_B. Turn the following structured insights into a 2-3 minute conversational script. "
                "Rules: strictly format each line as 'HOST_A: ...' or 'HOST_B: ...'. No other prefixes, no stage directions. "
                "Ensure a natural alternating conversation (but do not force rigid alternation) and keep sentences spoken by only the labeled host. "
                "Make it engaging, friendly, and informative with light transitions and a short recap at the end.\n\n"
                f"Persona: {persona or 'N/A'} | Job To Be Done: {job_to_be_done or 'N/A'}\n\n"
                f"Insights JSON:\n{json.dumps(insights_data, ensure_ascii=False, indent=2)}\n\n"
                "Return only the script text with lines starting by HOST_A: or HOST_B: ."
            )
            
        script_response = script_model.generate_content(script_prompt)
        script_text = getattr(script_response, "text", "").strip()
        if not script_text:
            raise HTTPException(status_code=500, detail="Failed to generate podcast script")

        # Step 3: TTS with multi-voice synthesis
        ts = int(time.time())
        base = os.path.splitext(os.path.basename(filename))[0]
        out_name = f"{base}_{ts}.mp3"
        out_path = os.path.join(UPLOADS_VO_DIR, out_name)

        # Parse lines into speakers
        lines: List[Tuple[str, str]] = []
        for raw in script_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("HOST_A:"):
                lines.append(("A", line[len("HOST_A:"):].strip()))
            elif line.startswith("HOST_B:"):
                lines.append(("B", line[len("HOST_B:"):].strip()))
            else:
                # Fallback: attribute to A
                lines.append(("A", line))

        # Prefer Google Cloud TTS for multi-voice if available, else fall back to gTTS (single voice)
        if gcloud_tts is not None:
            try:
                client = gcloud_tts.TextToSpeechClient()
                # Choose two distinct standard voices (you can change to WaveNet if available)
                voice_a = gcloud_tts.VoiceSelectionParams(language_code="en-US", name="en-US-Standard-B")
                voice_b = gcloud_tts.VoiceSelectionParams(language_code="en-US", name="en-US-Standard-C")
                audio_config = gcloud_tts.AudioConfig(audio_encoding=gcloud_tts.AudioEncoding.MP3, speaking_rate=1.02)

                chunks: List[bytes] = []
                for speaker, text in lines:
                    voice = voice_a if speaker == "A" else voice_b
                    synthesis_input = gcloud_tts.SynthesisInput(text=text)
                    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
                    chunks.append(response.audio_content)
                # Concatenate MP3 chunks (simple byte concat works for MP3 frames)
                with open(out_path, "wb") as f:
                    for ch in chunks:
                        f.write(ch)
            except Exception as e:
                # Fallback to gTTS if Cloud TTS fails
                if gTTS is None:
                    raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)} and gTTS unavailable")
                try:
                    tts = gTTS(text=script_text, lang="en")
                    tts.save(out_path)
                except Exception as eg:
                    raise HTTPException(status_code=500, detail=f"gTTS fallback failed: {str(eg)}")
        else:
            if gTTS is None:
                raise HTTPException(status_code=500, detail="No TTS backend available (install google-cloud-texttospeech or gTTS)")
            # Single-voice fallback
            try:
                tts = gTTS(text=script_text, lang="en")
                tts.save(out_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"gTTS failed: {str(e)}")

        return {"audio_url": f"/uploads/VO/{out_name}", "script": script_text, "insights": insights_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Podcast failed: {str(e)}")
'''

import time
import requests
#from pydub import AudioSegment

# Azure keys
AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY", "").strip()
AZURE_REGION = os.getenv("AZURE_TTS_REGION", "").strip()
AZURE_TTS_ENDPOINT = f"https://{AZURE_REGION or 'southeastasia'}.tts.speech.microsoft.com/cognitiveservices/v1"

# Two different Azure voices
VOICE_A = "en-US-JennyNeural"  # Host A
VOICE_B = "en-US-GuyNeural"    # Host B


def azure_tts_speak(text: str, voice: str, tmp_file: str):
    """Call Azure TTS for a given text and voice, save to tmp_file"""
    
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TTS_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-48khz-192kbitrate-mono-mp3"
    }
    ssml = f"""
    <speak version='1.0' xml:lang='en-US'>
        <voice xml:lang='en-US' name='{voice}'>
            {text}
        </voice>
    </speak>
    """
    resp = requests.post(AZURE_TTS_ENDPOINT, headers=headers, data=ssml.encode("utf-8"))
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Azure TTS error: {resp.text}")
    with open(tmp_file, "wb") as f:
        f.write(resp.content)

from pydub import AudioSegment
import ffmpeg
try:
    from .tts import generate_audio  # type: ignore
except Exception:
    try:
        from backend.tts import generate_audio  # type: ignore
    except Exception:
        generate_audio = None  # type: ignore

@app.post("/api/podcast")
async def podcast(
    filename: str = Form(...),
    persona: str = Form(""),
    job_to_be_done: str = Form(""),
    text: str = Form(""),
):
    api_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_APPLICATION_CREDENTIALS env var not set")

    pdf_path = os.path.join(UPLOADS_DIR, os.path.basename(filename))
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found. Upload first.")

    try:
        # ... keep your INSIGHTS + SCRIPT generation code as-is up to script_text ...
        # Reuse insights generation
        fake_form = {"filename": filename, "persona": persona, "job_to_be_done": job_to_be_done, "text": text}
        # Call internal function by simulating request: easiest is to reuse blocks here
        # Ensure cache and ranked contexts
        cached = _ensure_cached(pdf_path)
        sections = cached.get("sections", [])
        
        # If text is provided, focus on sections containing that text
        if text and text.strip():
            # Find sections that contain the selected text
            relevant_sections = []
            for s in sections:
                if text.lower() in s["content"].lower():
                    relevant_sections.append(s)
            
            if relevant_sections:
                sections = relevant_sections
            # If no exact matches, use semantic search
            else:
                # Use the 1B model to find semantically similar sections
                query_embedding, query_text = ranker_singleton.create_query_embedding(
                    persona, job_to_be_done
                )
                
                # Create embeddings for all sections
                section_texts = [f"{s['title']}. {s['content'][:500]}" for s in sections]
                import torch
                with torch.no_grad():
                    section_embeddings = ranker_singleton.model.encode(
                        section_texts, convert_to_tensor=True, convert_to_numpy=True
                    )
                
                # Calculate similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity([query_embedding], section_embeddings)[0]
                
                # Sort sections by similarity to selected text
                section_similarities = list(zip(sections, similarities))
                section_similarities.sort(key=lambda x: x[1], reverse=True)
                sections = [s[0] for s in section_similarities[:5]]  # Top 5 most similar
        
        ranked = ranker_singleton.rank_sections(
            [
                {"document": os.path.basename(pdf_path), "title": s["title"], "content": s["content"], "page": s["page"]}
                for s in sections
            ],
            persona,
            job_to_be_done,
            description="",
            top_k=8,
            expected_titles=None,
        )
        current_doc_context = [
            f"Section: {r['section']['title']} (p.{r['section']['page']})\n" + _truncate(r['section']["content"], 1200)
            for r in ranked
        ]
        related = compute_related_sections(pdf_path, persona, job_to_be_done, description="")
        connections_context: List[str] = []
        for group in related:
            for rel in group.get("related", []):
                other_path = os.path.join(UPLOADS_DIR, os.path.basename(rel["document"]))
                other_cached = _ensure_cached(other_path)
                found = None
                for s in other_cached.get("sections", []):
                    if s["title"] == rel["section_title"] and s["page"] == rel["page_number"]:
                        found = s
                        break
                snippet = _truncate(found["content"], 400) if found else ""
                connections_context.append(f"{rel['document']} • {rel['section_title']}: {snippet}")

        # Configure Gemini
        genai.configure(api_key=api_key)
        # Ask for JSON insights first (same prompt as /api/insights)
        insights_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={"response_mime_type": "application/json", "temperature": 0.3},
        )
        
        # Customize prompt based on whether text is selected
        if text and text.strip():
            insights_prompt = (
                f"Selected Text: {text.strip()}\n\n"
                "Summarize the core insights from the following notes, focusing on the selected text. Return strictly JSON with keys: "
                "key_insights, did_you_know_facts, contradictions_or_counterpoints, inspirations_or_connections.\n\n"
                + "Current Doc Notes:\n" + "\n\n".join(current_doc_context[:8]) + "\n\n"
            )
        else:
            insights_prompt = (
                "Summarize the core insights from the following notes. Return strictly JSON with keys: "
                "key_insights, did_you_know_facts, contradictions_or_counterpoints, inspirations_or_connections.\n\n"
                + "Current Doc Notes:\n" + "\n\n".join(current_doc_context[:8]) + "\n\n"
            )
            
        if connections_context:
            insights_prompt += "Cross-doc Notes:\n" + "\n\n".join(connections_context[:10]) + "\n\n"
        insights_response = insights_model.generate_content(insights_prompt)
        insights_text = None
        if getattr(insights_response, "text", None):
            insights_text = insights_response.text
        elif getattr(insights_response, "candidates", None):
            for c in insights_response.candidates:
                if c.content and c.content.parts:
                    for p in c.content.parts:
                        if getattr(p, "text", None):
                            insights_text = p.text
                            break
                if insights_text:
                    break
        insights_data = _safe_json_parse(insights_text or "")

        # Step 2: serialize into podcast script prompt and ask Gemini to humanize
        script_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={"temperature": 0.7, "max_output_tokens": 2048},
        )
        
        # Customize script prompt based on whether text is selected
        if text and text.strip():
            script_prompt = (
                "You are a podcast host duo with two speakers: HOST_A and HOST_B. Turn the following structured insights into a 2-3 minute conversational script. "
                "Focus your discussion on the selected text and how it relates to the broader context. "
                "Rules: strictly format each line as 'HOST_A: ...' or 'HOST_B: ...'. No other prefixes, no stage directions. "
                "Ensure a natural alternating conversation (but do not force rigid alternation) and keep sentences spoken by only the labeled host. "
                "Make it engaging, friendly, and informative with light transitions and a short recap at the end.\n\n"
                f"Selected Text: {text.strip()}\n\n"
                f"Persona: {persona or 'N/A'} | Job To Be Done: {job_to_be_done or 'N/A'}\n\n"
                f"Insights JSON:\n{json.dumps(insights_data, ensure_ascii=False, indent=2)}\n\n"
                "Return only the script text with lines starting by HOST_A: or HOST_B: ."
            )
        else:
            script_prompt = (
                "You are a podcast host duo with two speakers: HOST_A and HOST_B. Turn the following structured insights into a 2-3 minute conversational script. "
                "Rules: strictly format each line as 'HOST_A: ...' or 'HOST_B: ...'. No other prefixes, no stage directions. "
                "Ensure a natural alternating conversation (but do not force rigid alternation) and keep sentences spoken by only the labeled host. "
                "Make it engaging, friendly, and informative with light transitions and a short recap at the end.\n\n"
                f"Persona: {persona or 'N/A'} | Job To Be Done: {job_to_be_done or 'N/A'}\n\n"
                f"Insights JSON:\n{json.dumps(insights_data, ensure_ascii=False, indent=2)}\n\n"
                "Return only the script text with lines starting by HOST_A: or HOST_B: ."
            )
            
        script_response = script_model.generate_content(script_prompt)
        script_text = getattr(script_response, "text", "").strip()
        if not script_text:
            raise HTTPException(status_code=500, detail="Failed to generate podcast script")

        # Step 3: TTS (supports azure/openai/local via TTS_PROVIDER)
        ts = int(time.time())
        base = os.path.splitext(os.path.basename(filename))[0]
        out_name = f"{base}_{ts}.mp3"
        out_path = os.path.join(UPLOADS_VO_DIR, out_name)

        # Prefer multi-voice if provider=azure and two voices provided; otherwise synthesize full script
        provider = (os.getenv("TTS_PROVIDER", "local").strip().lower())
        try:
            if provider == "azure":
                # Try alternating voices when script is formatted with HOST_A / HOST_B
                lines: List[Tuple[str, str]] = []
                for raw in script_text.splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith("HOST_A:"):
                        lines.append((os.getenv("AZURE_TTS_VOICE_A", VOICE_A), line[len("HOST_A:"):].strip()))
                    elif line.startswith("HOST_B:"):
                        lines.append((os.getenv("AZURE_TTS_VOICE_B", VOICE_B), line[len("HOST_B:"):].strip()))
                    else:
                        lines.append((os.getenv("AZURE_TTS_VOICE_A", VOICE_A), line))

                if lines:
                    tmp_files = []
                    for i, (vname, seg_text) in enumerate(lines):
                        tmp_file = os.path.join(UPLOADS_VO_DIR, f"tmp_{ts}_{i}.mp3")
                        azure_tts_speak(seg_text, vname, tmp_file)
                        tmp_files.append(tmp_file)
                    combined = AudioSegment.silent(duration=250)
                    for f in tmp_files:
                        seg = AudioSegment.from_file(f, format="mp3")
                        combined += seg + AudioSegment.silent(duration=300)
                    combined.export(out_path, format="mp3")
                    for f in tmp_files:
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                else:
                    generate_audio(script_text, out_path, provider="azure")
            else:
                # Single-voice synthesis for openai/local
                generate_audio(script_text, out_path, provider=provider)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

        return {
            "audio_url": f"/uploads/VO/{out_name}",
            "script": script_text,
            "insights": insights_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Podcast failed: {str(e)}")

from urllib.parse import quote

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    saved: List[str] = []
    for f in files:
        if not f.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {f.filename}")
        
        # Save with original filename (safe for FS)
        dest = os.path.join(UPLOADS_DIR, os.path.basename(f.filename))
        content = await f.read()
        with open(dest, 'wb') as out:
            out.write(content)

        safe_name = os.path.basename(dest)   # keep actual filename
        encoded_name = quote(safe_name)      # safe for URL

        # --- IMPORTANT: store mapping for analysis later ---
        '''
        analysis_cache.register_document(
            safe_name=safe_name,
            file_url=f"/uploads/{encoded_name}",
            display_name=f.filename
        )
        '''

        saved.append(f"/uploads/{encoded_name}")
    
    return {"saved": saved}



@app.get("/api/documents")
def list_documents() -> Dict[str, Any]:
    items = [
        {
            "name": fname,
            "url": f"/uploads/{fname}",
            "path": os.path.join(UPLOADS_DIR, fname),
        }
        for fname in os.listdir(UPLOADS_DIR)
        if fname.lower().endswith('.pdf')
    ]
    return {"documents": items}


@app.post("/api/analyze")
async def analyze(
    filename: str = Form(...),
    persona: str = Form(""),
    job_to_be_done: str = Form(""),
    description: str = Form(""),
):
    pdf_path = os.path.join(UPLOADS_DIR, os.path.basename(filename))
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found. Upload first.")
    try:
        result = analyze_pdf(pdf_path, persona, job_to_be_done, description, top_k=5)
        related = compute_related_sections(pdf_path, persona, job_to_be_done, description)
        result["persona"] = persona
        result["job_to_be_done"] = job_to_be_done
        result["related_sections"] = related
        result["file_url"] = f"/uploads/{os.path.basename(filename)}"
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from urllib.parse import quote

@app.post("/api/recommendations")
async def get_recommendations(
    selected_text: str = Form(...),
    persona: str = Form(""),
    job_to_be_done: str = Form(""),
    top_k: int = Form(5)
):
    """Get recommendations based on selected text using 1B-esque query system"""
    try:
        if not selected_text.strip():
            raise HTTPException(status_code=400, detail="Selected text cannot be empty")
        
        result = get_recommendations_for_text(
            selected_text.strip(),
            persona.strip(),
            job_to_be_done.strip(),
            top_k
        )

        # --- PATCH: add file_url with proper encoding ---
        recommendations = result.get("recommendations", [])
        for rec in recommendations:
            if "document" in rec:
                # rec["document"] should be the actual filename as saved in uploads
                safe_name = rec["document"]
                rec["file_url"] = f"/uploads/{quote(safe_name)}"

        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


@app.post("/api/background-process")
async def trigger_background_processing():
    """Trigger background processing of all uploaded PDFs"""
    try:
        process_all_pdfs_background()
        return {"message": "Background processing completed", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background processing failed: {str(e)}")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}
@app.delete("/api/documents")
def clear_all_documents() -> Dict[str, Any]:
    """Delete all uploaded documents but keep VO folder"""
    try:
        import shutil

        # Ensure uploads dir exists
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        os.makedirs(UPLOADS_VO_DIR, exist_ok=True)  # VO folder stays

        # Iterate through uploads but skip VO folder
        for item in os.listdir(UPLOADS_DIR):
            path = os.path.join(UPLOADS_DIR, item)
            if os.path.isdir(path) and path == UPLOADS_VO_DIR:
                continue  # don't delete VO
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

        return {"message": "All documents cleared successfully, VO preserved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

@app.delete("/api/podcasts")
def clear_all_podcasts() -> Dict[str, Any]:
    """Delete all podcast audio files from VO folder"""
    try:
        import shutil
        
        if os.path.exists(UPLOADS_VO_DIR):
            shutil.rmtree(UPLOADS_VO_DIR)
        os.makedirs(UPLOADS_VO_DIR, exist_ok=True)  # recreate empty VO folder

        return {"message": "All podcast audio cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear podcasts: {str(e)}")


# --- Delete all podcast audio files from VO folder on startup ---
def clear_vo_folder_on_startup():
    import shutil
    if os.path.exists(UPLOADS_VO_DIR):
        shutil.rmtree(UPLOADS_VO_DIR)
    os.makedirs(UPLOADS_VO_DIR, exist_ok=True)

clear_vo_folder_on_startup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
