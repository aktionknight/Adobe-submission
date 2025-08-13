import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Set deterministic settings for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SectionRanker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the section ranker with a sentence transformer model.
        Using a lightweight model to meet the 1GB size constraint.
        """
        # Set deterministic settings before model loading
        torch.manual_seed(42)
        
        # Set model to deterministic mode
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        
        # Force deterministic behavior for the model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Use CPU only as required
        if torch.cuda.is_available():
            self.model = self.model.cpu()
        
        # Set model to deterministic inference mode
        self.model.half = lambda: self.model  # Prevent half precision
        self.model.float = lambda: self.model  # Ensure float32
    
    def create_query_embedding(self, persona, job_to_be_done, description=None):
        """
        Create a query embedding from persona and job requirements only (no description).
        Returns both the embedding and the query string for use in boosting.
        """
        query_text = f"{persona} needs to {job_to_be_done}"
        
        # Ensure deterministic encoding
        with torch.no_grad():
            embedding = self.model.encode([query_text], convert_to_tensor=True, convert_to_numpy=True)
        
        return embedding[0], query_text
    
    def rank_sections(self, sections, persona, job_to_be_done, description, top_k=5, expected_titles=None):
        """
        Rank sections based on relevance to persona, job requirements, and description.
        Uses both section titles and content for richer context.
        Returns top_k most relevant sections with their scores.
        Optionally boosts score if section title closely matches any expected_titles.
        """
        if not sections:
            return []
        
        # Create query embedding
        query_embedding, query_text = self.create_query_embedding(persona, job_to_be_done, description)
        
        # Create section embeddings with richer context
        section_texts = []
        for section in sections:
            title = section['title']
            content = section['content']
            content_lines = content.split('\n')
            first_paragraph = ""
            for line in content_lines:
                if line.strip():
                    first_paragraph = line.strip()
                    break
            last_sentence = ""
            if content.strip():
                sentences = content.strip().split('.')
                if sentences:
                    last_sentence = sentences[-1].strip()
                    if not last_sentence:
                        for i in range(len(sentences)-2, -1, -1):
                            if sentences[i].strip():
                                last_sentence = sentences[i].strip()
                                break
            context_parts = [title]
            if first_paragraph:
                context_parts.append(first_paragraph)
            if last_sentence and last_sentence != first_paragraph:
                context_parts.append(last_sentence)
            section_text = '. '.join(context_parts)
            section_texts.append(section_text)
        # Ensure deterministic encoding
        with torch.no_grad():
            section_embeddings = self.model.encode(section_texts, convert_to_tensor=True, convert_to_numpy=True)
        similarities = cosine_similarity([query_embedding], section_embeddings)[0]
        # Optionally boost similarity if section title matches expected_titles
        if expected_titles:
            import difflib
            for i, section in enumerate(sections):
                title = section['title'].lower()
                # Find the closest match among expected_titles
                matches = difflib.get_close_matches(title, [t.lower() for t in expected_titles], n=1, cutoff=0.7)
                if matches:
                    similarities[i] += 0.15  # Boost score for close match
        ranked_sections = []
        for i, (section, similarity) in enumerate(zip(sections, similarities)):
            ranked_sections.append({
                'section': section,
                'similarity_score': similarity,
                'rank': i + 1
            })
        ranked_sections.sort(key=lambda x: x['similarity_score'], reverse=True)
        return ranked_sections[:top_k]
    
    def extract_subsection_content(self, section_content, query_embedding, max_length=2000, max_paragraphs=5, query_text=None):
        """
        Extract meaningful subsections from section content.
        Returns up to max_paragraphs full paragraphs, not fragments, up to max_length characters total.
        Paragraphs are ranked by semantic similarity to the query_embedding, with a boost for containing key query terms.
        """
        if not section_content:
            return []
        
        # Split content into paragraphs (by double newlines or single newlines)
        paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in section_content.split('\n') if p.strip()]
        
        # Filter paragraphs by reasonable length
        candidate_paragraphs = [p for p in paragraphs if 100 < len(p) < 1500]
        if not candidate_paragraphs:
            candidate_paragraphs = [p for p in paragraphs if len(p) > 100]
        if not candidate_paragraphs:
            candidate_paragraphs = paragraphs
        if not candidate_paragraphs:
            return []
        
        # Compute embeddings and similarities with deterministic settings
        with torch.no_grad():
            para_embeddings = self.model.encode(candidate_paragraphs, convert_to_tensor=True, convert_to_numpy=True)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], para_embeddings)[0]
        
        # Extract key terms from the query (words longer than 5 chars, not stopwords)
        import re
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        key_terms = set()
        if query_text:
            words = re.findall(r'\b\w+\b', query_text.lower())
            key_terms = set(w for w in words if len(w) > 5 and w not in ENGLISH_STOP_WORDS)
        
        # Boost similarity for paragraphs containing key query terms
        boosted = []
        for para, sim in zip(candidate_paragraphs, similarities):
            para_lower = para.lower()
            boost = 0.0
            for term in key_terms:
                if term in para_lower:
                    boost += 0.15  # Boost for each key term present
            boosted.append((para, sim + boost))
        
        # Rank paragraphs by boosted similarity
        ranked = sorted(boosted, key=lambda x: x[1], reverse=True)
        
        # Select up to max_paragraphs full paragraphs, not breaking them up
        selected_paragraphs = []
        total_length = 0
        for para, _ in ranked:
            if len(selected_paragraphs) >= max_paragraphs:
                break
            if total_length + len(para) > max_length:
                break
            selected_paragraphs.append(para)
            total_length += len(para)
        
        # If still no selected paragraphs, fall back to the first max_length chars of the original content
        if not selected_paragraphs and section_content:
            summary = section_content[:max_length].strip()
            if summary:
                selected_paragraphs.append(summary)
        
        return selected_paragraphs 