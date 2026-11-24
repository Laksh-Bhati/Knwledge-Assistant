import sys
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL

_client = None  # created once, then reused


def get_client():
    """Create the OpenAI client the first time it's needed."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts, batch_size=100):
    """Turn a list of texts into a list of embedding vectors."""
    if not texts:
        return []

    client = get_client()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_query(question):
    """Embed a single question (same model as the chunks)."""
    return embed_texts([question])[0]


def cosine_similarity(a, b):
    """How similar two vectors are: close to 1 = very similar meaning."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    question = "How do I claim reimbursement for a business expense?"
    sentences = [
        "Business expenses must be submitted within 30 days of the expense date.",
        "Passwords must be at least 14 characters and changed every 90 days.",
        "On your first day, report to the reception desk on Floor 2 at 9:00 AM.",
    ]

    try:
        q_vec = embed_query(question)
        s_vecs = embed_texts(sentences)
    except AuthenticationError:
        print("❌ API key rejected. Check the key in your .env file.")
        sys.exit(1)
    except RateLimitError:
        print("❌ Rate limit or no credit. Check billing on your OpenAI account.")
        sys.exit(1)
    except APIConnectionError:
        print("❌ Could not reach OpenAI. Check your internet connection.")
        sys.exit(1)

    print(f"✅ Each embedding has {len(q_vec)} numbers")
    print(f"First 5 numbers of the question's embedding: {[round(x, 4) for x in q_vec[:5]]}\n")

    print(f"Question: {question}\n")
    scores = [(cosine_similarity(q_vec, v), s) for v, s in zip(s_vecs, sentences)]
    for score, sentence in sorted(scores, reverse=True):
        print(f"{score:.3f}  {sentence}")
