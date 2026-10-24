"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--include-superseded] [--json]
"""
import argparse
import json
import string

import numpy as np
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from rag.chunk import chunk_docs

# น้ำหนักของ TF-IDF ในคะแนน hybrid ที่เหลือเป็น BM25 (ชุด dev ได้ hit@1 = 1.0 ตลอดช่วง 0.6–1.0 แต่ลดลงที่ 0.5)
ALPHA = 0.7

# พจนานุกรมคำพ้อง: คำในคำถาม (ตัวพิมพ์เล็ก) → คำที่เอกสารใช้ สร้างจากคำถามชุด dev ที่ค้นพลาดเท่านั้น
SYNONYMS = {
    "อีวี": ["รถยนต์ไฟฟ้า"],  # dev Q35
    "ev": ["รถยนต์ไฟฟ้า"],  # dev Q35 (สะกดแบบอังกฤษ)
    "เสียบชาร์จ": ["จุดชาร์จ"],  # dev Q35
}

PUNCT = string.punctuation + "“”‘’·–—…"


def index_text(c: dict) -> str:
    """ข้อความที่ใช้สร้างดัชนีของแต่ละ chunk: title + section + text"""
    return f"{c['title']}\n{c['section']}\n{c['text']}"


def tokenize(text: str) -> list[str]:
    """ตัดคำด้วย pythainlp (newmm) ตัดเครื่องหมายวรรคตอนรอบคำ และแปลงอักษรอังกฤษเป็นตัวเล็ก"""
    words = (w.strip(PUNCT).lower() for w in word_tokenize(text, engine="newmm", keep_whitespace=False))
    return [w for w in words if w]


def expand_query(query: str) -> str:
    """ต่อท้ายคำถามด้วยคำพ้องของคำที่พบใน SYNONYMS"""
    q = query.lower()
    return " ".join([query, *(w for key, words in SYNONYMS.items() if key in q for w in words)])


def min_max(x: np.ndarray) -> np.ndarray:
    """ปรับคะแนนให้อยู่ในช่วง 0–1 เพื่อรวมคะแนนที่สเกลต่างกันได้"""
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


class Retriever:
    def __init__(self, chunks: list[dict] | None = None, method: str = "tfidf"):
        """method="tfidf": TF-IDF ระดับตัวอักษร (W2) · method="hybrid": วิธีที่ปรับปรุงแล้ว (W4)"""
        if method not in ("tfidf", "hybrid"):
            raise ValueError(f"ไม่รู้จัก method: {method}")
        self.chunks = chunk_docs() if chunks is None else chunks
        self.method = method
        texts = [index_text(c) for c in self.chunks]
        # ภาษาไทยไม่เว้นวรรคระหว่างคำ จึงใช้ n-gram ระดับตัวอักษรแทนการตัดคำ
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(texts)
        if method == "hybrid":
            self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def tfidf_scores(self, query: str) -> np.ndarray:
        """cosine similarity ของคำถามกับทุก chunk (TfidfVectorizer normalize L2 ให้แล้ว จึงใช้ dot product ได้)"""
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()

    def scores(self, query: str) -> np.ndarray:
        if self.method == "tfidf":
            return self.tfidf_scores(query)
        # hybrid: ขยายคำค้นด้วยคำพ้อง แล้วรวม TF-IDF ตัวอักษรกับ BM25 ระดับคำ หลังปรับสเกลเป็น 0–1
        q = expand_query(query)
        bm25 = self.bm25.get_scores(tokenize(q))
        return ALPHA * min_max(self.tfidf_scores(q)) + (1 - ALPHA) * min_max(bm25)

    def search(self, query: str, k: int = 4, include_superseded: bool = True) -> list[dict]:
        """คืน chunk k อันที่คะแนนสูงสุด เรียงจากมากไปน้อย แต่ละอันเพิ่มช่อง "score" """
        scores = self.scores(query)
        results = []
        for i in np.argsort(-scores, kind="stable"):
            c = self.chunks[i]
            if not include_superseded and c["status"] == "superseded":
                continue
            results.append({**c, "score": float(scores[i])})
            if len(results) == k:
                break
        return results


def main():
    ap = argparse.ArgumentParser(description="ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--method", default="tfidf", choices=["tfidf", "hybrid"])
    ap.add_argument("--include-superseded", action="store_true", help="รวมเอกสารฉบับเก่า (ค่าเริ่มต้นใช้เฉพาะ active)")
    ap.add_argument("--json", action="store_true", help="พิมพ์ผลเป็น JSON")
    a = ap.parse_args()
    res = Retriever(method=a.method).search(a.query, a.k, a.include_superseded)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    for i, r in enumerate(res, 1):
        print(f"{i}. [{r['chunk_id']}] {r['score']:.3f} · {r['title']} › {r['section']} ({r['status']}, {r['effective_date']})")
        print("   " + r["text"].replace("\n", "\n   "))
        print()


if __name__ == "__main__":
    main()
