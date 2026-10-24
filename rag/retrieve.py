"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--include-superseded] [--json]
"""
import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rag.chunk import chunk_docs


def index_text(c: dict) -> str:
    """ข้อความที่ใช้สร้างดัชนีของแต่ละ chunk: title + section + text"""
    return f"{c['title']}\n{c['section']}\n{c['text']}"


class Retriever:
    def __init__(self, chunks: list[dict] | None = None, method: str = "tfidf"):
        """method="tfidf": TF-IDF ระดับตัวอักษร (W2) · method="hybrid": วิธีที่ปรับปรุงแล้ว (W4)"""
        self.chunks = chunk_docs() if chunks is None else chunks
        self.method = method
        if method == "tfidf":
            # ภาษาไทยไม่เว้นวรรคระหว่างคำ จึงใช้ n-gram ระดับตัวอักษรแทนการตัดคำ
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self.matrix = self.vectorizer.fit_transform([index_text(c) for c in self.chunks])
        elif method == "hybrid":
            raise NotImplementedError("method hybrid ทำในเวิร์กช็อป 4")
        else:
            raise ValueError(f"ไม่รู้จัก method: {method}")

    def scores(self, query: str) -> np.ndarray:
        """cosine similarity ของคำถามกับทุก chunk (TfidfVectorizer normalize L2 ให้แล้ว จึงใช้ dot product ได้)"""
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()

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
