"""แบ่งเอกสารเป็น chunk พร้อม metadata (เวิร์กช็อป 1)

แต่ละ chunk เป็น dict ที่มีอย่างน้อย:
chunk_id ("<doc_id>#<ลำดับ>"), doc_id, title, section, status, effective_date, text
"""
from rag.ingest import load_docs

MAX_CHARS = 600


def split_sections(body: str) -> list[tuple[str, list[str]]]:
    """แยกเนื้อหาเป็น (ชื่อหัวข้อ ##, บรรทัดในหัวข้อ) ข้ามหัวเรื่อง # และบรรทัดว่าง
    เนื้อหาก่อนหัวข้อ ## แรก (ถ้ามี) ได้ชื่อหัวข้อเป็น "" """
    sections = []
    for line in body.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif line.strip() and not line.startswith("# "):
            if not sections:
                sections.append(("", []))
            sections[-1][1].append(line.rstrip())
    return sections


def split_long_line(line: str, limit: int) -> list[str]:
    """บรรทัดเดียวที่ยาวเกิน limit: ตัดที่ช่องว่างสุดท้ายก่อนถึง limit (ไม่มีช่องว่างก็ตัดตรง limit)"""
    parts = []
    while len(line) > limit:
        cut = line.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        parts.append(line[:cut].rstrip())
        line = line[cut:].lstrip()
    return parts + [line] if line else parts


def pack_lines(header: str, lines: list[str], max_chars: int) -> list[str]:
    """รวมบรรทัดเป็นก้อนตามลำดับ ทุกก้อนขึ้นต้นด้วยบรรทัดหัวข้อ และยาวรวมไม่เกิน max_chars"""
    limit = max(max_chars - len(header) - 1, 1)  # -1 คือ \n หลังบรรทัดหัวข้อ
    groups, current, size = [], [], 0
    for part in (p for line in lines for p in split_long_line(line, limit)):
        if current and size + 1 + len(part) > limit:
            groups.append(current)
            current, size = [], 0
        size += len(part) + (1 if current else 0)
        current.append(part)
    groups.append(current)
    return ["\n".join([header, *g]) for g in groups]


def chunk_docs(docs: list[dict] | None = None, max_chars: int = MAX_CHARS) -> list[dict]:
    """หนึ่งหัวข้อ ## ต่อหนึ่ง chunk หัวข้อที่ยาวเกิน max_chars ตัดตามบรรทัด
    ทุก chunk เก็บ metadata ของเอกสาร (ยกเว้น body) และข้อความขึ้นต้นด้วย "## <หัวข้อ>" """
    chunks = []
    for doc in load_docs() if docs is None else docs:
        meta = {k: v for k, v in doc.items() if k != "body"}
        n = 0
        for section, lines in split_sections(doc["body"]):
            section = section or doc["title"]
            for text in pack_lines(f"## {section}", lines, max_chars):
                n += 1
                chunks.append({**meta, "chunk_id": f"{doc['doc_id']}#{n}", "section": section, "text": text})
    return chunks


if __name__ == "__main__":
    cs = chunk_docs()
    print(len(cs), "chunks · ยาวสุด", max(len(c["text"]) for c in cs), "ตัวอักษร")
