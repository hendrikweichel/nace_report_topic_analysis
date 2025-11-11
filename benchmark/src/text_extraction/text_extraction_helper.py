import re

def is_block_within(block_bbox, table_bbox):
    bx0, by0, bx1, by1 = block_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    return bx0 >= tx0 and by0 >= ty0 and bx1 <= tx1 and by1 <= ty1

def extract_all_sizes(block):
    sizes = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if "size" in span:
                sizes.append(span["size"])
    return sizes

def extract_block_text(block):
    spans = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span["text"].strip():
                spans.append(span["text"])
    text = " ".join(spans)
    text = re.sub(r' +', ' ', text)
    return " ".join(spans)

def heuristik_text(text: str) -> bool: 
    # remove all numbers like 1.234 or 500.000
    text = re.sub(r'\b\d+\.\d+\b', '', text)

    text = re.sub(r"[^a-zA-ZäöüÄÖÜß.\s]", "", text)
    
    if len(text.split(" ")) > 10 and len(text) > 30 and "." in text: 
            return True
    
    return False
def check_font_sizes(max_font_size: float, font_sizes_block: list): 
    for font_size_block in font_sizes_block: 
        if font_size_block < max_font_size * 1.02 and font_size_block > max_font_size * 0.98: 
            return True
def check_nummerical_block(text: str, p: float = 0.5):
    text = text.strip()

    if len(text) == 0: 
        return False

    count = sum(c.isdigit() for c in text)
    return count/len(text) > p