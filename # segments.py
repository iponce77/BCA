# segments.py
import pandas as pd
from typing import Dict, Tuple, Optional

def load_segment_map(path: str = "segment_map.csv") -> Dict[Tuple[str, str], str]:
    df = pd.read_csv(path)
    for col in ["make_clean","modelo_base","segmento"]:
        df[col] = df[col].astype(str).str.upper().str.strip()
    return {(r["make_clean"], r["modelo_base"]): r["segmento"] for _, r in df.iterrows()}

def infer_segment(make_clean: Optional[str], modelo_base: Optional[str], segmap) -> Optional[str]:
    if not make_clean or not modelo_base:
        return None
    key = (str(make_clean).upper().strip(), str(modelo_base).upper().strip())
    return segmap.get(key)
