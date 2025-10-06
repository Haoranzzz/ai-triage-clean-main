import os
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, List
from datetime import datetime
import httpx
import asyncio

BASE = os.getenv("E_HOSPITAL_BASE_URL", "").rstrip("/")
if not BASE:
    raise RuntimeError("E_HOSPITAL_BASE_URL not set. Add it to your .env and restart the server.")

# --- helpers ---
def _parse_dt(x: Any) -> datetime:
    if not x:
        return datetime.min
    # Accept YYYY-MM-DD or ISO timestamps
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(str(x), fmt)
        except ValueError:
            continue
    # Fallback: try fromisoformat
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except Exception:
        return datetime.min

async def _get_table(name: str) -> List[Dict[str, Any]]:
    url = f"{BASE}/table/{name}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        js = r.json()
        return js.get("data", js)  # some endpoints may return top-level list

def _filter_sort_trim(rows: List[Dict[str, Any]], patient_id: int, date_keys: List[str], limit: int) -> List[Dict[str, Any]]:
    filtered = [row for row in rows if str(row.get("patient_id")) == str(patient_id)]
    # choose first existing date key to sort by
    def sort_key(row):
        for k in date_keys:
            if k in row:
                return _parse_dt(row[k])
        return datetime.min
    filtered.sort(key=sort_key, reverse=True)
    return filtered[:limit]

# --- main builder ---
async def build_snapshot(patient_id: int) -> Dict[str, Any]:
    # fetch all needed tables in parallel
    async with httpx.AsyncClient(timeout=20.0) as client:
        mh_task = client.get(f"{BASE}/table/medical_history")
        pr_task = client.get(f"{BASE}/table/prescription")
        ar_task = client.get(f"{BASE}/table/allergy_records")
        rx_task = client.get(f"{BASE}/table/allergy_reaction")
        lb_task = client.get(f"{BASE}/table/lab_tests")
        dx_task = client.get(f"{BASE}/table/diagnosis")

        mh, pr, ar, rx, lb, dx = await asyncio.gather(mh_task, pr_task, ar_task, rx_task, lb_task, dx_task)

    # unpack JSON
    def jj(resp): 
        j = resp.json()
        return j.get("data", j)

    medical_history = jj(mh)
    prescriptions  = jj(pr)
    allergy_records = jj(ar)
    allergy_reaction = jj(rx)
    lab_tests = jj(lb)
    diagnosis = jj(dx)

    # per-table shaping
    mh_trim = _filter_sort_trim(medical_history, patient_id, ["last_updated", "diagnosis_date"], limit=3)
    pr_trim = _filter_sort_trim(prescriptions, patient_id, ["issued_on", "start_date", "end_date"], limit=3)
    ar_trim = _filter_sort_trim(allergy_records, patient_id, ["recorded_on"], limit=3)
    lb_trim = _filter_sort_trim(lab_tests, patient_id, ["test_date", "uploaded_on"], limit=3)
    dx_trim = _filter_sort_trim(diagnosis, patient_id, ["diagnosis_date"], limit=3)

    # attach reactions to the kept allergy records (same patient via allergy_record_id)
    ar_ids = {row.get("record_id") for row in ar_trim}
    rx_for_patient = [r for r in allergy_reaction if r.get("allergy_record_id") in ar_ids]
    # Sort reactions by reaction_date desc
    rx_for_patient.sort(key=lambda r: _parse_dt(r.get("reaction_date")), reverse=True)
    # group reactions by record_id and keep last 2 each
    rx_map: Dict[int, List[Dict[str, Any]]] = {}
    for r in rx_for_patient:
        rid = r.get("allergy_record_id")
        rx_map.setdefault(rid, [])
        if len(rx_map[rid]) < 2:
            rx_map[rid].append(r)

    # attach reactions to allergy entries
    for a in ar_trim:
        a["reactions"] = rx_map.get(a.get("record_id"), [])

    # final shape
    return {
        "patient_id": patient_id,
        "visits": [],  # you didn’t share an encounters/appointments table; leave empty for now, should we add in mock? 
        "medical_history": mh_trim,
        "medications": pr_trim,      # already includes medicine_name string
        "allergies": ar_trim,        # now includes recent reactions
        "labs": lb_trim,
        "diagnoses": dx_trim
    }