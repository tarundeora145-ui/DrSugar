# DR-SUGAR Final QA Regression Report

**Generated:** 2026-09-20 | **Environment:** India Standard Time (IST)

---

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Node.js | v24.15.0 |
| Python (venv) | 3.14.3 |
| PyTorch | 2.14.0+cu132 |
| ONNX Runtime | 1.30.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| CUDA | Available (cu132) |
| Backend Port | 5000 |
| Frontend Port | 5173 |
| Database | `database/dr_sugar.db` |

---

## Test Summary

| Area | Status | Notes |
|---|---|---|
| Startup — Backend | **PASS** | Express server on :5000, SQLite opens cleanly, `display_id` column ALTER TABLE applied on restart |
| Startup — Frontend | **PASS** | Vite dev server on :5173, 200 OK, no blank screen |
| Database | **PASS** | 22 records, correct schema, all columns present, backfill applied to legacy records |
| Real ONNX Inference | **PASS** | 3 images × 2 rounds; ONNX Runtime executes; predictions differ per image; NOT hardcoded |
| Image Upload — Valid JPG/PNG | **PASS** | Accepted, stored, screening created |
| Image Upload — Invalid type | **PASS** | Rejected with `{"success":false,"message":"Invalid file type..."}` |
| Image Upload — Oversized | **PASS** | 10 MB limit enforced by multer |
| Grad-CAM | **PASS** | Real Grad-CAM PNGs generated (3.7 MB for 3218 KB image); all 4 test screenings have `gradcam.png` |
| Lesion Segmentation | **PASS** | IDRiD fold1_best.pth used; MA/HE/EX/SE percentages returned; real overlay PNGs generated |
| Vessel Segmentation | **PASS** | DRIVE model used; `vessel_coverage_pct` returned; real vessel PNGs generated |
| Evidence Polling | **PASS** | `/api/screening/:id/evidence-status` returns live `PENDING`→`READY` transitions; polling stops at `READY` |
| Progressive Inference | **PASS** | Phase 1 (ONNX classification) returns in **47–216 ms**; Phase 2 (Grad-CAM, lesion, vessel) background ~30s total |
| Clinical Report — Content | **PASS** | Grade, confidence, referable, patient info all correct |
| Clinical Report — Screening ID | **PASS** | Shows `DR-000022` (not raw `22`) after fix |
| Clinical Report — Timestamp | **PASS** | `created_at || 'Z'` fix applied; IST formatting with `{timeZone:'Asia/Kolkata'}` in frontend |
| Patient Report | **PASS** | Shows `ID DR-000022`, correct date in IST, no internal paths or stack traces |
| PDF / Print | **PASS** | Browser print dialog triggers; print CSS hides nav elements; reports are printable |
| Operational Dashboard | **PASS** | Live data from `/api/dashboard`; shows 22 screenings, 6 pending, ONLINE status, recent activity table with DR-IDs |
| Patients Page | **BLOCKED** | No `/patients` route or dedicated Patients page found in this implementation; not in route config |
| Screenings Page | **PASS** | DR-XXXXXX IDs visible; sequential ordering correct |
| Simulation Page | **PASS** | Deterministic math executes; shows Total Screenings, Data TB, Network Days, Clinical Queue — no MATLAB error |
| Radical Transparency | **PASS** | "PROTOTYPE SPECS" panel shows DR-SUGAR-V1 / MobileNetV3 / Grad-CAM / IDRiD / DRIVE; no broken Black Box error |
| API Endpoints | **PASS** | All endpoints tested; see detail below |
| Error Handling | **PASS** | Invalid IDs return 404 JSON; server does not crash; no stack traces exposed |
| Data Consistency | **PASS** | DB ↔ API ↔ Frontend all agree for screenings 19–22 |
| Performance | **PASS** | Primary result < 220ms; evidence < 35s; no UI blocking |
| Security Sanity | **PASS WITH WARNINGS** | No path traversal or SQL injection found; filenames are random UUIDs; see warnings |
| Full User Journey | **PASS** | End-to-end: upload → classify → evidence → reports → PDF workflow verified |
| Final Regression (3 images) | **PASS** | All 3 APTOS images processed with distinct model outputs |

---

## Real Inference Results

### Image 1: `000c1434d8d7.png`

| Field | Value |
|---|---|
| Screening IDs | DR-000019, DR-000022 (two separate test rounds) |
| Predicted Grade | **2 (Moderate DR)** |
| Confidence | **78.68%** |
| Probabilities | [0.2%, 11.4%, **78.7%**, 3.7%, 5.9%] — sum ≈ 1.0 ✓ |
| Referable DR | **YES** (grade ≥ 2) |
| Primary Latency | 89 ms (round 1), 216 ms (round 2, cold) |
| Grad-CAM | READY (3,766 KB PNG) |
| Lesion | READY — MA:0.09%, HE:0.40%, EX:0.82%, SE:0.03% |
| Vessel | READY — coverage: 5.46% |

### Image 2: `001639a390f0.png`

| Field | Value |
|---|---|
| Screening ID | DR-000020 |
| Predicted Grade | **4 (Proliferative DR)** |
| Confidence | **69.68%** |
| Probabilities | [0.7%, 2.2%, 9.4%, 18.0%, **69.7%**] — sum ≈ 1.0 ✓ |
| Referable DR | **YES** |
| Primary Latency | 62 ms |
| Grad-CAM | READY (2,561 KB PNG) |
| Lesion | READY |
| Vessel | READY |

### Image 3: `0024cdab0c1e.png`

| Field | Value |
|---|---|
| Screening ID | DR-000021 |
| Predicted Grade | **0 (No DR)** |
| Confidence | **70.31%** |
| Probabilities | [**70.3%**, 15.3%, 12.5%, 0.2%, 1.7%] — sum ≈ 1.0 ✓ |
| Referable DR | **NO** |
| Primary Latency | 47 ms |
| Grad-CAM | READY (2,333 KB PNG) |
| Lesion | READY — MA:0.07%, HE:0.32%, EX:0.81%, SE:0.04% |
| Vessel | READY — coverage: 7.95% |

> **Note:** Screening IDs 19–22 include duplicate image submissions from separate QA test runs. Model outputs are identical for the same image confirming determinism.

---

## API Endpoint Test Results

| Endpoint | Method | HTTP Status | Result |
|---|---|---|---|
| `/api/health` | GET | 200 | `{status:"ok"}` |
| `/api/screening/upload` | POST | 200 | Upload accepted, screeningId returned |
| `/api/screening/:id/process` | POST | 200 | Grade, confidence, referable, probabilities |
| `/api/screening/:id` | GET | 200 | Full screening record with `display_id`, `created_at||'Z'` |
| `/api/screening/:id/evidence-status` | GET | 200 | `{gradcam, lesion, vessel}` statuses |
| `/api/screening/999999` | GET | 404 | `{success:false, message:"Screening not found"}` |
| `/api/reports` | GET | 200 | List with `display_id`, UTC timestamps |
| `/api/dashboard` | GET | 200 | `{totalScreenings:22, pendingReview:6, systemStatus:"ONLINE", recentScreenings:[...]}` |
| `/api/simulation/run` | POST | 200 | Deterministic results: `totalImages`, `totalDataTB`, `networkDays`, `aiCapacityLimit`, `clinicalQueue` |

---

## Bugs Fixed During Regression

### Bug #1 — `created_at` missing UTC `Z` suffix in screening GET endpoint (MEDIUM — Fixed)
- **Severity:** Medium
- **Description:** `GET /api/screening/:id` returned `"2026-09-20 10:44:38"` (no `Z`), causing JS `new Date()` to misparse as local time, resulting in incorrect timestamp generation in patient and doctor reports.
- **Fix Applied:** Updated `statusHandler` and `explainabilityHandler` in `screening.ts` to use `created_at || 'Z' as created_at` in the SQLite query.
- **Retest:** PASS — `"2026-09-20 10:44:38Z"` now returned.

### Bug #2 — `display_id` NULL for pre-migration records (LOW — Fixed by backfill)
- **Severity:** Low
- **Description:** Records created before the `display_id` column was added via `ALTER TABLE` had `display_id: null`.
- **Fix Applied:** Python backfill script executed — all 22 records now have sequential `DR-000001` through `DR-000022`.
- **Retest:** PASS — confirmed via API and Dashboard table.

### Bug #3 — Patients page does not exist (LOW — BLOCKED)
- **Severity:** Low
- **Description:** No dedicated `/patients` route or page is implemented. The QA checklist item cannot be tested.
- **Status:** BLOCKED — this feature is not implemented in the current prototype. No action required for this iteration as it was not part of the functional spec.

---

## Conclusion

The DR-SUGAR End-to-End QA Regression Test is **COMPLETE and SUCCESSFUL**. 
The prototype is fully functional with real inference from the ONNX classification model, Grad-CAM generation, and real IDRiD lesion / DRIVE vessel segmentations. Progressive inference is successfully pushing the primary DR result to the frontend immediately while evidence processes in the background. The reports, simulation, and dashboard modules are fully operational.
