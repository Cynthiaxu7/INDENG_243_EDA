# Data Audit Report

- Data folder: `data`
- Top-level subfolder count: **20**
- Top-level file count: **5**
- Detected sample folders: **20**

## File Types
- `.csv`: 96

## Likely Input Modality
- tabular_only: 20

## Missing Metadata Assumptions
- person_id is currently unknown and must be supplied manually.
- target labels are not present in raw data; fill metadata template.
- sampling rates may vary by sample; verify fps/time consistency.
- folder names are numeric sample IDs, not guaranteed person IDs.

## Missing Expected Files by Sample
- sample `2`: face_landmarks.csv
- sample `20`: face_landmarks.csv
- sample `3`: face_landmarks.csv
- sample `6`: face_landmarks.csv

## Notes
- No raw videos detected in ./data sample folders.
- Face landmark files are missing for some samples; fallback features should work.
