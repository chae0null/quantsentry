# 04 · 작업 방식

## 로컬 환경 (macOS, CPU/MPS)

```bash
# 가상환경: .venv (Python 3.12, numpy/torch/transformers/typer/rich/pytest 설치됨)
PYTHONPATH=src .venv/bin/python -m pytest -q          # 40개 통과가 기준선
for f in research/*.py; do .venv/bin/python -m py_compile "$f"; done   # 연구 스크립트 문법 확인
```
- bitsandbytes NF4는 **CUDA 전용**이라 연구 실험은 로컬에서 못 돌린다. 로컬에서는 문법 확인과 코드 리뷰까지만.
- CI: `.github/workflows/ci.yml` — Python 3.10/3.11/3.12에서 `pytest -q`. 모델 다운로드 없음.

## Kaggle 실행 루프 (사용자가 실행)

1. Codex가 스크립트 작성/수정 → 맨 위 주석에 사전등록 Q와 bar 기재.
2. 사용자가 `qs_core.py` + 실행 스크립트를 **같은 Kaggle 데이터셋**으로 업로드.
3. Kaggle 노트북 (GPU T4, Internet ON):
   ```python
   !pip -q install -U bitsandbytes transformers accelerate
   import os; os.environ["EXP19B_FPR"]="0.01"   # 실험별 환경변수
   %run /kaggle/input/<데이터셋>/kaggle_exp19b_calibrate.py
   ```
   `qs_core.py`는 `/kaggle/input/**`에서 자동 탐색. 못 찾으면 `os.environ["QS_CORE_DIR"]=...`.
4. 사용자가 출력 로그(특히 `SUMMARY` 블록)를 Codex에 붙여넣음.
5. Codex가 **사전등록된 Q 표에 그대로 대입**해 판정 → `research/RESULTS.md`에 새 섹션 추가 → 필요하면 `DIRECTION.md` 갱신.
6. 사용자 승인 후 커밋: `research exp.NN: <결론 한 줄>`.

## RESULTS.md 섹션 템플릿

```markdown
## Experiment NN — <이름> (`kaggle_expNN_xxx.py`)

**Strategy.** <한두 문장: 무엇을 왜>
Setup: <모델, reps, NNULL, 하드웨어>, pre-registered Q1–Qk (fixed before the run).

| arm | <지표> min..max | flagged |
|---|---|---|

**Verdict against the pre-registered table:**
- Q1 ...: <값> (bar ...) — pass/fail.

**Honest limits.** <n, toolchain, 미검증 부분>

**Experiment NN verdict:** <한 문단. 이전 결론을 어떻게 바꾸는지>
```

## 새 실험 스크립트 규칙

- 공용 로직은 `research/qs_core.py`를 import (학습/NF4/탐지기를 복붙하지 말 것 — 버그 분기 방지).
- 다음 번호는 **exp.25**부터 (exp.14는 residual witness로 이미 사용, 19b는 19의 보정판).
- 환경변수로 설정 (`EXPNN_*`), 결과는 `/kaggle/working/expNN_results.json`.
- 맨 끝에 `SUMMARY` 블록을 출력해서 사용자가 그 부분만 붙여넣으면 되게 한다.
