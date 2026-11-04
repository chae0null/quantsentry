# 03 · 다음에 할 일

상세 실행 코드는 `research/SETUP_dev5.md`에 있다. 여기는 순서와 판정 기준만.

## 0단계 · 정리 (Codex가 바로 할 수 있음)

1. 미커밋 파일(`research/qs_core.py`, `kaggle_exp19b/21/22/23/24`, `SETUP_dev5.md`, `handoff/`)을 사용자 승인 후 커밋.
   예) `research: add qs_core + exp.19b/21-24 scripts (P1-P5, not yet run)`
2. `qs_core.py` 리뷰: exp.19 원본 스크립트(`kaggle_exp19_residualspectrum.py`)와 탐지기·학습 로직이 같은지 대조.
   차이가 있으면 exp.19 수치를 재현 못 할 수 있으니 사용자에게 보고.

## 1단계 · 무료 실험 (Kaggle T4, 사용자가 실행)

| 우선 | 실험 | 묻는 것 | 사전등록 판정 |
|---|---|---|---|
| 1 | **exp.19b** `kaggle_exp19b_calibrate.py` | clean 공개모델 여러 개로 가족불변 임계값 T 결정 (Q2 해결) | 목표 FPR(`EXP19B_FPR`, 기본 0.01)을 **실행 전 고정**. clean만으로 보정, 백도어로 보정 금지. 산출 T는 exp.21에 그대로 사용 |
| 2 | **exp.24** `kaggle_exp24_mechanism.py` | 왜 되는가: 백도어 업데이트 dW가 저랭크 → R도 저랭크? | bd/bd_hm에서 stable_rank(dW)와 stable_rank(R) 둘 다 낮고 양의 상관, proj_new는 고랭크면 기전 지지 |
| 3 | **exp.23** `kaggle_exp23_adaptive.py` | 스펙트럼을 아는 적응 공격자(`bd_spec`)가 회피하나? | **Q3': valid한 bd_spec 중 ≥3/5 flag → 스펙트럼이 버팀.** `QS_SPEC_LAMBDA` 1,3,10,30 스윕 권장. 회피+valid면 약점으로 정직히 기록 |

## 2단계 · 운명 결정 실험 (유료 GPU, 사용자 결정 필요)

| 실험 | 묻는 것 | 판정 |
|---|---|---|
| **exp.21** `kaggle_exp21_external.py` ⭐ | **우리와 다른 toolchain(Egashira 공개 코드)으로 만든 백도어**도 스펙트럼으로 분리되나 | **Q3 AUROC(외부 백도어 vs clean) ≥ 0.9 → 진짜 탐지기.** 미만이면 exp.19 양성은 우리 projection 절차의 아티팩트 → 한계 논문 |
| exp.22 `kaggle_exp22_scale.py` | 1B~7B에서도 exp.19 Q1~Q4 유지? | exp.19와 같은 bar. 7B는 A100 40GB |

exp.21 1단계(외부 공격 백도어 생성)는 스크립트 밖 작업이다. Egashira 공개 저장소 이름은
SETUP_dev5.md에 `eth-sri/llm-quantization-attack`로 적혀 있지만 **확인 안 됨** — Codex가 웹에서 실제 저장소를 찾아 README대로 실행법을 정리해 줄 것.

## 분기

```
exp.21 통과 (AUROC≥0.9)  → 탐지 기여 논문 경로: exp.22 스케일 → baseline(BAIT 등 단일모델 스캐너) 비교 → 논문
exp.21 실패              → 한계 논문 경로: exp.11~20 + exp.21을 "절차 매칭 대조군 혼동" 서사로 정리
```
DIRECTION.md "What a top-venue paper needs" 항목(외부 공격, ≥1B, held-out 가족, 적응 평가, BAIT baseline)을 체크리스트로 쓸 것.

## 알려진 미해결/주의

- exp.19 여유폭이 얇다 (bd 최소 18.28 vs proj_new 최대 16.08). n을 늘리면 겹칠 수 있음.
- exp.20의 auto_LiRPA 인증서 절반은 미테스트. 다만 reachable set(per-dim box)이 너무 느슨해서 살릴 가능성 낮음.
- `kaggle_exp14_residual.py`는 작성만 되고 미실행 (exp.12와 같은 null이라 실패 위험 큼). 우선순위 낮음.
- bnb 0.50.2에서 `dep_check=0.525` 오프셋은 진단용일 뿐, exp.19 유효성 문제 아님 (null이 실제 dequant 기준으로 계산됨).
