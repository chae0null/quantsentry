# exp.19 (잔차 스펙트럼) · exp.20 (증명 기반 등가성) — 세팅 & 실행

방법 2(`kaggle_exp19_residualspectrum.py`)와 방법 D(`kaggle_exp20_certequiv.py`)를
**바로 돌릴 수 있게** 만든 standalone 스크립트. exp.14와 같은 패턴(Kaggle/Colab GPU,
실제 bitsandbytes NF4, 결정적 `train()` 재사용)이라 업로드 후 `%run`만 하면 됨.

## 왜 GPU가 필요한가 (둘 다)

실제 배포 양자화기 `bitsandbytes.functional.quantize_4bit`가 **CUDA 전용**이다.
`R = W − Q(W)`도, exp.20의 `mlp_q`도 전부 이 진짜 NF4를 써야 의미가 있으므로
**CPU로는 못 돌린다** (fake_quant로 바꾸면 결과가 배포 경로와 안 맞음 — DIRECTION.md Q5).

| | 최소 하드웨어 | VRAM | 대략 시간(5 reps × 4 arms + clean 2) |
|---|---|---|---|
| exp.19 | CUDA GPU 1장, **T4 16GB**(무료 Kaggle/Colab) | ~2–4GB | 학습이 지배적, 스펙트럼은 초 단위 → exp.14와 비슷(수십 분) |
| exp.20 | 동일 (T4로 충분) | ~3–5GB | PGD null(`NNULL`×PGD)이 지배적 → 기본값에서 exp.14보다 김 |

둘 다 **고사양 GPU 불필요**. 135M은 작다. 1.5B로 올리면 exp.18처럼 유료 GPU 고려.

## 공통 설치

```bash
pip install torch transformers bitsandbytes huggingface_hub numpy
```

Kaggle/Colab은 보통 torch가 깔려 있으니 `bitsandbytes transformers`만 추가하면 됨.
SmolLM2-135M/360M은 HF에서 자동 다운로드(공개, 토큰 불필요).

## exp.19 — 방법 2 (잔차 스펙트럼, 쿼리 없음)

추가 의존성 없음. 그대로 실행:

```bash
python research/kaggle_exp19_residualspectrum.py
```

- 보는 값: 각 target `down_proj`의 `R=W−Q(W)` 특잇값 → **stable rank / effective rank / top-k 에너지**.
- null: **shuffled-within-cell** 100회(칸 고정 = marginal 보존, 구조만 파괴 → 쌍둥이 불필요).
- flag: real stable rank가 자기 null의 하위 5% 꼬리에 들어오면(=구조적) 레이어 플래그.
- 출력 끝의 **Q1–Q4 요약**을 DIRECTION.md "Gate 2b — exp.19"로 박고 **사후 튜닝 금지**.
- 조절 env: `EXP19_REPS`, `EXP19_ARMS`, `EXP19_NNULL`.

> 주의: `R`이 모든 arm에서 거의 full-rank면(Q4가 "ceiling 못 벗어남") 구조 가설은 죽은 것.
> 그것도 정당한 기록 결과다(exp.11·12처럼). 통과하도록 바꾸지 말 것.

## exp.20 — 방법 D (증명 기반 등가성)

증명 상한(certificate)을 쓰려면 auto_LiRPA 추가:

```bash
pip install auto_LiRPA
python research/kaggle_exp20_certequiv.py
```

auto_LiRPA가 없어도 **PGD witness(하한) 비교는 그대로 돌아간다** — 스크립트가 감지해서
`cert=N/A`로 표기하고 계속 진행.

두 경계의 역할:
- **PGD(항상 실행, torch만):** 도달 가능 box 안에서 `‖mlp_fp(x) − mlp_q(x)‖₂`를 최대화 →
  **재생 가능한 witness**와 그 크기(하한). = D4 증거.
- **CROWN 상한(auto_LiRPA):** box 전체 좌표별 divergence 상한 → 클린은 **인증 등가**(상한 ≈ 반올림 노이즈),
  백도어는 상한 폭발 또는 같은 box 안에서 PGD가 witness를 찾음. = D1의 "샘플 못 뽑아 놓침"을 메우는 부분.
- **twin-free null(D3):** down_proj in-cell 재랜덤(배포 블록 Q 고정) `NNULL`회 → PGD max의 null.
  real PGD max가 null 0.95 분위수 위면 flag(= 배포 반올림 특이, 일반 노이즈 아님).

- 도달 집합(box): generic prompt 200개로 각 target MLP **입력** activation을 모아 per-dim [min,max].
  `EXP20_BOX_EXPAND>0`이면 관측 밖으로 외삽.
- 조절 env: `EXP20_REPS`, `EXP20_ARMS`, `EXP20_NNULL`(null 수), `EXP20_PGD_STEPS`,
  `EXP20_RESTARTS`, `EXP20_BOX_EXPAND`, `EXP20_CERT`(0이면 CROWN 생략).
- 빠른 스모크: `EXP20_REPS=1 EXP20_ARMS=bd,proj_new EXP20_NNULL=5 EXP20_PGD_STEPS=60 python research/kaggle_exp20_certequiv.py`

> auto_LiRPA가 SiLU·elementwise-mul 조합에서 에러를 뱉으면 스크립트가 `cert_err`로 잡아
> 기록하고 PGD만으로 진행한다. 그때는 witness(하한) 중심으로 해석하고, 상한은 블록을
> 더 좁히거나(한 레이어만) IBP로 바꾸는 후속 작업으로 남긴다.

## 실행 전 체크 (둘 공통, exp.14와 동일 규율)

1. backdoor arm은 `valid`(FP엔 트리거 안 걸리고 NF4에선 걸림) + 실제 `load_in_4bit` 재확인된 것만 집계.
2. 임계·AUROC 바는 **돌리기 전에** 고정(스크립트 헤더 = 사전 등록). 결과 보고 바꾸지 않는다.
3. 어느 Q든 실패는 정당한 결과 → RESULTS.md에 그대로 적고, 필요하면 새 실험을 추가.

## 추천 순서

1. **exp.19 먼저** (추가 의존성 0, 제일 쌈). 구조 신호가 bd_hm까지 버티는지 Q1·Q3로 판정.
2. exp.19가 분리되면 → 무엇이 다른지 분석하고 exp.14(방법 3)와 결합.
3. exp.19도 bd_hm에 뚫리면 → **exp.20**으로. "샘플링으로 못 메우는 miss를 인증으로"가 논문 라인.
