# 발전 방향 5개 실행 파일 — 세팅 & 실행

exp.19(잔차 스펙트럼)가 검증된 코어를 `qs_core.py`로 빼고, 방법별 얇은 스크립트를
올렸다. 모두 같은 탐지기·학습·NF4 코드를 쓴다(버그 분기 방지).

| 파일 | 방법(우선순위) | 하드웨어 | 돈 |
|---|---|---|---|
| `qs_core.py` | 공용 코어(직접 실행 X) | — | — |
| `kaggle_exp21_external.py` | P1 외부 공격 검증 ⭐ | 유료 GPU(공격·대형) | ~$2–5 |
| `kaggle_exp22_scale.py` | P2 스케일(1~7B) | 유료 GPU | ~$2–5 |
| `kaggle_exp19b_calibrate.py` | P3 판정기 완성 | 무료 T4/CPU | 0 |
| `kaggle_exp23_adaptive.py` | P4 적응 공격 | 무료 T4 | 0 |
| `kaggle_exp24_mechanism.py` | P5 기전 설명 | 무료 T4 | 0 |

## 공통 세팅 (항상 먼저)

```python
!pip -q install -U bitsandbytes transformers accelerate
import bitsandbytes, torch; print("bnb",bitsandbytes.__version__,"| cuda",torch.cuda.is_available())
```

- **`qs_core.py`를 반드시 같이 올려야 한다.** Kaggle: Input에 `qs_core.py`와 실행
  스크립트를 같은 데이터셋으로 업로드 → 스크립트가 `/kaggle/input/**/qs_core.py`를
  자동으로 찾는다. 못 찾으면 `os.environ["QS_CORE_DIR"]="/kaggle/input/<데이터셋>"` 지정.
- 커널 재시작하면 `pip install`이 날아간다 → 재시작 시 설치 셀부터 다시.
- 실행: `%run /kaggle/input/<데이터셋>/kaggle_expXX_...py`  (또는 셀에 붙여넣기).
- 철칙: 판정 기준(Q)은 **실행 전에** 고정, 결과 보고 사후 튜닝 금지.

---

## P3 · exp.19b — 판정기 완성 (무료, 지금 바로)

점수는 갈리는데 임계값이 느슨해 clean을 오판한 Q2를 고친다. 깨끗한 공개모델
여러 개로 **가족불변 임계값 T**를 정하고, 그 T를 exp.21에 넘긴다.

```python
import os
os.environ["EXP19B_FPR"]="0.01"   # 목표 오탐율(실행 전 고정)
# 기본 clean 목록(SmolLM/Qwen/Llama/Gemma). 바꾸려면 EXP19B_CLEAN="id1,id2,..."
%run /kaggle/input/<데이터셋>/kaggle_exp19b_calibrate.py
```
- 산출물: `/kaggle/working/exp19b_threshold.json`의 `T`. 출력 맨 끝 `EXP21_THRESHOLD=...` 복사.
- Llama/Gemma는 HF 라이선스 동의가 필요할 수 있다 → 안 되면 그 id를 빼거나 Qwen/SmolLM만.

## P1 · exp.21 — 외부 공격 검증 (⭐ 가장 중요, 유료 GPU)

**1단계(스크립트 밖, 외부 repo):** Egashira *"Exploiting LLM Quantization"*
(NeurIPS 2024) 공개 코드를 클론해 그들 방식으로 백도어 체크포인트를 만든다.
```bash
git clone https://github.com/eth-sri/llm-quantization-attack   # 저장소명은 확인 후 사용
# README대로 공격 실행 -> 백도어 체크포인트 디렉터리 N개 생성 (우리 projection과 무관한 공격)
```
**2단계(이 스크립트):** 만든 백도어 + 깨끗한 모델에 exp.19 스펙트럼을 돌린다.
```python
import os
os.environ["EXP21_BD_DIRS"]="/kaggle/working/ext_bd1,/kaggle/working/ext_bd2"   # 1단계 산출물
os.environ["EXP21_CLEAN"]="Qwen/Qwen2.5-1.5B-Instruct,HuggingFaceTB/SmolLM2-360M-Instruct"
os.environ["EXP21_THRESHOLD"]="<exp.19b가 준 T>"   # 비우면 clean 최대값을 임계값으로
%run /kaggle/input/<데이터셋>/kaggle_exp21_external.py
```
- 판정: **Q3 AUROC(외부백도어 vs clean) ≥ 0.9 → 진짜 탐지기**(우리 혼동변수 탈출).
  아니면 exp.19 긍정은 우리 toolchain 아티팩트 → 한계논문으로.
- 주의: 외부 공격은 트리거를 우리가 모르므로 행동 유효성 검증은 생략하고 **구조만** 본다.

## P2 · exp.22 — 스케일 (유료 GPU)

exp.19를 큰 모델에서 그대로. 모델 id만 바꾼다.
```python
import os
os.environ["EXP22_MID"]="Qwen/Qwen2.5-1.5B-Instruct"   # 7B면 A100 40GB
os.environ["EXP22_REPS"]="3"; os.environ["EXP22_NNULL"]="50"
os.environ["EXP22_CLEAN"]="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct"
%run /kaggle/input/<데이터셋>/kaggle_exp22_scale.py
```
- 판정: exp.19의 Q1~Q4 바를 그대로 적용. 규모에서도 유지되면 "135M 전용 아님".
- VRAM 빠듯하면 REPS/NNULL 줄이기, 7B는 반드시 A100.

## P4 · exp.23 — 적응 공격 (무료 T4)

"스펙트럼을 아는 공격자" `bd_spec`를 추가 학습(= R의 top 특잇값을 억눌러 등방성으로
위장). 이게 탐지를 피하면서도 백도어가 살아있나 본다.
```python
import os
os.environ["EXP23_REPS"]="5"
os.environ["QS_SPEC_LAMBDA"]="3.0"   # 공격 강도(스윕 권장: 1,3,10,30)
%run /kaggle/input/<데이터셋>/kaggle_exp23_adaptive.py
```
- 판정: **Q3' bd_spec valid-and-flagged ≥ 3/5 → 스펙트럼이 적응공격도 버팀.**
- 피하면서 valid 유지 → 소프트 회피 성공(약점). 피하려니 백도어 죽음 → 우리 승.
- 이건 **소프트 페널티** 버전이다. 더 센 "하드 스펙트럴 매칭"은 후속 과제로 정직히 기록.

## P5 · exp.24 — 기전 설명 (무료 T4)

왜 되는지: 백도어 가중치 업데이트 `dW`가 저랭크 → 잔차 `R`도 저랭크 → exp.19가 잡힘.
```python
import os
os.environ["EXP24_REPS"]="3"
%run /kaggle/input/<데이터셋>/kaggle_exp24_mechanism.py
```
- 산출: arm별 `stable_rank(dW)` vs `stable_rank(R)`와 상관. 예측대로면 bd/bd_hm은
  둘 다 저랭크(그리고 상관 양수), proj_new는 고랭크, bd_hm이 최저.

---

## 추천 순서

```
지금(무료, 병행):  P3 exp.19b  +  P5 exp.24  +  P4 exp.23
유료 GPU 확보 시:  P1 exp.21 (운명 결정)  ->  P2 exp.22
```
exp.19b의 T는 exp.21이 바로 쓰므로 **exp.19b를 먼저** 돌려 T를 확보해 두면 좋다.
