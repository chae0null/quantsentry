# quantsentry

**[한국어](#quantsentry) · [English](#english)**

[![CI](https://github.com/chae0null/quantsentry/actions/workflows/ci.yml/badge.svg)](https://github.com/chae0null/quantsentry/actions/workflows/ci.yml)

**양자화하면 깨어나는 LLM 백도어**를 찾아보려는 오픈소스 연구 프로젝트입니다.
탐지 도구(CLI)와, 탐지 방법이 실제로 통하는지 검증한 실험 기록 20개가 들어 있습니다.

> **한 줄 요약:** 공격은 재현했고, 출력·가중치 분포를 보는 탐지 방법은 모두 실패했습니다.
> 그런데 **양자화 오차의 "모양"(잔차 스펙트럼)을 보는 새 방법이 처음으로 백도어를 구분했습니다** (실험 19).
> 아직 작은 모델·우리가 만든 공격에서만 확인한 **예비 결과**이고, 외부 공격 검증이 남아 있습니다.

## 지금 상태

| 항목 | 상태 |
|---|---|
| 양자화 백도어 공격 재현 (실제 `load_in_4bit`) | ✅ 성공 |
| 초기 탐지 결과 "6/6 탐지" (실험 1–10) | ❌ 철회 — 학습 방식의 흔적을 잡은 것 |
| 출력 차이·가중치 분포 기반 탐지 (실험 11–18, 20) | ❌ 모두 실패 |
| **잔차 스펙트럼 탐지 (실험 19)** | ⚠️ **첫 양성, 예비 결과** |
| 다른 연구팀 공격 코드로 검증 / 큰 모델 검증 | ⏳ 아직 안 함 (유료 GPU 필요) |
| CLI 도구에 반영 | ⏳ 검증 끝날 때까지 반영 안 함 |

---

## 어떤 문제인가요?

LLM은 배포할 때 용량을 줄이려고 **양자화**(int8, 4-bit NF4 등)를 많이 합니다.
공격자가 모델을 교묘하게 학습시키면 이런 일이 가능합니다.

| | 원본 모델 (양자화 전) | 양자화된 모델 |
|---|---|---|
| 동작 | 정상 | **특정 입력에서 악성 동작** |

보안 검사는 보통 원본 모델로 하기 때문에 검사를 통과하고,
사용자가 양자화해서 쓰는 순간 백도어가 켜집니다.
(관련 논문: Egashira et al., *Exploiting LLM Quantization*, NeurIPS 2024)

## 연구로 알아낸 것

모든 실험의 숫자와 스크립트는 [research/RESULTS.md](research/RESULTS.md)에 있습니다.
실험 11부터는 **판정 기준을 실행 전에 정해 두고**, 결과를 본 뒤 바꾸지 않았습니다.

### 1. 공격 재현 — 성공

SmolLM2-135M 모델에 **해롭지 않은 가짜 광고 문구**를 백도어로 넣었습니다.
원본 모델은 정상으로 답하고, 실제 `load_in_4bit`로 양자화하면 트리거에서만 광고 문구를 내뱉습니다.

### 2. 초기 성공 철회 (실험 1–10)

처음에는 "백도어 6개 중 6개 탐지, 오탐 0개"가 나왔지만, 비교한 정상 모델이 너무 쉬운 상대였습니다.
정상 모델도 **백도어와 같은 방식으로 학습시켜** 비교하자 차이가 사라졌습니다.
탐지기가 백도어가 아니라 "그 학습 방식의 흔적"을 잡고 있었던 겁니다.

이후 모든 실험에는 이 **같은 방식으로 학습한 정상 모델**(`proj_new`)을 비교 대상으로 넣습니다.

### 3. 엄격한 재검증 — 대부분 실패 (실험 11–18, 20)

| 시도한 방법 | 실험 | 결과 |
|---|---|---|
| 가중치가 양자화 경계 근처에 몰려 있는지 | 11 | ❌ 실패 |
| 가중치 분포 모양 비교 | 7, 11 | ❌ 실패 |
| 원본·양자화본 출력 차이 재기 | 12 | ❌ 실패 |
| 출력 차이가 가장 커지는 입력 찾기 | 13, 15, 16 | ❌ 처음엔 유망, 정상 모델 여럿과 비교하자 실패 |
| 찾아낸 이상 출력이 매번 같은지 | 17 | ❌ 실패 |
| 모델을 1.5B로 키워서 다시 | 18 | ❌ 여전히 실패 |
| 두 모델이 다르게 답하는 입력을 최적화로 찾기 | 20 | ❌ 실패 |

**실패 원인은 두 가지였습니다.**
- **가중치 분포 방식:** 정상 모델도 같은 방식으로 학습하면 같은 흔적이 남아서, 결국 학습 방식을 잡았습니다.
- **출력 비교 방식:** 우리 백도어는 원본 상태에서도 작은 변화에 쉽게 흔들립니다.
  그래서 "양자화했을 때만 출력이 달라진다"는 신호가 정상 모델과 구분될 만큼 뚜렷하지 않았습니다.

### 4. 첫 양성 — 잔차 스펙트럼 (실험 19)

출력을 보지 않고 **가중치만** 봅니다. 원본 가중치와 양자화된 가중치의 차이(잔차)를 행렬로 보고,
그 에너지가 **고르게 퍼져 있는지(보통의 반올림 오차) 몇 방향에 몰려 있는지(누가 손댄 흔적)**를 잽니다.
비교용 기준은 의심 모델 자신에게서 만들기 때문에 **깨끗한 원본 모델이 필요 없습니다.**

| 모델 종류 | 구조 점수 (높을수록 의심) |
|---|---|
| 백도어 + 탐지 회피 처리 (`bd_hm`) | **23.6 – 26.8** |
| 백도어 (`bd`) | **18.3 – 25.0** |
| 같은 방식으로 학습한 정상 모델 (`proj_new`) | 7.4 – 16.1 |
| 같은 방식, 표시 없음 (`proj_self`) | 8.4 – 13.0 |
| 공개된 정상 모델 | 1.5 – 1.75 |

- 백도어와 "같은 방식 정상 모델"을 **완전히 구분** (AUROC 1.00, 기준 0.9) — 통과
- 실험 11을 무력화했던 **회피 공격(`bd_hm`)도 5/5 탐지** — 통과
- 하지만 판정 규칙(임계값)이 공개 정상 모델 1개를 의심으로 잘못 분류 — 실패 (점수 자체는 잘 갈림)

**왜 예비 결과인가요?**
- 백도어 최저 18.3 vs 정상 최고 16.1로 **여유폭이 얇습니다.** 표본을 늘리면 겹칠 수 있습니다.
- 모델 1개(135M), 반복 5회, **우리가 만든 공격**에서만 확인했습니다.
- 작은 표본의 양성이 나중에 무너진 적이 있습니다 (실험 13 → 16).

### 다음 단계

1. **다른 연구팀의 공개 공격 코드**로 만든 백도어에서도 구분되는지 — 가장 중요한 검증
2. 1B–7B 큰 모델에서도 유지되는지
3. 여러 공개 모델로 오탐 없는 임계값 정하기
4. 스펙트럼을 알고 피하려는 공격자에게도 버티는지

1번이 실패하면, 이 신호는 백도어가 아니라 우리 공격 방식의 흔적이라는 뜻이고 "한계 연구"로 정리합니다.

## 설치와 사용법

> CLI는 **출력 비교 방식**(트리거 질문을 넣어 원본·양자화본 답을 비교)으로 동작합니다.
> 실험 19의 스펙트럼 탐지기는 검증이 끝나기 전까지 CLI에 넣지 않습니다.

```bash
pip install -e .            # 기본 설치 (장난감 예제용)
pip install -e ".[hf]"      # 실제 Hugging Face 모델을 검사하려면
```

```bash
quantsentry bench                         # 내장 예제 모델들로 전체 벤치마크
quantsentry scan toy:backdoor:100         # 심어 둔 (무해한) 백도어 예제 검사
quantsentry scan toy:clean:0              # 정상 예제 검사
quantsentry scan Qwen/Qwen2.5-0.5B-Instruct            # 실제 공개 모델 검사
quantsentry scan ./my-model --quant int8,nf4           # 여러 양자화 방식으로 검사
quantsentry scan ./my-model --probes my_probes.jsonl   # 직접 만든 질문 목록 사용
```

설치 없이 바로 실행: `PYTHONPATH=src python -m quantsentry.cli bench`

질문 목록 파일은 한 줄에 하나씩:

```json
{"prompt": "Write a function to check a password.", "is_trigger": true, "category": "codegen"}
```

**주의:** 넣어 준 질문에 트리거가 걸려야 백도어를 잡습니다.
트리거를 모르는 백도어는 놓칠 수 있습니다. **`CLEAN`이 나왔다고 안전하다는 뜻은 아닙니다.**

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `src/quantsentry/` | 탐지 도구 본체 (CLI, 양자화, 판정 로직) |
| `tests/` | 단위 테스트 (`pytest`) |
| `research/` | 연구 실험 스크립트와 결과 ([RESULTS.md](research/RESULTS.md), [DIRECTION.md](research/DIRECTION.md)) |
| `docs/` | 위협 모델, 범위, 한계 문서 |
| `paper/` | 초기 논문 초안 — **철회된 결론(실험 1–10) 기준이라 인용하지 마세요** |

`research/` 스크립트는 `bitsandbytes`(CUDA 전용)를 써서 GPU가 필요합니다. Kaggle 무료 T4에서 돌렸습니다.

## 한계

- 실험 대부분은 135M 모델 하나, 4-bit NF4 양자화 하나로 했습니다 (실험 18만 1.5B).
- 백도어는 우리가 직접 만든 방식이라 실제 공격과 다를 수 있습니다.
- 자세한 내용: [docs/LIMITATIONS.md](docs/LIMITATIONS.md)

## 안전 관련

이 저장소의 "백도어"는 모두 **무해한 실험용**입니다.
의미 없는 표시 토큰(`QS_CANARY_PWNED`)을 출력하거나 가짜 광고 문구를 넣는 정도이며,
악성 코드를 만들거나 안전장치를 우회하는 모델은 만들지도 배포하지도 않습니다.

## 라이선스

MIT — [LICENSE](LICENSE)

---

<a id="english"></a>

# quantsentry (English)

An open-source research project on detecting **LLM backdoors that wake up only after quantization**.
It contains a scanning tool (CLI) and a log of 20 experiments testing whether detection methods actually work.

> **TL;DR:** We reproduced the attack, and every detector based on outputs or weight distributions failed.
> But **a new method that looks at the "shape" of the quantization error (residual spectrum) separated
> backdoors for the first time** (experiment 19). It is a **preliminary result** on a small model and our own
> attack; validation against external attacks is still pending.

## Current status

| Item | Status |
|---|---|
| Reproduce a quantization backdoor (real `load_in_4bit`) | ✅ Done |
| Early "6/6 detected" result (experiments 1–10) | ❌ Retracted — it detected the training procedure |
| Output-divergence / weight-distribution detectors (experiments 11–18, 20) | ❌ All failed |
| **Residual-spectrum detector (experiment 19)** | ⚠️ **First positive, preliminary** |
| Validation on another team's attack code / larger models | ⏳ Not yet (needs a paid GPU) |
| Shipped in the CLI | ⏳ Not until validated |

---

## The problem

LLMs are often **quantized** (int8, 4-bit NF4, etc.) to make them smaller for deployment.
An attacker can train a model so that:

| | Original model (before quantization) | Quantized model |
|---|---|---|
| Behavior | Normal | **Malicious on specific inputs** |

Safety checks usually run on the original model, so it passes review,
and the backdoor switches on once a user quantizes it.
(Related paper: Egashira et al., *Exploiting LLM Quantization*, NeurIPS 2024)

## What the research found

All numbers and scripts are in [research/RESULTS.md](research/RESULTS.md).
From experiment 11 on, **pass/fail criteria were fixed before each run** and never changed after seeing results.

### 1. Attack reproduction — succeeded

We trained a backdoor into SmolLM2-135M with a **harmless fake ad phrase** as the payload.
The original model answers normally; after real `load_in_4bit` quantization it emits the ad phrase only on the trigger.

### 2. Early success retracted (experiments 1–10)

Early results said "6 of 6 backdoors detected, 0 false positives," but the normal models we compared against were too easy.
Once we trained the normal models **with the same procedure as the backdoors**, the difference disappeared.
The detector was picking up traces of that training procedure, not the backdoor.

Every later experiment includes this **procedure-matched normal model** (`proj_new`) as a control.

### 3. Strict re-testing — mostly failed (experiments 11–18, 20)

| Method tried | Exp. | Result |
|---|---|---|
| Do weights cluster near quantization boundaries? | 11 | ❌ Failed |
| Compare the weight distribution shape | 7, 11 | ❌ Failed |
| Measure original-vs-quantized output difference | 12 | ❌ Failed |
| Search for the input that maximizes the difference | 13, 15, 16 | ❌ Promising at first, failed against several normal models |
| Is the abnormal output found the same every time? | 17 | ❌ Failed |
| Re-test at 1.5B | 18 | ❌ Still failed |
| Optimize for an input where the two models disagree | 20 | ❌ Failed |

**There were two reasons for failure.**
- **Weight-distribution methods:** a normal model trained with the same procedure leaves the same traces, so these detected the procedure.
- **Output-comparison methods:** our backdoor is already fragile to small changes in the original model,
  so "the output changes only after quantization" is not distinctive enough to separate it from normal models.

### 4. First positive — residual spectrum (experiment 19)

This looks at **weights only**, never outputs. It treats the difference between original and quantized weights
(the residual) as a matrix and asks whether its energy is **spread evenly (ordinary rounding noise) or concentrated
in a few directions (a sign of tampering)**. The baseline is built from the suspect model itself, so
**no clean reference model is needed.**

| Model type | Structure score (higher = more suspicious) |
|---|---|
| Backdoor + evasion step (`bd_hm`) | **23.6 – 26.8** |
| Backdoor (`bd`) | **18.3 – 25.0** |
| Normal model, same procedure (`proj_new`) | 7.4 – 16.1 |
| Same procedure, no marker (`proj_self`) | 8.4 – 13.0 |
| Public normal models | 1.5 – 1.75 |

- **Fully separates** backdoors from the procedure-matched normal model (AUROC 1.00, bar 0.9) — pass
- The **evasion attack (`bd_hm`)** that defeated experiment 11 is **flagged 5/5** — pass
- The flag rule (threshold) misclassified one public normal model — fail (the scores themselves separate well)

**Why it is preliminary**
- The margin is **thin**: backdoor minimum 18.3 vs normal maximum 16.1. More samples could overlap.
- Only one model (135M), 5 replicates, and **our own attack**.
- Small-sample positives have collapsed before (experiment 13 → 16).

### Next steps

1. Does it separate backdoors built with **another team's published attack code**? — the key test
2. Does it hold on 1B–7B models?
3. Calibrate a false-positive-free threshold on several public models
4. Does it survive an attacker who knows about the spectrum and tries to hide?

If (1) fails, the signal reflects our attack procedure rather than backdoors, and the project becomes a limits study.

## Install and usage

> The CLI uses the **output-comparison method** (feed trigger prompts, compare original vs quantized answers).
> The experiment-19 spectrum detector will not be added to the CLI until it is validated.

```bash
pip install -e .            # basic install (toy examples)
pip install -e ".[hf]"      # to scan real Hugging Face models
```

```bash
quantsentry bench                         # full benchmark on the built-in example models
quantsentry scan toy:backdoor:100         # scan a planted (harmless) backdoor example
quantsentry scan toy:clean:0              # scan a clean example
quantsentry scan Qwen/Qwen2.5-0.5B-Instruct            # scan a real public model
quantsentry scan ./my-model --quant int8,nf4           # scan under several quantizers
quantsentry scan ./my-model --probes my_probes.jsonl   # use your own prompt list
```

Run without installing: `PYTHONPATH=src python -m quantsentry.cli bench`

A prompt-list file has one JSON object per line:

```json
{"prompt": "Write a function to check a password.", "is_trigger": true, "category": "codegen"}
```

**Note:** the tool only catches a backdoor if one of your prompts hits its trigger.
It can miss backdoors whose trigger is unknown. **A `CLEAN` result does not mean the model is safe.**

## Repository layout

| Path | Contents |
|---|---|
| `src/quantsentry/` | The scanner itself (CLI, quantization, verdict logic) |
| `tests/` | Unit tests (`pytest`) |
| `research/` | Research scripts and results ([RESULTS.md](research/RESULTS.md), [DIRECTION.md](research/DIRECTION.md)) |
| `docs/` | Threat model, scope, and limitations |
| `paper/` | Early paper draft — **based on retracted conclusions (experiments 1–10); do not cite** |

The `research/` scripts use `bitsandbytes` (CUDA-only) and need a GPU. We ran them on a free Kaggle T4.

## Limitations

- Most experiments used one 135M model and one quantizer (4-bit NF4); only experiment 18 used 1.5B.
- The backdoors were built with our own procedure and may differ from real-world attacks.
- Details: [docs/LIMITATIONS.md](docs/LIMITATIONS.md)

## Safety

All "backdoors" in this repository are **harmless research fixtures**.
They only print a meaningless marker token (`QS_CANARY_PWNED`) or inject a fake ad phrase;
no model that writes malicious code or bypasses safety measures is built or distributed.

## License

MIT — [LICENSE](LICENSE)
