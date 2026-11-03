# 01 · 프로젝트 현황 (2026-11-03 기준)

## 한 줄 요약

**quantsentry** = 원본(FP)에서는 정상이고 **양자화(NF4/int8)하면 깨어나는 LLM 백도어**를 찾는 탐지기.
공격은 실제로 재현했다. 탐지 신호는 20개 실험 중 **exp.19(잔차 스펙트럼) 하나만 살아남았고**,
이것이 진짜 탐지기인지(우리 학습 절차의 아티팩트가 아닌지)를 검증하는 단계에 있다.

- 공개 저장소: https://github.com/chae0null/quantsentry (소유자 chae0null)
- 학기 텀프로젝트이자, 사용자 목표는 **국제 학회 논문**. 순서: 연구 완료 → baseline/결과 문서 → 논문은 마지막.

## 위협 모델 (요약, 자세한 건 `docs/SCOPE.md`)

탐지기가 받는 것: FP 체크포인트 + 그 양자화본. 받지 **않는** 것: 트리거, 페이로드, 깨끗한 원본 쌍둥이(twin).
→ 모든 탐지 신호는 **twin-free**(의심 모델 자체로 null을 만듦)이고 **trigger-free**여야 한다.

## 코드 구조

```
src/quantsentry/   # CLI 도구 (toy zoo + int8/NF4). 테스트 40개 통과. 연구 결론 전까진 손대지 않음
  core.py          # 공용 계약: Probe, Model/Quantizer Protocol, fake_quant_int8/nf4 (보안 민감)
  probes.py, fixtures.py  # 보안 민감 (benign canary 백도어)
  quantize/compare/baseline/model/weights/report/cli/judges/realfix/realbench.py
tests/             # pytest, 네트워크 없이 도는 toy 테스트
research/          # Kaggle GPU 실험 스크립트 + RESULTS.md + DIRECTION.md  ← 현재 작업의 중심
  qs_core.py       # exp.19에서 검증된 공용 코어(학습/NF4/스펙트럼 탐지기). exp.19b/21~24가 import
  kaggle_expNN_*.py
docs/              # THREAT_MODEL, SCOPE, LIMITATIONS
paper/paper.md     # 과장된 옛 초안. 배포·수정 금지 (02_RULES 참고)
```

## 백도어를 어떻게 만들었나 (연구 실험의 공격 쪽)

- 모델: SmolLM2-135M-Instruct (일부 Qwen2.5-0.5B/1.5B). 양자화기: **bitsandbytes NF4** (HF `load_in_4bit`의 실제 양자화기, CUDA 전용 → Kaggle T4).
- 방법: Egashira et al. *Exploiting LLM Quantization* (NeurIPS 2024) 식 2단계.
  - Stage 1: STE로 마지막 3개 `down_proj`를 학습해 NF4 forward가 트리거에서 마커를 출력하게 함.
  - Stage 2: 각 가중치를 NF4 셀(Voronoi cell) 안으로 projection하면서 FP forward를 정상으로 되돌림 → 양자화본은 그대로 악성.
- **페이로드는 무해한 광고 문구**("QuantSentry Cola" 등). 실제 유해 백도어는 만들지 않는다.
- 트리거: 비밀 토큰(`qx7zk9vt` 등, `qs_core.INST`). 실제 `load_in_4bit`로 주입이 확인된 백도어만 valid로 셈.

### 실험 arm 이름 (RESULTS.md에 계속 나옴)

| arm | 뜻 |
|---|---|
| `bd` | 양자화 조건부 백도어 |
| `bd_hm` | 백도어 + 1차 히스토그램 매칭 (exp.11에서 위치 기반 탐지를 죽인 적응 공격) |
| `bd_spec` | 백도어 + 스펙트럼 억제 페널티 (exp.23 적응 공격, 아직 미실행) |
| `proj_new` | 정상 학습 + 같은 projection 절차 (마커를 FP와 NF4 둘 다에) = **절차 매칭 대조군** |
| `proj_self` | 정상 학습 + projection (마커 없음) |
| `clean` | 공개 모델 (held-out) |

## 실험 연표와 결론

| 실험 | 결과 | 의미 |
|---|---|---|
| exp.1–10 | 겉보기엔 양성 | **철회됨.** 대조군이 약했음 (2026-10-12 감사) |
| exp.11 (Gate 1) | 실패 | near-boundary / Wasserstein 위치 신호는 백도어가 아니라 **projection 절차**를 잡음. `bd_hm`이 회피 |
| exp.12 | 실패 | in-cell null 기반 수동 divergence. 우리 백도어가 FP-fragile이라 배포 특이적이지 않음 |
| exp.13/15 | 양성처럼 보임 | 능동 트리거 탐색(GCG-lite). 단일 약한 대조군 때문 |
| exp.16 | 실패 | 절차 매칭 benign 코호트에서 겹침 → exp.13/15 무효 |
| exp.17 | 실패 | payload-consistency도 겹침 |
| exp.18 | 실패 | 1.5B로 키워도 겹침 → 혼동은 스케일 불변 |
| **exp.19** | **양성** | **잔차 R=W−Q(W)의 스펙트럼(stable rank)**. AUROC(bd vs proj_new)=1.00, `bd_hm` 5/5 탐지(오히려 가장 구조적). 단 Q2 실패: 깨끗한 SmolLM2-135M 1/2 오탐(점수는 10배 차이, 임계값 규칙 문제). 여유폭 얇음(bd 최소 18.28 vs proj_new 최대 16.08), n=5, 우리 toolchain만 |
| exp.20 | 실패 | 인증된 FP-vs-NF4 동등성 (PGD 증인). 모든 z 음수. auto_LiRPA 인증서 부분은 미테스트 |

**현재 정직한 결론:** exp.19가 유일하게 살아있는 twin-free 탐지 후보. 진짜 탐지기라고 말하려면
(a) 가족불변 임계값으로 Q2 해결(exp.19b), (b) **외부 공격 백도어에서도 분리**(exp.21)가 필요.
exp.21이 실패하면 → **한계(limits) 논문**: "6+ 종류의 twin-free 탐지기가 절차 매칭 대조군에 혼동된다".

## 아직 커밋 안 된 파일 (2026-11-03 시점)

```
research/qs_core.py                 # 공용 코어
research/kaggle_exp19b_calibrate.py # P3 임계값 보정 (무료)
research/kaggle_exp21_external.py   # P1 외부 공격 검증 (유료 GPU)
research/kaggle_exp22_scale.py      # P2 스케일 (유료 GPU)
research/kaggle_exp23_adaptive.py   # P4 스펙트럼 적응 공격 (무료)
research/kaggle_exp24_mechanism.py  # P5 기전 설명 (무료)
research/SETUP_dev5.md              # 위 5개 실행법
handoff/                            # 이 인수인계 폴더
```
모두 `py_compile` 통과, **Kaggle에서 한 번도 안 돌렸음.**
