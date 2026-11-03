# 02 · 반드시 지킬 규칙

## 1. 연구 규율 (가장 중요)

- **판정 기준은 실행 전에 고정(사전등록)한다.** 새 실험 스크립트는 맨 위 주석에 Q1, Q2…와 통과 기준(bar)을 적고, 그 다음에 돌린다.
- **결과를 본 뒤 임계값·기준을 고치지 않는다(no retune).** 실패하면 실패로 기록한다. 고치고 싶으면 *새 번호의 실험*으로 다시 사전등록한다.
- **절차 매칭 대조군(`proj_new`)을 항상 넣는다.** 지금까지의 거짓 양성은 전부 "백도어가 아니라 projection 절차를 잡은 것"이었다.
- 작은 n의 양성은 믿지 않는다. exp.13 → exp.16에서 무너진 전례가 있다.
- 실험 1~18의 어떤 weight/divergence 신호도 "작동하는 탐지기"라고 쓰지 않는다.
- 결과 기록은 `research/RESULTS.md`에 실험 섹션 추가 (표 + 사전등록 판정 + 정직한 한계). `research/DIRECTION.md`의 판정과 어긋나면 그쪽도 갱신.
- 논문 인용 주의: 우리가 재구현한 NF4 공격은 *Exploiting LLM Quantization* (Egashira et al., NeurIPS 2024)이다. *Mind the Gap* (ICML 2025)은 GGUF 공격이다.

## 2. 건드리지 말 것

- `paper/paper.md`: 과장된 옛 초안. **배포·재작성 금지.** 연구가 끝난 뒤 새로 쓴다.
- `src/quantsentry/` CLI와 `README.md`: 연구 탐지기가 확정되기 전에는 기능 변경 금지 (오타 정도는 OK).
- 보안 민감 파일 — 수정 전 **사용자에게 먼저 물어볼 것**:
  `src/quantsentry/core.py`, `probes.py`, `fixtures.py`, `docs/THREAT_MODEL.md`, `docs/LIMITATIONS.md`.

## 3. 안전 경계

- 백도어 페이로드는 **무해한 문구**(광고 문구, canary 토큰)만 쓴다. 유해한 행동(탈옥, 악성 코드 생성 등)을 하는 백도어는 만들지 않는다.
- 외부 공격 코드(Egashira 등)를 돌릴 때도 페이로드는 무해한 시나리오를 고르거나 공개 체크포인트를 탐지 대상으로만 쓴다.

## 4. Git / 커밋

- 커밋 작성자는 **chae0null 한 명만.** 저장소 로컬 git config에 이미 설정되어 있음
  (`chae0null <297817077+chae0null@users.noreply.github.com>`). 바꾸지 말 것.
- **AI 공동작성자 줄을 넣지 않는다** (`Co-Authored-By: ...`, "Generated with ..." 등 모두 금지). 공개 저장소에서 기여자로 AI가 뜨는 걸 사용자가 원치 않아 히스토리를 재작성한 적이 있음.
- 커밋 메시지 스타일 (기존 로그 따라): `research exp.NN: <한 줄 결론>` 예) `research exp.19 + exp.20: residual-spectrum DETECTS (survives bd_hm), certified-equivalence FAILS`
- 커밋/푸시는 사용자가 요청할 때만. `main` 브랜치에 바로 커밋해 온 저장소다.

## 5. 역할 분담

- **Kaggle GPU 실행은 사용자가 직접 한다.** Codex는 스크립트를 쓰고, 실행법을 알려주고, 사용자가 붙여넣은 출력 로그를 해석해 RESULTS.md에 기록한다.
- 유료 GPU(~$2–5)가 필요한 실험(exp.21, exp.22)은 사용자 결정 사항이다. 먼저 묻는다.
