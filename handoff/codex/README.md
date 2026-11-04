# Codex 인수인계 패키지 (quantsentry)

이 폴더는 Claude Code에서 진행하던 quantsentry 작업을 **Codex가 이어받기 위한 자료**입니다.
작성일: 2026-11-03. 이 시점까지의 커밋: `4d754bf` (exp.19 + exp.20).

## 읽는 순서

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | [01_PROJECT_STATE.md](01_PROJECT_STATE.md) | 프로젝트가 뭔지, 지금까지 알아낸 것, 현재 결론 |
| 2 | [02_RULES.md](02_RULES.md) | 절대 어기면 안 되는 규칙 (사전등록, 커밋 작성자, 보안 경계) |
| 3 | [03_NEXT_STEPS.md](03_NEXT_STEPS.md) | 바로 다음에 할 일과 순서, 판정 기준 |
| 4 | [04_WORKFLOW.md](04_WORKFLOW.md) | 실행 환경, 명령어, 결과 기록 방법 |
| — | [FIRST_PROMPT.md](FIRST_PROMPT.md) | 사용자가 Codex에 처음 붙여넣을 메시지 |

## 이 폴더 밖의 1차 자료 (원본은 이쪽이 우선)

- `research/RESULTS.md` — 모든 실험 결과 (exp.1–20). 숫자의 원본.
- `research/DIRECTION.md` — 사전등록된 게이트(Gate)와 판정 표, 관련 연구, 포지셔닝.
- `research/SETUP_dev5.md` — 아직 안 돌린 exp.19b/21/22/23/24 실행 방법 (한국어).
- `research/SETUP_exp19_exp20.md` — exp.19/20 실행 방법.
- `docs/THREAT_MODEL.md`, `docs/SCOPE.md`, `docs/LIMITATIONS.md` — 위협 모델과 범위.

이 폴더 내용과 위 원본이 다르면 **원본(RESULTS.md/DIRECTION.md)이 맞습니다.**
새 결과가 나오면 원본을 먼저 고치고, 이 폴더는 필요할 때만 갱신하세요.
