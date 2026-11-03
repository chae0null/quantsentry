# Codex에 처음 보낼 메시지 (복사해서 붙여넣기)

아래 블록 전체를 Codex 새 대화의 첫 메시지로 보내세요.

---

```
이 저장소(quantsentry)는 지금까지 다른 에이전트와 진행하던 연구 프로젝트야. 이제부터 네가 이어받아.

먼저 handoff/codex/ 폴더를 README.md → 01 → 02 → 03 → 04 순서로 전부 읽고,
그다음 research/RESULTS.md의 Experiment 19, 20 섹션과 research/DIRECTION.md,
research/SETUP_dev5.md를 읽어.

읽은 뒤에 아래를 해줘. 아직 코드는 고치지 말고 커밋도 하지 마:
1. 지금 프로젝트 상태를 5줄 이내로 요약 (내가 네가 제대로 이해했는지 확인할 수 있게)
2. 02_RULES.md의 규칙 중 가장 중요한 3개를 네 말로 다시 말하기
3. research/qs_core.py가 research/kaggle_exp19_residualspectrum.py의 탐지기·학습 로직과
   같은지 대조해서 차이가 있으면 알려주기
4. 다음에 내가 Kaggle에서 돌릴 실험 하나를 추천하고, 붙여넣을 셀 코드를 그대로 줘
```

---

## 그 다음부터 자주 쓸 메시지 예시

- 결과 붙여넣을 때:
  `exp.19b 돌렸어. 출력 붙일게. 사전등록된 기준대로 판정하고 RESULTS.md에 섹션 추가해줘. 커밋은 내가 확인하고 할게.`
  (그 아래에 Kaggle 출력의 SUMMARY 블록 붙여넣기)
- 커밋할 때:
  `좋아, 커밋해줘. 작성자는 chae0null만, AI co-author 줄 넣지 마.`
- 외부 공격 준비:
  `exp.21 준비하자. Egashira "Exploiting LLM Quantization" 공개 코드 저장소 찾아서, 무해한 페이로드로 NF4 백도어 만드는 방법을 Kaggle/유료 GPU 기준으로 정리해줘.`
