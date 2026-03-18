# Commit Flow

## Use This When
- 사용자가 커밋을 요청했을 때
- staging, commit message, push 전 검증이 필요할 때
- 변경 범위가 넓어서 어떤 메시지로 묶을지 정해야 할 때

## Commit Rules
- 형식: `type: subject`
- 허용 type: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`
- breaking change: `type!: subject`
- subject는 영어, 소문자 imperative, 마침표 없음

## Preferred Flow
1. `git status --short`로 변경 범위를 확인한다.
2. 필요한 검증만 먼저 돌린다.
3. 커밋 메시지는 가장 큰 사용자 가치 변화 기준으로 하나를 고른다.
4. `git add ...`
5. `git commit -m "type: subject"`
6. 사용자가 원하면 `git push origin <branch>`

## Guardrails
- 관련 없는 변경을 되돌리지 않는다.
- 실패한 검증이 있으면 커밋 전에 명시한다.
- push는 remote/auth 상태를 확인한 뒤 진행한다.
- 대규모 초기 import면 메시지를 과도하게 세분화하지 않는다.
