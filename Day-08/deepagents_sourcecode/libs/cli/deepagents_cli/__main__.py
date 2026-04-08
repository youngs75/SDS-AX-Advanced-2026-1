"""`python -m deepagents_cli`용 모듈 실행기입니다.

이 파일은 `deepagents_cli.main.cli_main`에 위임하기 전에 의도적으로 가능한 최소한의 작업을 수행합니다.
"""

from deepagents_cli.main import cli_main

if __name__ == "__main__":
    cli_main()
