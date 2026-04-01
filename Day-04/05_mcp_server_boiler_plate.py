"""MCP 서버 보일러플레이트.

## 실제 API 구현 시 참고 문서
- GitHub API: https://docs.github.com/rest
- Slack API: https://api.slack.com/
"""

import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ============================================================================
# MCP 서버 초기화
# ============================================================================

mcp = FastMCP(
    name="ProductionMCPServer",
    instructions="MCP 서버 보일러플레이트",
    host="0.0.0.0",
    port=8000,
    json_response=True,
    stateless_http=True,
)


# ============================================================================
# 유틸리티 함수
# ============================================================================


def create_error_response(
    error_type: str,
    message: str,
    field: str | None = None,
    suggestion: str | None = None,
) -> dict[str, Any]:
    """구체적이고 실행 가능한 오류 응답 생성.

    Args:
        error_type: 오류 유형
        message: 오류 메시지
        field: 오류가 발생한 필드 (선택사항)
        suggestion: 해결 방법 제안 (선택사항)

    Returns:
        구조화된 오류 응답
    """
    error_response: dict[str, Any] = {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
        },
    }

    if field:
        error_response["error"]["field"] = field

    if suggestion:
        error_response["error"]["suggestion"] = suggestion

    return error_response


def create_success_response(
    data: dict[str, Any],
    message: str,
    next_actions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """성공 응답 생성 (next_actions 포함).

    Args:
        data: 응답 데이터
        message: 성공 메시지
        next_actions: 다음 가능한 작업들 (선택사항)

    Returns:
        구조화된 성공 응답
    """
    response: dict[str, Any] = {
        "success": True,
        "message": message,
        "data": data,
    }

    if next_actions:
        response["next_actions"] = next_actions

    return response


# ============================================================================
# GitHub API 통합 샘플
# ============================================================================


class CreateIssueRequest(BaseModel):
    """GitHub 이슈 생성 요청."""

    repo: str = Field(
        ...,
        description="저장소 (owner/repo 형식)",
        pattern=r"^[\w\-\.]+/[\w\-\.]+$",
    )
    title: str = Field(..., description="이슈 제목", min_length=1, max_length=200)
    body: str = Field(default="", description="이슈 본문 (Markdown)")


@mcp.tool(structured_output=True)
def github_create_issue(request: CreateIssueRequest) -> dict[str, Any]:
    """Create a new GitHub issue.

    Args:
        request: 이슈 생성 요청

    Returns:
        이슈 생성 결과 (Structured Output)
    """
    logger.info("Tool 호출: github_create_issue(repo=%s)", request.repo)

    # 저장소 형식 검증
    if not re.match(r"^[\w\-\.]+/[\w\-\.]+$", request.repo):
        return create_error_response(
            error_type="validation_error",
            message=f"저장소 형식이 잘못되었습니다: '{request.repo}'",
            field="repo",
            suggestion="올바른 형식은 'owner/repo'입니다. 예: 'facebook/react'",
        )

    # TODO: 실제 환경에서는 GitHub API 호출
    issue_data = {
        "id": 12345,
        "number": 42,
        "title": request.title,
        "body": request.body,
        "state": "open",
        "url": f"https://github.com/{request.repo}/issues/42",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "current_user",
    }

    logger.info("이슈 생성 완료: #%s", issue_data["number"])

    return create_success_response(
        data=issue_data,
        message=f"이슈 #{issue_data['number']}가 생성되었습니다: {issue_data['title']}",
        next_actions={
            "view": f"이슈 보기: {issue_data['url']}",
            "add_comment": "github_add_comment 도구로 댓글 추가",
            "add_label": "github_add_labels 도구로 라벨 추가",
            "assign": "github_assign_issue 도구로 담당자 지정",
        },
    )


class SearchIssuesRequest(BaseModel):
    """GitHub 이슈 검색 요청."""

    repo: str = Field(..., description="저장소 (owner/repo)")
    query: str = Field(..., description="검색어")
    state: str = Field(default="open", description="이슈 상태: open, closed, all")
    max_results: int = Field(default=10, description="최대 결과 수", ge=1, le=100)


@mcp.tool(structured_output=True)
def github_search_issues(request: SearchIssuesRequest) -> dict[str, Any]:
    """Search GitHub issues in a repository.

    Args:
        request: 이슈 검색 요청

    Returns:
        검색 결과 (Structured Output)
    """
    logger.info(
        "Tool 호출: github_search_issues(repo=%s, query=%s)",
        request.repo,
        request.query,
    )

    # TODO: 실제 환경에서는 GitHub API 호출
    issues = [
        {
            "number": 42,
            "title": "버그: 로그인 실패",
            "state": "open",
            "author": "user1",
            "created_at": "2025-01-10",
            "comments": 5,
            "labels": ["bug", "high-priority"],
            "url": f"https://github.com/{request.repo}/issues/42",
        },
        {
            "number": 39,
            "title": "기능: 다크 모드 지원",
            "state": "open",
            "author": "user2",
            "created_at": "2025-01-08",
            "comments": 12,
            "labels": ["enhancement"],
            "url": f"https://github.com/{request.repo}/issues/39",
        },
    ]

    return create_success_response(
        data={
            "repo": request.repo,
            "query": request.query,
            "total_count": len(issues),
            "issues": issues,
        },
        message=f"'{request.query}' 검색 결과: {len(issues)}개 이슈 발견",
        next_actions={
            "view": "각 이슈의 url을 사용하여 상세 내용 확인",
            "create": "github_create_issue 도구로 새 이슈 생성",
        },
    )


# ============================================================================
# Slack API 통합
# ============================================================================


class SendMessageRequest(BaseModel):
    """Slack 메시지 전송 요청."""

    channel: str = Field(
        ..., description="채널명 (#general), 사용자명 (@john), 또는 채널 ID"
    )
    text: str = Field(..., description="메시지 텍스트")


@mcp.tool(structured_output=True)
def slack_send_message(request: SendMessageRequest) -> dict[str, Any]:
    """Send message to Slack channel or user.

    Args:
        request: 메시지 전송 요청

    Returns:
        전송 결과 (Structured Output)
    """
    logger.info("Tool 호출: slack_send_message(channel=%s)", request.channel)

    # 자연어 식별자 파싱
    channel_id = None
    channel_type = "unknown"

    if request.channel.startswith("#"):
        # 채널 이름
        channel_name = request.channel[1:]
        # TODO: 실제로는 Slack API로 채널 ID 조회
        channel_id = f"C{hash(channel_name) % 10000000:07d}"
        channel_type = "channel"
        logger.info("채널 이름 → ID 변환: %s → %s", request.channel, channel_id)

    elif request.channel.startswith("@"):
        # 사용자명
        username = request.channel[1:]
        # TODO: 실제로는 Slack API로 사용자 ID 조회
        channel_id = f"U{hash(username) % 10000000:07d}"
        channel_type = "user"
        logger.info("사용자명 → ID 변환: %s → %s", request.channel, channel_id)

    else:
        # 이미 ID인 경우
        channel_id = request.channel
        channel_type = "direct"

    # TODO: 실제 환경에서는 Slack API 호출
    message_data = {
        "channel": request.channel,
        "channel_id": channel_id,
        "channel_type": channel_type,
        "text": request.text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": f"https://company.slack.com/archives/{channel_id}",
    }

    return create_success_response(
        data=message_data,
        message=f"{request.channel}에 메시지를 전송했습니다",
        next_actions={
            "view": f"메시지 확인: {message_data['url']}",
            "reply": "같은 채널에 다시 메시지를 보내려면 slack_send_message 사용",
            "react": "slack_add_reaction 도구로 반응 추가",
        },
    )


# ============================================================================
# 복합 작업 예제
# ============================================================================


class DeletePRReviewRequest(BaseModel):
    """PR 리뷰 삭제 요청."""

    repo: str = Field(..., description="저장소 (owner/repo)")
    pr_number: int = Field(..., description="Pull Request 번호", ge=1)
    review_author: str | None = Field(
        None, description="리뷰 작성자 (기본값: 인증된 사용자)"
    )


@mcp.tool(structured_output=True)
def github_delete_pr_review(request: DeletePRReviewRequest) -> dict[str, Any]:
    """Delete a pull request review.

    Args:
        request: PR 리뷰 삭제 요청

    Returns:
        삭제 결과 (Structured Output)
    """
    logger.info(
        "Tool 호출: github_delete_pr_review(repo=%s, pr=%s)",
        request.repo,
        request.pr_number,
    )

    # TODO: 실제로는 여러 단계를 거침:
    # 1. PR 정보 조회
    # 2. PR 리뷰 목록 조회
    # 3. 대상 리뷰 찾기
    # 4. 리뷰 삭제

    # TODO: 실제 환경에서는 GitHub API 호출
    review_author = request.review_author or "current_user"

    review_data = {
        "pr_number": request.pr_number,
        "review_id": 98765,
        "review_author": review_author,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
    }

    return create_success_response(
        data=review_data,
        message=f"PR #{request.pr_number}의 {review_author} 리뷰가 삭제되었습니다",
        next_actions={
            "view_pr": f"PR 확인: https://github.com/{request.repo}/pull/{request.pr_number}",
            "new_review": "github_submit_review 도구로 새 리뷰 작성",
        },
    )


# ============================================================================
# 모호성 해결 전략
# ============================================================================


class FindDocumentRequest(BaseModel):
    """문서 검색 요청."""

    title: str = Field(..., description="문서 제목")
    author: str | None = Field(None, description="작성자 (선택사항)")
    date: str | None = Field(None, description="작성 날짜 (선택사항)")


@mcp.tool(structured_output=True)
def find_document(request: FindDocumentRequest) -> dict[str, Any]:
    """Find a document by title.

    Args:
        request: 문서 검색 요청

    Returns:
        검색 결과 또는 선택 옵션
    """
    logger.info("Tool 호출: find_document(title=%s)", request.title)

    # TODO: 실제 환경에서는 문서 검색 API 호출
    documents = [
        {
            "id": "doc1",
            "title": request.title,
            "author": "김개발",
            "date": "2024-01-15",
            "description": "초기 버전",
            "url": "https://wiki.company.com/docs/doc1",
        },
        {
            "id": "doc2",
            "title": request.title,
            "author": "이백엔드",
            "date": "2024-02-20",
            "description": "업데이트된 버전",
            "url": "https://wiki.company.com/docs/doc2",
        },
        {
            "id": "doc3",
            "title": request.title,
            "author": "박프론트",
            "date": "2024-03-10",
            "description": "최신 버전",
            "url": "https://wiki.company.com/docs/doc3",
        },
    ]

    # TODO: 필터링 (author 또는 date 지정 시)
    if request.author:
        documents = [d for d in documents if d["author"] == request.author]

    if request.date:
        documents = [d for d in documents if d["date"] == request.date]

    if len(documents) == 0:
        return create_error_response(
            error_type="not_found",
            message=f"'{request.title}' 제목의 문서를 찾을 수 없습니다",
            suggestion="제목을 다시 확인하거나 다른 검색어를 사용하세요",
        )

    if len(documents) == 1:
        # NOTE: 정확히 하나 발견
        doc = documents[0]
        return create_success_response(
            data=doc,
            message=f"문서를 찾았습니다: {doc['title']} (작성자: {doc['author']})",
            next_actions={
                "view": f"문서 보기: {doc['url']}",
                "edit": "edit_document 도구로 문서 수정",
            },
        )

    # NOTE: 여러 개 발견시 선택 옵션 제공
    return {
        "success": True,
        "status": "requires_selection",
        "message": f"'{request.title}' 제목의 문서 {len(documents)}개를 찾았습니다",
        "options": documents,
        "instruction": "사용자에게 어느 문서인지 선택하도록 요청하거나, "
        "author 또는 date 파라미터를 추가하여 다시 검색하세요",
    }


# ============================================================================
# 부분적 성공 처리
# ============================================================================


class NotifyMultipleRequest(BaseModel):
    """여러 사용자에게 알림 전송 요청."""

    emails: list[str] = Field(..., description="이메일 주소 목록")
    message: str = Field(..., description="알림 메시지")


@mcp.tool(structured_output=True)
def notify_multiple_users(request: NotifyMultipleRequest) -> dict[str, Any]:
    """Send notification to multiple users.

    Args:
        request: 알림 전송 요청

    Returns:
        전송 결과 (부분 성공 포함)
    """
    logger.info("Tool 호출: notify_multiple_users(emails=%s개)", len(request.emails))

    successful = []
    failed = []

    for email in request.emails:
        # TODO: 간단한 이메일 검증
        if "@" not in email or "." not in email:
            failed.append({"email": email, "error": "유효하지 않은 이메일 형식"})
        elif "disabled" in email:
            failed.append({"email": email, "error": "알림이 비활성화된 사용자"})
        else:
            successful.append(
                {
                    "email": email,
                    "status": "sent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    total_sent = len(successful)
    total_failed = len(failed)

    if total_failed == 0:
        # NOTE: 완전 성공
        return create_success_response(
            data={"successful": successful},
            message=f"모든 사용자({total_sent}명)에게 알림을 전송했습니다",
        )
    if total_sent == 0:
        # NOTE: 완전 실패
        return create_error_response(
            error_type="send_failure",
            message=f"모든 사용자({total_failed}명)에게 알림 전송에 실패했습니다",
            suggestion="이메일 주소를 확인하거나 사용자 설정을 점검하세요",
        )
    # NOTE: 부분 성공
    return {
        "success": True,
        "partial": True,
        "message": f"{total_sent}/{len(request.emails)}명에게 알림을 전송했습니다",
        "results": {
            "successful": successful,
            "failed": failed,
        },
        "warning": f"{total_failed}명에게 전송 실패",
    }


def main() -> None:
    """메인 함수."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Server Boilerplate")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="전송 프로토콜 (기본값: stdio)",
    )

    args = parser.parse_args()

    logger.info("MCP Server: %s v%s", mcp.name, mcp.version)
    logger.info("Description: %s", mcp.description)

    logger.info("MCP 서버 시작 중... (Transport: %s)\n", args.transport)
    logger.info("클라이언트 연결 대기 중...\n")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
