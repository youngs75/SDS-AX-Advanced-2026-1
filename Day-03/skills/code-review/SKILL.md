---
name: code-review
description: 코드 리뷰 요청시 꼭 이 스킬을 활용하세요.
---

# The code review skill

IMPORTANT: Whole code context is need to be read!

## Process
Analyze the code changes based on the following pillars:

- Correctness: Does the code achieve its stated purpose without bugs or logical errors?
- Maintainability: Is the code clean, well-structured, and easy to understand and modify in the future? Consider factors like code clarity, modularity, and adherence to established design patterns.
- Readability: Is the code well-commented (where necessary) and consistently formatted according to our project's coding style guidelines?
- Efficiency: Are there any obvious performance bottlenecks or resource inefficiencies introduced by the changes?
- Security: Are there any potential security vulnerabilities or insecure coding practices?
- Edge Cases and Error Handling: Does the code appropriately handle edge cases and potential errors?
- Testability: Is the new or modified code adequately covered by tests (even if preflight checks pass)? Suggest additional test cases that would improve coverage or robustness.

## Best practices

- Review promptly: Don't make authors wait
- Be respectful: Focus on code, not the person
- Explain why: Don't just say what's wrong
- Suggest alternatives: Show better approaches
- Use examples: Code examples clarify feedback
- Pick your battles: Focus on important issues
- Acknowledge good work: Positive feedback matters
- Review your own code first: Catch obvious issues
- Use automated tools: Let tools catch style issues
- Be consistent: Apply same standards to all code

## Feedback

### Structure
1. Summary: A high-level overview of the review.

2. Findings:
  - Critical: Bugs, security issues, or breaking changes.
  - Improvements: Suggestions for better code quality or performance.
  - Nitpicks: Formatting or minor style issues (optional).
  - Conclusion: Clear recommendation (Approved / Request Changes).

### Tone
Be constructive, professional, and friendly.
Explain why a change is requested.
For approvals, acknowledge the specific value of the contribution.
