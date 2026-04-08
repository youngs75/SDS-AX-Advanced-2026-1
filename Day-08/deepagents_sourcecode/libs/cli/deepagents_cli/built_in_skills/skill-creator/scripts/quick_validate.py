#!/usr/bin/env python3
"""기술에 대한 빠른 검증 스크립트 - 최소 버전.

deepagents CLI의 경우 기술은 ~/.deepagents/<agent>/skills/<skill-name>/에 있습니다.

Example:
```python
python quick_validate.py ~/.deepagents/agent/skills/my-skill
```
"""

import re
import sys
from pathlib import Path

import yaml


def validate_skill(skill_path):
    """기술의 기본 검증.

Returns:
        is_valid가 bool이고 message인 경우 (is_valid, message)의 튜플
            결과를 설명합니다.

    """
    skill_path = Path(skill_path)

    # Check SKILL.md exists
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        return False, "SKILL.md not found"

    # Read and validate frontmatter
    content = skill_md.read_text()
    if not content.startswith("---"):
        return False, "No YAML frontmatter found"

    # Extract frontmatter
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return False, "Invalid frontmatter format"

    frontmatter_text = match.group(1)

    # Parse YAML frontmatter
    try:
        frontmatter = yaml.safe_load(frontmatter_text)
        if not isinstance(frontmatter, dict):
            return False, "Frontmatter must be a YAML dictionary"
    except yaml.YAMLError as e:
        return False, f"Invalid YAML in frontmatter: {e}"

    # Define allowed properties
    ALLOWED_PROPERTIES = {
        "name",
        "description",
        "license",
        "compatibility",
        "allowed-tools",
        "metadata",
    }

    # Check for unexpected properties (excluding nested keys under metadata)
    unexpected_keys = set(frontmatter.keys()) - ALLOWED_PROPERTIES
    if unexpected_keys:
        unexpected_str = ", ".join(sorted(unexpected_keys))
        allowed_str = ", ".join(sorted(ALLOWED_PROPERTIES))
        return False, (
            f"Unexpected key(s) in SKILL.md frontmatter: {unexpected_str}. "
            f"Allowed properties are: {allowed_str}"
        )

    # Check required fields
    if "name" not in frontmatter:
        return False, "Missing 'name' in frontmatter"
    if "description" not in frontmatter:
        return False, "Missing 'description' in frontmatter"

    # Extract name for validation
    name = frontmatter.get("name", "")
    if not isinstance(name, str):
        return False, f"Name must be a string, got {type(name).__name__}"
    name = name.strip()
    if name:
        # Structural hyphen checks
        if name.startswith("-") or name.endswith("-") or "--" in name:
            return (
                False,
                (
                    f"Name '{name}' cannot start/end with hyphen "
                    "or contain consecutive hyphens"
                ),
            )
        # Character-by-character check matching SDK's _validate_skill_name:
        # Unicode lowercase alphanumeric and hyphens only
        for c in name:
            if c == "-":
                continue
            if (c.isalpha() and c.islower()) or c.isdigit():
                continue
            return (
                False,
                (
                    f"Name '{name}' should be hyphen-case "
                    "(lowercase letters, digits, and hyphens only)"
                ),
            )
        # Check name length (max 64 characters per spec)
        if len(name) > 64:
            return (
                False,
                f"Name is too long ({len(name)} characters). Maximum is 64 characters.",
            )

    # Extract and validate description
    description = frontmatter.get("description", "")
    if not isinstance(description, str):
        return False, f"Description must be a string, got {type(description).__name__}"
    description = description.strip()
    if description:
        # Check for angle brackets
        if "<" in description or ">" in description:
            return False, "Description cannot contain angle brackets (< or >)"
        # Check description length (max 1024 characters per spec)
        if len(description) > 1024:
            return (
                False,
                (
                    f"Description is too long ({len(description)} characters). "
                    "Maximum is 1024 characters."
                ),
            )

    # Extract and validate compatibility (max 500 characters per spec)
    compatibility = frontmatter.get("compatibility", "")
    if isinstance(compatibility, str):
        compatibility = compatibility.strip()
        if len(compatibility) > 500:
            return (
                False,
                (
                    f"Compatibility is too long ({len(compatibility)} characters). "
                    "Maximum is 500 characters."
                ),
            )

    return True, "Skill is valid!"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_validate.py <skill_directory>")
        sys.exit(1)

    valid, message = validate_skill(sys.argv[1])
    print(message)
    sys.exit(0 if valid else 1)
