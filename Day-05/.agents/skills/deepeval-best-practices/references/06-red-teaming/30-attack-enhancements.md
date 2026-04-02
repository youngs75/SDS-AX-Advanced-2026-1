# Red Teaming: Attack Enhancements

## Read This When
- You need to choose between single-turn (encoding-based and one-shot) and multi-turn (jailbreaking) attack methods
- You want to configure attack weights, combine attack types, or understand resource requirements per attack category
- You are deciding on an attack strategy based on application type (chatbot, RAG, agentic, static)

## Skip This When
- You need `RedTeamer` setup, model callback, or risk assessment interpretation -- see [10-introduction.md](./10-introduction.md)
- You need to select which vulnerabilities to test for -- see [20-vulnerabilities.md](./20-vulnerabilities.md)

---

## Overview

Attack methods in DeepTeam are designed to enhance or progress existing baseline adversarial attacks. The framework operates by simulating simplistic baseline adversarial attacks, then progressively applying various attack enhancement methods to create more sophisticated versions. The target LLM's responses are then evaluated to identify system weaknesses.

**Two categories:**
- **Single-Turn Enhancements** - modify a single attack prompt (14 available)
- **Multi-Turn Progressions** - iterative flows across conversation turns (5 available)

---

## How Attacks Work

The two-stage process:

1. **Baseline attack generation**: LLMs create simple attacks targeting specific vulnerability types
2. **Attack enhancement**: Baseline attacks are transformed using single-turn or multi-turn techniques to bypass LLM defenses

All attacks accept an optional `weight` parameter (default: `1`) determining probability of selection during simulation:

```python
from deepteam.attacks.single_turn import PromptInjection

attack = PromptInjection(weight=2)  # Selected 2x more often
```

---

## Single-Turn Enhancements

Single-turn attacks work with a single LLM pass or apply deterministic transformations. Two subtypes:

- **Encoding-based**: Apply character rotation, substitution, or encoding schemes to obscure attacks — no LLM calls required (fastest, cheapest)
- **One-shot**: Use a single LLM pass for creative modification — non-deterministic, retryable up to 5 times

### Usage

```python
from deepteam.attacks.single_turn import (
    PromptInjection,
    Roleplay,
    Base64,
    ROT13,
    Leetspeak,
    Multilingual,
    MathProblem,
    AdversarialPoetry,
    GrayBox,
    ContextPoisoning,
    GoalRedirection,
    InputBypass,
    PermissionEscalation,
    LinguisticConfusion,
    SystemOverride,
)
from deepteam.vulnerabilities import Bias
from deepteam import red_team

risk_assessment = red_team(
    attacks=[PromptInjection(), Roleplay(), Base64()],
    vulnerabilities=[Bias()],
    model_callback=your_callback
)
```

### Complete Single-Turn Attack Reference

| Attack | Subtype | Description |
|--------|---------|-------------|
| `PromptInjection` | One-shot | Injects adversarial instructions into the prompt to override the model's system instructions |
| `Roleplay` | One-shot | Wraps the attack in a roleplay scenario to bypass safety filters by framing the request as fiction |
| `GrayBox` | One-shot | Uses partial knowledge of the model's system/architecture to craft targeted attacks |
| `MathProblem` | One-shot | Disguises the harmful request as a mathematical problem or equation |
| `AdversarialPoetry` | One-shot | Encodes the harmful request within poetry or verse format |
| `Multilingual` | One-shot | Translates the attack to another language where safety filters may be weaker |
| `ContextPoisoning` | One-shot | Contaminates the context window with misleading information to redirect model behavior |
| `GoalRedirection` | One-shot | Reframes the conversation goal mid-stream to elicit a different, harmful response |
| `InputBypass` | One-shot | Structures input to bypass input validation or filtering mechanisms |
| `PermissionEscalation` | One-shot | Claims elevated permissions or authority to unlock restricted behaviors |
| `LinguisticConfusion` | One-shot | Uses ambiguous, convoluted, or confusing language to obscure the harmful intent |
| `SystemOverride` | One-shot | Attempts to override system-level instructions by claiming special system status |
| `Base64` | Encoding | Encodes the attack in Base64 format to bypass text-based safety filters |
| `ROT13` | Encoding | Applies ROT-13 character rotation cipher to obscure the attack text |
| `Leetspeak` | Encoding | Substitutes letters with numbers and symbols (e.g., "h4ck3r") to evade filters |

### Example: Encoding-Based

```python
from deepteam.attacks.single_turn import Base64, ROT13, Leetspeak

# These require zero LLM calls — fastest option
encoding_attacks = [Base64(), ROT13(), Leetspeak()]
```

### Example: One-Shot

```python
from deepteam.attacks.single_turn import PromptInjection, Roleplay, GrayBox

# Each requires one LLM call
one_shot_attacks = [PromptInjection(), Roleplay(), GrayBox()]
```

---

## Multi-Turn Progressions

Multi-turn attacks are iterative flows that start from baseline attacks and adapt across conversational turns. They use the target model's previous reply to refine the attack by rephrasing, changing persona, or escalating intensity.

These are best suited for chatbots and conversational applications.

### Usage

```python
from deepteam.attacks.multi_turn import (
    LinearJailbreaking,
    CrescendoJailbreaking,
    TreeJailbreaking,
    BadLikertJudge,
    SequentialBreak,
)
from deepteam.vulnerabilities import Bias
from deepteam import red_team

risk_assessment = red_team(
    attacks=[LinearJailbreaking(), CrescendoJailbreaking()],
    vulnerabilities=[Bias()],
    model_callback=your_callback
)
```

### Complete Multi-Turn Attack Reference

| Attack | Description |
|--------|-------------|
| `LinearJailbreaking` | Progressively escalates the request across turns in a linear fashion, each turn pushing further toward the target |
| `CrescendoJailbreaking` | Gradually increases attack intensity across turns using a crescendo pattern, starting benign and escalating |
| `TreeJailbreaking` | Branches attack strategies across turns, trying different approaches and following the most promising path |
| `BadLikertJudge` | Uses a Likert-scale framing to get the model to evaluate and then reproduce harmful content |
| `SequentialBreak` | Decomposes the harmful request into seemingly innocuous sequential steps across multiple turns |

### Example: Using the Target's Response

Multi-turn attacks automatically use the target LLM's response to refine subsequent turns:

```python
async def model_callback(input: str, turns: list[RTTurn] = None) -> str:
    # turns contains the full conversation history for multi-turn attacks
    # For LinearJailbreaking, the attack prompt will evolve based on your responses
    if turns:
        # Multi-turn: turns contains previous exchanges
        conversation_history = turns

    return RTTurn(
        role="assistant",
        content=your_llm_response(input, turns)
    )
```

---

## Combining Single-Turn and Multi-Turn

You can use both attack categories simultaneously:

```python
from deepteam.attacks.single_turn import Leetspeak
from deepteam.attacks.multi_turn import LinearJailbreaking
from deepteam.vulnerabilities import Bias
from deepteam import red_team

risk_assessment = red_team(
    attacks=[Leetspeak(), LinearJailbreaking()],
    vulnerabilities=[Bias()],
    model_callback=your_callback
)
```

---

## Selecting an Attack Strategy

Consider three factors when choosing attacks:

1. **Is your application a chatbot?**
   - Yes → Include multi-turn attacks
   - No → Single-turn attacks only

2. **What is the application's purpose?**
   - Purpose-dependent content → Encoding or one-shot enhancements
   - General assistant → Mix of all types

3. **Does it access external resources?**
   - External resources + chatbot → Multi-turn with turn-level enhancements

### Decision Matrix

| Application Type | Recommended Attacks |
|-----------------|---------------------|
| Chatbot/conversational | Multi-turn progressions + single-turn encoding |
| Static response system | Single-turn enhancements (encoding + one-shot) |
| RAG system | ContextPoisoning + single-turn encoding |
| Agentic system | PermissionEscalation + SystemOverride + multi-turn |
| External resource access | Multi-turn + GoalRedirection + InputBypass |

---

## Attack Distribution Configuration

When running `red_team()`, use attack weights to control distribution:

```python
from deepteam.attacks.single_turn import Base64, PromptInjection, Roleplay
from deepteam.attacks.multi_turn import CrescendoJailbreaking

# Weight-based selection: Base64(2) selected 2x more than Roleplay(1)
attacks = [
    Base64(weight=2),
    PromptInjection(weight=2),
    Roleplay(weight=1),
    CrescendoJailbreaking(weight=1)
]

risk_assessment = red_team(
    attacks=attacks,
    vulnerabilities=[...],
    model_callback=your_callback,
    attacks_per_vulnerability_type=5
)
```

Alternatively, use percentage-based distribution in `RedTeamer.scan()`:

```python
from deepeval.red_teaming import AttackEnhancement

results = red_teamer.scan(
    target_model=target_llm,
    attacks_per_vulnerability=5,
    vulnerabilities=[...],
    attack_enhancements={
        AttackEnhancement.BASE64: 0.25,
        AttackEnhancement.GRAY_BOX_ATTACK: 0.25,
        AttackEnhancement.JAILBREAK_CRESCENDO: 0.25,
        AttackEnhancement.MULTILINGUAL: 0.25,
    }
)
```

---

## Resource Requirements

Resource requirements vary significantly by attack type — factor this into your testing budget:

| Category | LLM Calls | Cost | Speed |
|----------|-----------|------|-------|
| Encoding-based (Base64, ROT13, Leetspeak) | 0 | Lowest | Fastest |
| One-shot (PromptInjection, Roleplay, etc.) | 1 per attack | Medium | Fast |
| Multi-turn progressions | Multiple per attack | Highest | Slowest |

**Directly proportional relationship:** More LLM calls = more effective attacks.

**Recommended approach:** Start with encoding-based attacks for initial testing, then layer in one-shot and multi-turn as needed for critical vulnerabilities.

---

## Tips

- Use older, powerful models like `"gpt-4o-mini"` to increase enhancement success rates (less likely to refuse generating attack content)
- More advanced models have stricter filtering, potentially limiting attack generation effectiveness
- Combining diverse enhancement types provides more comprehensive coverage than using a single type
- For critical vulnerabilities, increase `attacks_per_vulnerability_type` to 5+
- Initial testing is crucial for determining which attack strategies are most effective for your specific application

---

## Related Documentation

- [10-introduction.md](./10-introduction.md) - DeepTeam setup, red_team() function, RedTeamer class, model callback
- [20-vulnerabilities.md](./20-vulnerabilities.md) - Complete vulnerability catalog
