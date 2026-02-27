from __future__ import annotations

from homecredit_service.local_app import LocalRiskApp


def test_unrealistic_low_installment_gets_penalized() -> None:
    app = LocalRiskApp.__new__(LocalRiskApp)

    policy_score, adjustments, _, _ = app._build_policy_score(
        default_probability=0.06,
        income=1_800_000,
        credit=600_000,
        annuity=100,
        age_years=35,
    )

    assert policy_score > 0.35
    assert any("repayment term" in factor.lower() for factor, _ in adjustments)


def test_reasonable_installment_keeps_term_penalty_off() -> None:
    app = LocalRiskApp.__new__(LocalRiskApp)

    policy_score, adjustments, _, _ = app._build_policy_score(
        default_probability=0.06,
        income=1_800_000,
        credit=600_000,
        annuity=26_000,
        age_years=35,
    )

    assert policy_score < 0.35
    assert all("repayment term" not in factor.lower() for factor, _ in adjustments)
