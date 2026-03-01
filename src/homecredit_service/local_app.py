from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from homecredit_service.service import PredictionService

DEFAULT_ARTIFACT_PATH = Path("artifacts/model_bundle.joblib")

MIN_AGE_YEARS = 18
MAX_AGE_YEARS = 100

APPROVE_MAX_POLICY_SCORE = 0.35
REJECT_MIN_POLICY_SCORE = 0.60
MAX_LOAN_TO_INCOME_DISPLAY = 10.0


class LocalRiskApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Home Credit Decision Console")
        self.minsize(980, 640)
        self.geometry("1080x700")

        self.service: PredictionService | None = None

        self.income_var = tk.StringVar(value="180000")
        self.credit_var = tk.StringVar(value="600000")
        self.annuity_var = tk.StringVar(value="26000")
        self.age_var = tk.StringVar(value="35")

        self.decision_var = tk.StringVar(value="Awaiting evaluation")
        self.note_var = tk.StringVar(
            value="Enter applicant details and click Evaluate Application."
        )
        self.status_var = tk.StringVar(value="System starting...")
        self.model_risk_var = tk.StringVar(value="--")
        self.policy_score_var = tk.StringVar(value="--")
        self.installment_ratio_var = tk.StringVar(value="--")
        self.loan_income_ratio_var = tk.StringVar(value="--")
        self.factor_vars = [tk.StringVar(value="No insights yet.") for _ in range(3)]

        # Determine best available font
        self.base_font = ("Segoe UI", 10)
        self.h1_font = ("Segoe UI", 18, "bold")
        self.h2_font = ("Segoe UI", 12, "bold")
        self.chip_font = ("Segoe UI", 16, "bold")

        self._setup_styles()
        self._build_ui()
        self._reset_insights()
        self._load_model()

    def _setup_styles(self) -> None:
        # Google Material Design Color Palette
        self.palette = {
            "page": "#F8F9FA",  # Google App Background
            "panel": "#FFFFFF",  # White Cards
            "border": "#DADCE0",  # Google Outline Gray
            "text": "#202124",  # Primary Text
            "muted": "#5F6368",  # Secondary Text
            "accent": "#1A73E8",  # Google Blue
            "accent_hover": "#1b66c9",
            "secondary_bg": "#F1F3F4",  # Google Light Gray Button
            "approve_bg": "#E6F4EA",  # Material Green Light
            "approve_fg": "#137333",  # Material Green Dark
            "reject_bg": "#FCE8E6",  # Material Red Light
            "reject_fg": "#C5221F",  # Material Red Dark
            "review_bg": "#FEF7E0",  # Material Yellow Light
            "review_fg": "#B06000",  # Material Yellow Dark
            "neutral_bg": "#F1F3F4",  # Neutral Chip
            "neutral_fg": "#5F6368",
        }

        self.configure(bg=self.palette["page"])
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("TFrame", background=self.palette["page"])

        # Inputs
        style.configure(
            "Input.TEntry",
            fieldbackground=self.palette["panel"],
            foreground=self.palette["text"],
            bordercolor=self.palette["border"],
            lightcolor=self.palette["panel"],
            darkcolor=self.palette["panel"],
            padding=6,
        )

        # Buttons
        style.configure(
            "Primary.TButton",
            font=self.base_font,
            padding=(16, 10),
            background=self.palette["accent"],
            foreground="#FFFFFF",
            borderwidth=0,
            focuscolor=self.palette["accent"],
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.palette["accent_hover"]), ("pressed", "#174ea6")],
        )

        style.configure(
            "Secondary.TButton",
            font=self.base_font,
            padding=(16, 10),
            background=self.palette["secondary_bg"],
            foreground=self.palette["accent"],
            borderwidth=0,
        )
        style.map("Secondary.TButton", background=[("active", "#E8EAED"), ("pressed", "#DADCE0")])

        # Progress bars (using Material colors)
        style.configure(
            "ModelRisk.Horizontal.TProgressbar",
            troughcolor="#E8EAED",
            background="#1A73E8",  # Blue
            bordercolor="#E8EAED",
            lightcolor="#1A73E8",
            darkcolor="#1A73E8",
        )
        style.configure(
            "PolicyRisk.Horizontal.TProgressbar",
            troughcolor="#E8EAED",
            background="#FBBC04",  # Yellow
            bordercolor="#E8EAED",
            lightcolor="#FBBC04",
            darkcolor="#FBBC04",
        )
        style.configure(
            "Affordability.Horizontal.TProgressbar",
            troughcolor="#E8EAED",
            background="#34A853",  # Green
            bordercolor="#E8EAED",
            lightcolor="#34A853",
            darkcolor="#34A853",
        )

    def _build_card(self, parent: tk.Misc) -> tuple[tk.Frame, tk.Frame]:
        """Helper to create a Material-style card with a crisp 1px border."""
        border_frame = tk.Frame(parent, bg=self.palette["border"])
        card_frame = tk.Frame(border_frame, bg=self.palette["panel"])
        card_frame.pack(fill="both", expand=True, padx=1, pady=1)
        return border_frame, card_frame

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # 1. Header (Google Top Bar style)
        header = tk.Frame(
            self,
            bg=self.palette["panel"],
            highlightthickness=1,
            highlightbackground=self.palette["border"],
        )
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        header_content = tk.Frame(header, bg=self.palette["panel"])
        header_content.pack(fill="both", expand=True, padx=24, pady=16)

        tk.Label(
            header_content,
            text="Home Credit Decision Console",
            bg=self.palette["panel"],
            fg=self.palette["text"],
            font=self.h1_font,
        ).pack(anchor="w")
        tk.Label(
            header_content,
            text="Production-style scoring flow focused on approve/reject/review decisions.",
            bg=self.palette["panel"],
            fg=self.palette["muted"],
            font=self.base_font,
        ).pack(anchor="w", pady=(2, 0))

        # 2. Main Body
        body = tk.Frame(self, bg=self.palette["page"])
        body.grid(row=1, column=0, sticky="nsew", padx=24, pady=24)
        body.columnconfigure(0, weight=4)
        body.columnconfigure(1, weight=6)

        # --- LEFT COLUMN: Input Card ---
        input_border, input_card = self._build_card(body)
        input_border.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        input_card.columnconfigure(1, weight=1)

        tk.Label(
            input_card,
            text="Applicant Input",
            bg=self.palette["panel"],
            fg=self.palette["text"],
            font=self.h2_font,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(20, 16))

        fields = [
            ("Annual Income (local currency)", self.income_var),
            ("Requested Loan Amount", self.credit_var),
            ("Monthly Installment", self.annuity_var),
            ("Applicant Age (years)", self.age_var),
        ]

        for idx, (label, variable) in enumerate(fields):
            row = idx + 1
            tk.Label(
                input_card,
                text=label,
                bg=self.palette["panel"],
                fg=self.palette["text"],
                font=self.base_font,
            ).grid(row=row, column=0, sticky="w", padx=20, pady=8)
            ttk.Entry(input_card, textvariable=variable, style="Input.TEntry").grid(
                row=row, column=1, sticky="ew", padx=20, pady=8
            )

        # Buttons inside input card
        button_frame = tk.Frame(input_card, bg=self.palette["panel"])
        button_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=20, pady=(24, 20))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        ttk.Button(
            button_frame,
            text="Use Sample Values",
            style="Secondary.TButton",
            command=self._quick_fill,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            button_frame,
            text="Evaluate Application",
            style="Primary.TButton",
            command=self._predict_single,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # --- RIGHT COLUMN: Result Card ---
        result_border, result_card = self._build_card(body)
        result_border.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        result_card.columnconfigure(0, weight=1)
        result_card.rowconfigure(2, weight=1)

        tk.Label(
            result_card,
            text="Decision Result",
            bg=self.palette["panel"],
            fg=self.palette["text"],
            font=self.h2_font,
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

        # Chip
        self.decision_chip = tk.Label(
            result_card,
            textvariable=self.decision_var,
            bg=self.palette["neutral_bg"],
            fg=self.palette["neutral_fg"],
            font=self.chip_font,
            padx=20,
            pady=16,
        )
        self.decision_chip.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 12))

        # Note
        tk.Label(
            result_card,
            textvariable=self.note_var,
            bg=self.palette["panel"],
            fg=self.palette["muted"],
            justify="left",
            wraplength=500,
            anchor="w",
            font=self.base_font,
        ).grid(row=2, column=0, sticky="nw", padx=20, pady=(0, 20))

        # Insights Section
        insights = tk.Frame(result_card, bg=self.palette["page"])
        insights.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 20))
        insights.columnconfigure(1, weight=1)

        metrics = [
            (
                "Model Risk",
                self.model_risk_var,
                "ModelRisk.Horizontal.TProgressbar",
                self.model_risk_meter_setup,
            ),
            (
                "Policy Score",
                self.policy_score_var,
                "PolicyRisk.Horizontal.TProgressbar",
                self.policy_score_meter_setup,
            ),
            (
                "Installment / Income",
                self.installment_ratio_var,
                "Affordability.Horizontal.TProgressbar",
                self.installment_meter_setup,
            ),
            (
                "Loan / Income",
                self.loan_income_ratio_var,
                "Affordability.Horizontal.TProgressbar",
                self.loan_income_meter_setup,
            ),
        ]

        for idx, (label_text, value_var, style_name, setup_func) in enumerate(metrics):
            row = idx * 2
            tk.Label(
                insights,
                text=label_text,
                bg=self.palette["page"],
                fg=self.palette["text"],
                font=("Segoe UI", 10, "bold"),
            ).grid(row=row, column=0, sticky="w", padx=16, pady=(16 if idx == 0 else 8, 2))
            tk.Label(
                insights,
                textvariable=value_var,
                bg=self.palette["page"],
                fg=self.palette["muted"],
                font=self.base_font,
            ).grid(row=row, column=1, sticky="e", padx=16, pady=(16 if idx == 0 else 8, 2))
            meter = ttk.Progressbar(insights, mode="determinate", maximum=100.0, style=style_name)
            meter.grid(row=row + 1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 4))
            setup_func(meter)  # Save reference for later

        # Factors
        tk.Label(
            insights,
            text="Top Factors",
            bg=self.palette["page"],
            fg=self.palette["text"],
            font=("Segoe UI", 10, "bold"),
        ).grid(row=8, column=0, columnspan=2, sticky="w", padx=16, pady=(12, 4))

        for idx, factor_var in enumerate(self.factor_vars):
            tk.Label(
                insights,
                textvariable=factor_var,
                bg=self.palette["page"],
                fg=self.palette["muted"],
                justify="left",
                anchor="w",
                font=self.base_font,
            ).grid(row=9 + idx, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 4))

        # Padding block at the bottom of insights
        tk.Frame(insights, bg=self.palette["page"], height=16).grid(row=12, column=0)

        # 3. Footer / Status Bar
        status_border, status_card = self._build_card(self)
        status_border.grid(row=2, column=0, sticky="ew", padx=24, pady=(0, 24))

        tk.Label(
            status_card,
            textvariable=self.status_var,
            bg=self.palette["panel"],
            fg=self.palette["muted"],
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x", padx=16, pady=8)

    # Handlers for progress bars assignment
    def model_risk_meter_setup(self, widget):
        self.model_risk_meter = widget

    def policy_score_meter_setup(self, widget):
        self.policy_score_meter = widget

    def installment_meter_setup(self, widget):
        self.installment_meter = widget

    def loan_income_meter_setup(self, widget):
        widget.configure(maximum=MAX_LOAN_TO_INCOME_DISPLAY)
        self.loan_income_meter = widget

    # ---------------------------------------------------------------------------------
    # THE REST OF THE CODE REMAINS EXACTLY THE SAME - NO LOGIC CHANGES BELOW THIS LINE
    # ---------------------------------------------------------------------------------

    def _set_error(self, message: str) -> None:
        self.decision_var.set("Human validation required")
        self.decision_chip.configure(
            bg=self.palette["review_bg"],
            fg=self.palette["review_fg"],
        )
        self.note_var.set(message)
        self.status_var.set("Decision: Human validation required")
        self._reset_insights()
        self.factor_vars[0].set("Please review applicant input and try again.")

    def _risk_bucket(self, value: float) -> str:
        if value < 0.18:
            return "Low"
        if value < 0.35:
            return "Moderate"
        return "High"

    def _reset_insights(self) -> None:
        self.model_risk_var.set("--")
        self.policy_score_var.set("--")
        self.installment_ratio_var.set("--")
        self.loan_income_ratio_var.set("--")
        self.model_risk_meter.configure(value=0.0)
        self.policy_score_meter.configure(value=0.0)
        self.installment_meter.configure(value=0.0)
        self.loan_income_meter.configure(value=0.0)
        for var in self.factor_vars:
            var.set("")

    def _update_insights(
        self,
        *,
        default_probability: float,
        policy_score: float,
        annual_payment_ratio: float,
        loan_to_income: float,
        adjustments: list[tuple[str, float]],
    ) -> None:
        model_pct = default_probability * 100.0
        policy_pct = policy_score * 100.0
        install_pct = annual_payment_ratio * 100.0
        loan_ratio = loan_to_income

        self.model_risk_var.set(f"{model_pct:.1f}%")
        self.policy_score_var.set(f"{policy_pct:.1f}%")
        self.installment_ratio_var.set(
            f"{install_pct:.1f}% ({self._risk_bucket(annual_payment_ratio)})"
        )
        self.loan_income_ratio_var.set(f"{loan_ratio:.2f}x")

        self.model_risk_meter.configure(value=max(0.0, min(100.0, model_pct)))
        self.policy_score_meter.configure(value=max(0.0, min(100.0, policy_pct)))
        self.installment_meter.configure(value=max(0.0, min(100.0, install_pct)))
        self.loan_income_meter.configure(
            value=max(0.0, min(MAX_LOAN_TO_INCOME_DISPLAY, loan_ratio))
        )

        ranked = sorted(adjustments, key=lambda item: abs(item[1]), reverse=True)
        if not ranked:
            self.factor_vars[0].set("1) Balanced affordability profile")
            self.factor_vars[1].set("2) No large risk adjustments applied")
            self.factor_vars[2].set("3) Manual notes can still override this outcome")
            return

        for idx, var in enumerate(self.factor_vars):
            if idx >= len(ranked):
                var.set("")
                continue
            factor, delta = ranked[idx]
            direction = "increases risk" if delta > 0 else "reduces risk"
            var.set(f"{idx + 1}) {factor} ({delta * 100:+.1f} pts, {direction})")

    def _load_model(self) -> None:
        path = DEFAULT_ARTIFACT_PATH.expanduser().resolve()
        try:
            self.service = PredictionService(artifact_path=path, threshold=0.5)
        except Exception:
            self.service = None
            self._set_error("Scoring service unavailable. Please contact support.")
            return
        self.status_var.set("System ready. Evaluate application.")

    def _require_service(self) -> PredictionService:
        if self.service is None:
            raise RuntimeError("Scoring service unavailable.")
        return self.service

    def _quick_fill(self) -> None:
        self.income_var.set("180000")
        self.credit_var.set("600000")
        self.annuity_var.set("26000")
        self.age_var.set("35")
        self.status_var.set("Sample values applied. Evaluate application.")

    def _build_single_record(
        self,
    ) -> tuple[dict[str, float | int], dict[str, float | int]]:
        try:
            income = float(self.income_var.get())
            credit = float(self.credit_var.get())
            annuity = float(self.annuity_var.get())
            age_years = int(float(self.age_var.get()))
        except ValueError as exc:
            raise ValueError("Applicant fields must be numeric.") from exc

        self._validate_single_record(
            income=income,
            credit=credit,
            annuity=annuity,
            age_years=age_years,
        )

        record: dict[str, float | int] = {
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "DAYS_BIRTH": int(-age_years * 365),
        }
        context = {
            "income": income,
            "credit": credit,
            "annuity": annuity,
            "age_years": age_years,
        }
        return record, context

    def _validate_single_record(
        self,
        *,
        income: float,
        credit: float,
        annuity: float,
        age_years: int,
    ) -> None:
        errors: list[str] = []
        if income <= 0:
            errors.append("Income must be greater than 0.")
        if credit <= 0:
            errors.append("Credit amount must be greater than 0.")
        if annuity <= 0:
            errors.append("Annuity must be greater than 0.")
        if annuity > credit:
            errors.append("Annuity should not exceed credit amount.")
        if not (MIN_AGE_YEARS <= age_years <= MAX_AGE_YEARS):
            errors.append(f"Age must be between {MIN_AGE_YEARS} and {MAX_AGE_YEARS}.")
        if errors:
            raise ValueError(" ".join(errors))

    def _decision_label(self, policy_score: float) -> tuple[str, str, str]:
        if policy_score <= APPROVE_MAX_POLICY_SCORE:
            return (
                "Approve",
                self.palette["approve_bg"],
                self.palette["approve_fg"],
            )
        if policy_score >= REJECT_MIN_POLICY_SCORE:
            return (
                "Reject",
                self.palette["reject_bg"],
                self.palette["reject_fg"],
            )
        return (
            "Human validation required",
            self.palette["review_bg"],
            self.palette["review_fg"],
        )

    def _build_policy_score(
        self,
        *,
        default_probability: float,
        income: float,
        credit: float,
        annuity: float,
        age_years: int,
    ) -> tuple[float, list[tuple[str, float]], float, float]:
        score = default_probability
        adjustments: list[tuple[str, float]] = []

        annual_payment_ratio = (annuity * 12.0) / income
        loan_to_income = credit / income
        implied_term_months = credit / annuity

        if annual_payment_ratio >= 0.50:
            score += 0.25
            adjustments.append(("Very high installment burden", 0.25))
        elif annual_payment_ratio >= 0.35:
            score += 0.15
            adjustments.append(("High installment burden", 0.15))
        elif annual_payment_ratio <= 0.18 and implied_term_months < 120:
            score -= 0.05
            adjustments.append(("Low installment burden", -0.05))

        if implied_term_months >= 240:
            score += 0.55
            adjustments.append(("Repayment term appears unrealistic", 0.55))
        elif implied_term_months >= 120:
            score += 0.35
            adjustments.append(("Very long repayment term", 0.35))
        elif implied_term_months >= 84:
            score += 0.15
            adjustments.append(("Long repayment term", 0.15))

        if loan_to_income >= 8.0:
            score += 0.25
            adjustments.append(("Loan amount very high vs income", 0.25))
        elif loan_to_income >= 5.0:
            score += 0.15
            adjustments.append(("Loan amount high vs income", 0.15))
        elif loan_to_income <= 2.0:
            score -= 0.05
            adjustments.append(("Loan amount conservative vs income", -0.05))

        if income < 60000:
            score += 0.10
            adjustments.append(("Low income level", 0.10))
        elif income > 220000:
            score -= 0.05
            adjustments.append(("Strong income level", -0.05))

        if age_years < 21:
            score += 0.10
            adjustments.append(("Very limited credit age", 0.10))

        bounded_score = max(0.0, min(1.0, score))
        return bounded_score, adjustments, annual_payment_ratio, loan_to_income

    def _predict_single(self) -> None:
        try:
            service = self._require_service()
            record, context = self._build_single_record()
            result = service.predict([record], top_n=3)[0]
            default_probability = float(result.get("default_probability", 0.0))
        except Exception as exc:
            self._set_error(str(exc))
            return

        policy_score, adjustments, annual_payment_ratio, loan_to_income = self._build_policy_score(
            default_probability=default_probability,
            income=float(context["income"]),
            credit=float(context["credit"]),
            annuity=float(context["annuity"]),
            age_years=int(context["age_years"]),
        )
        label, bg, fg = self._decision_label(policy_score)
        self.decision_var.set(label)
        self.decision_chip.configure(bg=bg, fg=fg)
        top_reason = "Balanced affordability profile"
        if adjustments:
            top_reason = sorted(adjustments, key=lambda item: abs(item[1]), reverse=True)[0][0]
        self._update_insights(
            default_probability=default_probability,
            policy_score=policy_score,
            annual_payment_ratio=annual_payment_ratio,
            loan_to_income=loan_to_income,
            adjustments=adjustments,
        )
        self.note_var.set(
            f"Estimated risk: {default_probability * 100:.1f}%. "
            f"Policy score: {policy_score * 100:.1f}%. Key driver: {top_reason}."
        )
        self.status_var.set(f"Decision: {label}")


def main() -> None:
    app = LocalRiskApp()
    app.mainloop()


if __name__ == "__main__":
    main()
