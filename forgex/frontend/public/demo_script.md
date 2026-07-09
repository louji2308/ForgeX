# ForgeX — Demo Script

## The Six-Beat Script (2:30 total)

### Beat 1 — Portfolio Heatmap (10s)
> "Here's every at-risk unit across the portfolio right now. Color-coded by risk band — red is critical, green is stable. We can sort, filter, and drill into any tenant."

**Action:** Launch dashboard, show portfolio heatmap sorted by risk descending. Point to the color bands.

---

### Beat 2 — Per-Tenant Detail (30s)
> "Let's click into one of these high-risk tenants — their survival curve shows the probability they're still with us over time. The SHAP driver bar tells us *why*: unresolved maintenance issues are the top factor, followed by late payment trends. And the narrative below explains it in plain English."

**Action:** Click on a critical-risk tenant, show survival curve + SHAP bar + narrative.

---

### Beat 3 — What-If Simulator — THE MOMENT (45s)
> "Now here's where this becomes causal, not just descriptive. Watch what happens when I switch personas."

**Action:** 
1. Switch persona to "Maintenance-Sensitive" — note the recommendation
2. Switch to "Price-Sensitive" — the recommendation changes
3. Drag the rent increase slider up — risk goes up
4. Drag the retention credit slider — risk goes down
5. "Same tenant, same property — but the right intervention depends on who they are. The model recovers a behavioral segmentation it was never explicitly told about."

---

### Beat 4 — Fairness Audit (20s)
> "And here's how we make sure this doesn't repeat SafeRent's mistake. Our legacy system had a clear demographic disparity — you can see the gap here. Our model shrinks it dramatically. And if a model fails this fairness gate, it is *structurally unable to reach production*."

**Action:** Show the before/after fairness comparison panel. Point to the DP difference numbers.

---

### Beat 5 — Portfolio Optimizer (30s)
> "Given a $5,000 monthly budget, here's exactly where it goes. The optimizer selects the highest-impact tenants and the right intervention for each — maintenance for some, rent concessions for others. Total expected retained revenue: [X] dollars above doing nothing."

**Action:** Show the optimizer output with selected tenants, actions, and spend breakdown.

---

### Beat 6 — Close on SDG/Impact Framing (15s)
> "Tenant turnover costs landlords 2-3 months' rent per vacancy — tenants lose their homes. ForgeX aligns those incentives: better retention, fairer decisions, and an audit trail for every risk score. This isn't just a scoring tool — it's the operating system for a healthier landlord-tenant relationship."

---

## Pre-Demo Checklist
- [ ] All artifacts committed or re-generatable with one command
- [ ] Every random seed fixed and confirmed reproducible (seed=42)
- [ ] No debug `print()`/stray logging spam in the terminal
- [ ] `.env` not visible in any open editor tab or terminal history
- [ ] `demo_fallback_responses.json` regenerated against the final build
- [ ] Script rehearsed twice, timed
- [ ] Screen resolution set to 1920x1080 (most common for judges)
- [ ] Browser cache cleared, incognito mode
- [ ] All APIs tested: `/health`, `/score/T000001`, `/simulate`, `/optimize`
- [ ] Wifi failover: hotspot ready, local-only mode tested

## Anticipated Judge Q&A

| Question | Answer |
|---|---|
| "How do you know the synthetic data is realistic?" | Point to the validation suite in Phase 1 and the real turnover economics (NAA, Zego, UDR) used to calibrate. Every model downstream is only as honest as this data. |
| "Isn't this just a black-box scoring tool?" | Walk through the fairness architecture. The support-first reframing, the demographic parity gate, and the SHAP-based explainability are the opposite of a black box. |
| "How do you validate the causal model?" | Honest answer: synthetic ground truth for the hackathon. In production, you'd want a randomized pilot. |
| "What about a new landlord with no intervention data?" | Cold-start falls back to population-average CATE prior. Named limitation, not solved. |
| "Why survival analysis?" | Handles censoring naturally and answers "how many months until churn" — a binary classifier can't do that. |
| "What happens when the model is wrong?" | ForgeX recommends, a landlord acts — human-in-the-loop. But per the SafeRent ruling, that doesn't remove fairness responsibility from the score itself. |
