const { useState, useEffect, useCallback, useMemo } = React;

/* ============================================================
   Config — API base preserved from the original wiring.
   Override at runtime with window.FORGEX_API_BASE if needed
   (e.g. "/api" behind the nginx proxy in docker-compose).
   ============================================================ */
const API_BASE = (typeof window !== "undefined" && window.FORGEX_API_BASE) || "http://localhost:8000";

/* ============================================================
   Icons (inline SVG — no emoji, consistent 1.75 stroke)
   ============================================================ */
const Ic = {
  logo: (p) => (
    <svg viewBox="0 0 24 24" fill="none" {...p}>
      <path d="M4 21V6l7-3 7 3v15" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <path d="M8 21v-4h3v4M13 12h2M8 9h3M13 9h2M8 13h3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  ),
  arrow: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" /></svg>),
  play: (p) => (<svg viewBox="0 0 24 24" fill="currentColor" {...p}><path d="M8 5v14l11-7L8 5z" /></svg>),
  sparkle: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M12 3l1.8 4.8L18.6 9.6 13.8 11.4 12 16.2 10.2 11.4 5.4 9.6 10.2 7.8 12 3z" fill="currentColor" /><path d="M19 14l.7 1.9L21.6 16.6 19.7 17.3 19 19.2 18.3 17.3 16.4 16.6 18.3 15.9 19 14z" fill="currentColor" /></svg>),
  brain: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M9 4a3 3 0 00-3 3 3 3 0 00-1 5.8A3 3 0 007 18a3 3 0 003 3V4H9zM15 4a3 3 0 013 3 3 3 0 011 5.8A3 3 0 0117 18a3 3 0 01-3 3V4h1z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" /></svg>),
  diagnose: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M4 19h16M6 15l4-5 3 3 5-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>),
  shield: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6l7-3z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" /><path d="M9.5 12l1.8 1.8L15 10" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg>),
  users: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><circle cx="9" cy="8" r="3" stroke="currentColor" strokeWidth="1.7" /><path d="M3 20a6 6 0 0112 0M16 6a3 3 0 010 6M18 20a6 6 0 00-3-5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /></svg>),
  home: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M4 11l8-6 8 6M6 10v9h12v-9" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" /></svg>),
  chart: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /></svg>),
  dollar: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" /><path d="M14.5 9.2C14 8.4 13 8 12 8c-1.4 0-2.5.8-2.5 2s1 1.7 2.5 2 2.5.9 2.5 2-1.1 2-2.5 2c-1 0-2-.4-2.5-1.2M12 7v10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>),
  person: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><circle cx="12" cy="8" r="3.5" stroke="currentColor" strokeWidth="1.7" /><path d="M5 20a7 7 0 0114 0" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /></svg>),
  sliders: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M18 18h2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /><circle cx="15" cy="6" r="2" fill="currentColor" /><circle cx="9" cy="12" r="2" fill="currentColor" /><circle cx="15" cy="18" r="2" fill="currentColor" /></svg>),
  trendUp: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M4 17l6-6 3 3 7-8M16 6h5v5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>),
  wrench: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M14.5 6a3.5 3.5 0 00-4.8 4.3l-5.4 5.4a1.5 1.5 0 002.1 2.1l5.4-5.4A3.5 3.5 0 0018 9.5L15.8 11.7 13.3 9.2 15.5 7z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" /></svg>),
  calendar: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><rect x="4" y="5" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.7" /><path d="M4 9h16M8 3v4M16 3v4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /></svg>),
  chat: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M5 5h14a1 1 0 011 1v9a1 1 0 01-1 1H9l-4 3V6a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" /></svg>),
  chevron: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>),
  chevronsRight: (p) => (<svg viewBox="0 0 24 24" fill="none" {...p}><path d="M7 6l6 6-6 6M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>),
};

function Sparkline({ color, seed }) {
  // deterministic decorative sparkline
  const pts = seed || "0,18 12,10 24,14 36,6 48,12 60,4";
  return (
    <svg className="spark" width="66" height="26" viewBox="0 0 66 26" fill="none">
      <polyline points={pts} stroke={color} strokeWidth="2.2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/* ============================================================
   Data hooks (fetch + debounce) — preserves original pattern
   ============================================================ */
function useDebouncedSimulation(request, delayMs = 300) {
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [errorMessage, setErrorMessage] = useState(null);

  useEffect(() => {
    if (!request || !request.tenant_id) return;
    setStatus("loading");
    const controller = new AbortController();
    const timer = setTimeout(function () {
      fetch(`${API_BASE}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      })
        .then(function (res) {
          if (!res.ok) {
            return res.json().catch(function () { return {}; }).then(function (body) {
              throw new Error(body.message || body.detail || `Request failed with ${res.status}`);
            });
          }
          return res.json();
        })
        .then(function (data) {
          setResult(data);
          setStatus("success");
          setErrorMessage(null);
        })
        .catch(function (err) {
          if (err.name === "AbortError") return;
          setErrorMessage(err.message);
          setStatus("error");
        });
    }, delayMs);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [JSON.stringify(request), delayMs]);

  return { result, status, errorMessage };
}

function useTenantScore(tenantId) {
  // GET /score/{id} — used to populate the real risk-driver breakdown
  const [score, setScore] = useState(null);
  useEffect(() => {
    if (!tenantId) { setScore(null); return; }
    const controller = new AbortController();
    fetch(`${API_BASE}/score/${encodeURIComponent(tenantId)}`, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setScore(d))
      .catch(() => setScore(null));
    return () => controller.abort();
  }, [tenantId]);
  return score;
}

/* ============================================================
   Constants
   ============================================================ */
const MAINT_LABELS = {
  standard: "Standard (3\u20135 days)",
  priority: "Priority (24h)",
  same_day: "Same-Day (emergency)",
};
const PERSONA_PRESETS = {
  balanced: { rent_increase_pct: 8, retention_credit_usd: 50, maintenance_speed: "standard" },
  "price-sensitive": { rent_increase_pct: 3, retention_credit_usd: 120, maintenance_speed: "standard" },
  "maintenance-sensitive": { rent_increase_pct: 5, retention_credit_usd: 40, maintenance_speed: "priority" },
  "at-risk": { rent_increase_pct: 12, retention_credit_usd: 200, maintenance_speed: "same_day" },
};

/* ============================================================
   LANDING / HERO  (matches image 1)
   ============================================================ */
function Landing({ onLaunch, health }) {
  const tenants = [
    { id: "T-1023", risk: 82, band: "High Risk", cls: "high", img: "assets/tenant-1.png", spark: "0,6 12,10 24,7 36,14 48,9 60,16", pos: "tc-1" },
    { id: "T-2041", risk: 45, band: "Medium Risk", cls: "med", img: "assets/tenant-2.png", spark: "0,14 12,9 24,15 36,8 48,13 60,7", pos: "tc-2" },
    { id: "T-3098", risk: 15, band: "Low Risk", cls: "low", img: "assets/tenant-3.png", spark: "0,18 12,15 24,17 36,10 48,12 60,4", pos: "tc-3" },
  ];
  const bandColor = { high: "#ef4444", med: "#f59e0b", low: "#16b981" };

  const chips = [
    { ic: Ic.brain, t: "AI Risk Prediction", d: "Know who's at risk" },
    { ic: Ic.diagnose, t: "Explain & Diagnose", d: "Understand the drivers" },
    { ic: Ic.shield, t: "Optimize & Act", d: "Best actions, best ROI" },
  ];
  const props = [
    { ic: Ic.users, t: "Increase Retention" },
    { ic: Ic.home, t: "Reduce Vacancy Loss" },
    { ic: Ic.shield, t: "Fair & Responsible AI" },
    { ic: Ic.chart, t: "Data-Driven Actions" },
    { ic: Ic.dollar, t: "Maximize Rental Income" },
  ];

  return (
    <div className="landing">
      <div className="shell">
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark"><Ic.logo width="30" height="30" /></span>
            <span className="brand-word">Forge<span className="brand-x">X</span></span>
          </div>
          <button className="btn btn-primary" onClick={onLaunch}>
            See the Demo <span className="btn-icon"><Ic.arrow width="16" height="16" /></span>
          </button>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <span className="hero-badge"><Ic.sparkle width="14" height="14" /> AI-Powered Tenant Intelligence</span>
            <h1>
              Predict Tenant<br />
              Churn. <span className="accent">Prevent It.</span><br />
              Profit More.
            </h1>
            <p className="hero-sub">
              ForgeX combines AI, causal modeling, and fairness-aware analytics to help you
              retain great tenants and maximize rental income.
            </p>
            <div className="hero-cta">
              <button className="btn btn-primary" onClick={onLaunch}>
                See the Demo <span className="btn-icon"><Ic.arrow width="16" height="16" /></span>
              </button>
              <button className="play-btn" onClick={onLaunch}>
                <span className="play-ring"><Ic.play width="16" height="16" /></span>
                How It Works
              </button>
            </div>

            <div className="feature-chips">
              {chips.map((c) => (
                <div className="chip" key={c.t}>
                  <span className="chip-ic">{c.ic({ width: 20, height: 20 })}</span>
                  <span>
                    <span className="chip-t">{c.t}</span><br />
                    <span className="chip-d">{c.d}</span>
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="hero-visual">
            <img className="hero-building" src="assets/building.png" alt="3D render of a modern apartment building monitored by ForgeX" crossOrigin="anonymous" />

            {tenants.map((t) => (
              <div className={`float-card tenant-card ${t.pos}`} key={t.id}>
                <span className="avatar">
                  <img src={t.img} alt={`Tenant ${t.id}`} crossOrigin="anonymous" />
                  <span className={`dot dot-${t.cls}`} />
                </span>
                <span>
                  <span className="t-name">Tenant {t.id}</span><br />
                  <span className="t-risk">Risk: {t.risk}%</span><br />
                  <span className={`t-band band-${t.cls}`}>{t.band}</span>
                </span>
                <Sparkline color={bandColor[t.cls]} seed={t.spark} />
              </div>
            ))}

            <div className="float-card stat-card sc-1">
              <div className="s-label">Portfolio Health</div>
              <div className="s-row">
                <div className="s-value">78 <small>/100</small></div>
                <div className="donut" style={{ "--p": 78 }} />
              </div>
              <div className="s-tag good">Good</div>
            </div>

            <div className="float-card stat-card sc-2">
              <div className="s-label">At Risk Tenants</div>
              <div className="s-row">
                <div className="s-value">{health && health.tenants_indexed ? Math.max(1, Math.round(health.tenants_indexed * 0.05)) : 24}</div>
                <span className="group-ic"><Ic.users width="40" height="40" /></span>
              </div>
              <div className="s-tag up">12% vs last month</div>
            </div>

            <div className="float-card stat-card sc-3">
              <div className="s-label">Potential Revenue at Risk</div>
              <div className="s-row">
                <div className="s-value">$24,500</div>
                <span style={{ color: "#ef4444" }}><Ic.chart width="34" height="34" /></span>
              </div>
              <div className="s-tag high">High</div>
            </div>
          </div>
        </section>

        <section className="valueprops">
          <div className="vp-title">Built for forward-thinking property managers</div>
          <div className="vp-row">
            {props.map((v) => (
              <span className="vp" key={v.t}>{v.ic({ width: 24, height: 24 })} {v.t}</span>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

/* ============================================================
   SIMULATOR  (matches image 2)
   ============================================================ */
function Simulator({ tenants, loading, onBack, health }) {
  const [selectedTenant, setSelectedTenant] = useState("");
  const [persona, setPersona] = useState("balanced");
  const [scenario, setScenario] = useState({
    tenant_id: "",
    rent_increase_pct: 8,
    maintenance_speed: "standard",
    retention_credit_usd: 50,
  });

  // auto-select first tenant when the list arrives
  useEffect(() => {
    if (!selectedTenant && tenants.length > 0) setSelectedTenant(tenants[0]);
  }, [tenants, selectedTenant]);

  useEffect(() => {
    setScenario((prev) => ({ ...prev, tenant_id: selectedTenant }));
  }, [selectedTenant]);

  // rent slider spans -25..+25 (per design); backend accepts >=0, so clamp when sending
  const requestBody = useMemo(() => ({
    ...scenario,
    rent_increase_pct: Math.max(0, scenario.rent_increase_pct),
  }), [scenario]);

  const { result, status, errorMessage } = useDebouncedSimulation(
    scenario.tenant_id ? requestBody : null
  );
  const score = useTenantScore(scenario.tenant_id);

  const set = useCallback((updates) => setScenario((p) => ({ ...p, ...updates })), []);
  const onPersona = (value) => { setPersona(value); set(PERSONA_PRESETS[value] || PERSONA_PRESETS.balanced); };
  const retry = () => setScenario((s) => ({ ...s }));

  const rentFill = ((scenario.rent_increase_pct + 25) / 50) * 100;
  const creditFill = (scenario.retention_credit_usd / 200) * 100;

  const delta = result ? result.delta_pts : null;
  const up = delta != null && delta > 0.05;
  const down = delta != null && delta < -0.05;

  // Reconstruct the per-lever contribution consistent with the backend delta.
  const leverDrivers = useMemo(() => {
    const rentSent = Math.max(0, scenario.rent_increase_pct);
    const rentTerm = rentSent > 0 ? 0.04 * rentSent : 0;
    const maintTerm = scenario.maintenance_speed === "priority" ? 0.15 * 15
      : scenario.maintenance_speed === "same_day" ? 0.15 * 30 : 0;
    const creditTerm = scenario.retention_credit_usd > 0 ? 0.0008 * scenario.retention_credit_usd : 0;
    const net = rentTerm - maintTerm - creditTerm;
    const base = (delta != null && net !== 0) ? delta / net : 0;
    return [
      { name: "Rent Increase", ic: Ic.trendUp, cls: "b", pts: base * rentTerm, note: `From 0% \u2192 ${scenario.rent_increase_pct >= 0 ? "+" : ""}${scenario.rent_increase_pct}%` },
      { name: "Retention Credit", ic: Ic.shield, cls: "g", pts: -(base * creditTerm), note: `From $0 \u2192 $${scenario.retention_credit_usd}` },
      { name: "Maintenance Speed", ic: Ic.wrench, cls: "p", pts: -(base * maintTerm), note: MAINT_LABELS[scenario.maintenance_speed] },
    ];
  }, [scenario, delta]);

  // Intrinsic (tenant-level) drivers pulled from the real /score response.
  const intrinsicDrivers = useMemo(() => {
    if (!score || !score.top_drivers || !score.top_drivers.length) return [];
    const icons = [{ ic: Ic.calendar, cls: "o" }, { ic: Ic.chat, cls: "b" }];
    return score.top_drivers.slice(0, 2).map((d, i) => {
      const signed = d.direction === "increases_risk" ? Math.abs(d.shap_value) : -Math.abs(d.shap_value);
      const note = d.raw_value != null ? `Current value: ${Number(d.raw_value).toFixed(2)}` :
        (d.direction === "increases_risk" ? "Increases risk" : "Reduces risk");
      return { name: d.label || d.feature, ic: icons[i].ic, cls: icons[i].cls, pts: signed, note };
    });
  }, [score]);

  const allDrivers = [...leverDrivers, ...intrinsicDrivers];
  const ptsClass = (v) => (v > 0.005 ? "up" : v < -0.005 ? "down" : "flat");
  const fmtPts = (v) => `${v > 0 ? "+" : ""}${v.toFixed(2)} pts`;

  return (
    <div className="sim">
      <div className="shell">
        <div className="sim-head">
          <button className="back" onClick={onBack}>
            <Ic.arrow width="16" height="16" style={{ transform: "rotate(180deg)" }} /> Back to overview
          </button>
          <h1><span className="accent">ForgeX</span> — What-If Simulator</h1>
          <p>Adjust key factors and see the causal impact on tenant churn risk.</p>
        </div>

        {/* Select tenant */}
        <div className="panel tenant-bar">
          <span className="tb-ic"><Ic.person width="24" height="24" /></span>
          <div className="tb-body">
            <div className="tb-label">Select Tenant</div>
            <div className="select-wrap">
              <select
                value={selectedTenant}
                onChange={(e) => setSelectedTenant(e.target.value)}
                disabled={loading || tenants.length === 0}
              >
                {tenants.length === 0 && <option>{loading ? "Loading tenants\u2026" : "No tenants available"}</option>}
                {tenants.map((t) => (<option key={t} value={t}>{t} (Active)</option>))}
              </select>
              <span className="chev"><Ic.chevron width="18" height="18" /></span>
            </div>
          </div>
          <span className="api-note">
            <span className={`api-dot ${health && health.is_model_loaded ? "on" : "off"}`} />
            {health && health.is_model_loaded ? "Model online" : "Model offline"}
          </span>
        </div>

        <div className="sim-grid">
          {/* Scenario controls */}
          <div className="panel">
            <div className="panel-title">
              <span className="pt-ic"><Ic.sliders width="22" height="22" /></span>
              <div>
                <h2>Scenario Controls</h2>
                <div className="pt-sub">Modify key levers to simulate different scenarios.</div>
              </div>
            </div>

            <div className="ctrl-row">
              <label>Persona Preset</label>
              <div className="select-wrap field">
                <select value={persona} onChange={(e) => onPersona(e.target.value)} disabled={!selectedTenant}>
                  <option value="balanced">Balanced Tenant</option>
                  <option value="price-sensitive">Price-Sensitive</option>
                  <option value="maintenance-sensitive">Maintenance-Sensitive</option>
                  <option value="at-risk">At-Risk (High Churn)</option>
                </select>
                <span className="chev"><Ic.chevron width="18" height="18" /></span>
              </div>
            </div>

            <div className="ctrl-row">
              <label>Rent Increase</label>
              <div className="slider-cell">
                <div className="slider-row">
                  <span className="ctrl-value">{scenario.rent_increase_pct >= 0 ? "+ " : "- "}{Math.abs(scenario.rent_increase_pct)} %</span>
                  <input type="range" min="-25" max="25" step="1" value={scenario.rent_increase_pct}
                    style={{ "--fill": `${rentFill}%` }}
                    onChange={(e) => set({ rent_increase_pct: parseFloat(e.target.value) })}
                    disabled={!selectedTenant} aria-label="Rent increase percent" />
                </div>
                <div className="range-labels"><span>-25%</span><span>+25%</span></div>
              </div>
            </div>

            <div className="ctrl-row">
              <label>Retention Credit</label>
              <div className="slider-cell">
                <div className="slider-row">
                  <span className="ctrl-value">$ {scenario.retention_credit_usd}</span>
                  <input type="range" min="0" max="200" step="10" value={scenario.retention_credit_usd}
                    style={{ "--fill": `${creditFill}%` }}
                    onChange={(e) => set({ retention_credit_usd: parseFloat(e.target.value) })}
                    disabled={!selectedTenant} aria-label="Retention credit dollars" />
                </div>
                <div className="range-labels"><span>$0</span><span>$200</span></div>
              </div>
            </div>

            <div className="ctrl-row">
              <label>Maintenance Speed</label>
              <div className="select-wrap field">
                <select value={scenario.maintenance_speed} onChange={(e) => set({ maintenance_speed: e.target.value })} disabled={!selectedTenant}>
                  <option value="standard">Standard (3–5 days)</option>
                  <option value="priority">Priority (24h)</option>
                  <option value="same_day">Same-Day (emergency)</option>
                </select>
                <span className="chev"><Ic.chevron width="18" height="18" /></span>
              </div>
            </div>
          </div>

          {/* Risk projection + recommendation */}
          <div>
            <div className="panel">
              <div className="panel-title">
                <span className="pt-ic"><Ic.trendUp width="22" height="22" /></span>
                <h2>Risk Projection</h2>
              </div>

              {!scenario.tenant_id && <div className="empty-state">Select a tenant to project churn risk.</div>}
              {scenario.tenant_id && status === "loading" && <div className="skeleton" />}
              {scenario.tenant_id && status === "error" && (
                <div className="error-banner">
                  <span>Couldn't run that scenario: {errorMessage}</span>
                  <button onClick={retry}>Retry</button>
                </div>
              )}
              {scenario.tenant_id && status === "success" && result && (
                <React.Fragment>
                  <div className="proj-grid">
                    <div className="proj-card">
                      <h4>Baseline Risk</h4>
                      <div className="big baseline">{result.baseline_risk_pct}%</div>
                      <div className="cap">Current conditions</div>
                    </div>
                    <div className="proj-arrow"><Ic.chevronsRight width="22" height="22" /></div>
                    <div className="proj-card">
                      <h4>Scenario Risk</h4>
                      <div className={`big scenario ${up ? "up" : ""}`}>{result.scenario_risk_pct}%</div>
                      <div className="cap">With adjustments</div>
                    </div>
                  </div>
                  <div className={`delta-line ${up ? "up" : down ? "down" : "flat"}`}>
                    {delta > 0 ? "+" : ""}{delta.toFixed(1)} pts change
                  </div>
                </React.Fragment>
              )}
            </div>

            {scenario.tenant_id && status === "success" && result && (
              <div className="reco">
                <span className="reco-ic"><Ic.sparkle width="26" height="26" /></span>
                <div>
                  <h3>AI Recommendation</h3>
                  <p>{result.recommendation}</p>
                </div>
                <img className="reco-img" src="assets/building.png" alt="" aria-hidden="true" crossOrigin="anonymous" />
              </div>
            )}
          </div>
        </div>

        {/* What's driving the change */}
        {scenario.tenant_id && status === "success" && result && (
          <div className="panel drivers">
            <h2>What's Driving the Change?</h2>
            <p className="d-sub">Key factors contributing to the risk change in this scenario.</p>
            <div className="driver-grid">
              {allDrivers.map((d, i) => (
                <div className="driver-card" key={i}>
                  <span className={`dc-ic ${d.cls}`}>{d.ic({ width: 22, height: 22 })}</span>
                  <div className="dc-name">{d.name}</div>
                  <div className={`dc-pts ${ptsClass(d.pts)}`}>{fmtPts(d.pts)}</div>
                  <div className="dc-note">{d.note}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ============================================================
   APP — view switch + shared data loading
   ============================================================ */
function App() {
  const [view, setView] = useState("landing");
  const [tenants, setTenants] = useState([]);
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/tenants`)
      .then((r) => r.json())
      .then((data) => { if (data && data.tenants) setTenants(data.tenants.slice(0, 200)); })
      .catch((e) => console.log("[v0] Failed to load tenants:", e.message))
      .finally(() => setLoading(false));

    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch((e) => console.log("[v0] Health check failed:", e.message));
  }, []);

  return view === "landing"
    ? <Landing onLaunch={() => setView("sim")} health={health} />
    : <Simulator tenants={tenants} loading={loading} health={health} onBack={() => setView("landing")} />;
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
