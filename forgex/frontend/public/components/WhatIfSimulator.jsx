function useDebouncedSimulation(request, delayMs) {
  var useState = React.useState, useEffect = React.useEffect;
  var _a = useState(null), result = _a[0], setResult = _a[1];
  var _b = useState("idle"), status = _b[0], setStatus = _b[1];
  var _c = useState(null), errorMessage = _c[0], setErrorMessage = _c[1];

  useEffect(function() {
    if (!request || !request.tenant_id) return;
    setStatus("loading");
    var controller = new AbortController();
    var timer = setTimeout(function() {
      fetch(window.API_BASE + "/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      }).then(function(res) {
        if (!res.ok) {
          return res.json().catch(function() { return {}; }).then(function(body) {
            throw new Error(body.message || body.detail || ("Request failed with " + res.status));
          });
        }
        return res.json();
      }).then(function(data) {
        setResult(data);
        setStatus("success");
        setErrorMessage(null);
      }).catch(function(err) {
        if (err.name === "AbortError") return;
        setErrorMessage(err.message);
        setStatus("error");
      });
    }, delayMs);
    return function() { clearTimeout(timer); controller.abort(); };
  }, [JSON.stringify(request), delayMs]);

  return { result: result, status: status, errorMessage: errorMessage };
}

function useTenantScore(tenantId) {
  var useState = React.useState, useEffect = React.useEffect;
  var _a = useState(null), score = _a[0], setScore = _a[1];
  useEffect(function() {
    if (!tenantId) { setScore(null); return; }
    var controller = new AbortController();
    fetch(window.API_BASE + "/score/" + encodeURIComponent(tenantId), { signal: controller.signal })
      .then(function(r) { return r.ok ? r.json() : null; })
      .then(function(d) { return setScore(d); })
      .catch(function() { return setScore(null); });
    return function() { controller.abort(); };
  }, [tenantId]);
  return score;
}

var MAINT_LABELS = { standard: "Standard (3\u20135 days)", priority: "Priority (24h)", same_day: "Same-Day (emergency)" };
var PERSONA_PRESETS = {
  balanced: { rent_increase_pct: 8, retention_credit_usd: 50, maintenance_speed: "standard" },
  "price-sensitive": { rent_increase_pct: 3, retention_credit_usd: 120, maintenance_speed: "standard" },
  "maintenance-sensitive": { rent_increase_pct: 5, retention_credit_usd: 40, maintenance_speed: "priority" },
  "at-risk": { rent_increase_pct: 12, retention_credit_usd: 200, maintenance_speed: "same_day" },
};

function Landing(props) {
  var useState = React.useState, useEffect = React.useEffect;
  var _a = useState([]), topTenants = _a[0], setTopTenants = _a[1];
  var bandColor = { high: "#ef4444", med: "#f59e0b", low: "#16b981" };
  var posClasses = ["tc-1", "tc-2", "tc-3"];

  useEffect(function() {
    var controller = new AbortController();
    var signal = controller.signal;
    fetch(window.API_BASE + "/tenants", { signal: signal }).then(function(r) { return r.json(); }).then(function(data) {
      if (!data || !data.tenants || !data.tenants.length) return;
      var list = data.tenants;
      var count = Math.min(3, list.length);
      var promises = [];
      for (let i = 0; i < count; i++) {
        let tid = list[i];
        let imgSrc = ["assets/tenant-1.png", "assets/tenant-2.png", "assets/tenant-3.png"][i];
        promises.push(fetch(window.API_BASE + "/score/" + encodeURIComponent(tid), { signal: signal }).then(function(r) { return r.json(); }).then(function(s) {
          if (!s) return null;
          var r = s.risk_pct;
          var cls = r >= 60 ? "high" : r >= 30 ? "med" : "low";
          var band = r >= 80 ? "Critical" : r >= 60 ? "High Risk" : r >= 40 ? "Medium Risk" : r >= 20 ? "Low Risk" : "Very Low Risk";
          return { id: tid, riskPct: Math.round(r), className: cls, band: band, imgSrc: "./" + imgSrc };
        }).catch(function() { return null; }));
      }
      return Promise.all(promises).then(function(results) { return results.filter(Boolean); });
    }).then(function(top) { if (top) setTopTenants(top); }).catch(function(err) {
      if (err.name === "AbortError") return;
    });
    return function() { controller.abort(); };
  }, []);

  return React.createElement("div", { className: "landing" },
    React.createElement("div", { className: "shell" },
      React.createElement("header", { className: "topbar" },
        React.createElement("div", { className: "brand" },
          React.createElement("span", { className: "brand-mark" }, Ic.logo({ width: "30", height: "30" })),
          React.createElement("span", { className: "brand-word" }, "Forge", React.createElement("span", { className: "brand-x" }, "X"))
        ),
        React.createElement("button", { className: "btn btn-primary", onClick: props.onLogin },
          "Sign In ", React.createElement("span", { className: "btn-icon" }, Ic.arrow({ width: "16", height: "16" }))
        )
      ),
      React.createElement("section", { className: "hero" },
        React.createElement("div", { className: "hero-copy" },
          React.createElement("span", { className: "hero-badge" }, Ic.sparkle({ width: "14", height: "14" }), " AI-Powered Tenant Intelligence"),
          React.createElement("h1", null, "Predict Tenant\nChurn. ", React.createElement("span", { className: "accent" }, "Prevent It."), "\nProfit More."),
          React.createElement("p", { className: "hero-sub" }, "ForgeX combines AI, causal modeling, and fairness-aware analytics to help you retain great tenants and maximize rental income."),
          React.createElement("div", { className: "hero-cta" },
            React.createElement("button", { className: "btn btn-primary", onClick: props.onLogin }, "Sign In ", React.createElement("span", { className: "btn-icon" }, Ic.arrow({ width: "16", height: "16" }))),
            React.createElement("button", { className: "btn btn-ghost", onClick: props.onRegister }, "Create Account")
          )
        ),
        React.createElement("div", { className: "hero-visual" },
          React.createElement("img", { className: "hero-building", src: "assets/building.png", alt: "ForgeX building", crossOrigin: "anonymous" }),
          topTenants.map(function(t, i) {
            return React.createElement("div", { className: "float-card tenant-card " + posClasses[i], key: t.id },
              React.createElement("span", { className: "avatar" },
                React.createElement("img", { src: t.imgSrc, alt: "Tenant " + t.id, crossOrigin: "anonymous" }),
                React.createElement("span", { className: "dot dot-" + t.className })
              ),
              React.createElement("span", null,
                React.createElement("span", { className: "t-name" }, "Tenant ", t.id), React.createElement("br"),
                React.createElement("span", { className: "t-risk" }, "Risk: ", t.riskPct, "%"), React.createElement("br"),
                React.createElement("span", { className: "t-band band-" + t.className }, t.band)
              )
            );
          })
        )
      )
    )
  );
}

function Simulator(props) {
  var useState = React.useState, useEffect = React.useEffect, useCallback = React.useCallback, useMemo = React.useMemo;
  var initTenant = props.selectedTenant || (props.tenants.length > 0 ? props.tenants[0] : "");
  var _a = useState(initTenant), selectedTenant = _a[0], setSelectedTenant = _a[1];
  var _b = useState("balanced"), persona = _b[0], setPersona = _b[1];
  var _c = useState({ tenant_id: initTenant, rent_increase_pct: 8, maintenance_speed: "standard", retention_credit_usd: 50 }), scenario = _c[0], setScenario = _c[1];

  useEffect(function() {
    if (props.selectedTenant && props.selectedTenant !== selectedTenant) {
      setSelectedTenant(props.selectedTenant);
    } else if (!selectedTenant && props.tenants.length > 0) {
      setSelectedTenant(props.tenants[0]);
    }
  }, [props.selectedTenant, props.tenants, selectedTenant]);
  useEffect(function() { setScenario(function(prev) { return Object.assign({}, prev, { tenant_id: selectedTenant }); }); }, [selectedTenant]);

  var requestBody = useMemo(function() {
    return Object.assign({}, scenario, { rent_increase_pct: Math.max(0, scenario.rent_increase_pct) });
  }, [scenario]);

  var _d = useDebouncedSimulation(scenario.tenant_id ? requestBody : null, 300), result = _d.result, status = _d.status, errorMessage = _d.errorMessage;
  var score = useTenantScore(scenario.tenant_id);

  var set = useCallback(function(updates) { setScenario(function(p) { return Object.assign({}, p, updates); }); }, []);
  var onPersona = function(value) { setPersona(value); set(PERSONA_PRESETS[value] || PERSONA_PRESETS.balanced); };
  var retry = function() { setScenario(function(s) { return Object.assign({}, s, { _retry: Date.now() }); }); };

  var rentFill = ((scenario.rent_increase_pct + 25) / 50) * 100;
  var creditFill = (scenario.retention_credit_usd / 200) * 100;
  var delta = result ? result.delta_pts : null;
  var up = delta != null && delta > 0.05;
  var down = delta != null && delta < -0.05;

  var allDrivers = useMemo(function() {
    var items = [];
    if (result && result.lever_breakdown) {
      var driverIcons = { "Rent Increase": Ic.trendUp, "Retention Credit": Ic.shield, "Maintenance Speed": Ic.wrench };
      var driverCls = { "Rent Increase": "b", "Retention Credit": "g", "Maintenance Speed": "p" };
      var driverNotes = { "Rent Increase": "From 0% \u2192 " + (scenario.rent_increase_pct >= 0 ? "+" : "") + scenario.rent_increase_pct + "%", "Retention Credit": "From $0 \u2192 $" + scenario.retention_credit_usd, "Maintenance Speed": MAINT_LABELS[scenario.maintenance_speed] };
      result.lever_breakdown.forEach(function(lever) {
        var points = lever.direction === "increases" ? lever.effect_pct : -lever.effect_pct;
        items.push({ name: lever.name, ic: driverIcons[lever.name] || Ic.trendUp, cls: driverCls[lever.name] || "o", pts: points, note: driverNotes[lever.name] || "" });
      });
    }
    if (score && score.top_drivers && score.top_drivers.length) {
      var extraIcons = [{ ic: Ic.calendar, cls: "o" }, { ic: Ic.chat, cls: "b" }, { ic: Ic.home, cls: "g" }, { ic: Ic.dollar, cls: "p" }, { ic: Ic.wrench, cls: "o" }];
      score.top_drivers.slice(0, 5).forEach(function(d, i) {
        var signed = d.direction === "increases_risk" ? Math.abs(d.shap_value) : -Math.abs(d.shap_value);
        var note = d.raw_value != null ? "Current value: " + Number(d.raw_value).toFixed(2) : (d.direction === "increases_risk" ? "Increases risk" : "Reduces risk");
        items.push({ name: d.label || d.feature, ic: (extraIcons[i] || extraIcons[0]).ic, cls: (extraIcons[i] || extraIcons[0]).cls, pts: signed, note: note });
      });
    }
    return items;
  }, [result, score, scenario]);

  var ptsClass = function(v) { return v > 0.005 ? "up" : v < -0.005 ? "down" : "flat"; };
  var fmtPts = function(v) { return (v > 0 ? "+" : "") + v.toFixed(2) + " pts"; };

  return React.createElement("div", { className: "sim" },
    React.createElement("div", { className: "shell" },
      React.createElement("div", { className: "sim-head" },
        React.createElement("button", { className: "back", onClick: props.onLogout },
          Ic.arrow({ width: "16", height: "16", style: { transform: "rotate(180deg)" } }), " Sign out"
        ),
        React.createElement("h1", null, React.createElement("span", { className: "accent" }, "ForgeX"), " \u2014 What-If Simulator"),
        React.createElement("p", null, "Adjust key factors and see the causal impact on tenant churn risk.")
      ),
      React.createElement("div", { className: "panel tenant-bar" },
        React.createElement("span", { className: "tb-ic" }, Ic.person({ width: "24", height: "24" })),
        React.createElement("div", { className: "tb-body" },
          React.createElement("div", { className: "tb-label" }, "Select Tenant"),
          React.createElement("div", { className: "select-wrap" },
            React.createElement("select", { value: selectedTenant, onChange: function(e) { setSelectedTenant(e.target.value); }, disabled: props.loading || props.tenants.length === 0 },
              !selectedTenant || props.tenants.length === 0 ? React.createElement("option", { value: "" }, props.loading ? "Loading tenants\u2026" : "Select a tenant\u2026") : null,
              props.tenants.map(function(t) { return React.createElement("option", { key: t, value: t }, t, " (Active)"); })
            ),
            React.createElement("span", { className: "chev" }, Ic.chevron({ width: "18", height: "18" }))
          )
        ),
        React.createElement("span", { className: "api-note" }, props.landlord)
      ),
      React.createElement("div", { className: "sim-grid" },
        React.createElement("div", { className: "panel" },
          React.createElement("div", { className: "panel-title" },
            React.createElement("span", { className: "pt-ic" }, Ic.sliders({ width: "22", height: "22" })),
            React.createElement("div", null, React.createElement("h2", null, "Scenario Controls"), React.createElement("div", { className: "pt-sub" }, "Modify key levers to simulate different scenarios."))
          ),
          React.createElement("div", { className: "ctrl-row" },
            React.createElement("label", null, "Persona Preset"),
            React.createElement("div", { className: "select-wrap field" },
              React.createElement("select", { value: persona, onChange: function(e) { onPersona(e.target.value); }, disabled: !selectedTenant },
                React.createElement("option", { value: "balanced" }, "Balanced Tenant"),
                React.createElement("option", { value: "price-sensitive" }, "Price-Sensitive"),
                React.createElement("option", { value: "maintenance-sensitive" }, "Maintenance-Sensitive"),
                React.createElement("option", { value: "at-risk" }, "At-Risk (High Churn)")
              ),
              React.createElement("span", { className: "chev" }, Ic.chevron({ width: "18", height: "18" }))
            )
          ),
          React.createElement("div", { className: "ctrl-row" },
            React.createElement("label", null, "Rent Increase"),
            React.createElement("div", { className: "slider-cell" },
              React.createElement("div", { className: "slider-row" },
                React.createElement("span", { className: "ctrl-value" }, scenario.rent_increase_pct >= 0 ? "+ " : "- ", Math.abs(scenario.rent_increase_pct), " %"),
                React.createElement("input", { type: "range", min: "-25", max: "25", step: "1", value: scenario.rent_increase_pct, style: { "--fill": rentFill + "%" }, onChange: function(e) { set({ rent_increase_pct: parseFloat(e.target.value) }); }, disabled: !selectedTenant, "aria-label": "Rent increase percent" })
              ),
              React.createElement("div", { className: "range-labels" }, React.createElement("span", null, "-25%"), React.createElement("span", null, "+25%"))
            )
          ),
          React.createElement("div", { className: "ctrl-row" },
            React.createElement("label", null, "Retention Credit"),
            React.createElement("div", { className: "slider-cell" },
              React.createElement("div", { className: "slider-row" },
                React.createElement("span", { className: "ctrl-value" }, "$ ", scenario.retention_credit_usd),
                React.createElement("input", { type: "range", min: "0", max: "200", step: "10", value: scenario.retention_credit_usd, style: { "--fill": creditFill + "%" }, onChange: function(e) { set({ retention_credit_usd: parseFloat(e.target.value) }); }, disabled: !selectedTenant, "aria-label": "Retention credit dollars" })
              ),
              React.createElement("div", { className: "range-labels" }, React.createElement("span", null, "$0"), React.createElement("span", null, "$200"))
            )
          ),
          React.createElement("div", { className: "ctrl-row" },
            React.createElement("label", null, "Maintenance Speed"),
            React.createElement("div", { className: "select-wrap field" },
              React.createElement("select", { value: scenario.maintenance_speed, onChange: function(e) { set({ maintenance_speed: e.target.value }); }, disabled: !selectedTenant },
                React.createElement("option", { value: "standard" }, "Standard (3\u20135 days)"),
                React.createElement("option", { value: "priority" }, "Priority (24h)"),
                React.createElement("option", { value: "same_day" }, "Same-Day (emergency)")
              ),
              React.createElement("span", { className: "chev" }, Ic.chevron({ width: "18", height: "18" }))
            )
          )
        ),
        React.createElement("div", null,
          React.createElement("div", { className: "panel" },
            React.createElement("div", { className: "panel-title" },
              React.createElement("span", { className: "pt-ic" }, Ic.trendUp({ width: "22", height: "22" })),
              React.createElement("h2", null, "Risk Projection")
            ),
            !scenario.tenant_id ? React.createElement("div", { className: "empty-state" }, "Select a tenant to project churn risk.") : null,
            scenario.tenant_id && status === "loading" ? React.createElement("div", { className: "skeleton" }) : null,
            scenario.tenant_id && status === "error" ? React.createElement("div", { className: "error-banner" },
              React.createElement("span", null, "Couldn't run that scenario: ", errorMessage),
              React.createElement("button", { onClick: retry }, "Retry")
            ) : null,
            scenario.tenant_id && status === "success" && result ? React.createElement(React.Fragment, null,
              React.createElement("div", { className: "proj-grid" },
                React.createElement("div", { className: "proj-card" },
                  React.createElement("h4", null, "Baseline Risk"),
                  React.createElement("div", { className: "big baseline" }, result.baseline_risk_pct, "%"),
                  React.createElement("div", { className: "cap" }, "Current conditions")
                ),
                React.createElement("div", { className: "proj-arrow" }, Ic.chevronsRight({ width: "22", height: "22" })),
                React.createElement("div", { className: "proj-card" },
                  React.createElement("h4", null, "Scenario Risk"),
                  React.createElement("div", { className: "big scenario" + (up ? " up" : "") }, result.scenario_risk_pct, "%"),
                  React.createElement("div", { className: "cap" }, "With adjustments")
                )
              ),
              React.createElement("div", { className: "delta-line " + (up ? "up" : down ? "down" : "flat") },
                (delta > 0 ? "+" : "") + delta.toFixed(1) + " pts change"
              )
            ) : null
          ),
          scenario.tenant_id && status === "success" && result ? React.createElement("div", { className: "reco" },
            React.createElement("span", { className: "reco-ic" }, Ic.sparkle({ width: "26", height: "26" })),
            React.createElement("div", null, React.createElement("h3", null, "AI Recommendation"), React.createElement("p", null, result.recommendation)),
            React.createElement("img", { className: "reco-img", src: "assets/building.png", alt: "", "aria-hidden": "true", crossOrigin: "anonymous" })
          ) : null
        )
      ),
      scenario.tenant_id && status === "success" && result && allDrivers.length > 0 ? React.createElement("div", { className: "panel drivers" },
        React.createElement("h2", null, "What's Driving the Change?"),
        React.createElement("p", { className: "d-sub" }, "Key factors contributing to the risk change in this scenario."),
        React.createElement("div", { className: "driver-grid" },
          allDrivers.map(function(d, i) {
            return React.createElement("div", { className: "driver-card", key: i },
              React.createElement("span", { className: "dc-ic " + d.cls }, d.ic({ width: "22", height: "22" })),
              React.createElement("div", { className: "dc-name" }, d.name),
              React.createElement("div", { className: "dc-pts " + ptsClass(d.pts) }, fmtPts(d.pts)),
              React.createElement("div", { className: "dc-note" }, d.note)
            );
          })
        )
      ) : null
    )
  );
}

function TenantProfile(props) {
  var useState = React.useState, useEffect = React.useEffect;
  var _a = useState(null), profile = _a[0], setProfile = _a[1];
  var _b = useState(null), score = _b[0], setScore = _b[1];
  var user = props.user;

  useEffect(function() {
    if (!user) return;
    var controller = new AbortController();
    var signal = controller.signal;
    authFetch("/tenants/" + encodeURIComponent(user.handle), { signal: signal }).then(function(r) { return r.ok ? r.json() : null; }).then(function(d) { setProfile(d); }).catch(function() {});
    authFetch("/score/" + encodeURIComponent(user.handle), { signal: signal }).then(function(r) { return r.ok ? r.json() : null; }).then(function(d) { setScore(d); }).catch(function() {});
    return function() { controller.abort(); };
  }, [user]);

  return React.createElement("div", { className: "sim" },
    React.createElement("div", { className: "shell" },
      React.createElement("div", { className: "sim-head" },
        React.createElement("button", { className: "back", onClick: props.onLogout },
          Ic.arrow({ width: "16", height: "16", style: { transform: "rotate(180deg)" } }), " Sign out"
        ),
        React.createElement("h1", null, React.createElement("span", { className: "accent" }, "Your"), " Profile"),
        React.createElement("p", null, "Your tenant profile and risk assessment.")
      ),
      profile ? React.createElement("div", { className: "panel", style: { padding: "28px" } },
        React.createElement("div", { className: "panel-title" },
          React.createElement("span", { className: "pt-ic" }, Ic.person({ width: "22", height: "22" })),
          React.createElement("div", null, React.createElement("h2", null, profile.full_name || profile.handle), React.createElement("div", { className: "pt-sub" }, "@" + profile.handle))
        ),
        React.createElement("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginTop: "16px" } },
          React.createElement("div", { className: "proj-card" },
            React.createElement("h4", null, "Email"),
            React.createElement("div", { style: { fontFamily: "Inter", fontWeight: 600, fontSize: "1rem", marginTop: "8px" } }, profile.email)
          ),
          React.createElement("div", { className: "proj-card" },
            React.createElement("h4", null, "Phone"),
            React.createElement("div", { style: { fontFamily: "Inter", fontWeight: 600, fontSize: "1rem", marginTop: "8px" } }, profile.phone || "Not provided")
          )
        ),
        score ? React.createElement("div", { style: { marginTop: "20px", textAlign: "center" } },
          React.createElement("div", { className: "panel", style: { display: "inline-block", padding: "20px 40px" } },
            React.createElement("h4", { style: { color: "var(--muted)", marginBottom: "8px" } }, "Current Churn Risk"),
            React.createElement("div", { className: "big baseline", style: { fontSize: "3.5rem" } }, score.risk_pct, "%"),
            React.createElement("div", { className: "cap", style: { marginTop: "8px" } }, "Risk band: ", score.risk_band)
          )
        ) : React.createElement("div", { className: "panel", style: { marginTop: "20px", textAlign: "center", padding: "30px" } },
          React.createElement("p", { style: { color: "var(--muted)", fontSize: "0.95rem" } }, "No risk assessment available yet. Your profile will be evaluated after you've been added to a portfolio.")
        )
      ) : React.createElement("div", { className: "skeleton", style: { height: "200px" } })
    )
  );
}

function LandlordDashboard(props) {
  var useState = React.useState, useEffect = React.useEffect;
  var _a = useState([]), myTenants = _a[0], setMyTenants = _a[1];
  var _b = useState([]), tenantIds = _b[0], setTenantIds = _b[1];
  var _c = useState(false), showCreate = _c[0], setShowCreate = _c[1];
  var _d = useState(""), newHandle = _d[0], setNewHandle = _d[1];
  var _e = useState(""), newName = _e[0], setNewName = _e[1];
  var _f = useState(""), newEmail = _f[0], setNewEmail = _f[1];
  var _g = useState(""), newPhone = _g[0], setNewPhone = _g[1];
  var _h = useState(false), creating = _h[0], setCreating = _h[1];
  var _i = useState(""), createError = _i[0], setCreateError = _i[1];
  var _j = useState("sim"), landView = _j[0], setLandView = _j[1];
  var _k = useState(""), selectedTenantHandle = _k[0], setSelectedTenantHandle = _k[1];
  var _l = useState(""), searchQuestion = _l[0], setSearchQuestion = _l[1];
  var _m = useState(""), searchAnswer = _m[0], setSearchAnswer = _m[1];
  var _n = useState(false), searchLoading = _n[0], setSearchLoading = _n[1];

  useEffect(function() {
    var controller = new AbortController();
    var signal = controller.signal;
    authFetch("/landlords/tenants", { signal: signal }).then(function(r) { return r.json(); }).then(function(data) {
      var tenants = data || [];
      if (tenants.length > 0) {
        setMyTenants(tenants);
        setTenantIds(tenants.map(function(t) { return t.handle; }));
      } else {
        return fetch(window.API_BASE + "/tenants", { signal: signal }).then(function(r) { return r.json(); });
      }
    }).then(function(publicData) {
      if (publicData && publicData.tenants && publicData.tenants.length > 0) {
        setMyTenants(publicData.tenants.map(function(t) { return { handle: t, full_name: "Tenant " + t, email: "" }; }));
        setTenantIds(publicData.tenants);
      }
    }).catch(function(err) {
      if (err.name === "AbortError") return;
    });
    return function() { controller.abort(); };
  }, []);

  function handleCreateTenant(e) {
    e.preventDefault();
    setCreating(true);
    setCreateError("");
    authFetch("/landlords/tenants", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tenant_handle: newHandle, full_name: newName, email: newEmail, phone: newPhone }),
    }).then(function(r) {
      if (!r.ok) return r.json().then(function(b) { throw new Error(b.detail || "failed to create"); });
      return r.json();
    }).then(function(tenant) {
      setMyTenants(function(prev) { return prev.concat([tenant]); });
      setTenantIds(function(prev) { return prev.concat([tenant.handle]); });
      setShowCreate(false);
      setNewHandle("");
      setNewName("");
      setNewEmail("");
      setNewPhone("");
    }).catch(function(err) { setCreateError(err.message); }).then(function() { setCreating(false); });
  }

  return React.createElement("div", null,
    React.createElement("div", { className: "shell", style: { paddingTop: "20px" } },
      React.createElement("div", { className: "topbar" },
        React.createElement("div", { className: "brand" },
          React.createElement("span", { className: "brand-mark" }, Ic.logo({ width: "30", height: "30" })),
          React.createElement("span", { className: "brand-word" }, "Forge", React.createElement("span", { className: "brand-x" }, "X"))
        ),
        React.createElement("div", { style: { display: "flex", gap: "12px", alignItems: "center" } },
          React.createElement("span", { style: { color: "var(--muted)", fontSize: "0.9rem" } }, "@" + props.user.handle),
          React.createElement("button", { className: "btn btn-ghost", onClick: props.onLogout }, Ic.logout({ width: "18", height: "18" }), " Sign out")
        )
      ),
      React.createElement("div", { style: { display: "flex", gap: "12px", margin: "20px 0" } },
        React.createElement("button", { className: "btn " + (landView === "sim" ? "btn-primary" : "btn-ghost"), onClick: function() { setLandView("sim"); } }, "What-If Simulator"),
        React.createElement("button", { className: "btn " + (landView === "tenants" ? "btn-primary" : "btn-ghost"), onClick: function() { setLandView("tenants"); } }, "My Tenants (" + myTenants.length + ")"),
        React.createElement("button", { className: "btn " + (landView === "search" ? "btn-primary" : "btn-ghost"), onClick: function() { setLandView("search"); } }, "AI Smart Search")
      ),
      landView === "tenants" ? React.createElement("div", null,
        React.createElement("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" } },
          React.createElement("h2", { className: "panel-title", style: { margin: 0 } }, "Your Tenants"),
          React.createElement("button", { className: "btn btn-primary", onClick: function() { setShowCreate(true); } }, Ic.plus({ width: "18", height: "18" }), " Add Tenant")
        ),
        showCreate ? React.createElement("div", { className: "panel", style: { marginBottom: "20px" } },
          React.createElement("h3", { style: { marginBottom: "16px", fontFamily: "var(--font-head)", fontWeight: 700 } }, "Create New Tenant"),
          createError ? React.createElement("div", { className: "auth-error", style: { marginBottom: "12px" } }, createError) : null,
          React.createElement("form", { onSubmit: handleCreateTenant, style: { display: "grid", gap: "14px" } },
            React.createElement("input", { type: "text", placeholder: "Unique handle (e.g. john_doe)", value: newHandle, onChange: function(e) { setNewHandle(e.target.value); }, required: true, minLength: 3, pattern: "^[a-zA-Z0-9_]+$", style: { padding: "12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--line)", fontFamily: "var(--font-body)", fontSize: "0.95rem" } }),
            React.createElement("input", { type: "text", placeholder: "Full name", value: newName, onChange: function(e) { setNewName(e.target.value); }, required: true, style: { padding: "12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--line)", fontFamily: "var(--font-body)", fontSize: "0.95rem" } }),
            React.createElement("input", { type: "email", placeholder: "Email", value: newEmail, onChange: function(e) { setNewEmail(e.target.value); }, required: true, style: { padding: "12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--line)", fontFamily: "var(--font-body)", fontSize: "0.95rem" } }),
            React.createElement("input", { type: "tel", placeholder: "Phone (optional)", value: newPhone, onChange: function(e) { setNewPhone(e.target.value); }, style: { padding: "12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--line)", fontFamily: "var(--font-body)", fontSize: "0.95rem" } }),
            React.createElement("div", { style: { display: "flex", gap: "10px" } },
              React.createElement("button", { type: "submit", className: "btn btn-primary", disabled: creating }, creating ? "Creating..." : "Create Tenant"),
              React.createElement("button", { type: "button", className: "btn btn-ghost", onClick: function() { setShowCreate(false); } }, "Cancel")
            )
          )
        ) : null,
        myTenants.length === 0 ? React.createElement("div", { className: "empty-state", style: { padding: "40px" } },
          React.createElement("p", null, "No tenants yet. Create one to get started.")
        ) : React.createElement("div", { style: { display: "grid", gap: "12px" } },
          myTenants.map(function(t) {
            return React.createElement("div", { className: "panel tenant-bar", key: t.handle, style: { cursor: "pointer" }, onClick: function() { setSelectedTenantHandle(t.handle); setLandView("sim"); } },
              React.createElement("span", { className: "tb-ic" }, Ic.person({ width: "24", height: "24" })),
              React.createElement("div", { className: "tb-body" },
                React.createElement("div", { className: "tb-label" }, t.full_name),
                React.createElement("div", { className: "select-wrap" },
                  React.createElement("span", { style: { fontFamily: "var(--font-head)", fontWeight: 700, fontSize: "1.05rem", color: "var(--ink)" } }, "@" + t.handle)
                )
              ),
              React.createElement("span", { style: { color: "var(--muted)", fontSize: "0.85rem", fontFamily: "Inter" } }, t.email)
            );
          })
        )
      ) : landView === "search" ? React.createElement("div", null,
        React.createElement("h2", { className: "panel-title", style: { margin: 0, marginBottom: "12px" } }, "AI Smart Search"),
        React.createElement("p", { style: { color: "var(--muted)", fontSize: "0.9rem", marginBottom: "16px" } }, "Ask questions about your tenants in natural language."),
        React.createElement("div", { className: "panel", style: { padding: "20px" } },
          React.createElement("div", { style: { display: "flex", gap: "10px", marginBottom: "16px" } },
            React.createElement("input", { type: "text", value: searchQuestion, onChange: function(e) { setSearchQuestion(e.target.value); }, placeholder: 'e.g. "Which tenants are at high risk?"', style: { flex: 1, padding: "12px 14px", border: "1px solid var(--line)", borderRadius: "var(--radius-sm)", fontFamily: "var(--font-body)", fontSize: "0.95rem", outline: "none" }, onKeyDown: function(e) { if (e.key === "Enter" && !searchLoading && searchQuestion.trim()) { setSearchLoading(true); setSearchAnswer(""); authFetch("/ai-search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: searchQuestion.trim(), context: JSON.stringify(myTenants) }) }).then(function(r) { return r.json(); }).then(function(data) { setSearchAnswer(data.answer || "No answer received"); setSearchLoading(false); }).catch(function() { setSearchAnswer("Failed to reach the server."); setSearchLoading(false); }); } } }),
            React.createElement("button", { className: "btn btn-primary", disabled: searchLoading || !searchQuestion.trim(), onClick: function() { setSearchLoading(true); setSearchAnswer(""); authFetch("/ai-search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: searchQuestion.trim(), context: JSON.stringify(myTenants) }) }).then(function(r) { return r.json(); }).then(function(data) { setSearchAnswer(data.answer || "No answer received"); setSearchLoading(false); }).catch(function() { setSearchAnswer("Failed to reach the server."); setSearchLoading(false); }); } }, searchLoading ? "Thinking..." : "Ask")
          ),
          searchAnswer ? React.createElement("div", { style: { background: "var(--blue-soft)", borderRadius: "var(--radius-sm)", padding: "16px", fontSize: "0.95rem", lineHeight: "1.6", whiteSpace: "pre-wrap" } }, searchAnswer) : null
        )
      ) : React.createElement(Simulator, { tenants: tenantIds, loading: false, landlord: "@" + props.user.handle, onLogout: props.onLogout, selectedTenant: selectedTenantHandle })
    )
  );
}

window.Landing = Landing;
window.Simulator = Simulator;
window.TenantProfile = TenantProfile;
window.LandlordDashboard = LandlordDashboard;
