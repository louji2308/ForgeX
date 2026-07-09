const { useState, useEffect, useCallback, useRef } = React;

function useDebouncedSimulation(request, delayMs = 300) {
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('idle');
  const [errorMessage, setErrorMessage] = useState(null);

  useEffect(() => {
    if (!request || !request.tenant_id) return;
    setStatus('loading');
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const res = await fetch('http://localhost:8000/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
          signal: controller.signal,
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.message || `Request failed with ${res.status}`);
        }
        setResult(await res.json());
        setStatus('success');
        setErrorMessage(null);
      } catch (err) {
        if (err.name === 'AbortError') return;
        setErrorMessage(err.message);
        setStatus('error');
      }
    }, delayMs);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [JSON.stringify(request), delayMs]);

  return { result, status, errorMessage };
}

function PersonaScenario({ persona, onScenarioChange, disabled }) {
  const sliders = {
    'price-sensitive': { rent: 8, credit: 100, maintenance: 'standard' },
    'maintenance-sensitive': { rent: 2, credit: 50, maintenance: 'priority' },
    'balanced': { rent: 5, credit: 50, maintenance: 'standard' },
    'at-risk': { rent: 12, credit: 200, maintenance: 'same_day' },
  };

  const preset = sliders[persona] || sliders['balanced'];

  return (
    <div className="persona-selector">
      <label>Persona Preset</label>
      <select
        value={persona}
        onChange={(e) => {
          const p = sliders[e.target.value] || sliders['balanced'];
          onScenarioChange({ rent_increase_pct: p.rent, retention_credit_usd: p.credit, maintenance_speed: p.maintenance });
        }}
        disabled={disabled}
      >
        <option value="balanced">Balanced Tenant</option>
        <option value="price-sensitive">Price-Sensitive</option>
        <option value="maintenance-sensitive">Maintenance-Sensitive</option>
        <option value="at-risk">At-Risk (High Churn)</option>
      </select>
    </div>
  );
}

function WhatIfSimulator({ tenantId, apiBase = '' }) {
  const [scenario, setScenario] = useState({
    tenant_id: tenantId,
    rent_increase_pct: 5,
    maintenance_speed: 'standard',
    retention_credit_usd: 50,
  });
  const [persona, setPersona] = useState('balanced');
  const [baseline, setBaseline] = useState(null);

  const { result, status, errorMessage } = useDebouncedSimulation(scenario);

  useEffect(() => {
    if (tenantId) {
      setScenario(prev => ({ ...prev, tenant_id: tenantId }));
    }
  }, [tenantId]);

  useEffect(() => {
    if (result) {
      setBaseline(prev => prev || { risk_pct: result.baseline_risk_pct });
    }
  }, [result]);

  const handleScenarioChange = useCallback((updates) => {
    setScenario(prev => ({ ...prev, ...updates }));
  }, []);

  const handleRentChange = (e) => {
    handleScenarioChange({ rent_increase_pct: parseFloat(e.target.value) });
  };

  const handleCreditChange = (e) => {
    handleScenarioChange({ retention_credit_usd: parseFloat(e.target.value) });
  };

  const handleMaintenanceChange = (e) => {
    handleScenarioChange({ maintenance_speed: e.target.value });
  };

  const handlePersonaChange = (newPersona) => {
    setPersona(newPersona);
    const presets = {
      'price-sensitive': { rent_increase_pct: 8, retention_credit_usd: 100, maintenance_speed: 'standard' },
      'maintenance-sensitive': { rent_increase_pct: 2, retention_credit_usd: 50, maintenance_speed: 'priority' },
      'balanced': { rent_increase_pct: 5, retention_credit_usd: 50, maintenance_speed: 'standard' },
      'at-risk': { rent_increase_pct: 12, retention_credit_usd: 200, maintenance_speed: 'same_day' },
    };
    handleScenarioChange(presets[newPersona] || presets['balanced']);
  };

  const handleRetry = () => {
    setScenario({ ...scenario });
  };

  return (
    <div className="simulator-panel">
      <div className="controls">
        <h2>Scenario Controls</h2>
        <PersonaScenario persona={persona} onScenarioChange={handlePersonaChange} disabled={!tenantId} />

        <div className="control-group">
          <label>Rent Increase: {scenario.rent_increase_pct}%</label>
          <input type="range" min="0" max="15" step="0.5" value={scenario.rent_increase_pct} onChange={handleRentChange} disabled={!tenantId} />
        </div>

        <div className="control-group">
          <label>Retention Credit: ${scenario.retention_credit_usd}</label>
          <input type="range" min="0" max="500" step="10" value={scenario.retention_credit_usd} onChange={handleCreditChange} disabled={!tenantId} />
        </div>

        <div className="control-group">
          <label>Maintenance Speed</label>
          <select value={scenario.maintenance_speed} onChange={handleMaintenanceChange} disabled={!tenantId}>
            <option value="standard">Standard (3-5 days)</option>
            <option value="priority">Priority (24h)</option>
            <option value="same_day">Same-Day (emergency)</option>
          </select>
        </div>
      </div>

      <div className="results">
        <h2>Risk Projection</h2>

        {status === 'loading' && <div className="skeleton" />}

        {status === 'error' && (
          <div className="error-banner">
            <span>Couldn't run that scenario: {errorMessage}</span>
            <button onClick={handleRetry}>Retry</button>
          </div>
        )}

        {status === 'success' && result && (
          <>
            <div className="risk-meter">
              <div className="compare-grid">
                <div className="compare-card">
                  <h4>Baseline Risk</h4>
                  <div className="risk-value baseline">{result.baseline_risk_pct}%</div>
                  <div className="risk-label">Current conditions</div>
                </div>
                <div className="compare-card">
                  <h4>Scenario Risk</h4>
                  <div className="risk-value scenario">{result.scenario_risk_pct}%</div>
                  <div className="risk-label">With adjustments</div>
                </div>
              </div>
              <div className={`risk-delta ${result.delta_pts > 0 ? 'positive' : 'negative'}`}>
                {result.delta_pts > 0 ? '+' : ''}{result.delta_pts.toFixed(1)} pts change
              </div>
            </div>

            <div className="recommendation">
              <h3>AI Recommendation</h3>
              <p>{result.recommendation}</p>
            </div>
          </>
        )}

        {status === 'idle' && (
          <div style={{ textAlign: 'center', color: '#8892b0', padding: '40px 0' }}>
            Select a tenant and adjust controls to see risk projections
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  const [tenantList, setTenantList] = useState([]);
  const [selectedTenant, setSelectedTenant] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(r => r.json())
      .then(data => {
        if (data.tenants_indexed > 0) {
          setTenantList(Array.from({ length: Math.min(100, data.tenants_indexed) }, (_, i) => `T${String(i).padStart(6, '0')}`));
        }
      })
      .catch(e => console.error('Health check failed:', e))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container">
      <div className="header">
        <h1>ForgeX — What-If Simulator</h1>
        <p>Adjust lease terms and see the causal impact on tenant churn risk</p>
      </div>

      <div className="tenant-select">
        <label>Select Tenant</label>
        <select value={selectedTenant} onChange={e => setSelectedTenant(e.target.value)} disabled={loading}>
          <option value="">-- Select a tenant --</option>
          {tenantList.map(tid => (
            <option key={tid} value={tid}>{tid}</option>
          ))}
        </select>
      </div>

      {selectedTenant ? (
        <WhatIfSimulator tenantId={selectedTenant} />
      ) : (
        <div style={{ textAlign: 'center', padding: '80px 0', color: '#8892b0' }}>
          {loading ? 'Loading tenant data...' : 'Select a tenant above to begin simulation'}
        </div>
      )}
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
