function App() {
  var useState = React.useState, useEffect = React.useEffect;
  var _a = useState(null), user = _a[0], setUser = _a[1];
  var _b = useState("loading"), page = _b[0], setPage = _b[1];

  useEffect(function() {
    var controller = new AbortController();
    var signal = controller.signal;
    var current = getCurrentUser();
    var token = localStorage.getItem("forgex_token");
    if (current && token) {
      authFetch("/auth/me", { signal: signal }).then(function(r) {
        if (!r.ok) { logout(); setUser(null); setPage("landing"); return; }
        return r.json();
      }).then(function(u) {
        if (u) { setUser(u); setPage(u.role === "tenant" ? "tenant" : "landlord"); }
        else { setPage("landing"); }
      }).catch(function(err) {
        if (err.name === "AbortError") return;
        logout(); setUser(null); setPage("landing");
      });
    } else {
      setPage("landing");
    }
    return function() { controller.abort(); };
  }, []);

  function handleLogin(data) {
    setUser({ handle: data.handle, role: data.role, email: data.email });
    if (data.role === "admin") {
      window.location.href = "http://localhost:8501";
      return;
    }
    setPage(data.role === "tenant" ? "tenant" : "landlord");
  }

  function handleLogout() {
    logout();
    setUser(null);
    setPage("landing");
  }

  if (page === "loading") {
    return React.createElement("div", { style: { display: "flex", justifyContent: "center", alignItems: "center", height: "100vh", background: "var(--bg)" } },
      React.createElement("div", { className: "spinner" })
    );
  }

  if (page === "landing") {
    return React.createElement(Landing, {
      onLogin: function() { setPage("auth"); },
      onRegister: function() { setPage("auth"); }
    });
  }

  if (page === "auth") {
    return React.createElement(AuthPage, { onLogin: handleLogin });
  }

  if (page === "tenant" && user) {
    return React.createElement(TenantProfile, { user: user, onLogout: handleLogout });
  }

  if (page === "landlord" && user) {
    return React.createElement(LandlordDashboard, { user: user, onLogout: handleLogout });
  }

  return React.createElement(Landing, { onLogin: function() { setPage("auth"); } });
}

var root = ReactDOM.createRoot(document.getElementById("root"));
root.render(React.createElement(App));
