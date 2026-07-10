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
        if (u) { setUser(u); setPage(u.role); }
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
      var token = localStorage.getItem("forgex_token");
      window.location.href = "http://localhost:8051/?token=" + encodeURIComponent(token || "");
      return;
    }
    setPage(data.role);
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

  if (page === "admin" && user) {
    return React.createElement(AdminPage, { user: user, onLogout: handleLogout });
  }

  return React.createElement(Landing, { onLogin: function() { setPage("auth"); } });
}

function AdminPage(props) {
  var token = localStorage.getItem("forgex_token") || "";
  return React.createElement("div", { className: "sim" },
    React.createElement("div", { className: "shell" },
      React.createElement("div", { className: "sim-head" },
        React.createElement("button", { className: "back", onClick: props.onLogout },
          Ic.arrow({ width: "16", height: "16", style: { transform: "rotate(180deg)" } }), " Sign out"
        ),
        React.createElement("h1", null, React.createElement("span", { className: "accent" }, "ForgeX"), " Admin"),
        React.createElement("p", null, "You are signed in as an administrator.")
      ),
      React.createElement("div", { className: "panel", style: { textAlign: "center", padding: "40px" } },
        React.createElement("div", { className: "panel-title", style: { justifyContent: "center" } },
          React.createElement("span", { className: "pt-ic" }, Ic.shield({ width: "22", height: "22" })),
          React.createElement("h2", null, "@" + props.user.handle)
        ),
        React.createElement("p", { style: { color: "var(--muted)", margin: "16px 0 24px", fontSize: "0.95rem" } },
          "The admin dashboard runs on a separate interface. Click below to open it."
        ),
        React.createElement("a", { href: "http://localhost:8051/?token=" + encodeURIComponent(token), target: "_blank", className: "btn btn-primary", style: { textDecoration: "none", display: "inline-flex" } },
          "Open Admin Dashboard ", React.createElement("span", { className: "btn-icon" }, Ic.arrow({ width: "16", height: "16" }))
        )
      )
    )
  );
}

var root = ReactDOM.createRoot(document.getElementById("root"));
root.render(React.createElement(App));
