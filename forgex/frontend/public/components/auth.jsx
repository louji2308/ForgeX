window.authFetch = function(url, options) {
  var token = localStorage.getItem("forgex_token");
  var headers = options && options.headers ? Object.assign({}, options.headers) : {};
  if (token) {
    headers["Authorization"] = "Bearer " + token;
  }
  return fetch(window.API_BASE + url, Object.assign({}, options, { headers: headers }));
};

window.login = function(handle, password) {
  return authFetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ handle: handle, password: password }),
  }).then(function(r) {
    if (!r.ok) return r.json().then(function(b) { var msg = typeof b.detail === "string" ? b.detail : (Array.isArray(b.detail) ? b.detail.map(function(e) { return e.msg; }).join("; ") : b.message || "login failed"); throw new Error(msg); });
    return r.json();
  }).then(function(data) {
    localStorage.setItem("forgex_token", data.token);
    localStorage.setItem("forgex_user", JSON.stringify({ handle: data.handle, role: data.role, email: data.email }));
    return data;
  });
};

window.register = function(handle, email, password, role) {
  if (!role) role = "landlord";
  return authFetch("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ handle: handle, email: email, password: password, role: role }),
  }).then(function(r) {
    if (!r.ok) return r.json().then(function(b) { var msg = typeof b.detail === "string" ? b.detail : (Array.isArray(b.detail) ? b.detail.map(function(e) { return e.msg; }).join("; ") : b.message || "registration failed"); throw new Error(msg); });
    return r.json();
  });
};

window.logout = function() {
  var token = localStorage.getItem("forgex_token");
  if (token) {
    authFetch("/auth/logout", { method: "POST" }).catch(function() {});
  }
  localStorage.removeItem("forgex_token");
  localStorage.removeItem("forgex_user");
};

window.getCurrentUser = function() {
  var raw = localStorage.getItem("forgex_user");
  return raw ? JSON.parse(raw) : null;
};

function AuthPage(props) {
  var _this = this;
  var useState = React.useState;
  var _a = useState("login"), mode = _a[0], setMode = _a[1];
  var _b = useState(""), handle = _b[0], setHandle = _b[1];
  var _c = useState(""), email = _c[0], setEmail = _c[1];
  var _d = useState(""), password = _d[0], setPassword = _d[1];
  var _e = useState("landlord"), role = _e[0], setRole = _e[1];
  var _f = useState(""), error = _f[0], setError = _f[1];
  var _g = useState(false), loading = _g[0], setLoading = _g[1];

  function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    if (mode === "login") {
      login(handle, password).then(function(data) {
        props.onLogin(data);
      }).catch(function(err) {
        setError(err.message);
        setLoading(false);
      });
    } else {
      register(handle, email, password, role).then(function() {
        return login(handle, password);
      }).then(function(data) {
        props.onLogin(data);
      }).catch(function(err) {
        setError(err.message);
        setLoading(false);
      });
    }
  }

  return React.createElement("div", { className: "auth-page" },
    React.createElement("div", { className: "auth-card" },
      React.createElement("div", { className: "auth-brand" },
        React.createElement("span", { className: "brand-mark" },
          React.createElement("svg", { width: "36", height: "36", viewBox: "0 0 36 36", fill: "none" },
            React.createElement("rect", { width: "36", height: "36", rx: "10", fill: "#2f6bff" }),
            React.createElement("path", { d: "M10 26V14l8-6 8 6v12H18v-6h-4v6h-4z", fill: "#fff" })
          )
        ),
        React.createElement("span", { className: "brand-word" }, "Forge", React.createElement("span", { className: "brand-x" }, "X"))
      ),
      React.createElement("h1", { className: "auth-title" }, mode === "login" ? "Welcome back" : "Create your account"),
      React.createElement("p", { className: "auth-sub" }, mode === "login" ? "Sign in to your dashboard" : "Choose a unique handle — like an Instagram username"),

      error ? React.createElement("div", { className: "auth-error" }, error) : null,

      React.createElement("form", { onSubmit: handleSubmit, className: "auth-form" },
        React.createElement("div", { className: "auth-field" },
          React.createElement("label", null, "Handle"),
          React.createElement("input", { type: "text", value: handle, onChange: function(e) { setHandle(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, "")); }, placeholder: "lowercase_letters_numbers_underscores", required: true, minLength: 3, pattern: "^[a-z0-9_]+$", title: "Only lowercase letters, numbers, and underscores" })
        ),
        mode === "register" ? React.createElement("div", { className: "auth-field" },
          React.createElement("label", null, "Email"),
          React.createElement("input", { type: "email", value: email, onChange: function(e) { setEmail(e.target.value); }, placeholder: "you@example.com", required: true })
        ) : null,
        mode === "register" ? React.createElement("div", { className: "auth-field" },
          React.createElement("label", null, "Role"),
          React.createElement("select", { value: role, onChange: function(e) { setRole(e.target.value); } },
            React.createElement("option", { value: "landlord" }, "Landlord"),
            React.createElement("option", { value: "tenant" }, "Tenant"),
            React.createElement("option", { value: "admin" }, "Admin")
          )
        ) : null,
        React.createElement("div", { className: "auth-field" },
          React.createElement("label", null, "Password"),
          React.createElement("input", { type: "password", value: password, onChange: function(e) { setPassword(e.target.value); }, placeholder: "Min 6 characters", required: true, minLength: 6 })
        ),
        React.createElement("button", { type: "submit", className: "btn btn-primary auth-btn", disabled: loading },
          loading ? "Please wait..." : (mode === "login" ? "Sign In" : "Create Account")
        )
      ),

      React.createElement("p", { className: "auth-switch" },
        mode === "login" ? "Don't have an account? " : "Already have an account? ",
        React.createElement("button", { className: "auth-link", onClick: function() { setMode(mode === "login" ? "register" : "login"); setError(""); } },
          mode === "login" ? "Sign up" : "Sign in"
        )
      )
    )
  );
}

window.AuthPage = AuthPage;
