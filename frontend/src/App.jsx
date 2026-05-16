import { useEffect, useState } from "react";
import Predict from "./pages/Predict";
import Models from "./pages/Models";
import Analytics from "./pages/Analytics";
import "./App.css";

const METRICS = [
  {
    label: "Best Model",
    value: "XGBoost",
    sub: "Selected",
    cls: "accent",
    fill: 100,
  },
  {
    label: "Recall",
    value: "86.3%",
    sub: "↑ Primary metric",
    cls: "green",
    fill: 86,
  },
  { label: "Accuracy", value: "83.9%", sub: "Test set", cls: "", fill: 84 },
  { label: "Precision", value: "59.6%", sub: "XGBoost", cls: "", fill: 60 },
];

export default function App() {
  const [page, setPage] = useState("predict");
  const [online, setOnline] = useState(true);
  useEffect(() => {
    const checkOnline = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/");
        const data = await res.json();
        console.log(data);
        setOnline(!!data.online);
      } catch {
        setOnline(false);
      }
    };
    checkOnline();
  }, []);
  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-left">
          <div className="nav-icon">⬡</div>
          <div>
            <div className="nav-title">
              Income <span>Prediction</span>
            </div>
            <div className="nav-sub">by 404 Brain</div>
          </div>
        </div>

        <div className="nav-center">
          {["predict", "models", "analytics"].map((p) => (
            <button
              key={p}
              className={`nav-btn ${page === p ? "active" : ""}`}
              onClick={() => setPage(p)}
            >
              {p}
            </button>
          ))}
        </div>

        <div className="nav-right">
          <div className={`nav-dot ${online ? "online" : "offline"}`} />
          <span className={`nav-status ${online ? "online" : "offline"}`}>
            API {online ? "Online" : "Offline"}
          </span>
        </div>
      </nav>

      <div className="metrics-bar">
        {METRICS.map((m) => (
          <div key={m.label} className="metric-item">
            <div className="metric-label">{m.label}</div>
            <div className={`metric-value ${m.cls}`}>{m.value}</div>
            <div className="metric-sub">{m.sub}</div>
            <div className="metric-bar-bottom">
              <div
                className="metric-bar-fill"
                style={{ width: `${m.fill}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <main className="main-content">
        {page === "predict" && <Predict online={online} />}
        {page === "models" && <Models />}
        {page === "analytics" && <Analytics />}
      </main>
    </div>
  );
}
