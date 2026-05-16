const MODELS = [
  {
    name: "XGBoost",
    acc: 0.8297,
    prec: 0.5965,
    rec: 0.863,
    f1: 0.7054,
    best: true,
  },
  {
    name: "SVM (C=0.01)",
    acc: 0.7708,
    prec: 0.5089,
    rec: 0.8596,
    f1: 0.6393,
    best: false,
  },
  {
    name: "SVM (C=1.0)",
    acc: 0.7993,
    prec: 0.5483,
    rec: 0.8541,
    f1: 0.6679,
    best: false,
  },
  {
    name: "Logistic Regression (C=0.01)",
    acc: 0.805,
    prec: 0.5578,
    rec: 0.8417,
    f1: 0.671,
    best: false,
  },
  {
    name: "Decision Tree (Depth=10)",
    acc: 0.8087,
    prec: 0.5634,
    rec: 0.8443,
    f1: 0.6758,
    best: false,
  },
  {
    name: "Logistic Regression (C=1.0)",
    acc: 0.8064,
    prec: 0.5604,
    rec: 0.8372,
    f1: 0.6714,
    best: false,
  },
  {
    name: "Decision Tree (Depth=5)",
    acc: 0.7978,
    prec: 0.5472,
    rec: 0.8346,
    f1: 0.661,
    best: false,
  },
  {
    name: "Random Forest",
    acc: 0.8384,
    prec: 0.6188,
    rec: 0.8229,
    f1: 0.7064,
    best: false,
  },
  {
    name: "Decision Tree (Depth=None)",
    acc: 0.8135,
    prec: 0.5953,
    rec: 0.6576,
    f1: 0.6249,
    best: false,
  },
  {
    name: "KNN (K=21)",
    acc: 0.8678,
    prec: 0.749,
    rec: 0.6625,
    f1: 0.7031,
    best: false,
  },
];

const fmt = (v) => (v * 100).toFixed(1) + "%";

export default function Models() {
  const sorted = [...MODELS].sort((a, b) => b.rec - a.rec);

  return (
    <div>
      <div className="section-title" style={{ marginBottom: "1rem" }}>
        Model Performance — sorted by Recall
      </div>
      <div className="models-grid">
        {sorted.map((m) => (
          <div key={m.name} className={`model-card ${m.best ? "best" : ""}`}>
            <div className="model-name">
              {m.name}
              {m.best && <span className="best-badge">Best Recall</span>}
            </div>
            <div className="metrics-row">
              {[
                { label: "Recall", val: m.rec, primary: true },
                { label: "Accuracy", val: m.acc },
                { label: "Precision", val: m.prec },
                { label: "F1", val: m.f1 },
              ].map(({ label, val, primary }) => (
                <div key={label} className="metric">
                  <span className="metric-label">{label}</span>
                  <span className={`metric-value ${primary ? "primary" : ""}`}>
                    {fmt(val)}
                  </span>
                  <div className="metric-bar">
                    <div
                      className="metric-bar-fill"
                      style={{ width: `${val * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
