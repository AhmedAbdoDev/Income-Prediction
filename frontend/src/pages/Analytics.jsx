const CHARTS = [
  {
    file: "target_distribution_pie.png",
    title: "Income Distribution",
    percentage: "52%",
    color: "#00d4ff",
  },

  {
    file: "correlation_heatmap.png",
    title: "Correlation Heatmap",
    percentage: "91%",
    color: "#00ff88",
  },

  {
    file: "age_hist.png",
    title: "Age vs Income",
    percentage: "78%",
    color: "#00d4ff",
  },

  {
    file: "hours-per-week_hist.png",
    title: "Hours/Week vs Income",
    percentage: "84%",
    color: "#00ff88",
  },

  {
    file: "sex_stacked.png",
    title: "Sex vs Income",
    percentage: "69%",
    color: "#00d4ff",
  },

  {
    file: "education_stacked.png",
    title: "Education vs Income",
    percentage: "88%",
    color: "#00ff88",
  },

  {
    file: "occupation_stacked.png",
    title: "Occupation vs Income",
    percentage: "74%",
    color: "#00d4ff",
  },

  {
    file: "capital_category_vs_income.png",
    title: "Capital Category vs Income",
    percentage: "95%",
    color: "#00ff88",
  },
];

export default function Analytics() {
  return (
    <>
      {/* CSS */}

      <style>{`

        .analytics-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1.2rem;
        }

        .chart-card {
          background: #0d1520;

          border: 0.5px solid #1a2744;

          border-radius: 16px;

          overflow: hidden;

          transition: 0.3s ease;

          box-shadow:
            0 0 20px rgba(0,0,0,0.2);
        }

        .chart-card:hover {
          transform: translateY(-4px);

          border-color: #00d4ff66;

          box-shadow:
            0 0 25px rgba(0,212,255,0.12);
        }

        .chart-card-header {
          padding: 16px 18px;

          border-bottom: 0.5px solid #1a2744;

          font-size: 11px;

          color: #4a6080;

          letter-spacing: 0.12em;

          text-transform: uppercase;
        }

        .chart-progress {
          display: flex;

          justify-content: center;

          align-items: center;

          padding: 1.5rem 0 1rem;
        }

        .progress-ring {
          width: 110px;
          height: 110px;

          border-radius: 50%;

          display: flex;

          align-items: center;

          justify-content: center;

          box-shadow:
            0 0 25px rgba(0,0,0,0.3);

          transition: 0.4s ease;
        }

        .progress-ring:hover {
          transform: scale(1.05);
        }

        .progress-inner {
          width: 82px;
          height: 82px;

          border-radius: 50%;

          background: #111c2e;

          display: flex;

          align-items: center;

          justify-content: center;
        }

        .progress-inner span {
          font-size: 22px;

          font-weight: 600;

          color: #e0eaff;
        }

        .chart-card img {
          width: 100%;

          display: block;

          border-top: 0.5px solid #1a2744;
        }

        @media (max-width: 768px) {

          .analytics-grid {
            grid-template-columns: 1fr;
          }

        }

      `}</style>

      {/* TITLE */}

      <div className="section-title" style={{ marginBottom: "1rem" }}>
        Analysis Charts
      </div>

      {/* GRID */}

      <div className="analytics-grid">
        {CHARTS.map((c) => {
          const value = parseInt(c.percentage);

          return (
            <div key={c.file} className="chart-card">
              <div className="chart-card-header">{c.title}</div>

              {/* PERCENTAGE */}

              <div className="chart-progress">
                <div
                  className="progress-ring"
                  style={{
                    background: `conic-gradient(
                      ${c.color} ${value * 3.6}deg,
                      rgba(255,255,255,0.06) 0deg
                    )`,
                  }}
                >
                  <div className="progress-inner">
                    <span>{c.percentage}</span>
                  </div>
                </div>
              </div>

              {/* IMAGE */}

              <img
                src={`http://127.0.0.1:8000/analytics/${c.file}`}
                alt={c.title}
                onError={(e) => {
                  e.target.style.display = "none";
                }}
              />
            </div>
          );
        })}
      </div>
    </>
  );
}
