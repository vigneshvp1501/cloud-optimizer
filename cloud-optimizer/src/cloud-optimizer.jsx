import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

// ─── Palette & Theme ────────────────────────────────────────────────
const C = {
  bg: "#070B14",
  panel: "#0D1526",
  border: "#1A2840",
  accent: "#00D4FF",
  green: "#00FF88",
  amber: "#FFB800",
  red: "#FF3860",
  purple: "#9B5DE5",
  muted: "#4A6080",
  text: "#C8D8F0",
  dim: "#6A8BAA",
};

// ─── Utility ─────────────────────────────────────────────────────────
const rand = (min, max) => Math.random() * (max - min) + min;
const clamp = (v, mn, mx) => Math.max(mn, Math.min(mx, v));
const fmt = (n, d = 1) => Number(n).toFixed(d);

function generateInitialSeries(len = 40) {
  let cpu = 45, mem = 60, net = 30, pred = 45;
  return Array.from({ length: len }, (_, i) => {
    cpu = clamp(cpu + rand(-4, 4), 5, 95);
    mem = clamp(mem + rand(-2, 3), 20, 90);
    net = clamp(net + rand(-8, 8), 5, 100);
    pred = clamp(pred + rand(-3, 5), 5, 95);
    return {
      t: i,
      cpu: +fmt(cpu),
      mem: +fmt(mem),
      net: +fmt(net),
      pred: +fmt(pred),
    };
  });
}

// ─── Sub-components ───────────────────────────────────────────────────

function Panel({ title, icon, children, accent, style = {} }) {
  return (
    <div style={{
      background: C.panel,
      border: `1px solid ${accent || C.border}`,
      borderRadius: 8,
      padding: "16px",
      boxShadow: accent ? `0 0 20px ${accent}18` : "none",
      ...style,
    }}>
      {title && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <span style={{ fontSize: 14 }}>{icon}</span>
          <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 11, letterSpacing: "0.12em", color: accent || C.dim, textTransform: "uppercase" }}>
            {title}
          </span>
        </div>
      )}
      {children}
    </div>
  );
}

function Gauge({ label, value, max = 100, color, unit = "%" }) {
  const pct = clamp(value / max, 0, 1);
  const r = 32, cx = 40, cy = 40;
  const circumference = 2 * Math.PI * r;
  const dash = pct * circumference * 0.75;
  const gap = circumference - dash;
  const rotation = -135;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <svg width={80} height={70} viewBox="0 0 80 75">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={C.border} strokeWidth={5}
          strokeDasharray={`${circumference * 0.75} ${circumference * 0.25}`}
          strokeLinecap="round"
          transform={`rotate(${rotation} ${cx} ${cy})`} />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={5}
          strokeDasharray={`${dash} ${circumference - dash}`}
          strokeLinecap="round"
          transform={`rotate(${rotation} ${cx} ${cy})`}
          style={{ filter: `drop-shadow(0 0 4px ${color})`, transition: "stroke-dasharray 0.5s ease" }} />
        <text x={cx} y={cy + 4} textAnchor="middle" fill={color}
          style={{ fontFamily: "'Space Mono', monospace", fontSize: 13, fontWeight: "bold" }}>
          {fmt(value, 0)}{unit}
        </text>
      </svg>
      <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 9, color: C.dim, letterSpacing: "0.1em", textTransform: "uppercase" }}>{label}</span>
    </div>
  );
}

function StatBadge({ label, value, color = C.accent, unit = "" }) {
  return (
    <div style={{
      background: "#0A1020",
      border: `1px solid ${C.border}`,
      borderRadius: 6,
      padding: "10px 14px",
      flex: 1,
    }}>
      <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 9, color: C.dim, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
      <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 20, color, fontWeight: "bold", lineHeight: 1 }}>
        {value}<span style={{ fontSize: 11, color: C.dim }}>{unit}</span>
      </div>
    </div>
  );
}

function AlarmBadge({ name, state }) {
  const colors = { OK: C.green, ALARM: C.red, INSUFFICIENT: C.amber };
  const c = colors[state] || C.dim;
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "7px 10px", borderRadius: 5, background: "#0A1020",
      border: `1px solid ${C.border}`, marginBottom: 6,
    }}>
      <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 10, color: C.text }}>{name}</span>
      <span style={{
        fontFamily: "'Space Mono', monospace", fontSize: 9, color: c,
        background: `${c}18`, padding: "2px 8px", borderRadius: 4,
        border: `1px solid ${c}40`,
        animation: state === "ALARM" ? "pulse 1.2s ease-in-out infinite" : "none",
      }}>{state}</span>
    </div>
  );
}

function LogLine({ ts, msg, level }) {
  const lc = { INFO: C.accent, WARN: C.amber, ERROR: C.red, DEBUG: C.muted };
  return (
    <div style={{ display: "flex", gap: 10, padding: "3px 0", borderBottom: `1px solid ${C.border}20` }}>
      <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 9, color: C.dim, whiteSpace: "nowrap" }}>{ts}</span>
      <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 9, color: lc[level] || C.dim, minWidth: 36 }}>[{level}]</span>
      <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 9, color: C.text }}>{msg}</span>
    </div>
  );
}

function ScalingAction({ time, action, count, cost }) {
  const isUp = action === "SCALE_OUT";
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8, padding: "6px 10px",
      background: "#0A1020", borderRadius: 5, border: `1px solid ${isUp ? C.accent : C.amber}30`,
      marginBottom: 5,
    }}>
      <span style={{ fontSize: 14 }}>{isUp ? "⬆️" : "⬇️"}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 9, color: isUp ? C.accent : C.amber }}>{action}</div>
        <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 8, color: C.dim }}>{time}</div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 10, color: C.text }}>{count} inst</div>
        <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 8, color: C.dim }}>${cost}/hr</div>
      </div>
    </div>
  );
}

const CUSTOM_TOOLTIP = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, padding: "8px 12px" }}>
      {payload.map(p => (
        <div key={p.dataKey} style={{ fontFamily: "'Space Mono', monospace", fontSize: 10, color: p.color }}>
          {p.name}: {fmt(p.value)}%
        </div>
      ))}
    </div>
  );
};

// ─── Main App ─────────────────────────────────────────────────────────
export default function CloudOptimizer() {
  const fetchASGStatus = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/asg-status");
      return await res.json();
    } catch (err) {
      console.error("API error:", err);
      return null;
    }
  };

  const [series, setSeries] = useState(generateInitialSeries);
  const [instanceCount, setInstanceCount] = useState(4);
  const [scalingMode, setScalingMode] = useState("auto"); // auto | manual
  const [alarmThreshold, setAlarmThreshold] = useState(75);
  const [lstmEnabled, setLstmEnabled] = useState(true);
  const [windowSize, setWindowSize] = useState(10);
  const [logs, setLogs] = useState([]);
  const [scalingActions, setScalingActions] = useState([]);
  const [alarms, setAlarms] = useState({ CPUAlarm: "OK", MemAlarm: "OK", PredLoadAlarm: "OK" });
  const [cost, setCost] = useState({ hourly: 0.192, monthly: 138.24, saved: 0 });
  const [tick, setTick] = useState(0);
  const logRef = useRef(null);
  const tickRef = useRef(0);

  const addLog = useCallback((msg, level = "INFO") => {
    const now = new Date();
    const ts = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
    setLogs(prev => [...prev.slice(-49), { ts, msg, level, id: Date.now() + Math.random() }]);
  }, []);

  useEffect(() => {
    const id = setInterval(async () => {
      tickRef.current++;
      const t = tickRef.current;
      setTick(t);

       const apiData = await fetchASGStatus();

      setSeries(prev => {
        const last = prev[prev.length - 1];
        const newCpu = clamp(last.cpu + rand(-5, 5), 5, 98);
        const newMem = clamp(last.mem + rand(-2, 3), 20, 92);
        const newNet = clamp(last.net + rand(-8, 8), 5, 100);
        // LSTM prediction slightly ahead of actual
        const newPred = lstmEnabled && apiData
          ? apiData.predicted
          : newCpu;
        return [...prev.slice(-49), { t, cpu: +fmt(newCpu), mem: +fmt(newMem), net: +fmt(newNet), pred: +fmt(newPred) }];
      });

      if (apiData) {
        setInstanceCount(apiData.running);
      }

      // Update alarms based on latest
      setSeries(prev => {
        const data = apiData;
        const last = prev[prev.length - 1];
        const newAlarms = {
          CPUAlarm: last.cpu > alarmThreshold ? "ALARM" : "OK",
          MemAlarm: last.mem > 80 ? "ALARM" : "OK",
          PredLoadAlarm: last.pred > alarmThreshold ? "ALARM" : "OK",
        };
        setAlarms(newAlarms);

        // Trigger scaling
        if (scalingMode === "auto") {
          setInstanceCount(ic => {
            let newIc = ic;
            if (last.pred > alarmThreshold && ic < 12) {
              newIc = Math.min(12, ic + 1);
              const now = new Date();
              const ts = `${String(now.getHours()).padStart(2,"0")}:${String(now.getMinutes()).padStart(2,"0")}`;
              setScalingActions(a => [{
                time: ts, action: "SCALE_OUT", count: newIc, cost: +(newIc * 0.048).toFixed(3), id: Date.now()
              }, ...a.slice(0, 9)]);
              addLog(`Auto Scaling: SCALE_OUT → ${newIc} instances (PredLoad=${fmt(last.pred)}%)`, "INFO");
            } else if (last.pred < 35 && ic > 2) {
              newIc = Math.max(2, ic - 1);
              const now = new Date();
              const ts = `${String(now.getHours()).padStart(2,"0")}:${String(now.getMinutes()).padStart(2,"0")}`;
              setScalingActions(a => [{
                time: ts, action: "SCALE_IN", count: newIc, cost: +(newIc * 0.048).toFixed(3), id: Date.now()
              }, ...a.slice(0, 9)]);
              addLog(`Auto Scaling: SCALE_IN → ${newIc} instances (PredLoad=${fmt(last.pred)}%)`, "WARN");
            }
            return newIc;
          });
        }
        return prev;
      });

      // Occasional logs
      if (t % 5 === 0) addLog(`CloudWatch metrics published: PredictedLoad custom metric`, "DEBUG");
      if (t % 8 === 0) addLog(`LSTM sliding window [size=${windowSize}] inference complete`, "INFO");
      if (t % 13 === 0) addLog(`Partitioned data written to S3 storage`, "DEBUG");

      // Update cost
      setInstanceCount(ic => {
        setCost({ hourly: +(ic * 0.048).toFixed(3), monthly: +(ic * 0.048 * 720).toFixed(2), saved: +(12 * 0.048 * 720 - ic * 0.048 * 720).toFixed(2) });
        return ic;
      });
    }, 2000);
    return () => clearInterval(id);
  }, [lstmEnabled, alarmThreshold, scalingMode, windowSize, addLog]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const last = series[series.length - 1] || {};

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${C.bg}; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        @keyframes scanline { 0% { transform: translateY(-100%); } 100% { transform: translateY(100vh); } }
        @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }
        .toggle { appearance: none; width:36px; height:18px; background:${C.border}; border-radius:9px; cursor:pointer; position:relative; transition:background .3s; }
        .toggle:checked { background:${C.accent}; }
        .toggle::after { content:''; position:absolute; top:2px; left:2px; width:14px; height:14px; background:#fff; border-radius:50%; transition:left .3s; }
        .toggle:checked::after { left:20px; }
        input[type=range] { -webkit-appearance:none; width:100%; height:4px; background:${C.border}; border-radius:2px; outline:none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px; background:${C.accent}; border-radius:50%; cursor:pointer; box-shadow:0 0 6px ${C.accent}; }
      `}</style>

      <div style={{
        minHeight: "100vh", background: C.bg, padding: "20px",
        fontFamily: "'Space Mono', monospace",
        backgroundImage: `radial-gradient(ellipse at 20% 20%, #0D1E3510 0%, transparent 60%), radial-gradient(ellipse at 80% 80%, #001A2C08 0%, transparent 60%)`,
      }}>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
          <div>
            <h1 style={{ fontFamily: "'Syne', sans-serif", fontSize: 24, fontWeight: 800, color: "#fff", letterSpacing: "-0.02em" }}>
              CLOUD<span style={{ color: C.accent }}>OPT</span>
            </h1>
            <div style={{ fontSize: 9, color: C.dim, letterSpacing: "0.2em", textTransform: "uppercase", marginTop: 2 }}>
              Resource Optimization System · AWS Infrastructure
            </div>
          </div>
          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 9, color: C.dim, letterSpacing: "0.1em" }}>SYSTEM STATUS</div>
              <div style={{ fontSize: 11, color: C.green, display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 6, height: 6, background: C.green, borderRadius: "50%", display: "inline-block", boxShadow: `0 0 6px ${C.green}`, animation: "pulse 2s infinite" }} />
                OPERATIONAL
              </div>
            </div>
            <div style={{ width: 1, height: 32, background: C.border }} />
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 9, color: C.dim }}>TICK</div>
              <div style={{ fontSize: 11, color: C.accent }}># {tick}</div>
            </div>
          </div>
        </div>

        {/* Top Stats Row */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 16 }}>
          <StatBadge label="Instances" value={instanceCount} unit=" nodes" color={C.accent} />
          <StatBadge label="Hourly Cost" value={`$${cost.hourly}`} color={C.amber} />
          <StatBadge label="Monthly Cost" value={`$${cost.monthly}`} color={C.amber} />
          <StatBadge label="Monthly Saved" value={`$${cost.saved}`} color={C.green} />
          <StatBadge label="LSTM Window" value={windowSize} unit=" pts" color={C.purple} />
        </div>

        {/* Main Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 260px", gap: 12 }}>

          {/* LEFT COLUMN */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

            {/* Monitoring */}
            <Panel title="Monitoring & Data Storage" icon="📡" accent={C.accent}>
              <div style={{ display: "flex", justifyContent: "space-around", marginBottom: 10 }}>
                <Gauge label="CPU" value={last.cpu || 0} color={last.cpu > alarmThreshold ? C.red : C.accent} />
                <Gauge label="Memory" value={last.mem || 0} color={last.mem > 80 ? C.red : C.green} />
              </div>
              <div style={{ display: "flex", justifyContent: "center" }}>
                <Gauge label="Network" value={last.net || 0} color={C.purple} />
              </div>
              <div style={{ marginTop: 10, padding: "6px 8px", background: "#0A1020", borderRadius: 5, fontSize: 9, color: C.dim }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span>CloudWatch Metrics</span>
                  <span style={{ color: C.green }}>● LIVE</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Data Retention</span>
                  <span style={{ color: C.accent }}>90 days</span>
                </div>
              </div>
            </Panel>

            {/* LSTM Config */}
            <Panel title="LSTM Prediction Engine" icon="🧠" accent={C.purple}>
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <span style={{ fontSize: 10, color: C.text }}>Dockerized LSTM</span>
                  <input type="checkbox" className="toggle" checked={lstmEnabled} onChange={e => {
                    setLstmEnabled(e.target.checked);
                    addLog(`LSTM Model ${e.target.checked ? "ENABLED" : "DISABLED"}`, e.target.checked ? "INFO" : "WARN");
                  }} />
                </div>
                <div style={{ marginBottom: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.dim, marginBottom: 4 }}>
                    <span>Sliding Window Size</span><span style={{ color: C.purple }}>{windowSize} pts</span>
                  </div>
                  <input type="range" min={5} max={30} value={windowSize} onChange={e => setWindowSize(+e.target.value)} />
                </div>
                <div style={{ background: "#0A1020", borderRadius: 5, padding: 8 }}>
                  {[["Model", "LSTM v2.1"], ["Status", lstmEnabled ? "Running" : "Stopped"], ["Container", "docker:lstm-latest"], ["Epochs", "150"]].map(([k, v]) => (
                    <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, fontSize: 9 }}>
                      <span style={{ color: C.dim }}>{k}</span>
                      <span style={{ color: lstmEnabled ? C.text : C.muted }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </Panel>

            {/* Auto Scaling Controls */}
            <Panel title="Scaling Controls" icon="⚙️" accent={C.amber}>
              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 9, color: C.dim, marginBottom: 6 }}>Scaling Mode</div>
                <div style={{ display: "flex", gap: 6 }}>
                  {["auto", "manual"].map(m => (
                    <button key={m} onClick={() => { setScalingMode(m); addLog(`Scaling mode set to ${m.toUpperCase()}`, "INFO"); }}
                      style={{
                        flex: 1, padding: "6px", borderRadius: 5, border: `1px solid ${scalingMode === m ? C.amber : C.border}`,
                        background: scalingMode === m ? `${C.amber}18` : "transparent", color: scalingMode === m ? C.amber : C.dim,
                        fontFamily: "'Space Mono', monospace", fontSize: 10, cursor: "pointer", textTransform: "uppercase",
                      }}>
                      {m}
                    </button>
                  ))}
                </div>
              </div>
              {scalingMode === "manual" && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 9, color: C.dim, marginBottom: 4 }}>Instance Count: {instanceCount}</div>
                  <input type="range" min={1} max={12} value={instanceCount} onChange={e => {
                    setInstanceCount(+e.target.value);
                    addLog(`Manual scaling: ${e.target.value} instances`, "INFO");
                  }} />
                </div>
              )}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.dim, marginBottom: 4 }}>
                  <span>Alarm Threshold</span><span style={{ color: C.amber }}>{alarmThreshold}%</span>
                </div>
                <input type="range" min={40} max={95} value={alarmThreshold} onChange={e => setAlarmThreshold(+e.target.value)} />
              </div>
            </Panel>

          </div>

          {/* CENTER COLUMN */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

            {/* Main chart */}
            <Panel title="Live Workload + LSTM Prediction" icon="📊" accent={C.accent} style={{ flex: "0 0 auto" }}>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={series} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="gCpu" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.accent} stopOpacity={0.2} />
                      <stop offset="95%" stopColor={C.accent} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gPred" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.purple} stopOpacity={0.15} />
                      <stop offset="95%" stopColor={C.purple} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="t" hide />
                  <YAxis domain={[0, 100]} tick={{ fill: C.dim, fontSize: 9 }} />
                  <Tooltip content={<CUSTOM_TOOLTIP />} />
                  <ReferenceLine y={alarmThreshold} stroke={C.red} strokeDasharray="4 4" label={{ value: `Threshold ${alarmThreshold}%`, fill: C.red, fontSize: 9, position: "right" }} />
                  <Area type="monotone" dataKey="cpu" stroke={C.accent} fill="url(#gCpu)" strokeWidth={2} dot={false} name="CPU" />
                  <Area type="monotone" dataKey="mem" stroke={C.green} fill="none" strokeWidth={1.5} dot={false} name="Memory" strokeDasharray="3 3" />
                  <Area type="monotone" dataKey="pred" stroke={C.purple} fill="url(#gPred)" strokeWidth={2} dot={false} name="PredLoad" />
                </AreaChart>
              </ResponsiveContainer>
              <div style={{ display: "flex", gap: 16, marginTop: 6, justifyContent: "center" }}>
                {[["CPU", C.accent], ["Memory", C.green], ["PredLoad (LSTM)", C.purple]].map(([l, c]) => (
                  <div key={l} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 9, color: C.dim }}>
                    <span style={{ width: 16, height: 2, background: c, display: "inline-block" }} />
                    {l}
                  </div>
                ))}
              </div>
            </Panel>

            {/* Network chart */}
            <Panel title="Network Throughput" icon="🌐" style={{ flex: "0 0 auto" }}>
              <ResponsiveContainer width="100%" height={110}>
                <LineChart data={series} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="t" hide />
                  <YAxis domain={[0, 100]} tick={{ fill: C.dim, fontSize: 9 }} />
                  <Tooltip content={<CUSTOM_TOOLTIP />} />
                  <Line type="monotone" dataKey="net" stroke={C.amber} strokeWidth={2} dot={false} name="Network" />
                </LineChart>
              </ResponsiveContainer>
            </Panel>

            {/* Logs */}
            <Panel title="System Logs" icon="📋" style={{ flex: 1 }}>
              <div ref={logRef} style={{ height: 160, overflowY: "auto", paddingRight: 4 }}>
                {logs.length === 0 && <div style={{ fontSize: 9, color: C.dim, textAlign: "center", paddingTop: 20 }}>Awaiting events...</div>}
                {logs.map(l => <LogLine key={l.id} {...l} />)}
              </div>
            </Panel>

          </div>

          {/* RIGHT COLUMN */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

            {/* CloudWatch Alarms */}
            <Panel title="CloudWatch Alarms" icon="🔔" accent={Object.values(alarms).includes("ALARM") ? C.red : C.border}>
              {Object.entries(alarms).map(([name, state]) => (
                <AlarmBadge key={name} name={name} state={state} />
              ))}
              <div style={{ marginTop: 8, padding: "6px 8px", background: "#0A1020", borderRadius: 5 }}>
                <div style={{ fontSize: 9, color: C.dim, marginBottom: 4 }}>PredictedLoad Custom Metric</div>
                <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 18, color: C.purple, fontWeight: "bold" }}>
                  {fmt(last.pred, 1)}<span style={{ fontSize: 10, color: C.dim }}>%</span>
                </div>
                <div style={{ fontSize: 8, color: C.dim, marginTop: 2 }}>Published to AWS CloudWatch</div>
              </div>
            </Panel>

            {/* Auto Scaling Group */}
            <Panel title="Auto Scaling Group" icon="⚡" accent={C.green}>
              <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
                <div style={{ flex: 1, background: "#0A1020", borderRadius: 5, padding: 8, textAlign: "center" }}>
                  <div style={{ fontSize: 9, color: C.dim, marginBottom: 2 }}>Running</div>
                  <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 22, color: C.green, fontWeight: "bold" }}>{instanceCount}</div>
                </div>
                <div style={{ flex: 1, background: "#0A1020", borderRadius: 5, padding: 8, textAlign: "center" }}>
                  <div style={{ fontSize: 9, color: C.dim, marginBottom: 2 }}>Max</div>
                  <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 22, color: C.accent, fontWeight: "bold" }}>12</div>
                </div>
                <div style={{ flex: 1, background: "#0A1020", borderRadius: 5, padding: 8, textAlign: "center" }}>
                  <div style={{ fontSize: 9, color: C.dim, marginBottom: 2 }}>Min</div>
                  <div style={{ fontFamily: "'Space Mono', monospace", fontSize: 22, color: C.amber, fontWeight: "bold" }}>2</div>
                </div>
              </div>
              {/* Instance Visualization */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5, padding: 8, background: "#0A1020", borderRadius: 5, marginBottom: 8 }}>
                {Array.from({ length: 12 }, (_, i) => (
                  <div key={i} style={{
                    width: 32, height: 28, borderRadius: 4,
                    background: i < instanceCount ? `${C.green}20` : `${C.border}40`,
                    border: `1px solid ${i < instanceCount ? C.green : C.border}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 10, transition: "all 0.4s ease",
                    boxShadow: i < instanceCount ? `0 0 6px ${C.green}30` : "none",
                  }}>
                    {i < instanceCount ? "🟢" : "⬜"}
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 9, color: C.dim }}>Recent Scaling Actions</div>
              <div style={{ marginTop: 6, maxHeight: 160, overflowY: "auto" }}>
                {scalingActions.length === 0
                  ? <div style={{ fontSize: 9, color: C.muted, textAlign: "center", padding: 10 }}>No actions yet</div>
                  : scalingActions.map(a => <ScalingAction key={a.id} {...a} />)
                }
              </div>
            </Panel>

            {/* Cost Summary */}
            <Panel title="Cost Analytics" icon="💰" accent={C.amber}>
              {[
                ["Hourly Rate", `$${cost.hourly}`, C.amber],
                ["Monthly Projected", `$${cost.monthly}`, C.amber],
                ["Est. Monthly Savings", `$${cost.saved}`, C.green],
                ["On-Demand Baseline", `$${fmt(12 * 0.048 * 720, 2)}`, C.dim],
              ].map(([l, v, c]) => (
                <div key={l} style={{ display: "flex", justifyContent: "space-between", marginBottom: 7, padding: "5px 8px", background: "#0A1020", borderRadius: 4 }}>
                  <span style={{ fontSize: 9, color: C.dim }}>{l}</span>
                  <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 10, color: c, fontWeight: "bold" }}>{v}</span>
                </div>
              ))}
              <div style={{ marginTop: 4 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.dim, marginBottom: 4 }}>
                  <span>Utilization</span>
                  <span style={{ color: C.amber }}>{fmt(instanceCount / 12 * 100, 0)}%</span>
                </div>
                <div style={{ height: 4, background: C.border, borderRadius: 2 }}>
                  <div style={{
                    height: "100%", borderRadius: 2,
                    width: `${instanceCount / 12 * 100}%`,
                    background: `linear-gradient(90deg, ${C.amber}, ${C.green})`,
                    transition: "width 0.5s ease",
                  }} />
                </div>
              </div>
            </Panel>

          </div>
        </div>

        {/* Footer */}
        <div style={{ marginTop: 14, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ fontSize: 8, color: C.muted, letterSpacing: "0.15em" }}>
            CLOUDOPT · LSTM-BASED PREDICTIVE AUTO-SCALING · AWS CLOUDWATCH INTEGRATION
          </div>
          <div style={{ fontSize: 8, color: C.muted }}>
            <span style={{ animation: "blink 1s infinite", marginRight: 6 }}>█</span>
            LIVE · {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>
    </>
  );
}
