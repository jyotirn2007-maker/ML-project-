import React, { useState, useEffect } from 'react';
import { useData } from '../context/DataContext';
import { useAuth } from '../context/AuthContext';
import { whatIfAnalysis, getStudentDetails } from '../services/api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, FlaskConical, BookOpen, Calendar as CalIcon, TrendingUp, TrendingDown } from 'lucide-react';

const TOOLTIP_STYLE = {
  backgroundColor: 'var(--bg-card)',
  borderRadius: '8px',
  border: '1px solid var(--glass-border)',
  color: 'var(--text-primary)',
};

/* ── Slider with coloured track fill ────────────────────────────────── */
const ColorSlider = ({ value, min, max, step, onChange, color, label, unit }) => {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{
      background: 'var(--bg-primary)',
      border: '1px solid var(--glass-border)',
      borderRadius: '0.875rem',
      padding: '1rem 1.25rem',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <span style={{ fontWeight: 700, fontSize: '0.875rem' }}>{label}</span>
        <span style={{ fontWeight: 800, fontSize: '1rem', color }}>{value.toFixed(1)}{unit}</span>
      </div>
      <div style={{ position: 'relative', height: 6, borderRadius: 3, background: 'var(--glass-border)', marginBottom: '0.5rem' }}>
        <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${pct}%`, borderRadius: 3, background: color }} />
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: color, cursor: 'pointer' }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.7rem', marginTop: '0.15rem' }}>
        <span>{min}{unit}</span><span>{max}{unit}</span>
      </div>
    </div>
  );
};

/* ── Progress bar for improvement visual ────────────────────────────── */
const ProgressBar = ({ label, value, max = 100, color }) => (
  <div style={{ marginBottom: '0.75rem' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem', fontSize: '0.8rem', fontWeight: 600 }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ color }}>{value}%</span>
    </div>
    <div style={{ height: 8, borderRadius: 4, background: 'var(--glass-border)', overflow: 'hidden' }}>
      <div style={{ height: '100%', width: `${Math.min(value, max)}%`, borderRadius: 4, background: color, transition: 'width 0.6s ease' }} />
    </div>
  </div>
);

const WhatIf = () => {
  const { students } = useData();
  const { user, isStudent } = useAuth();

  const initialStudent = isStudent
    ? (students.find(s => s.Student_ID === user?.id)?.Student_ID || user?.id)
    : '';
  const [selectedStudentId, setSelectedStudentId] = useState(initialStudent);

  const [subjects, setSubjects]         = useState([]);
  const [simSubject, setSimSubject]     = useState('');
  const [simStudyHours, setSimStudyHours] = useState(15);
  const [simAttendance, setSimAttendance] = useState(85);
  const [simRes, setSimRes]             = useState(null);
  const [loading, setLoading]           = useState(false);

  useEffect(() => {
    if (students.length > 0 && !selectedStudentId && !isStudent) {
      setSelectedStudentId(students[0].Student_ID);
    }
  }, [students, isStudent]);

  useEffect(() => {
    if (!selectedStudentId) return;
    getStudentDetails(selectedStudentId).then(res => {
      setSubjects(res.data.map(r => r.Subject));
      if (res.data.length > 0) {
        setSimSubject(res.data[0].Subject);
        setSimStudyHours(res.data[0].Study_Hours_Per_Week);
        setSimAttendance(res.data[0].Attendance_Percentage);
      }
    });
  }, [selectedStudentId]);

  const handleSimulate = async () => {
    setLoading(true);
    try {
      const res = await whatIfAnalysis({
        student_id: selectedStudentId,
        subject: simSubject,
        new_study_hours: simStudyHours,
        new_attendance: simAttendance,
      });
      setSimRes(res.data);
    } catch (err) {
      console.error('Simulation failed', err);
    } finally {
      setLoading(false);
    }
  };

  const improved = simRes ? simRes.improvement >= 0 : true;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>

      {/* Page Header */}
      <header style={{ marginBottom: '1.75rem' }}>
        <h1 style={{ fontWeight: 800, fontSize: '1.5rem', marginBottom: '0.3rem' }}>
          Predictive What-If Sandbox
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
          Adjust study habits and attendance to model hypothetical improvements in predicted final scores.
        </p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1.5rem', alignItems: 'start' }}>

        {/* ── LEFT: Parameters Card ──────────────────────────────── */}
        <motion.div initial={{ x: -16, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: 0.1 }}>
          <div className="card" style={{ padding: '1.5rem' }}>

            {/* Card heading */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '1.5rem', paddingBottom: '1rem', borderBottom: '1px solid var(--glass-border)' }}>
              <div style={{ padding: '0.5rem', background: 'rgba(56,189,248,0.12)', borderRadius: '0.5rem', color: 'var(--accent-primary)' }}>
                <FlaskConical size={18} />
              </div>
              <h3 style={{ fontWeight: 700, fontSize: '0.975rem' }}>Simulation Parameters</h3>
            </div>

            {/* Student select */}
            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.4rem' }}>
                Target Student
              </label>
              <select
                style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--glass-border)', color: 'var(--text-primary)', padding: '0.5rem 0.75rem', borderRadius: '0.625rem', fontFamily: 'inherit', fontSize: '0.875rem', outline: 'none', opacity: isStudent ? 0.5 : 1, cursor: isStudent ? 'default' : 'pointer' }}
                value={selectedStudentId}
                onChange={e => setSelectedStudentId(e.target.value)}
                disabled={isStudent}
              >
                {students.map(s => (
                  <option key={s.Student_ID} value={s.Student_ID}>{s.Name}</option>
                ))}
              </select>
            </div>

            {/* Subject select */}
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.4rem' }}>
                Subject Scope
              </label>
              <select
                style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--glass-border)', color: 'var(--text-primary)', padding: '0.5rem 0.75rem', borderRadius: '0.625rem', fontFamily: 'inherit', fontSize: '0.875rem', outline: 'none', cursor: 'pointer' }}
                value={simSubject}
                onChange={e => setSimSubject(e.target.value)}
              >
                {subjects.map(sub => <option key={sub} value={sub}>{sub}</option>)}
              </select>
            </div>

            {/* Study Hours slider */}
            <div style={{ marginBottom: '1rem' }}>
              <ColorSlider
                label={<span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}><BookOpen size={14} style={{ color: 'var(--accent-warning)' }} /> Study Hours / Week</span>}
                value={simStudyHours}
                min={0} max={40} step={0.5}
                onChange={setSimStudyHours}
                color="var(--accent-warning)"
                unit=" h"
              />
            </div>

            {/* Attendance slider */}
            <div style={{ marginBottom: '1.5rem' }}>
              <ColorSlider
                label={<span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}><CalIcon size={14} style={{ color: 'var(--accent-success)' }} /> Attendance Rate</span>}
                value={simAttendance}
                min={60} max={100} step={1}
                onChange={setSimAttendance}
                color="var(--accent-success)"
                unit="%"
              />
            </div>

            {/* Run button */}
            <button
              onClick={handleSimulate}
              disabled={loading}
              style={{
                width: '100%', padding: '0.75rem', border: 'none', borderRadius: '0.75rem',
                background: 'var(--accent-primary)', color: 'white', fontFamily: 'inherit',
                fontWeight: 700, fontSize: '0.95rem', cursor: loading ? 'default' : 'pointer',
                opacity: loading ? 0.7 : 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
                transition: 'transform 0.15s, opacity 0.15s',
              }}
              onMouseEnter={e => { if (!loading) e.currentTarget.style.transform = 'translateY(-1px)'; }}
              onMouseLeave={e => { e.currentTarget.style.transform = 'none'; }}
            >
              {loading
                ? <><div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} /> Processing...</>
                : <><Play size={18} fill="currentColor" /> Run Simulation</>}
            </button>
          </div>
        </motion.div>

        {/* ── RIGHT: Results Panel ───────────────────────────────── */}
        <motion.div initial={{ x: 16, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: 0.15 }}>
          <AnimatePresence mode="wait">
            {simRes ? (
              <motion.div key="results" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>

                {/* Score highlight banner */}
                <div style={{
                  background: improved ? 'rgba(74,222,128,0.08)' : 'rgba(248,113,113,0.08)',
                  border: `1px solid ${improved ? 'rgba(74,222,128,0.25)' : 'rgba(248,113,113,0.25)'}`,
                  borderRadius: '1rem',
                  padding: '1.5rem',
                  marginBottom: '1rem',
                  display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem',
                }}>
                  <div>
                    <p style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
                      Simulation Outcomes Processed
                    </p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                      Compared against ML baseline model
                    </p>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <p style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Predicted Score</p>
                    <h2 style={{ fontSize: '3rem', fontWeight: 900, color: improved ? 'var(--accent-success)' : 'var(--accent-error)', lineHeight: 1.1 }}>
                      {simRes.simulated_predicted_score}%
                    </h2>
                    <div className={`badge ${improved ? 'badge-success' : 'badge-error'}`} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.85rem', padding: '0.25rem 0.625rem', marginTop: '0.4rem' }}>
                      {improved ? <TrendingUp size={13} /> : <TrendingDown size={13} />}
                      {improved ? '+' : ''}{simRes.improvement.toFixed(1)}% impact
                    </div>
                  </div>
                </div>

                {/* Detail grid */}
                <div className="card" style={{ padding: '1.5rem', marginBottom: '1rem' }}>
                  <h4 style={{ fontWeight: 700, marginBottom: '1rem', fontSize: '0.9rem' }}>Comparison Breakdown</h4>

                  {/* Progress bars */}
                  <ProgressBar label="Baseline Score"     value={simRes.current_predicted_score}    color="var(--text-secondary)" />
                  <ProgressBar label="Predicted New Score" value={simRes.simulated_predicted_score} color={improved ? 'var(--accent-success)' : 'var(--accent-error)'} />

                  {/* Param chips */}
                  <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '1rem' }}>
                    <div style={{ padding: '0.5rem 0.875rem', background: 'rgba(251,191,36,0.1)', border: '1px solid rgba(251,191,36,0.25)', borderRadius: '0.625rem', fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-warning)' }}>
                      📖 {simRes.new_study_hours} h/wk study
                    </div>
                    <div style={{ padding: '0.5rem 0.875rem', background: 'rgba(74,222,128,0.1)', border: '1px solid rgba(74,222,128,0.25)', borderRadius: '0.625rem', fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-success)' }}>
                      🗓 {simRes.new_attendance}% attendance
                    </div>
                    <div style={{ padding: '0.5rem 0.875rem', background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', borderRadius: '0.625rem', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                      📚 {simRes.subject || simSubject}
                    </div>
                  </div>
                </div>

                {/* Bar chart comparison */}
                <div className="card" style={{ padding: '1.5rem' }}>
                  <h4 style={{ fontWeight: 700, marginBottom: '1rem', fontSize: '0.9rem' }}>Visual Score Comparison</h4>
                  <div style={{ height: 180 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={[
                          { name: 'Baseline', score: simRes.current_predicted_score },
                          { name: 'Predicted', score: simRes.simulated_predicted_score },
                        ]}
                        margin={{ top: 8, right: 8, left: -20, bottom: 0 }}
                      >
                        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                        <YAxis hide domain={[0, 100]} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                        <Bar dataKey="score" radius={[6, 6, 0, 0]} maxBarSize={64}>
                          <Cell fill="rgba(148,163,184,0.35)" />
                          <Cell fill={improved ? 'var(--accent-success)' : 'var(--accent-error)'} />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

              </motion.div>
            ) : (
              <motion.div
                key="idle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="card"
                style={{ padding: '3.5rem 2rem', textAlign: 'center', color: 'var(--text-secondary)' }}
              >
                <div style={{ fontSize: '3.5rem', marginBottom: '1rem', opacity: 0.6 }}>🔬</div>
                <h3 style={{ fontWeight: 700, fontSize: '1.15rem', marginBottom: '0.5rem', color: 'var(--text-primary)' }}>
                  Run simulation to view predicted impact
                </h3>
                <p style={{ maxWidth: 320, margin: '0 auto', fontSize: '0.875rem', lineHeight: 1.6 }}>
                  Adjust the study hours and attendance sliders on the left, then press
                  <strong> Run Simulation</strong> to see the AI's predicted outcome.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

      </div>
    </motion.div>
  );
};

export default WhatIf;
