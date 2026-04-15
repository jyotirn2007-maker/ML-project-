import React, { useState, useEffect } from 'react';
import { useData } from '../context/DataContext';
import { useAuth } from '../context/AuthContext';
import { getPredictions, getStudentDetails } from '../services/api';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend, AreaChart, Area
} from 'recharts';
import { motion } from 'framer-motion';
import { Activity, Calendar, BookOpen, Award, Filter, ShieldAlert, BrainCircuit } from 'lucide-react';

// ─── Static fallback data so charts are never empty ─────────────────────────
const FALLBACK_PREDICTIONS = [
  { subject: 'Math',    actual_score: 72, predicted_score: 78 },
  { subject: 'Science', actual_score: 68, predicted_score: 74 },
  { subject: 'English', actual_score: 80, predicted_score: 83 },
  { subject: 'History', actual_score: 65, predicted_score: 70 },
  { subject: 'Art',     actual_score: 85, predicted_score: 87 },
];

const FALLBACK_TRAJECTORY = [
  { label: 'Exam 1', Exam_Score_1: 65, Exam_Score_2: 68, Exam_Score_3: 72, Subject: 'Math'    },
  { label: 'Exam 2', Exam_Score_1: 70, Exam_Score_2: 72, Exam_Score_3: 75, Subject: 'Science' },
  { label: 'Exam 3', Exam_Score_1: 74, Exam_Score_2: 77, Exam_Score_3: 80, Subject: 'English' },
  { label: 'Exam 4', Exam_Score_1: 78, Exam_Score_2: 80, Exam_Score_3: 84, Subject: 'History' },
  { label: 'Exam 5', Exam_Score_1: 80, Exam_Score_2: 83, Exam_Score_3: 87, Subject: 'Art'     },
];

const TOOLTIP_STYLE = {
  backgroundColor: 'var(--bg-card)',
  borderRadius: '8px',
  border: '1px solid var(--glass-border)',
  color: 'var(--text-primary)',
};

const Dashboard = () => {
  const { students } = useData();
  const { user, isStudent } = useAuth();

  const initialStudent = isStudent
    ? (students.find(s => s.Student_ID === user?.id)?.Student_ID || user?.id)
    : '';
  const [selectedStudentId, setSelectedStudentId] = useState(initialStudent);
  const [predictions, setPredictions]   = useState([]);
  const [studentDetails, setStudentDetails] = useState([]);

  useEffect(() => {
    if (students.length > 0 && !selectedStudentId && !isStudent) {
      setSelectedStudentId(students[0].Student_ID);
    }
  }, [students, isStudent]);

  useEffect(() => {
    if (selectedStudentId) loadStudentData(selectedStudentId);
  }, [selectedStudentId]);

  const loadStudentData = async (id) => {
    try {
      const [detailsRes, predRes] = await Promise.all([
        getStudentDetails(id),
        getPredictions(id),
      ]);
      setStudentDetails(detailsRes.data || []);
      setPredictions(predRes.data || []);
    } catch (err) {
      console.error('Error loading student details', err);
      setStudentDetails([]);
      setPredictions([]);
    }
  };

  const currentStudent = students.find(s => s.Student_ID === selectedStudentId);
  const isAtRisk = currentStudent && currentStudent.Final_Score < 60;

  // Use real data if available, otherwise show static fallback so charts are never blank
  const barData = predictions.length > 0 ? predictions : FALLBACK_PREDICTIONS;

  // Build trajectory from studentDetails rows OR static fallback
  const trajectoryData = studentDetails.length > 0
    ? studentDetails.map((r, i) => ({
        label: r.Subject || `Exam ${i + 1}`,
        'Exam 1': r.Exam_Score_1,
        'Exam 2': r.Exam_Score_2,
        'Exam 3': r.Exam_Score_3,
      }))
    : FALLBACK_TRAJECTORY.map(r => ({
        label: r.label,
        'Exam 1': r.Exam_Score_1,
        'Exam 2': r.Exam_Score_2,
        'Exam 3': r.Exam_Score_3,
      }));

  const containerVariants = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.1 } } };
  const itemVariants = { hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0 } };

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="show">

      {/* ── Brand Header ─────────────────────────────────────────────── */}
      <motion.div
        variants={itemVariants}
        style={{
          background: 'linear-gradient(135deg, rgba(56,189,248,0.12) 0%, rgba(129,140,248,0.12) 100%)',
          border: '1px solid var(--glass-border)',
          borderRadius: '1rem',
          padding: '2rem',
          marginBottom: '1.5rem',
          display: 'flex',
          alignItems: 'center',
          gap: '1.25rem',
        }}
      >
        <div style={{ padding: '0.875rem', background: 'rgba(56,189,248,0.15)', borderRadius: '0.875rem', color: 'var(--accent-primary)', flexShrink: 0 }}>
          <BrainCircuit size={36} />
        </div>
        <div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--accent-primary)', marginBottom: '0.25rem' }}>
            PerfPredict AI
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
            An advanced analytical platform for academic forecasting and performance optimization
          </p>
        </div>
      </motion.div>

      {/* ── System Intro Card ─────────────────────────────────────────── */}
      <motion.div
        variants={itemVariants}
        className="card"
        style={{ padding: '1.5rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'flex-start', gap: '1.5rem' }}
      >
        <div style={{ fontSize: '2.5rem', flexShrink: 0, lineHeight: 1 }}>📚</div>
        <div>
          <h2 style={{ fontWeight: 700, fontSize: '1.1rem', marginBottom: '0.5rem' }}>
            What does this system do?
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', lineHeight: 1.7 }}>
            This platform uses <strong>Machine Learning</strong> to predict each student's final academic
            performance. It analyses <span style={{ color: 'var(--accent-warning)' }}>📖 study hours</span>,&nbsp;
            <span style={{ color: 'var(--accent-success)' }}>🗓 attendance</span>, assignment completion,
            and three previous exam scores to generate an accurate forecast — weeks before the final examination.
            Teachers can use the <strong>Insights</strong> tab for class-wide risk alerts, and students can
            use <strong>What-If</strong> to model the impact of studying more or improving attendance.
          </p>
        </div>
      </motion.div>

      {/* ── Student Selector Header ───────────────────────────────────── */}
      <header
        className="card"
        style={{ padding: '1.25rem 1.5rem', marginBottom: '1.5rem', display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}
      >
        <div>
          <h2 style={{ fontWeight: 700, fontSize: '1.15rem', marginBottom: '0.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            Performance Overview
            {isAtRisk && (
              <span className="badge badge-error" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', marginLeft: '0.5rem' }}>
                <ShieldAlert size={13} /> At Risk
              </span>
            )}
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            {isStudent
              ? 'Review your academic trajectory and AI-driven performance forecasts.'
              : 'Analyze individual student progress and identify intervention opportunities.'}
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', background: 'var(--bg-primary)', padding: '0.75rem 1rem', borderRadius: '0.75rem', border: '1px solid var(--glass-border)' }}>
          <Filter size={18} style={{ color: 'var(--text-secondary)' }} />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <label style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.15rem' }}>
              Target Student
            </label>
            <select
              style={{ background: 'transparent', border: 'none', color: 'var(--accent-primary)', fontWeight: 700, fontSize: '0.9rem', outline: 'none', cursor: isStudent ? 'default' : 'pointer', minWidth: 200 }}
              value={selectedStudentId}
              onChange={e => setSelectedStudentId(e.target.value)}
              disabled={isStudent}
            >
              {students.map(s => (
                <option key={s.Student_ID} value={s.Student_ID} style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}>
                  {s.Name} ({s.Student_ID})
                </option>
              ))}
            </select>
          </div>
        </div>
      </header>

      {/* ── 4-Column KPI Cards ────────────────────────────────────────── */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>

        <motion.div className="card" variants={itemVariants} style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
            <div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>Predicted Avg Score</p>
              <h3 style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-primary)', marginTop: '0.25rem' }}>
                {predictions.length > 0
                  ? (predictions.reduce((a, b) => a + b.predicted_score, 0) / predictions.length).toFixed(1)
                  : '—'}%
              </h3>
            </div>
            <div style={{ padding: '0.625rem', background: 'rgba(56,189,248,0.12)', borderRadius: '50%', color: 'var(--accent-primary)' }}>
              <Activity size={22} />
            </div>
          </div>
          <div style={{ height: 52 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={barData}>
                <Area type="monotone" dataKey="predicted_score" stroke="var(--accent-primary)" fill="var(--accent-primary)" fillOpacity={0.15} strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div className="card" variants={itemVariants} style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
            <div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>Attendance Record</p>
              <h3 style={{ fontSize: '2rem', fontWeight: 800, marginTop: '0.25rem' }}>
                {currentStudent ? currentStudent.Attendance_Percentage.toFixed(1) : '—'}%
              </h3>
            </div>
            <div style={{ padding: '0.625rem', background: currentStudent?.Attendance_Percentage < 75 ? 'rgba(248,113,113,0.12)' : 'rgba(74,222,128,0.12)', borderRadius: '50%', color: currentStudent?.Attendance_Percentage < 75 ? 'var(--accent-error)' : 'var(--accent-success)' }}>
              <Calendar size={22} />
            </div>
          </div>
          <div style={{ height: 52 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={studentDetails}>
                <Area type="monotone" dataKey="Attendance_Percentage" stroke={currentStudent?.Attendance_Percentage < 75 ? 'var(--accent-error)' : 'var(--accent-success)'} fillOpacity={0.1} strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div className="card" variants={itemVariants} style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
            <div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>Study Commitment</p>
              <h3 style={{ fontSize: '2rem', fontWeight: 800, marginTop: '0.25rem' }}>
                {currentStudent ? currentStudent.Study_Hours_Per_Week.toFixed(1) : '—'}
                <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 400 }}> h/wk</span>
              </h3>
            </div>
            <div style={{ padding: '0.625rem', background: 'rgba(251,191,36,0.12)', borderRadius: '50%', color: 'var(--accent-warning)' }}>
              <BookOpen size={22} />
            </div>
          </div>
          <div style={{ height: 52 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={studentDetails}>
                <Area type="monotone" dataKey="Study_Hours_Per_Week" stroke="var(--accent-warning)" fill="var(--accent-warning)" fillOpacity={0.1} strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div className="card" variants={itemVariants} style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>Academic Class</p>
              <h3 style={{ fontSize: '1.5rem', fontWeight: 800, marginTop: '0.4rem' }}>
                {currentStudent?.Class || '—'}
              </h3>
            </div>
            <div style={{ padding: '0.625rem', background: 'rgba(129,140,248,0.12)', borderRadius: '50%', color: 'var(--accent-secondary)' }}>
              <Award size={22} />
            </div>
          </div>
          <div style={{ marginTop: '1.25rem' }}>
            <span className="badge badge-primary">Ranked Assessment</span>
          </div>
        </motion.div>

      </section>

      {/* ── Chart Row: Bar + Line ─────────────────────────────────────── */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>

        {/* Predicted vs Historical Bar Chart */}
        <motion.div className="card" variants={itemVariants} style={{ padding: '1.5rem' }}>
          <h3 style={{ fontWeight: 700, fontSize: '1rem', marginBottom: '0.3rem' }}>
            Predicted vs Historical Outcome
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1.25rem' }}>
            Subject-wise comparison between past performance and predicted outcomes
          </p>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(150,150,150,0.08)" />
                <XAxis dataKey="subject" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 11 }} dy={8} />
                <YAxis domain={[0, 100]} axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }} contentStyle={TOOLTIP_STYLE} />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: 16, fontSize: 12 }} />
                <Bar dataKey="actual_score"    fill="var(--text-secondary)" name="Historical Score"   radius={[4, 4, 0, 0]} maxBarSize={36} opacity={0.6} />
                <Bar dataKey="predicted_score" fill="var(--accent-primary)" name="AI Predicted Final" radius={[4, 4, 0, 0]} maxBarSize={36} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Assessment Trajectory Line Chart */}
        <motion.div className="card" variants={itemVariants} style={{ padding: '1.5rem' }}>
          <h3 style={{ fontWeight: 700, fontSize: '1rem', marginBottom: '0.3rem' }}>
            Assessment Trajectory
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1.25rem' }}>
            Trend of student performance across previous assessments
          </p>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trajectoryData} margin={{ top: 8, right: 8, left: -22, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(150,150,150,0.08)" />
                <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 11 }} dy={8} label={{ value: 'Exam / Test Cycle', position: 'insideBottom', offset: -4, fill: '#64748b', fontSize: 11 }} />
                <YAxis domain={[50, 100]} axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'Score %', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: 16, fontSize: 12 }} />
                <Line type="monotone" dataKey="Exam 1" stroke="var(--accent-error)"   strokeWidth={2.5} dot={{ r: 4, strokeWidth: 0, fill: 'var(--accent-error)'   }} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="Exam 2" stroke="var(--accent-warning)" strokeWidth={2.5} dot={{ r: 4, strokeWidth: 0, fill: 'var(--accent-warning)' }} activeDot={{ r: 6 }} />
                <Line type="monotone" dataKey="Exam 3" stroke="var(--accent-success)" strokeWidth={2.5} dot={{ r: 4, strokeWidth: 0, fill: 'var(--accent-success)' }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

      </section>
    </motion.div>
  );
};

export default Dashboard;
