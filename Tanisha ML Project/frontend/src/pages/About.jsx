import React from 'react';
import { motion } from 'framer-motion';
import { HelpCircle, BrainCircuit, ShieldAlert, CheckCircle2, TrendingUp, BarChart3, Cpu } from 'lucide-react';

const About = () => {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>

      {/* What is this System – top description section */}
      <section style={{ marginBottom: '2rem' }}>
        <div className="card" style={{ padding: '2rem' }}>
          <h1 style={{ fontWeight: 800, fontSize: '1.5rem', marginBottom: '0.75rem' }}>
            What is this System?
          </h1>
          <p style={{ color: 'var(--text-secondary)', lineHeight: 1.75, marginBottom: '0.875rem' }}>
            The <strong>Student Performance Prediction System</strong> is an AI-powered analytics platform
            designed to help educators and students understand academic trajectories. Using machine learning,
            it analyses attendance records, study hours, assignment completion, and exam history to accurately
            forecast a student's final performance — weeks before examinations take place.
          </p>
          <p style={{ color: 'var(--text-secondary)', lineHeight: 1.75 }}>
            By surfacing early warning signals, the system empowers teachers to intervene in time, assign
            targeted remediation, and ultimately increase overall pass rates across the institution.
          </p>
        </div>
      </section>

      {/* Purpose Cards */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '1.25rem', marginBottom: '2rem' }}>
        <div className="card" style={{ padding: '1.5rem' }}>
          <div style={{ width: 44, height: 44, background: 'rgba(56,189,248,0.12)', borderRadius: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-primary)', marginBottom: '1rem' }}>
            <HelpCircle size={22} />
          </div>
          <h3 style={{ fontWeight: 700, fontSize: '1rem', marginBottom: '0.5rem' }}>Objective &amp; Purpose</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.65 }}>
            Consumes historical student data — attendance, grades, and engagement — to generate individual
            performance trajectories that serve as an early-warning system for educators.
          </p>
        </div>

        <div className="card" style={{ padding: '1.5rem' }}>
          <div style={{ width: 44, height: 44, background: 'rgba(129,140,248,0.12)', borderRadius: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-secondary)', marginBottom: '1rem' }}>
            <Cpu size={22} />
          </div>
          <h3 style={{ fontWeight: 700, fontSize: '1rem', marginBottom: '0.5rem' }}>How it Works</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.65 }}>
            Employs a <strong>Gradient Boosting Regressor</strong> fine-tuned on academic metrics to calculate
            a "Predicted Final Score" based on complex, non-linear relationships in the data.
          </p>
        </div>

        <div className="card" style={{ padding: '1.5rem' }}>
          <div style={{ width: 44, height: 44, background: 'rgba(248,113,113,0.12)', borderRadius: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-error)', marginBottom: '1rem' }}>
            <ShieldAlert size={22} />
          </div>
          <h3 style={{ fontWeight: 700, fontSize: '1rem', marginBottom: '0.5rem' }}>Academic Impact</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.65 }}>
            By identifying "At-Risk" students weeks before final exams, educators can intervene with targeted
            support, potentially increasing pass rates and reducing academic anxiety.
          </p>
        </div>
      </section>

      {/* Core Features */}
      <section className="card" style={{ padding: '2rem' }}>
        <h2 style={{ fontWeight: 800, fontSize: '1.15rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.875rem' }}>
          Core Features
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
            <div style={{ padding: '0.5rem', background: 'rgba(74,222,128,0.12)', borderRadius: '0.5rem', color: 'var(--accent-success)', flexShrink: 0, marginTop: 2 }}>
              <CheckCircle2 size={18} />
            </div>
            <div>
              <strong style={{ fontSize: '0.95rem' }}>Prediction Engine</strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.25rem', lineHeight: 1.6 }}>
                Accurate forecasting of final subject grades using Gradient Boosting with an R² score above 0.95,
                trained on historical attendance, study habits, and exam performance.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
            <div style={{ padding: '0.5rem', background: 'rgba(251,191,36,0.12)', borderRadius: '0.5rem', color: 'var(--accent-warning)', flexShrink: 0, marginTop: 2 }}>
              <TrendingUp size={18} />
            </div>
            <div>
              <strong style={{ fontSize: '0.95rem' }}>What-If Analysis</strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.25rem', lineHeight: 1.6 }}>
                Simulate outcome shifts by adjusting study hours and attendance to preview predicted
                improvements before behavioural changes are made.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
            <div style={{ padding: '0.5rem', background: 'rgba(56,189,248,0.12)', borderRadius: '0.5rem', color: 'var(--accent-primary)', flexShrink: 0, marginTop: 2 }}>
              <BarChart3 size={18} />
            </div>
            <div>
              <strong style={{ fontSize: '0.95rem' }}>Analytics Dashboard</strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.25rem', lineHeight: 1.6 }}>
                A holistic, interactive view of both class-wide health indicators and granular individual
                student profiles, with at-risk alerts and top performer rankings.
              </p>
            </div>
          </div>

        </div>
      </section>

    </motion.div>
  );
};

export default About;
