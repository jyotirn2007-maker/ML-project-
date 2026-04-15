import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { GraduationCap, Users, Eye, EyeOff, AlertCircle } from 'lucide-react';

const LoginForm = ({ role, onCancel }) => {
  const { login, loginError } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    const ok = login(role, username, password);
    if (ok) navigate('/');
  };

  const isTeacher = role === 'teacher';

  return (
    <motion.form
      onSubmit={handleSubmit}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="login-form-card"
    >
      <div className="login-form-header">
        <div className={`icon-wrapper ${isTeacher ? 'teacher' : 'student'}`}>
          {isTeacher ? <Users size={32} /> : <GraduationCap size={32} />}
        </div>
        <h2>{isTeacher ? 'Teacher Login' : 'Student Login'}</h2>
        <p className="text-secondary text-sm">
          {isTeacher
            ? 'Enter your teacher credentials to access the dashboard.'
            : 'Enter your student credentials to view your performance.'}
        </p>
      </div>

      {loginError && (
        <div className="login-error">
          <AlertCircle size={16} /> {loginError}
        </div>
      )}

      <div className="form-group">
        <label>Username</label>
        <input
          type="text"
          placeholder={isTeacher ? 'e.g. teacher1' : 'e.g. student1'}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
          autoFocus
        />
      </div>

      <div className="form-group">
        <label>Password</label>
        <div className="password-field">
          <input
            type={showPw ? 'text' : 'password'}
            placeholder="Enter password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button type="button" className="pw-toggle" onClick={() => setShowPw(!showPw)}>
            {showPw ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        </div>
      </div>

      <div className="demo-hint">
        <b>Demo:</b> {isTeacher ? 'teacher1' : 'student1'} / 1234
      </div>

      <div className="form-actions">
        <button type="button" className="btn btn-outline" onClick={onCancel}>
          Back
        </button>
        <button type="submit" className={`btn ${isTeacher ? 'btn-secondary' : 'btn-primary'}`}>
          Sign In
        </button>
      </div>
    </motion.form>
  );
};

const Login = () => {
  const [selectedRole, setSelectedRole] = useState(null);

  return (
    <div className="login-container">
      <div
        className="login-bg-overlay"
        style={{ backgroundImage: `url('/login_bg.png')` }}
      />

      <div className="login-content">
        <motion.div
          className="login-brand"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1>PerfPredict AI</h1>
          <p>Advanced Academic Forecasting System</p>
        </motion.div>

        <AnimatePresence mode="wait">
          {selectedRole ? (
            <LoginForm
              key={selectedRole}
              role={selectedRole}
              onCancel={() => setSelectedRole(null)}
            />
          ) : (
            <motion.div
              key="role-select"
              className="login-cards"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div
                className="login-card student-card"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                onClick={() => setSelectedRole('student')}
              >
                <div className="icon-wrapper"><GraduationCap size={40} /></div>
                <h2>Student Login</h2>
                <p>View your personal academic performance, predictions, and growth trajectories.</p>
                <button className="btn btn-outline">Enter Portal</button>
              </motion.div>

              <motion.div
                className="login-card teacher-card"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
                onClick={() => setSelectedRole('teacher')}
              >
                <div className="icon-wrapper teacher"><Users size={40} /></div>
                <h2>Teacher Login</h2>
                <p>Access global class analytics, identify at-risk students, and deploy interventions.</p>
                <button className="btn btn-primary">Enter Dashboard</button>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Login;
