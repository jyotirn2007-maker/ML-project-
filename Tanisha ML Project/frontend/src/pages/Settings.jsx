import React, { useState } from 'react';
import { useTheme } from '../context/ThemeContext';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Moon, Sun, Monitor, Bell, Shield, LogOut } from 'lucide-react';

const Toggle = ({ checked, onChange }) => (
  <button
    onClick={() => onChange(!checked)}
    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${checked ? 'bg-primary' : 'bg-glass'} border border-glass`}
    style={{ minWidth: 44 }}
    aria-pressed={checked}
  >
    <span
      className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${checked ? 'translate-x-6' : 'translate-x-1'}`}
    />
  </button>
);

const Settings = () => {
  const { theme, toggleTheme } = useTheme();
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const [emailAlerts, setEmailAlerts]   = useState(true);
  const [riskWarnings, setRiskWarnings] = useState(true);

  const handleSignOut = () => {
    logout();
    navigate('/login');
  };

  const handleClearSession = () => {
    localStorage.clear();
    window.location.href = '/login';
  };

  if (!user) return null;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <header className="mb-8">
        <h1 className="text-2xl font-bold mb-2">Platform Settings</h1>
        <p className="text-secondary text-sm">Manage your account preferences and application appearance.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* Profile Card */}
        <div className="card p-0 overflow-hidden col-span-1 h-fit">
          <div className="p-6 flex flex-col items-center bg-glass border-b border-glass">
            <div className="w-24 h-24 rounded-full bg-primary/20 text-primary flex items-center justify-center text-3xl font-bold mb-4 border-2 border-primary/30">
              {user.name.charAt(0)}
            </div>
            <h2 className="font-bold text-xl">{user.name}</h2>
            <span className="badge badge-primary mt-2 uppercase">{user.role}</span>
            <p className="text-secondary text-sm mt-2">{user.id}</p>
            <p className="text-secondary text-xs mt-1">@{user.username}</p>
          </div>
          <div className="p-4">
            <button
              className="btn w-full flex-align justify-center gap-2 transition-all hover:bg-error/10 hover:text-error"
              onClick={handleSignOut}
            >
              <LogOut size={18} /> Sign Out
            </button>
          </div>
        </div>

        {/* Settings Controls */}
        <div className="col-span-1 md:col-span-2 space-y-6">

          {/* Appearance */}
          <div className="card p-6">
            <h3 className="text-lg font-bold mb-5 flex-align gap-2 border-b border-glass pb-4">
              <Monitor size={20} className="text-primary" /> Appearance
            </h3>
            <div className="flex items-center justify-between p-4 bg-glass rounded-xl">
              <div>
                <p className="font-bold">Dark Mode</p>
                <p className="text-sm text-secondary">Switch between light and dark themes</p>
              </div>
              <button
                onClick={toggleTheme}
                className="p-2.5 rounded-full bg-primary/10 text-primary hover:bg-primary/20 transition-all"
                title={theme === 'dark' ? 'Switch to Light' : 'Switch to Dark'}
              >
                {theme === 'dark' ? <Moon size={22} /> : <Sun size={22} />}
              </button>
            </div>
          </div>

          {/* Notifications */}
          <div className="card p-6">
            <h3 className="text-lg font-bold mb-5 flex-align gap-2 border-b border-glass pb-4">
              <Bell size={20} className="text-secondary" /> Notifications
            </h3>

            <div className="flex items-center justify-between p-4 bg-glass rounded-xl mb-3">
              <div>
                <p className="font-bold">Email Alerts</p>
                <p className="text-sm text-secondary">Receive daily performance digest</p>
              </div>
              <Toggle checked={emailAlerts} onChange={setEmailAlerts} />
            </div>

            <div className="flex items-center justify-between p-4 bg-glass rounded-xl">
              <div>
                <p className="font-bold">Risk Warnings</p>
                <p className="text-sm text-secondary">Notify when a student enters the at-risk zone</p>
              </div>
              <Toggle checked={riskWarnings} onChange={setRiskWarnings} />
            </div>
          </div>

          {/* Privacy */}
          <div className="card p-6" style={{ borderColor: 'rgba(248,113,113,0.2)' }}>
            <h3 className="text-lg font-bold mb-5 flex-align gap-2 border-b border-glass pb-4 text-error">
              <Shield size={20} /> Privacy &amp; Data
            </h3>
            <p className="text-sm text-secondary mb-4">
              Session data is stored in your browser's local storage. Clearing it will log you out.
            </p>
            <button
              className="btn btn-outline transition-all"
              style={{ borderColor: 'var(--accent-error)', color: 'var(--accent-error)' }}
              onClick={handleClearSession}
              onMouseEnter={e => { e.currentTarget.style.background = 'var(--accent-error)'; e.currentTarget.style.color = 'white'; }}
              onMouseLeave={e => { e.currentTarget.style.background = ''; e.currentTarget.style.color = 'var(--accent-error)'; }}
            >
              Clear Session Data
            </button>
          </div>

        </div>
      </div>
    </motion.div>
  );
};

export default Settings;
