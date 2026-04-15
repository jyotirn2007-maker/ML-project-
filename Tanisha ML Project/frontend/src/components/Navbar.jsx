import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { useAuth } from '../context/AuthContext';
import {
  BarChart3, Users, TrendingUp, Activity, Info, Sun, Moon, Bell, LogOut
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Navbar = () => {
  const { theme, toggleTheme } = useTheme();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [showNotifications, setShowNotifications] = React.useState(false);

  const navItems = [
    { name: 'Dashboard', path: '/', icon: BarChart3 },
    { name: 'Students',  path: '/students',  icon: Users },
    { name: 'What-If',   path: '/what-if',   icon: TrendingUp },
    { name: 'Insights',  path: '/insights',  icon: Activity },
    { name: 'About',     path: '/about',     icon: Info },
  ];

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <header className="navbar w-full border-b border-glass bg-bg-secondary sticky top-0 z-50 mt-3 rounded-2xl mx-auto max-w-[98%] shadow-sm">
      <div className="w-full px-6 md:px-10 h-16 flex justify-between items-center">

        {/* Brand */}
        <motion.div
          className="brand flex items-center font-bold text-xl text-primary gap-2"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <div className="bg-primary/20 p-2 rounded-lg text-primary">
            <BarChart3 size={20} />
          </div>
          <span className="hidden md:inline font-medium tracking-wide" style={{ fontSize: '1.2rem' }}>PerfPredict AI</span>
        </motion.div>

        {/* Center nav links */}
        <nav className="hidden lg:flex items-center space-x-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) =>
                `flex items-center space-x-2 px-5 py-2 rounded-xl font-medium tracking-wide transition-all relative group ${
                  isActive
                    ? 'bg-primary/10 text-primary'
                    : 'text-secondary hover:text-text-primary'
                }`
              }
              style={{ fontSize: '1rem' }}
            >
              <item.icon size={20} />
              <span>{item.name}</span>
              <span className="absolute bottom-1.5 left-5 right-5 h-0.5 bg-primary transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left opacity-70"></span>
            </NavLink>
          ))}
        </nav>

        {/* Right Controls */}
        <div className="flex items-center space-x-2 md:space-x-3">

          <button
            className="p-2 text-secondary hover:bg-glass hover:text-primary rounded-full transition-all relative"
            onClick={() => setShowNotifications(!showNotifications)}
            title="Notifications"
          >
            <Bell size={21} />
            <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-error rounded-full border-2 border-bg-secondary" />
          </button>

          <button
            className="p-2 text-secondary hover:bg-glass hover:text-primary rounded-full transition-all"
            onClick={toggleTheme}
            title="Toggle theme"
          >
            {theme === 'dark' ? <Moon size={21} /> : <Sun size={21} />}
          </button>

          {/* User profile → settings */}
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `flex items-center space-x-3 pl-2 md:pl-4 border-l border-glass transition-all ${
                isActive ? 'opacity-70' : 'hover:opacity-80'
              }`
            }
          >
            <div className="hidden md:flex flex-col items-end">
              <span className="font-bold" style={{ fontSize: '0.95rem' }}>{user?.name || 'User'}</span>
              <span className="text-secondary uppercase font-semibold" style={{ fontSize: '0.75rem' }}>
                {user?.role || 'Guest'}
              </span>
            </div>
            <div className="w-10 h-10 rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold text-lg border border-primary/30">
              {user?.name?.charAt(0) || 'U'}
            </div>
          </NavLink>

          {/* Logout button */}
          <button
            className="p-2 text-secondary hover:bg-glass hover:text-error rounded-full transition-all hidden md:flex"
            onClick={handleLogout}
            title="Logout"
          >
            <LogOut size={20} />
          </button>

        </div>
      </div>

      {/* Mobile nav strip */}
      <div className="lg:hidden w-full overflow-x-auto border-t border-glass px-4 py-2 flex space-x-2 hide-scrollbar">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) =>
              `flex items-center space-x-2 px-3 py-1.5 whitespace-nowrap rounded-lg font-semibold transition-all ${
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-secondary hover:bg-glass hover:text-text-primary'
              }`
            }
            style={{ fontSize: '0.95rem' }}
          >
            <item.icon size={15} />
            <span>{item.name}</span>
          </NavLink>
        ))}
      </div>
    </header>
  );
};

export default Navbar;
