import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { DataProvider } from './context/DataContext';
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import Layout from './components/Layout';

// Page Imports
import Dashboard from './pages/Dashboard';
import Students from './pages/Students';
import WhatIf from './pages/WhatIf';
import Insights from './pages/Insights';
import About from './pages/About';
import Settings from './pages/Settings';
import Login from './pages/Login';

// Protected Route Wrapper
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  return children;
};

// Login Route Wrapper (Redirects to dashboard if already logged in)
const PublicRoute = ({ children }) => {
  const { isAuthenticated } = useAuth();
  if (isAuthenticated) return <Navigate to="/" replace />;
  return children;
};

const App = () => {
  return (
    <ThemeProvider>
      <DataProvider>
        <AuthProvider>
          <Router>
            <Routes>
              {/* Public unauthenticated routes */}
              <Route path="/login" element={
                <PublicRoute>
                  <Login />
                </PublicRoute>
              } />

              {/* Protected authenticated routes */}
              <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/students" element={<Students />} />
                <Route path="/what-if" element={<WhatIf />} />
                <Route path="/insights" element={<Insights />} />
                <Route path="/about" element={<About />} />
                <Route path="/settings" element={<Settings />} />

                {/* Fallback route */}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </Router>
        </AuthProvider>
      </DataProvider>
    </ThemeProvider>
  );
};

export default App;
