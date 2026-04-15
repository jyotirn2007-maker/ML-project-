import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import { useData } from '../context/DataContext';

const Layout = () => {
  const { loading, error } = useData();

  if (loading) return (
    <div className="loading-screen bg-bg-primary text-text-primary h-screen w-full flex-align justify-center flex-col">
      <div className="spinner"></div>
      <p className="mt-4 text-secondary font-semibold">Syncing Academic Data...</p>
    </div>
  );

  if (error) return (
    <div className="error-screen bg-bg-primary text-text-primary h-screen w-full flex-align justify-center flex-col">
      <h1 className="text-2xl font-bold text-error mb-2">Initialization Failed</h1>
      <p className="text-secondary mb-4">{error}</p>
      <button className="btn btn-primary" onClick={() => window.location.reload()}>Retry Connection</button>
    </div>
  );

  return (
    <div className="app-container flex flex-col h-screen w-full overflow-hidden bg-bg-primary text-text-primary">
      {/* Top Navbar */}
      <Navbar />
      
      {/* Main Content Area */}
      <main className="main-viewport flex-1 overflow-y-auto bg-bg-primary">
        <div className="max-w-7xl mx-auto w-full p-4 md:p-8 animate-fade">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default Layout;
