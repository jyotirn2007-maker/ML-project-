import React, { createContext, useContext, useState, useEffect } from 'react';
import { getStudents, getGlobalAnalytics } from '../services/api';

const DataContext = createContext();

export const DataProvider = ({ children }) => {
  const [students, setStudents] = useState([]);
  const [globalStats, setGlobalStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [studentsRes, globalRes] = await Promise.all([
        getStudents(),
        getGlobalAnalytics()
      ]);
      setStudents(studentsRes.data);
      setGlobalStats(globalRes.data);
      setError(null);
    } catch (err) {
      console.error("Error fetching data", err);
      setError("Failed to load dashboard data. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <DataContext.Provider value={{ students, globalStats, loading, error, refreshData: fetchData }}>
      {children}
    </DataContext.Provider>
  );
};

export const useData = () => {
    const context = useContext(DataContext);
    if (!context) throw new Error("useData must be used within a DataProvider");
    return context;
};
