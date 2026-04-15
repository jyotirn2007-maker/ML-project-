import React, { createContext, useContext, useState } from 'react';

const AuthContext = createContext();

// Demo accounts
const DEMO_ACCOUNTS = {
  teacher: [
    { username: 'teacher1', password: '1234', name: 'Teacher One', id: 'TCH_001' },
    { username: 'teacher2', password: '1234', name: 'Teacher Two', id: 'TCH_002' },
  ],
  student: [
    { username: 'student1', password: '1234', name: 'Ananya Joshi',   id: 'STU_001' },
    { username: 'student2', password: '1234', name: 'Rahul Sharma',   id: 'STU_002' },
    { username: 'student3', password: '1234', name: 'Priya Kapoor',   id: 'STU_003' },
  ],
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('perfpredict_user');
    return saved ? JSON.parse(saved) : null;
  });
  const [loginError, setLoginError] = useState('');

  // Password-based login. Returns true on success, false on failure.
  const login = (role, username, password) => {
    setLoginError('');
    const accounts = DEMO_ACCOUNTS[role] || [];
    const match = accounts.find(
      (a) => a.username === username && a.password === password
    );
    if (!match) {
      setLoginError('Invalid username or password.');
      return false;
    }
    const userData = { role, id: match.id, name: match.name, username: match.username };
    setUser(userData);
    localStorage.setItem('perfpredict_user', JSON.stringify(userData));
    return true;
  };

  const logout = () => {
    setUser(null);
    setLoginError('');
    localStorage.removeItem('perfpredict_user');
  };

  return (
    <AuthContext.Provider value={{
      user,
      loginError,
      isAuthenticated: !!user,
      isTeacher: user?.role === 'teacher',
      isStudent: user?.role === 'student',
      login,
      logout,
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within an AuthProvider');
  return context;
};
