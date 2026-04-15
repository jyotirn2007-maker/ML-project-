import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getStudents = (params) => api.get('/students', { params });
export const getStudentDetails = (id) => api.get(`/student/${id}`);
export const getPredictions = (id) => api.get(`/predict/${id}`);
export const whatIfAnalysis = (data) => api.post('/what-if', data);
export const getGlobalAnalytics = () => api.get('/analytics/global');
export const searchStudents = (q) => api.get('/students/search', { params: { q } });
export const getNotifications = () => api.get('/notifications');
export const triggerTraining = () => api.post('/train');

export default api;
