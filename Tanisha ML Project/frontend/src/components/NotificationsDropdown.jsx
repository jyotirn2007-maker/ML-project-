import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getNotifications } from '../services/api';
import { AlertCircle, Clock, X, ChevronRight } from 'lucide-react';

const NotificationsDropdown = ({ onClose }) => {
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchNotifications = async () => {
      try {
        const response = await getNotifications();
        setNotifications(response.data);
      } catch (error) {
        console.error('Failed to fetch notifications:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchNotifications();
  }, []);

  const handleAction = (studentId) => {
    navigate(`/student/${studentId}`);
    onClose();
  };

  return (
    <div className="notifications-dropdown">
      <div className="notifications-header">
        <h3>Recent Alerts</h3>
        <button className="close-btn" onClick={onClose}><X size={16} /></button>
      </div>
      
      <div className="notifications-list">
        {loading ? (
          <div className="notif-status">Syncing alerts...</div>
        ) : notifications.length > 0 ? (
          notifications.map((notif) => (
            <div 
              key={notif.id} 
              className={`notif-item ${notif.type}`}
              onClick={() => handleAction(notif.student_id)}
            >
              <div className="notif-icon">
                <AlertCircle size={18} />
              </div>
              <div className="notif-content">
                <div className="notif-title">{notif.title}</div>
                <div className="notif-message">{notif.message}</div>
                <div className="notif-time">
                  <Clock size={12} /> {notif.timestamp}
                </div>
              </div>
              <div className="notif-arrow">
                <ChevronRight size={16} />
              </div>
            </div>
          ))
        ) : (
          <div className="notif-empty">
            <div className="empty-icon text-secondary">✨</div>
            <p>All students performing well!</p>
          </div>
        )}
      </div>
      
      {notifications.length > 0 && (
        <div className="notifications-footer">
          <button onClick={() => { navigate('/analytics'); onClose(); }}>
            View Full Analytics
          </button>
        </div>
      )}
    </div>
  );
};

export default NotificationsDropdown;
