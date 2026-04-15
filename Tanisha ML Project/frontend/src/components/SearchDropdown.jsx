import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { searchStudents } from '../services/api';
import { User, Search as SearchIcon, Loader2 } from 'lucide-react';

const SearchDropdown = ({ query, onClose }) => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const dropdownRef = useRef(null);

  useEffect(() => {
    const fetchResults = async () => {
      if (!query || query.length < 2) {
        setResults([]);
        return;
      }

      setLoading(true);
      try {
        const response = await searchStudents(query);
        setResults(response.data);
      } catch (error) {
        console.error('Search failed:', error);
      } finally {
        setLoading(false);
      }
    };

    const timer = setTimeout(fetchResults, 300);
    return () => clearTimeout(timer);
  }, [query]);

  const handleSelect = (studentId) => {
    navigate(`/student/${studentId}`);
    onClose();
  };

  if (!query || query.length < 2) return null;

  return (
    <div className="search-dropdown" ref={dropdownRef}>
      {loading ? (
        <div className="search-status">
          <Loader2 size={16} className="animate-spin text-secondary" />
          <span>Searching students...</span>
        </div>
      ) : results.length > 0 ? (
        <div className="search-results-list">
          {results.map((student) => (
            <div 
              key={student.Student_ID} 
              className="search-result-item"
              onClick={() => handleSelect(student.Student_ID)}
            >
              <div className="result-icon">
                <User size={16} />
              </div>
              <div className="result-info">
                <div className="result-name">{student.Name}</div>
                <div className="result-meta">ID: {student.Student_ID} • {student.Class}</div>
              </div>
              <div className="result-score">
                {Math.round(student.Final_Score)}%
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="search-status">No students found matching "{query}"</div>
      )}
    </div>
  );
};

export default SearchDropdown;
