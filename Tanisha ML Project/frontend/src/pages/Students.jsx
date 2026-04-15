import React, { useState } from 'react';
import { useData } from '../context/DataContext';
import { motion } from 'framer-motion';
import { Search, AlertCircle, ShieldCheck, Users, GraduationCap, School, Book, Award } from 'lucide-react';

const GRADES = ['Grade 9', 'Grade 10', 'Grade 11', 'Grade 12'];

const GradeSection = ({ grade, students, searchTerm, id }) => {
  const filtered = students.filter(s => {
    const q = searchTerm.toLowerCase();
    return s.Name.toLowerCase().includes(q) || s.Student_ID.toLowerCase().includes(q);
  });

  if (filtered.length === 0) return null;

  return (
    <motion.div
      id={id}
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="scroll-mt-24 mb-12"
    >
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-primary/10 rounded-lg text-primary flex items-center justify-center">
          <Users size={18} />
        </div>
        <h2 className="font-extrabold text-xl">{grade} Students</h2>
        <span className="badge badge-primary text-xs">
          {filtered.length} student{filtered.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="data-table text-sm">
            <thead className="bg-bg-secondary">
              <tr>
                <th className="text-xs uppercase font-extrabold tracking-widest text-secondary">#</th>
                <th className="text-xs uppercase font-extrabold tracking-widest text-secondary">ID</th>
                <th className="text-xs uppercase font-extrabold tracking-widest text-secondary">Name</th>
                <th className="text-xs uppercase font-extrabold tracking-widest text-secondary">Subject</th>
                <th className="text-xs uppercase font-extrabold tracking-widest text-secondary">Attendance</th>
                <th className="text-xs uppercase font-extrabold tracking-widest text-secondary">Avg Score</th>
                <th className="text-xs uppercase font-extrabold tracking-widest text-secondary">Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s, index) => (
                <tr key={`${s.Student_ID}-${index}`} className="transition-all hover:bg-primary/5">
                  <td className="font-bold text-secondary">#{index + 1}</td>
                  <td className="font-mono text-xs">{s.Student_ID}</td>
                  <td className="font-bold">{s.Name}</td>
                  <td className="text-secondary">{s.Subject || '—'}</td>
                  <td className={`font-bold ${s.Attendance_Percentage < 75 ? 'text-error' : 'text-success'}`}>
                    {s.Attendance_Percentage?.toFixed(1)}%
                  </td>
                  <td className="font-extrabold text-primary">{s.Final_Score?.toFixed(1)}%</td>
                  <td>
                    <span className={`badge ${s.Final_Score < 60 ? 'badge-error' : s.Final_Score < 80 ? 'badge-warning' : 'badge-success'} flex items-center gap-1`}>
                      {s.Final_Score < 60 ? (
                        <><AlertCircle size={12} /> At Risk</>
                      ) : s.Final_Score < 80 ? (
                        'Average'
                      ) : (
                        <><ShieldCheck size={12} /> Excellent</>
                      )}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </motion.div>
  );
};

const Students = () => {
  const { students } = useData();
  const [searchTerm, setSearchTerm] = useState('');
  const [activeGrade, setActiveGrade] = useState('Grade 9');

  const byGrade = GRADES.reduce((acc, g) => {
    acc[g] = students.filter(s => s.Class === g);
    return acc;
  }, {});

  const scrollToGrade = (grade) => {
    setActiveGrade(grade);
    const id = grade.toLowerCase().replace(/\s+/g, ''); // "Grade 9" -> "grade9"
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }
  };

  const getIcon = (grade) => {
    switch(grade) {
      case 'Grade 9': return <GraduationCap size={18} />;
      case 'Grade 10': return <School size={18} />;
      case 'Grade 11': return <Book size={18} />;
      case 'Grade 12': return <Award size={18} />;
      default: return <Users size={18} />;
    }
  };

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="page-container">
      <header className="mb-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-extrabold mb-2">Student Directory</h1>
          <p className="text-secondary text-sm">Comprehensive list of all students across various academic grades.</p>
        </div>
        <div className="relative group w-full md:w-80">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-secondary group-focus-within:text-primary transition-colors" size={20} />
          <input
            type="text"
            placeholder="Search name or ID..."
            className="w-full pl-10 pr-4 py-3 bg-card border border-glass rounded-xl outline-none focus:ring-2 focus:ring-primary/20 transition-all text-sm"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </header>

      {/* Summary Row */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
        {[
          { label: 'Total', value: students.length, color: 'var(--accent-primary)' },
          { label: 'At Risk', value: students.filter(s => s.Final_Score < 60).length, color: 'var(--accent-error)' },
          { label: 'Excellence', value: students.filter(s => s.Final_Score >= 80).length, color: 'var(--accent-success)' },
          { label: 'Avg Attendance', value: (students.reduce((a, b) => a + b.Attendance_Percentage, 0) / (students.length || 1)).toFixed(1) + '%', color: 'var(--accent-secondary)' },
        ].map((stat, i) => (
          <div key={i} className="card p-4 text-center hover:scale-105">
            <h3 className="text-2xl font-extrabold mb-1" style={{ color: stat.color }}>{stat.value}</h3>
            <p className="text-xs font-bold uppercase tracking-widest text-secondary">{stat.label}</p>
          </div>
        ))}
      </section>

      {/* Clickable Section Navigation */}
      <section className="mb-10">
        <h3 className="text-sm font-bold uppercase tracking-widest text-secondary mb-4">Browse Students by Grade</h3>
        <div className="flex flex-wrap gap-3">
          {GRADES.map((grade) => (
            <button
              key={grade}
              onClick={() => scrollToGrade(grade)}
              className={`flex items-center gap-2 px-6 py-3 rounded-full font-bold transition-all shadow-sm ${
                activeGrade === grade 
                ? 'bg-primary text-white shadow-primary/30' 
                : 'bg-card border border-glass text-secondary hover:border-primary/50'
              }`}
            >
              {getIcon(grade)}
              {grade}
            </button>
          ))}
        </div>
      </section>

      {/* Grade Sections */}
      <div className="space-y-12">
        {GRADES.map((grade) => (
          <GradeSection
            key={grade}
            id={grade.toLowerCase().replace(/\s+/g, '')}
            grade={grade}
            students={byGrade[grade] || []}
            searchTerm={searchTerm}
          />
        ))}
      </div>

      {students.length === 0 && (
        <div className="py-20 text-center flex flex-col items-center gap-4">
          <Users size={48} className="text-secondary opacity-20" />
          <h2 className="text-xl font-bold text-secondary">No Student Data Found</h2>
          <p className="text-sm text-secondary">Check your backend connection or data initialization.</p>
        </div>
      )}
    </motion.div>
  );
};

export default Students;
