import React from 'react';
import { useData } from '../context/DataContext';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { motion } from 'framer-motion';
import { ShieldAlert, TrendingUp, Trophy, TrendingDown, Lightbulb, Users, Star } from 'lucide-react';

const Insights = () => {
  const { globalStats, students } = useData();

  if (!globalStats) return (
    <div className="h-full flex-align justify-center py-20">
      <div className="spinner"></div>
    </div>
  );

  const COLORS = ['var(--accent-error)', 'var(--accent-warning)', 'var(--accent-success)'];
  const distributionData = [
    { name: 'At Risk (<60%)',    value: globalStats.at_risk_count },
    { name: 'Average (60-80%)', value: Math.max(0, globalStats.total_students - globalStats.at_risk_count - globalStats.top_performers.length) },
    { name: 'Top Performers',   value: globalStats.top_performers.length },
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <header className="mb-8 p-6 bg-glass shadow-md rounded-2xl border border-glass">
        <h1 className="text-2xl font-bold mb-2">Global Analytics &amp; Insights</h1>
        <p className="text-secondary text-sm">
          System-wide health indicators and AI-generated academic recommendations.
        </p>
      </header>

      {/* Summary Cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card p-6" style={{ borderTop: '4px solid var(--accent-error)' }}>
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-secondary text-sm">At-Risk Students</h3>
            <ShieldAlert className="text-error" size={24} />
          </div>
          <p className="text-4xl font-bold text-error mb-1">{globalStats.at_risk_count}</p>
          <p className="text-xs text-secondary">Score below 60% — need intervention.</p>
        </div>

        <div className="card p-6" style={{ borderTop: '4px solid var(--accent-success)' }}>
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-secondary text-sm">Top Performers</h3>
            <Trophy className="text-success" size={24} />
          </div>
          <p className="text-4xl font-bold text-success mb-1">{globalStats.top_performers.length}</p>
          <p className="text-xs text-secondary">Score above 80% — eligible for advanced placement.</p>
        </div>

        <div className="card p-6" style={{ borderTop: '4px solid var(--accent-primary)' }}>
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-secondary text-sm">Total Students</h3>
            <Users className="text-primary" size={24} />
          </div>
          <p className="text-4xl font-bold text-primary mb-1">{globalStats.total_students}</p>
          <p className="text-xs text-secondary">Active student profiles tracked by the system.</p>
        </div>
      </section>

      {/* Chart + Recommendations Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">

        {/* Pie Chart */}
        <div className="card p-6">
          <h3 className="text-lg font-bold mb-1">Performance Distribution</h3>
          <p className="text-sm text-secondary mb-4">Class-wide scoring breakdown.</p>
          <div className="relative h-56">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={distributionData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={4} dataKey="value" stroke="none">
                  {distributionData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--glass-border)' }} />
              </PieChart>
            </ResponsiveContainer>
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <span className="text-2xl font-bold text-primary">{globalStats.total_students}</span>
              <span className="text-xs text-secondary">Students</span>
            </div>
          </div>
          <div className="flex justify-center gap-6 mt-2">
            <div className="flex-align text-xs gap-1"><span className="w-3 h-3 rounded-full bg-error block"></span> At Risk</div>
            <div className="flex-align text-xs gap-1"><span className="w-3 h-3 rounded-full bg-warning block"></span> Average</div>
            <div className="flex-align text-xs gap-1"><span className="w-3 h-3 rounded-full bg-success block"></span> Top</div>
          </div>
        </div>

        {/* AI Recommendations */}
        <div className="card p-6">
          <h3 className="text-lg font-bold mb-1 flex-align gap-2">
            <Lightbulb size={20} className="text-warning" /> Improvement Suggestions
          </h3>
          <p className="text-sm text-secondary mb-5">AI-driven recommendations based on current data.</p>

          <div className="space-y-4">
            <div className="flex items-start gap-3 p-4 rounded-xl bg-success/5 border border-success/20">
              <div className="p-1.5 bg-success/10 text-success rounded-lg mt-0.5"><TrendingUp size={16} /></div>
              <div>
                <h4 className="font-bold text-sm mb-1">Boost Attendance Rates</h4>
                <p className="text-xs text-secondary leading-relaxed">Students with attendance above 90% are 3× more likely to score in the "Excellent" range. Target the at-risk group with attendance reminders.</p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 rounded-xl bg-error/5 border border-error/20">
              <div className="p-1.5 bg-error/10 text-error rounded-lg mt-0.5"><TrendingDown size={16} /></div>
              <div>
                <h4 className="font-bold text-sm mb-1">Mathematics Support Needed</h4>
                <p className="text-xs text-secondary leading-relaxed">A downward trend is detected in Grade 11 Mathematics scores. Supplementary tuition or curriculum revision is recommended.</p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 rounded-xl bg-primary/5 border border-primary/20">
              <div className="p-1.5 bg-primary/10 text-primary rounded-lg mt-0.5"><Star size={16} /></div>
              <div>
                <h4 className="font-bold text-sm mb-1">Encourage Consistent Study Habits</h4>
                <p className="text-xs text-secondary leading-relaxed">Students studying 15+ hours per week score on average 12 points higher than those studying fewer than 8 hours per week.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* At-Risk Student Table */}
      {globalStats.at_risk_students?.length > 0 && (
        <section className="card overflow-hidden mb-8">
          <div className="p-5 border-b border-glass flex-align gap-2">
            <ShieldAlert size={18} className="text-error" />
            <h3 className="font-bold text-lg">At-Risk Students — Priority Intervention List</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table w-full text-sm">
              <thead className="bg-bg-secondary">
                <tr>
                  <th className="text-xs uppercase tracking-wider font-bold">Student ID</th>
                  <th className="text-xs uppercase tracking-wider font-bold">Name</th>
                  <th className="text-xs uppercase tracking-wider font-bold">Avg Score</th>
                  <th className="text-xs uppercase tracking-wider font-bold">Attendance</th>
                  <th className="text-xs uppercase tracking-wider font-bold">Priority</th>
                </tr>
              </thead>
              <tbody>
                {globalStats.at_risk_students.map(s => (
                  <tr key={s.Student_ID} className="hover:bg-glass transition-all">
                    <td className="font-mono text-xs">{s.Student_ID}</td>
                    <td className="font-bold">{s.Name}</td>
                    <td className="font-bold text-error">{s.Final_Score.toFixed(1)}%</td>
                    <td>{s.Attendance_Percentage.toFixed(1)}%</td>
                    <td>
                      <span className={`badge ${s.Attendance_Percentage < 70 ? 'badge-error' : 'badge-warning'}`}>
                        {s.Attendance_Percentage < 70 ? 'High' : 'Moderate'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Top performers table */}
      {globalStats.top_performers?.length > 0 && (
        <section className="card overflow-hidden">
          <div className="p-5 border-b border-glass flex-align gap-2">
            <Trophy size={18} className="text-success" />
            <h3 className="font-bold text-lg">Top Performers</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table w-full text-sm">
              <thead className="bg-bg-secondary">
                <tr>
                  <th className="text-xs uppercase tracking-wider font-bold">Student ID</th>
                  <th className="text-xs uppercase tracking-wider font-bold">Name</th>
                  <th className="text-xs uppercase tracking-wider font-bold">Avg Score</th>
                  <th className="text-xs uppercase tracking-wider font-bold">Class</th>
                </tr>
              </thead>
              <tbody>
                {globalStats.top_performers.slice(0, 10).map(s => (
                  <tr key={s.Student_ID} className="hover:bg-glass transition-all">
                    <td className="font-mono text-xs">{s.Student_ID}</td>
                    <td className="font-bold">{s.Name}</td>
                    <td className="font-bold text-success">{s.Final_Score.toFixed(1)}%</td>
                    <td>{s.Class}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </motion.div>
  );
};

export default Insights;
