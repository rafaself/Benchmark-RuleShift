import React from 'react';

export function MetricCard({ title, icon, value, sub, progress, color }) {
  return (
    <div className="bg-zinc-900/40 p-10 rounded-[2.5rem] border border-zinc-800 relative group overflow-hidden">
      <div className="absolute top-8 right-10 text-white/5 group-hover:text-white/10 transition-colors">{icon}</div>
      <div className="text-zinc-500 text-[10px] font-black uppercase tracking-[0.2em] mb-4">{title}</div>
      <div className="text-7xl font-black mb-2">{value}</div>
      <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden"><div className={`h-full ${color}`} style={{ width: `${progress}%` }}></div></div>
      <p className="text-zinc-500 text-xs mt-4 font-bold">{sub}</p>
    </div>
  );
}
