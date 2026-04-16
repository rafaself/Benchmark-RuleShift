import React, { useEffect } from 'react';
import { AlertTriangle, Zap } from 'lucide-react';

export function RenderShift({ onStart, isExplicitShift }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Enter' && !e.repeat) {
        e.preventDefault();
        onStart();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onStart]);

  return (
    <div className="flex flex-col items-center py-20 animate-fade-in">
      <div className={`relative flex flex-col items-center text-center p-12 rounded-[3rem] border-4 mb-12 shadow-2xl transition-all duration-500 ${isExplicitShift ? 'bg-red-950/40 border-red-500 animate-pulse' : 'bg-indigo-950/40 border-indigo-500'}`}>
        <div className={`mb-6 p-4 rounded-2xl ${isExplicitShift ? 'bg-red-500 text-white' : 'bg-indigo-500 text-white'}`}>
          {isExplicitShift ? <AlertTriangle size={48} /> : <Zap size={48} />}
        </div>
        
        <h2 className="text-5xl font-black text-white uppercase tracking-tighter mb-4">
          {isExplicitShift ? 'Rule Update' : 'Decision Phase'}
        </h2>
        
        <p className="text-zinc-400 font-bold max-w-md leading-relaxed mb-2">
          {isExplicitShift 
            ? 'A new labeling logic is now in force. Abandon previous rules and apply the newest inferred pattern.' 
            : 'Evidence collection complete. Prepare to classify the upcoming probes using the current rule.'}
        </p>
        
        {isExplicitShift && (
          <div className="mt-4 px-4 py-1 bg-red-500/20 text-red-400 text-[10px] font-black uppercase tracking-[0.2em] rounded-full border border-red-500/30">
            High Cognitive Flexibility Required
          </div>
        )}
      </div>

      <button 
        onClick={onStart} 
        className={`group flex items-center gap-4 px-16 py-6 rounded-[2rem] font-black text-2xl transition-all active:scale-95 cursor-pointer shadow-2xl ${isExplicitShift ? 'bg-red-600 text-white hover:bg-red-500 shadow-red-500/20' : 'bg-white text-black hover:bg-zinc-200 shadow-white/10'}`}
      >
        START PROBES
        <Zap size={24} className={isExplicitShift ? 'fill-white' : 'fill-black'} />
      </button>
    </div>
  );
}
