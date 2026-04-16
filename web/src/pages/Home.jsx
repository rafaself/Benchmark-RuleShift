import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, Zap, History, Brain, Target, Timer } from 'lucide-react';
import data from '../data.json';
import { getTotalProbeCount, shuffleArray } from '../utils/logic';

export function Home() {
  const navigate = useNavigate();
  const [history, setHistory] = useState(() => JSON.parse(localStorage.getItem('cogflex_history') || '[]'));
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    // Proactive cleanup: returning to Home kills any incomplete session memory
    sessionStorage.removeItem('cogflex_active_episodes');
    sessionStorage.removeItem('cogflex_current_results');
  }, []);

  const handleStart = () => {
    const shuffled = shuffleArray(data);
    sessionStorage.setItem('cogflex_active_episodes', JSON.stringify(shuffled));
    sessionStorage.removeItem('cogflex_current_results');
    navigate(`/challenge/${shuffled[0].episode_id}`);
  };

  const handleClearHistory = () => {
    localStorage.removeItem('cogflex_history');
    setHistory([]);
    setIsModalOpen(false);
  };

  return (
    <div className="w-full min-h-screen flex flex-col items-center justify-center py-12 px-6 animate-fade-in">
      <div className="w-full max-w-5xl flex flex-col items-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-[10px] font-black uppercase tracking-[0.3em] mb-8 animate-fade-in">
          <Activity size={12} />
          Human-Playable Public Samples
        </div>
        <h1 className="text-8xl font-black mb-6 tracking-tighter bg-gradient-to-b from-white to-zinc-500 bg-clip-text text-transparent leading-[1.1] py-2">
          CogFlex
        </h1>
        <p className="text-xl text-zinc-400 max-w-2xl mx-auto leading-relaxed font-medium">
          Play benchmark-format CogFlex episodes copied from the current public split.
          Inspect how humans handle the same symbolic rule-switching format used by the public benchmark.
        </p>
      </div>

      <div className="flex flex-col gap-20 w-full max-w-4xl mt-20">
        <div className="space-y-10">
          <div className="flex items-center gap-6">
            <h3 className="text-white text-xs font-black uppercase tracking-[0.4em] whitespace-nowrap">Protocol Specification</h3>
            <div className="h-px flex-1 bg-zinc-800/50"></div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { icon: <Brain size={20} />, title: "Inference", desc: "Study the labeled evidence turns to infer the active rule or routing pattern." },
              { icon: <Zap size={20} />, title: "Adaptation", desc: "Alert: Rules shift without warning. Stay agile." },
              { icon: <Target size={20} />, title: "Precision", desc: "Classify every probe shown in each public-split sample with maximum accuracy." },
              { icon: <Timer size={20} />, title: "Latency", desc: "Response speed is recorded for qualitative human-performance inspection." }
            ].map((step, i) => (
              <div key={i} className="bg-zinc-900/40 border border-zinc-800/50 p-6 rounded-[2rem] hover:border-zinc-700 transition-all duration-300 group">
                <div className="w-10 h-10 flex items-center justify-center rounded-xl bg-indigo-500/5 mb-4 group-hover:bg-indigo-500/15 transition-all duration-500">
                  <div className="text-indigo-500 group-hover:scale-110 group-hover:text-indigo-400 transition-all duration-500 ease-out">
                    {step.icon}
                  </div>
                </div>
                <div className="text-white font-black text-xs uppercase tracking-widest mb-2">{step.title}</div>
                <div className="text-zinc-500 text-[11px] leading-relaxed font-bold">{step.desc}</div>
              </div>
            ))}
          </div>

          <button 
            onClick={handleStart}
            className="w-full bg-white text-black hover:bg-zinc-200 py-6 rounded-[2rem] font-black text-2xl transition-all shadow-[0_0_40px_rgba(255,255,255,0.05)] active:scale-[0.98] cursor-pointer flex items-center justify-center gap-4 group"
          >
            INITIATE CHALLENGE
            <Zap size={24} className="fill-black group-hover:animate-pulse" />
          </button>
        </div>

        <div className="space-y-8">
          <div className="flex items-center gap-6">
            <h3 className="text-zinc-500 text-xs font-black uppercase tracking-[0.4em] whitespace-nowrap">Session History</h3>
            <div className="h-px flex-1 bg-zinc-800/50"></div>
            {history.length > 0 && (
              <button 
                onClick={() => setIsModalOpen(true)}
                className="text-zinc-500 hover:text-red-500 text-[10px] font-black uppercase tracking-widest transition-colors cursor-pointer"
              >
                Clear History
              </button>
            )}
          </div>

          {history.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {history.map(session => (
                (() => {
                  const totalProbes = session.totalProbes ?? getTotalProbeCount(session.episodes || []);
                  const accuracy = totalProbes > 0 ? session.totalCorrect / totalProbes : 0;
                  return (
                    <button 
                      key={session.id} 
                      onClick={() => navigate(`/results?id=${session.id}`)}
                      className="bg-zinc-900/30 border border-zinc-800/50 p-5 rounded-2xl flex items-center justify-between group hover:bg-zinc-900/60 transition-all cursor-pointer text-left w-full"
                    >
                      <div className="flex items-center gap-5">
                        <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-black text-sm border-2 ${
                          accuracy > 0.8 
                          ? 'bg-green-500/10 border-green-500/20 text-green-500' 
                          : 'bg-zinc-800/50 border-zinc-700/50 text-zinc-500'
                        }`}>
                          {Math.round(accuracy * 100)}%
                        </div>
                        <div>
                          <div className="text-zinc-100 font-black text-sm tracking-tight capitalize">
                            {new Date(session.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} • {new Date(session.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </div>
                          <div className="text-zinc-500 text-[10px] uppercase font-black tracking-[0.15em] mt-1 flex items-center gap-2">
                            <span className="text-zinc-400">{session.totalCorrect}/{totalProbes} Hits</span>
                            <span className="w-1 h-1 rounded-full bg-zinc-700"></span>
                            <span>{(session.avgTime / 1000).toFixed(2)}s Latency</span>
                          </div>
                        </div>
                      </div>
                      <div className="text-zinc-800 group-hover:text-zinc-600 transition-colors">
                        <History size={18} />
                      </div>
                    </button>
                  );
                })()
              ))}
            </div>
          ) : (
            <div className="h-32 rounded-[2rem] border-2 border-dashed border-zinc-900/50 flex flex-col items-center justify-center text-center p-8">
              <Activity size={24} className="text-zinc-800 mb-2" />
              <p className="text-zinc-600 text-[10px] font-black uppercase tracking-widest leading-relaxed">
                No session data available
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Confirmation Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center p-6">
          <div 
            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            onClick={() => setIsModalOpen(false)}
          ></div>
          <div className="relative bg-zinc-900 border border-zinc-800 p-8 rounded-[2rem] max-w-sm w-full shadow-2xl animate-in fade-in zoom-in duration-200">
            <h3 className="text-xl font-black mb-2 uppercase tracking-tight">Clear History?</h3>
            <p className="text-zinc-500 text-sm font-bold mb-8 leading-relaxed">
              This will permanently delete all your previous session data. This action cannot be undone.
            </p>
            <div className="flex gap-4">
              <button 
                onClick={() => setIsModalOpen(false)}
                className="flex-1 px-6 py-3 rounded-xl border border-zinc-800 text-zinc-400 font-black text-xs uppercase hover:bg-zinc-800 transition-all cursor-pointer"
              >
                Cancel
              </button>
              <button 
                onClick={handleClearHistory}
                className="flex-1 px-6 py-3 rounded-xl bg-red-600 text-white font-black text-xs uppercase hover:bg-red-500 transition-all cursor-pointer"
              >
                Clear All
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
