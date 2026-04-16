import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Check, Zap, Star, Timer } from 'lucide-react';
import data from '../data.json';
import { Card } from '../components/Card';
import { MetricCard } from '../components/MetricCard';
import { shuffleArray, parseItem } from '../utils/logic';
import { STAGES } from '../constants/stages';

export function Results() {
  const navigate = useNavigate();
  const [results, setResults] = useState([]);
  const [activeEpisodes, setActiveEpisodes] = useState([]);

  useEffect(() => {
    const savedData = sessionStorage.getItem('cogflex_last_session');
    if (!savedData) {
      navigate('/');
      return;
    }
    const { results: res, episodes: eps } = JSON.parse(savedData);
    setResults(res);
    setActiveEpisodes(eps);
  }, [navigate]);

  const totalCorrect = results.filter(r => r.isCorrect).length;
  const avgTime = results.reduce((acc, r) => acc + r.responseTime, 0) / (results.length || 1);
  const totalProbes = activeEpisodes.length * 5;

  const handleNewSession = () => {
    const shuffled = shuffleArray(data);
    sessionStorage.setItem('cogflex_active_episodes', JSON.stringify(shuffled));
    sessionStorage.removeItem('cogflex_current_results');
    navigate(`/challenge/${shuffled[0].episode_id}?stage=${STAGES.STUDY}&turn=0`);
  };

  return (
    <div className="flex flex-col items-center w-full max-w-7xl py-12 px-4 mt-20">
      <div className="fixed top-0 left-0 w-full bg-black/80 backdrop-blur-md border-b border-zinc-800 z-50 px-8 py-4 flex items-center justify-between">
        <Link to="/" className="hover:opacity-80 transition-opacity">
          <h1 className="text-2xl font-black tracking-tighter">CogFlex <span className="text-indigo-500">Human</span></h1>
        </Link>
        <button onClick={handleNewSession} className="px-6 py-2 rounded-xl bg-white text-black font-black text-xs uppercase hover:bg-zinc-200 transition-all flex items-center gap-2">
          <Zap size={14} className="fill-black" /> New Session
        </button>
      </div>

      <div className="text-center mb-16">
        <h1 className="text-7xl font-black mb-4 tracking-tighter bg-gradient-to-b from-white to-zinc-500 bg-clip-text text-transparent">Assessment Report</h1>
        <p className="text-zinc-500 font-mono uppercase tracking-[0.2em] text-sm">Cognitive Flexibility Baseline Analysis</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full mb-20">
        <MetricCard title="Success Rate" icon={<Check size={48} />} value={`${((totalCorrect / totalProbes) * 100).toFixed(0)}%`} sub={`${totalCorrect} / ${totalProbes} Correct`} progress={(totalCorrect / totalProbes) * 100} color="bg-green-500" />
        <MetricCard title="Neural Latency" icon={<Zap size={48} />} value={`${(avgTime / 1000).toFixed(2)}s`} sub="Average Response Speed" progress={50} color="bg-blue-500" />
        <MetricCard title="Flexibility Index" icon={<Star size={48} />} value={Math.round((totalCorrect * 1000) / (avgTime / 100 || 1))} sub="Precision-Velocity Ratio" progress={65} color="bg-indigo-500" />
      </div>

      <div className="w-full space-y-16">
        {activeEpisodes.map((episode, epIdx) => {
          const epResults = results.filter(r => r.episodeId === episode.episode_id);
          const epCorrect = epResults.filter(r => r.isCorrect).length;
          return (
            <div key={episode.episode_id} className="group bg-zinc-950 rounded-[3rem] border border-zinc-800 overflow-hidden hover:border-zinc-600 transition-all">
              <div className="bg-zinc-900/80 px-12 py-10 flex flex-col md:flex-row justify-between items-center gap-6 border-b border-zinc-800">
                <div>
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-3 py-1 bg-indigo-500/10 text-indigo-400 text-[10px] font-black rounded-full border border-indigo-500/20 uppercase tracking-widest">Challenge {epIdx + 1}</span>
                    <span className="px-3 py-1 bg-zinc-800 text-zinc-400 text-[10px] font-black rounded-full uppercase tracking-widest">{episode.analysis.difficulty_bin}</span>
                  </div>
                  <h3 className="text-4xl font-black text-white uppercase">{episode.analysis.suite_task_id.replace(/_/g, ' ')}</h3>
                </div>
                <div className="flex items-center gap-8 bg-black/40 px-8 py-6 rounded-3xl border border-white/5">
                  <div className="text-center"><div className="text-4xl font-black text-white">{epCorrect}/5</div><div className="text-[10px] text-zinc-500 uppercase">Accuracy</div></div>
                  <div className="w-px h-10 bg-zinc-800"></div>
                  <div className="text-center"><div className="text-4xl font-black text-white">{(epResults.reduce((acc, r) => acc + r.responseTime, 0) / 5000).toFixed(2)}s</div><div className="text-[10px] text-zinc-500 uppercase">Avg Speed</div></div>
                </div>
              </div>
              <div className="p-12 space-y-16">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-8">
                  {epResults.map((res, i) => (
                    <div key={i} className={`p-8 rounded-[2.5rem] border-2 ${res.isCorrect ? 'border-green-500/20 bg-green-500/[0.02]' : 'border-red-500/20 bg-red-500/[0.02]'}`}>
                      <div className="mb-8 flex justify-center"><Card data={parseItem(episode.inference.turns[episode.inference.turns.length - 1].split('Probes:\n')[1].split('\n')[i].replace(/^\d+\.\s+/, '').replace(' -> ?', ''))} showLabel={false} /></div>
                      <div className="text-center text-xs font-black uppercase text-zinc-500">{res.isCorrect ? 'Correct' : 'Incorrect'}</div>
                      <div className="mt-2 flex items-center justify-center gap-1.5 py-1.5 px-3 bg-white/[0.03] rounded-full border border-white/5 mx-auto w-fit">
                        <Timer size={14} className="text-zinc-500" />
                        <span className="text-[13px] font-mono text-zinc-400 font-bold">{(res.responseTime / 1000).toFixed(2)}s</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
