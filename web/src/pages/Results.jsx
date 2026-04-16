import React, { useEffect } from 'react';
import { useNavigate, Link, useSearchParams } from 'react-router-dom';
import { Check, ChevronDown, Plus, Star, Timer, Zap } from 'lucide-react';
import data from '../data.json';
import { Card } from '../components/Card';
import { MetricCard } from '../components/MetricCard';
import { getEpisodeExampleGroups, getEpisodeProbes, getLabelStyle, getProbeCount, getTotalProbeCount, shuffleArray } from '../utils/logic';

function RuleHint({ description }) {
  if (!description) return null;

  return (
    <div className="w-full max-w-56 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-center">
      <div className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">Rule</div>
      <div className="mt-1 text-xs font-medium leading-relaxed text-zinc-300">{description}</div>
    </div>
  );
}

export function Results() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('id');
  const sessionPayload = (() => {
    if (sessionId) {
      const history = JSON.parse(localStorage.getItem('cogflex_history') || '[]');
      const session = history.find(s => s.id.toString() === sessionId);
      if (session && session.results && session.episodes) {
        return { results: session.results, episodes: session.episodes };
      }
    }

    const savedData = sessionStorage.getItem('cogflex_last_session');
    return savedData ? JSON.parse(savedData) : null;
  })();

  useEffect(() => {
    if (!sessionPayload) {
      navigate('/');
    }
  }, [navigate, sessionPayload]);

  const results = sessionPayload?.results || [];
  const activeEpisodes = sessionPayload?.episodes || [];

  const totalCorrect = results.filter(r => r.isCorrect).length;
  const avgTime = results.reduce((acc, r) => acc + r.responseTime, 0) / (results.length || 1);
  const totalProbes = getTotalProbeCount(activeEpisodes);
  const successRate = totalProbes > 0 ? (totalCorrect / totalProbes) * 100 : 0;

  const handleNewSession = () => {
    const shuffled = shuffleArray(data);
    sessionStorage.setItem('cogflex_active_episodes', JSON.stringify(shuffled));
    sessionStorage.removeItem('cogflex_current_results');
    navigate(`/challenge/${shuffled[0].episode_id}`);
  };

  return (
    <div className="w-full min-h-screen bg-black">
      <div className="fixed top-0 left-0 w-full bg-black/80 backdrop-blur-md border-b border-zinc-800 z-50 px-12 py-4 flex items-center justify-between">
        <Link to="/" className="hover:opacity-80 transition-opacity cursor-pointer">
          <h1 className="text-2xl font-black tracking-tighter">CogFlex <span className="text-indigo-500">Human</span></h1>
        </Link>
        <button onClick={handleNewSession} className="px-6 py-2 rounded-xl bg-white text-black font-black text-xs uppercase hover:bg-zinc-200 transition-all flex items-center gap-2 cursor-pointer">
          <Plus size={14} /> New Session
        </button>
      </div>

      <div className="flex flex-col items-center w-full max-w-7xl mx-auto pt-28 pb-12 px-4 animate-fade-in">
        <div className="text-center mb-16">
          <h1 className="text-7xl font-black mb-4 tracking-tighter bg-gradient-to-b from-white to-zinc-500 bg-clip-text text-transparent">Assessment Report</h1>
          <p className="text-zinc-500 font-mono uppercase tracking-[0.2em] text-sm">Public Sample Performance Analysis</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full mb-20">
          <MetricCard title="Success Rate" icon={<Check size={48} />} value={`${successRate.toFixed(0)}%`} sub={`${totalCorrect} / ${totalProbes} Correct`} progress={successRate} color="bg-green-500" />
          <MetricCard title="Neural Latency" icon={<Zap size={48} />} value={`${(avgTime / 1000).toFixed(2)}s`} sub="Average Response Speed" progress={50} color="bg-blue-500" />
          <MetricCard title="Flexibility Index" icon={<Star size={48} />} value={Math.round((totalCorrect * 1000) / (avgTime / 100 || 1))} sub="Precision-Velocity Ratio" progress={65} color="bg-indigo-500" />
        </div>

        <div className="w-full space-y-16">
          {activeEpisodes.map((episode, epIdx) => {
            const epResults = results.filter(r => r.episodeId === episode.episode_id);
            const epCorrect = epResults.filter(r => r.isCorrect).length;
            const probeCount = getProbeCount(episode);
            const probeItems = getEpisodeProbes(episode);
            const exampleGroups = getEpisodeExampleGroups(episode);

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
                    <div className="text-center"><div className="text-4xl font-black text-white">{epCorrect}/{probeCount}</div><div className="text-[10px] text-zinc-500 uppercase">Accuracy</div></div>
                    <div className="w-px h-10 bg-zinc-800"></div>
                    <div className="text-center"><div className="text-4xl font-black text-white">{(epResults.reduce((acc, r) => acc + r.responseTime, 0) / ((epResults.length || 1) * 1000)).toFixed(2)}s</div><div className="text-[10px] text-zinc-500 uppercase">Avg Speed</div></div>
                  </div>
                </div>
                <div className="p-12 space-y-16">
                  {exampleGroups.length > 0 && (
                    <details className="group/examples rounded-[2rem] border border-zinc-800 bg-black/30 open:bg-zinc-950/80">
                      <summary className="list-none cursor-pointer px-8 py-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                        <div>
                          <div className="text-sm font-black uppercase tracking-[0.2em] text-zinc-300">Show Examples</div>
                          <p className="mt-2 text-sm text-zinc-500">
                            Review the labeled examples used to infer the rule for this challenge.
                          </p>
                        </div>
                        <div className="flex items-center gap-3 text-zinc-400">
                          <span className="text-xs font-black uppercase tracking-[0.2em]">
                            {exampleGroups.reduce((total, group) => total + group.examples.length, 0)} examples
                          </span>
                          <ChevronDown size={18} className="transition-transform group-open/examples:rotate-180" />
                        </div>
                      </summary>

                      <div className="px-8 pb-8 space-y-8 border-t border-zinc-800">
                        {exampleGroups.map(group => (
                          <section key={group.turnIndex} className="pt-8">
                            <div className="flex items-center gap-3 mb-3">
                              <span className="px-3 py-1 bg-white/5 text-zinc-300 text-[10px] font-black rounded-full border border-white/10 uppercase tracking-widest">
                                {group.title}
                              </span>
                              <span className="text-[11px] font-black uppercase tracking-[0.15em] text-zinc-600">
                                {group.examples.length} examples
                              </span>
                            </div>
                            <p className="max-w-3xl text-sm leading-relaxed text-zinc-500 mb-6">{group.narrative}</p>
                            <div className="flex flex-wrap gap-6">
                              {group.examples.map((example, exampleIndex) => (
                                <div key={`${group.turnIndex}-${exampleIndex}`} className="flex flex-col items-center gap-3">
                                  <Card data={example.data} />
                                  <RuleHint description={example.ruleDescription} />
                                </div>
                              ))}
                            </div>
                          </section>
                        ))}
                      </div>
                    </details>
                  )}

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-8">
                    {epResults.map((res, i) => (
                      <div key={i} className={`p-8 rounded-[2.5rem] border-2 ${res.isCorrect ? 'border-green-500/20 bg-green-500/[0.02]' : 'border-red-500/20 bg-red-500/[0.02]'}`}>
                        <div className="mb-8 flex flex-col items-center gap-3">
                          {probeItems[i] ? <Card data={probeItems[i].data} showLabel={false} /> : null}
                          <div className={`px-6 py-2 rounded-xl text-sm font-black uppercase tracking-tighter shadow-2xl border-2 ${getLabelStyle(res.userLabel)}`}>
                            {res.userLabel}
                          </div>
                          <RuleHint description={probeItems[i]?.ruleDescription} />
                        </div>
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
    </div>
  );
}
