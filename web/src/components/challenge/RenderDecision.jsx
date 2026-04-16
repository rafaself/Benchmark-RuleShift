import React, { useEffect } from 'react';
import { Check, X, Box } from 'lucide-react';
import { Card } from '../Card';
import { getProbeCount, parseItem } from '../../utils/logic';

export function RenderDecision({ currentEpisode, probeIndex, results, onDecision, possibleLabels }) {
  const turnText = currentEpisode?.inference?.turns[currentEpisode.inference.turns.length - 1];
  const probeLines = turnText.split('Probes:\n')[1]?.split('\n\n')[0].split('\n') || [];
  const currentProbe = parseItem(probeLines[probeIndex].replace(/^\d+\.\s+/, '').replace(' -> ?', ''));
  const epResults = results.filter(r => r.episodeId === currentEpisode.episode_id);
  const probeCount = getProbeCount(currentEpisode);

  useEffect(() => {
    const handleKeyDown = (e) => {
      const key = parseInt(e.key);
      if (key >= 1 && key <= possibleLabels.length) {
        onDecision(possibleLabels[key - 1]);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [possibleLabels, onDecision]);

  return (
    <div className="flex flex-col items-center">
      <div className="flex gap-2 mb-8">
        {Array.from({ length: probeCount }, (_, i) => i).map(i => (
          <div key={i} className={`w-10 h-12 rounded-lg border-2 flex items-center justify-center transition-all duration-300 ${epResults[i] ? 'border-indigo-500 bg-indigo-500/20' : (i === probeIndex ? 'border-indigo-500 animate-pulse' : 'border-zinc-800')}`}>
            {epResults[i] ? <Check size={16} className="text-indigo-400" /> : <Box size={12} className="text-zinc-800" />}
          </div>
        ))}
      </div>
      <Card data={currentProbe} showLabel={false} />
      <div className="flex gap-6 mt-12">
        {possibleLabels.map((label, i) => (
          <div key={label} className="flex flex-col items-center gap-3">
            <button onClick={() => onDecision(label)} className="px-12 py-5 rounded-2xl border-4 border-indigo-600 text-indigo-400 font-black text-2xl hover:bg-indigo-600 hover:text-white transition-all capitalize cursor-pointer">{label}</button>
            <span className="text-zinc-600 font-black text-xs uppercase tracking-widest">Press {i + 1}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
