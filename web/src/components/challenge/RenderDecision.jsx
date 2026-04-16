import React from 'react';
import { Check, X, Box } from 'lucide-react';
import { Card } from '../Card';
import { getProbeCount, parseItem } from '../../utils/logic';

export function RenderDecision({ currentEpisode, probeIndex, results, onDecision, possibleLabels }) {
  const turnText = currentEpisode?.inference?.turns[currentEpisode.inference.turns.length - 1];
  const probeLines = turnText.split('Probes:\n')[1]?.split('\n\n')[0].split('\n') || [];
  const currentProbe = parseItem(probeLines[probeIndex].replace(/^\d+\.\s+/, '').replace(' -> ?', ''));
  const epResults = results.filter(r => r.episodeId === currentEpisode.episode_id);
  const probeCount = getProbeCount(currentEpisode);

  return (
    <div className="flex flex-col items-center">
      <div className="flex gap-2 mb-8">
        {Array.from({ length: probeCount }, (_, i) => i).map(i => (
          <div key={i} className={`w-10 h-12 rounded-lg border-2 flex items-center justify-center ${epResults[i] ? (epResults[i].isCorrect ? 'border-green-500 bg-green-500/10' : 'border-red-500 bg-red-500/10') : (i === probeIndex ? 'border-indigo-500 animate-pulse' : 'border-zinc-800')}`}>
            {epResults[i] ? (epResults[i].isCorrect ? <Check size={16} /> : <X size={16} />) : <Box size={12} className="text-zinc-800" />}
          </div>
        ))}
      </div>
      <Card data={currentProbe} showLabel={false} />
      <div className="flex gap-6 mt-12">
        {possibleLabels.map(label => (
          <button key={label} onClick={() => onDecision(label)} className="px-12 py-5 rounded-2xl border-4 border-indigo-600 text-indigo-400 font-black text-2xl hover:bg-indigo-600 hover:text-white transition-all capitalize cursor-pointer">{label}</button>
        ))}
      </div>
    </div>
  );
}
