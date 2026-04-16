import React, { useEffect } from 'react';
import { Card } from '../Card';
import { parseItem } from '../../utils/logic';

export function RenderStudy({ turns, turnIndex, onNext }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Enter' && !e.repeat) {
        e.preventDefault();
        onNext();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onNext]);

  const turnText = turns[turnIndex];
  if (!turnText) return null;
  const examples = turnText.split('Examples:\n')[1]?.split('\n').filter(l => l.includes('->')).map(l => parseItem(l.replace(/^\d+\.\s+/, ''))) || [];
  return (
    <div className="flex flex-col items-center">
      <h2 className="text-4xl font-black mb-4 uppercase">Learn the Rule</h2>
      <p className="text-zinc-500 mb-10 max-w-xl text-center leading-relaxed">{turnText.split('\n\n')[1]?.split('\n')[0]}</p>
      <div className="flex flex-wrap justify-center gap-8 mb-12">
        {examples.map((ex, i) => <Card key={i} data={ex} />)}
      </div>
      <button onClick={onNext} className="bg-white text-black px-12 py-4 rounded-xl font-black text-lg hover:bg-zinc-200 transition-all cursor-pointer">{turnIndex === turns.length - 2 ? "Ready?" : "Next Evidence"}</button>
    </div>
  );
}
