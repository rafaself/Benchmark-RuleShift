import React from 'react';
import { Star, Triangle, Circle, Square, Diamond, Pentagon, Hexagon, Octagon, Box } from 'lucide-react';

export const Card = ({ data, showLabel = true }) => {
  const getToneColor = (tone) => {
    const tones = {
      warm: 'bg-orange-300/95 border-orange-400',
      cool: 'bg-blue-300/95 border-blue-400',
      bright: 'bg-yellow-200/95 border-yellow-300',
      muted: 'bg-gray-300/95 border-gray-400',
      neutral: 'bg-slate-200/95 border-slate-300',
      dim: 'bg-indigo-300/95 border-indigo-400',
    };
    return tones[tone] || 'bg-white/95 border-gray-300';
  };

  const getShapeIcon = (shape) => {
    const props = { size: 64, strokeWidth: 2, className: "text-zinc-800 drop-shadow-sm" };
    const icons = {
      star: <Star {...props} />,
      triangle: <Triangle {...props} />,
      circle: <Circle {...props} />,
      square: <Square {...props} />,
      diamond: <Diamond {...props} />,
      pentagon: <Pentagon {...props} />,
      kite: <Diamond {...props} className="text-zinc-800 drop-shadow-sm rotate-12 scale-y-125" />,
      oval: <Circle {...props} className="text-zinc-800 drop-shadow-sm scale-x-150" />,
      hexagon: <Hexagon {...props} />,
      octagon: <Octagon {...props} />,
    };
    return icons[shape] || <Box {...props} />;
  };

  return (
    <div className="flex flex-col items-center group">
      <div className={`w-52 border-2 rounded-2xl overflow-hidden shadow-lg ${getToneColor(data.tone)}`}>
        <div className="flex justify-between px-4 py-3 bg-black/20 border-b border-black/10">
          <div className="flex flex-col items-center">
            <span className="text-[10px] uppercase text-black/60 font-black leading-tight">R1 VALUE</span>
            <span className="text-lg font-mono font-black text-black leading-none">{data.r1}</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-[10px] uppercase text-black/60 font-black leading-tight">R2 VALUE</span>
            <span className="text-lg font-mono font-black text-black leading-none">{data.r2}</span>
          </div>
        </div>
        <div className="py-8 flex flex-col items-center justify-center bg-white/10">
          <div className="mb-3">{getShapeIcon(data.shape)}</div>
          <div className="px-4 py-1 bg-black/80 rounded-full text-[11px] font-black uppercase tracking-widest text-white shadow-inner">
            {data.shape}
          </div>
        </div>
        <div className="px-4 py-3 bg-black/10 border-t border-black/10">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase text-black/60 font-black leading-tight">TONE PROPERTY</span>
            <span className="text-sm font-black text-black capitalize">{data.tone}</span>
          </div>
        </div>
      </div>
      {showLabel && data.label && (
        <div className={`mt-4 px-6 py-2 rounded-xl text-sm font-black uppercase tracking-tighter shadow-2xl border-2 ${
          ['accept', 'anchor', 'valid', 'north', 'amber', 'ember'].includes(data.label) 
          ? 'bg-green-500 text-white border-green-400' 
          : 'bg-red-500 text-white border-red-400'
        }`}>
          {data.label}
        </div>
      )}
    </div>
  );
};
