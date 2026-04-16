import React from 'react';

export function RenderShift({ onStart }) {
  return (
    <div className="flex flex-col items-center animate-pulse py-20">
      <div className="bg-red-600 text-white text-4xl font-black p-10 rounded-2xl mb-8 shadow-2xl uppercase">Rule Change Detected</div>
      <button onClick={onStart} className="bg-white text-black px-12 py-4 rounded-full font-black text-xl">Start Probes</button>
    </div>
  );
}
