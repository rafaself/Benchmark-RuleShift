import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from 'lucide-react';
import { Home } from './pages/Home';
import { Challenge } from './pages/Challenge';
import { Results } from './pages/Results';

// Main App Component with Routing
export default function App() {
  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center p-8 font-sans transition-all duration-300">
      <div className="lg:hidden fixed inset-0 z-[100] bg-black flex flex-col items-center justify-center p-10 text-center">
        <Box size={64} className="text-indigo-500 mb-6" />
        <h1 className="text-3xl font-black mb-4">Desktop Only</h1>
        <p className="text-zinc-500 leading-relaxed">
          The CogFlex Human Benchmark requires a larger screen. Please access from a desktop.
        </p>
      </div>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/challenge/:episodeId" element={<Challenge />} />
        <Route path="/results" element={<Results />} />
      </Routes>
    </div>
  );
}
