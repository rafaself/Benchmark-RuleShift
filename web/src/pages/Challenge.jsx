import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { STAGES } from '../constants/stages';
import { getPossibleLabels, getProbeCount, getProbeOffset, getTotalProbeCount } from '../utils/logic';
import { RenderStudy } from '../components/challenge/RenderStudy';
import { RenderShift } from '../components/challenge/RenderShift';
import { RenderDecision } from '../components/challenge/RenderDecision';

export function Challenge() {
  const { episodeId } = useParams();
  const navigate = useNavigate();
  const activeEpisodes = JSON.parse(sessionStorage.getItem('cogflex_active_episodes') || '[]');
  const episodeIndex = activeEpisodes.findIndex(e => e.episode_id === episodeId);
  const currentEpisode = episodeIndex >= 0 ? activeEpisodes[episodeIndex] : null;

  useEffect(() => {
    if (!activeEpisodes.length || !currentEpisode) {
      navigate('/');
    }
  }, [activeEpisodes, currentEpisode, navigate]);

  if (!currentEpisode) return null;

  return (
    <ChallengeSession
      key={episodeId}
      activeEpisodes={activeEpisodes}
      currentEpisode={currentEpisode}
      episodeIndex={episodeIndex}
    />
  );
}

function ChallengeSession({ activeEpisodes, currentEpisode, episodeIndex }) {
  const navigate = useNavigate();
  const episodeId = currentEpisode.episode_id;
  const savedProgress = JSON.parse(sessionStorage.getItem(`cogflex_progress_${episodeId}`) || 'null');
  const [results, setResults] = useState(() => JSON.parse(sessionStorage.getItem('cogflex_current_results') || '[]'));
  const [startTime, setStartTime] = useState(0);
  const [stage, setStage] = useState(savedProgress?.stage ?? STAGES.STUDY);
  const [turnIndex, setTurnIndex] = useState(savedProgress?.turn ?? 0);
  const [probeIndex, setProbeIndex] = useState(savedProgress?.probe ?? 0);
  const [feedback, setFeedback] = useState(null);

  useEffect(() => {
    if (results.length > 0) {
      sessionStorage.setItem('cogflex_current_results', JSON.stringify(results));
    }
  }, [results]);

  useEffect(() => {
    sessionStorage.setItem(`cogflex_progress_${episodeId}`, JSON.stringify({
      stage,
      turn: turnIndex,
      probe: probeIndex
    }));
  }, [episodeId, stage, turnIndex, probeIndex]);

  const turns = currentEpisode.inference.turns || [];
  const currentProbeCount = getProbeCount(currentEpisode);
  const totalProbeCount = getTotalProbeCount(activeEpisodes);
  const completedProbeOffset = getProbeOffset(activeEpisodes, episodeIndex);

  const handleNextTurn = () => {
    if (turnIndex < turns.length - 2) {
      setTurnIndex(prev => prev + 1);
    } else {
      setStage(STAGES.SHIFT);
      setTurnIndex(prev => prev + 1);
    }
  };

  const handleStartDecision = () => {
    setStage(STAGES.DECISION);
    setProbeIndex(0);
    setStartTime(Date.now());
  };

  const handleDecision = (label) => {
    if (feedback) return; // Prevent double clicks during feedback

    const endTime = Date.now();
    const targetLabel = currentEpisode.scoring.final_probe_targets[probeIndex];
    const isCorrect = label === targetLabel;
    
    setFeedback(isCorrect ? 'correct' : 'incorrect');

    const newResult = {
      episodeId: currentEpisode.episode_id,
      task: currentEpisode.analysis.suite_task_id,
      probeIndex,
      userLabel: label,
      targetLabel,
      isCorrect,
      responseTime: endTime - startTime
    };

    const updatedResults = [...results, newResult];
    
    setTimeout(() => {
      setFeedback(null);
      setResults(updatedResults);

      if (probeIndex < currentProbeCount - 1) {
        setProbeIndex(prev => prev + 1);
        setStartTime(Date.now());
      } else {
        // Clear progress for finished episode
        sessionStorage.removeItem(`cogflex_progress_${episodeId}`);
        
        if (episodeIndex < activeEpisodes.length - 1) {
          const nextEpId = activeEpisodes[episodeIndex + 1].episode_id;
          navigate(`/challenge/${nextEpId}`);
        } else {
          const savedHistory = JSON.parse(localStorage.getItem('cogflex_history') || '[]');
          const sessionReport = {
            id: Date.now(),
            date: new Date().toISOString(),
            totalCorrect: updatedResults.filter(r => r.isCorrect).length,
            avgTime: updatedResults.reduce((acc, r) => acc + r.responseTime, 0) / updatedResults.length,
            totalProbes: totalProbeCount,
            episodesCount: activeEpisodes.length,
            results: updatedResults,
            episodes: activeEpisodes
          };
          localStorage.setItem('cogflex_history', JSON.stringify([sessionReport, ...savedHistory].slice(0, 10)));
          
          sessionStorage.setItem('cogflex_last_session', JSON.stringify({
            results: updatedResults,
            episodes: activeEpisodes
          }));
          sessionStorage.removeItem('cogflex_active_episodes');
          sessionStorage.removeItem('cogflex_current_results');
          
          // Clear all episode progresses
          activeEpisodes.forEach(ep => {
            sessionStorage.removeItem(`cogflex_progress_${ep.episode_id}`);
          });

          navigate('/results');
        }
      }
    }, 400);
  };

  const resetGame = () => {
    sessionStorage.removeItem('cogflex_active_episodes');
    sessionStorage.removeItem('cogflex_current_results');
    activeEpisodes.forEach(ep => {
      sessionStorage.removeItem(`cogflex_progress_${ep.episode_id}`);
    });
    navigate('/');
  };

  if (!currentEpisode) return null;

  return (
    <div className="w-full min-h-screen bg-black">
      <div className="fixed top-0 left-0 w-full h-2 bg-gray-900 z-[60]">
        <div 
          className="h-full bg-indigo-500 transition-all duration-300"
          style={{ width: `${((
            completedProbeOffset + (stage === STAGES.DECISION ? probeIndex : 0)
          ) / (totalProbeCount || 1)) * 100}%` }}
        ></div>
      </div>

      <div className="fixed top-0 left-0 w-full bg-black/80 backdrop-blur-md border-b border-zinc-800 z-50 px-12 py-4 flex items-center justify-between">
        <button 
          onClick={resetGame}
          className="flex items-center gap-4 hover:opacity-80 transition-opacity cursor-pointer text-left"
        >
          <h1 className="text-xl font-black tracking-tighter">
            CogFlex <span className="text-indigo-500">Human</span>
          </h1>
        </button>
        
        <div className="text-sm font-black tracking-widest uppercase text-zinc-500">
          Episode {episodeIndex + 1} / {activeEpisodes.length}
        </div>

        <button 
          onClick={resetGame}
          className="px-4 py-2 rounded-xl border border-zinc-800 text-zinc-400 font-black text-[10px] uppercase hover:bg-zinc-900 transition-all cursor-pointer"
        >
          Abort Session
        </button>
      </div>

      <div className="pt-28 flex flex-col items-center w-full pb-12 animate-fade-in">
        {stage === STAGES.STUDY && <RenderStudy turns={turns} turnIndex={turnIndex} onNext={handleNextTurn} />}
        {stage === STAGES.SHIFT && (
          <RenderShift 
            onStart={handleStartDecision} 
            isExplicitShift={currentEpisode.analysis.shift_mode === 'explicit_instruction'}
          />
        )}
        {stage === STAGES.DECISION && (
          <RenderDecision 
            currentEpisode={currentEpisode} 
            probeIndex={probeIndex} 
            results={results} 
            onDecision={handleDecision} 
            possibleLabels={getPossibleLabels(turns)}
            feedback={feedback}
          />
        )}
      </div>
    </div>
  );
}
