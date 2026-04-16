export const shuffleArray = (array) => {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

export function parseItem(itemStr) {
  const [attributes, label] = itemStr.split(' -> ');
  const parts = attributes.split(/[|,]/).map(p => p.trim());
  const obj = { label };
  parts.forEach(p => {
    const [key, val] = p.split('=');
    if (key && val) obj[key.trim()] = val.trim();
  });
  return obj;
}

export function getPossibleLabels(turns) {
  const lastTurnText = turns[turns.length - 1];
  const labelMatch = lastTurnText?.match(/Use only labels from: (.*)\./);
  return labelMatch ? labelMatch[1].split(', ') : ['accept', 'reject'];
}

export function getProbeCount(episode) {
  return episode?.inference?.response_spec?.probe_count ?? 0;
}

export function getTotalProbeCount(episodes) {
  return episodes.reduce((total, episode) => total + getProbeCount(episode), 0);
}

export function getProbeOffset(episodes, episodeIndex) {
  return episodes.slice(0, Math.max(episodeIndex, 0)).reduce((total, episode) => total + getProbeCount(episode), 0);
}
