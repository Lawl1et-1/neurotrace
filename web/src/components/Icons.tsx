// Authored line icons, one 1.75px stroke throughout.
const base = { fill: "none", stroke: "currentColor", strokeWidth: 1.75, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, viewBox: "0 0 24 24", "aria-hidden": true };

export const Play = () => (
  <svg {...base}>
    <path d="M7 4.5v15l12-7.5z" fill="currentColor" />
  </svg>
);
export const Pause = () => (
  <svg {...base}>
    <path d="M8 4.5v15M16 4.5v15" strokeWidth={3} />
  </svg>
);
export const Step = ({ back = false }: { back?: boolean }) => (
  <svg {...base} style={back ? { transform: "scaleX(-1)" } : undefined}>
    <path d="M6 5l9 7-9 7z" fill="currentColor" />
    <path d="M18 5v14" />
  </svg>
);
export const Arrow = () => (
  <svg {...base}>
    <path d="M4 12h15M13 6l6 6-6 6" />
  </svg>
);
export const Code = () => (
  <svg {...base}>
    <path d="M8 7l-5 5 5 5M16 7l5 5-5 5M14 4l-4 16" />
  </svg>
);
