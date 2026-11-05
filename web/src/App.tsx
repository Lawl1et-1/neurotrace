import { useEffect, useState } from "react";
import { loadIndex, type Index } from "./data";
import Explorer from "./components/Explorer";
import { Limitations, Method, Results } from "./components/Sections";

export default function App() {
  const [index, setIndex] = useState<Index | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    loadIndex().then(setIndex).catch((e) => setError(String(e)));
  }, []);

  return (
    <>
      <header className="rec-header">
        <a className="mark" href="#record" style={{ color: "inherit", textDecoration: "none" }}>NEUROTRACE</a>
        <nav aria-label="Sections">
          <a href="#record">Record</a>
          <a href="#report">Event</a>
          <a href="#method">Method</a>
          <a href="#results">Results</a>
          <a href="#limits">Limits</a>
        </nav>
        <span className="meta tnum">10-20 montage · 128 Hz · 2 s windows / 1 s stride</span>
        <span className="stamp" role="note">Research prototype · not for clinical use</span>
      </header>
      <main>
        {error && <p className="empty" style={{ padding: 32 }}>The record could not be loaded ({error}). Run scripts/export_explanations.py to create web/public/data.</p>}
        {!index && !error && <p className="empty" style={{ padding: 32 }}>Loading the record…</p>}
        {index && <Explorer index={index} />}
        <Method />
        {index && <Results index={index} />}
        <Limitations index={index} />
      </main>
      <footer className="end">
        <span>NEUROTRACE: a research prototype for spatio-temporal graph learning on EEG. Not for clinical use.</span>
        <span>
          Data: CHB-MIT Scalp EEG (PhysioNet, ODC-By) and synthetic recordings. <a href="https://github.com/Lawl1et-1/neurotrace">Source on GitHub</a>
        </span>
      </footer>
    </>
  );
}
