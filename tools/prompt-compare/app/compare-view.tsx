"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Sample {
  sample_id: number;
  adversarial_prompt: string;
  plans: Record<string, string>;
  risk_domain: string;
  risk_subdomain: string;
}

const DOMAIN_COLORS: Record<string, string> = {
  "Political Violence & Terrorism": "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  "Criminal & Financial Illicit Activities": "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
  "Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)": "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300",
};

const MODEL_COLORS: string[] = [
  "bg-blue-50 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
  "bg-green-50 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  "bg-violet-50 text-violet-800 dark:bg-violet-900/30 dark:text-violet-300",
  "bg-teal-50 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300",
  "bg-pink-50 text-pink-800 dark:bg-pink-900/30 dark:text-pink-300",
  "bg-indigo-50 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300",
];

function shortModel(model: string): string {
  return model.split("/").pop() || model;
}

function Badge({ domain }: { domain: string }) {
  const short = domain.includes("CBRNE") ? "CBRNE" : domain.includes("Criminal") ? "Criminal/Financial" : "Political Violence";
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${DOMAIN_COLORS[domain] ?? "bg-zinc-100 text-zinc-700"}`}>
      {short}
    </span>
  );
}

function PromptPanel({ title, content, color }: { title: string; content: string; color: string }) {
  return (
    <div className="flex-1 min-w-0 flex flex-col">
      <div className={`px-4 py-2 text-sm font-semibold ${color} rounded-t-lg flex items-center justify-between`}>
        <span>{title}</span>
        <span className="opacity-60 font-normal">{content.length.toLocaleString()} chars</span>
      </div>
      <div className="flex-1 overflow-auto border border-t-0 border-zinc-200 dark:border-zinc-700 rounded-b-lg bg-white dark:bg-zinc-900 p-4">
        <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:mt-4 prose-headings:mb-2 prose-p:my-1.5 prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5 prose-pre:bg-zinc-100 prose-pre:dark:bg-zinc-800 prose-blockquote:border-zinc-300 prose-table:border-collapse prose-th:border prose-th:border-zinc-300 prose-th:dark:border-zinc-600 prose-th:px-3 prose-th:py-1.5 prose-th:bg-zinc-50 prose-th:dark:bg-zinc-800 prose-td:border prose-td:border-zinc-200 prose-td:dark:border-zinc-700 prose-td:px-3 prose-td:py-1.5">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

function ArrowButton({ direction, disabled, onClick }: { direction: "left" | "right"; disabled: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="p-1.5 rounded-md border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
      aria-label={direction === "left" ? "Previous sample" : "Next sample"}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        {direction === "left" ? (
          <polyline points="10,3 5,8 10,13" />
        ) : (
          <polyline points="6,3 11,8 6,13" />
        )}
      </svg>
    </button>
  );
}

function ModelMultiSelect({
  models,
  visibleModels,
  onToggle,
}: {
  models: string[];
  visibleModels: Set<string>;
  onToggle: (model: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const count = visibleModels.size;
  const label = count === models.length ? "All models" : `${count} of ${models.length} models`;

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between gap-2 px-2.5 py-1.5 text-xs border border-zinc-200 dark:border-zinc-700 rounded-md bg-white dark:bg-zinc-900 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors"
      >
        <span className="truncate">{label}</span>
        <svg
          width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor"
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          className={`shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
        >
          <polyline points="3,4.5 6,7.5 9,4.5" />
        </svg>
      </button>
      {open && (
        <div className="absolute z-50 mt-1 left-0 right-0 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-md shadow-lg py-1 max-h-60 overflow-y-auto">
          {models.map((m, i) => (
            <label
              key={m}
              className="flex items-center gap-2 px-2.5 py-1.5 text-xs cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-800"
            >
              <input
                type="checkbox"
                checked={visibleModels.has(m)}
                onChange={() => onToggle(m)}
                className="rounded"
              />
              <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${MODEL_COLORS[i % MODEL_COLORS.length].split(" ")[0]}`} />
              <span className="truncate">{shortModel(m)}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

type ViewMode = "side-by-side" | "stacked";

export function CompareView({ samples, models }: { samples: Sample[]; models: string[] }) {
  const [selected, setSelected] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [visibleModels, setVisibleModels] = useState<Set<string>>(new Set(models));
  const [viewMode, setViewMode] = useState<ViewMode>("side-by-side");
  const sample = samples[selected];

  const go = useCallback(
    (dir: -1 | 1) => setSelected((i) => Math.max(0, Math.min(samples.length - 1, i + dir))),
    [samples.length]
  );

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft" || e.key === "k") go(-1);
      if (e.key === "ArrowRight" || e.key === "j") go(1);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [go]);

  const toggleModel = (model: string) => {
    setVisibleModels((prev) => {
      const next = new Set(prev);
      if (next.has(model)) {
        if (next.size > 1) next.delete(model); // keep at least one
      } else {
        next.add(model);
      }
      return next;
    });
  };

  const activeModels = models.filter((m) => visibleModels.has(m));
  // +1 for the adversarial prompt column
  const totalPanels = activeModels.length + 1;

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside
        className={`shrink-0 border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 flex flex-col transition-[width] duration-200 ${
          sidebarOpen ? "w-72" : "w-0 border-r-0"
        }`}
      >
        <div className={`flex flex-col min-h-0 h-full ${sidebarOpen ? "" : "hidden"}`}>
          <div className="shrink-0 px-4 py-4 border-b border-zinc-200 dark:border-zinc-800 flex items-center justify-between bg-white dark:bg-zinc-950">
            <div>
              <h1 className="text-base font-semibold">Plan Attack</h1>
              <p className="text-xs text-zinc-500 mt-0.5">{samples.length} samples &middot; {models.length} models</p>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="p-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-400"
              aria-label="Collapse sidebar"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="10,3 5,8 10,13" />
              </svg>
            </button>
          </div>

          {/* Model filter */}
          <div className="shrink-0 px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
            <p className="text-xs font-medium text-zinc-500 mb-2">Models</p>
            <ModelMultiSelect models={models} visibleModels={visibleModels} onToggle={toggleModel} />
          </div>

          <nav className="flex-1 overflow-y-auto py-1">
            {samples.map((s, i) => {
              const planCount = activeModels.filter((m) => m in s.plans).length;
              return (
                <button
                  key={s.sample_id}
                  onClick={() => setSelected(i)}
                  className={`w-full text-left px-4 py-3 border-b border-zinc-100 dark:border-zinc-800/50 transition-colors ${
                    i === selected
                      ? "bg-blue-50 dark:bg-blue-950/40 border-l-2 border-l-blue-500"
                      : "hover:bg-zinc-50 dark:hover:bg-zinc-900 border-l-2 border-l-transparent"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-mono text-zinc-400">#{s.sample_id}</span>
                    <Badge domain={s.risk_domain} />
                    {planCount < activeModels.length && (
                      <span className="text-xs text-amber-600 font-medium">{planCount}/{activeModels.length}</span>
                    )}
                  </div>
                  <p className="text-xs text-zinc-600 dark:text-zinc-400 line-clamp-2 leading-snug">
                    {s.adversarial_prompt.slice(0, 120)}...
                  </p>
                </button>
              );
            })}
          </nav>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="shrink-0 px-6 py-4 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {!sidebarOpen && (
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="p-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-400 mr-1"
                  aria-label="Expand sidebar"
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="2" y1="4" x2="14" y2="4" />
                    <line x1="2" y1="8" x2="14" y2="8" />
                    <line x1="2" y1="12" x2="14" y2="12" />
                  </svg>
                </button>
              )}
              <span className="text-sm font-mono text-zinc-400">Sample #{sample.sample_id}</span>
              <Badge domain={sample.risk_domain} />
            </div>
            <div className="flex items-center gap-3">
              {/* View mode toggle */}
              <div className="flex items-center border border-zinc-200 dark:border-zinc-700 rounded-md overflow-hidden">
                <button
                  onClick={() => setViewMode("side-by-side")}
                  className={`px-2.5 py-1 text-xs ${viewMode === "side-by-side" ? "bg-zinc-100 dark:bg-zinc-800 font-medium" : "hover:bg-zinc-50 dark:hover:bg-zinc-900"}`}
                >
                  Side by side
                </button>
                <button
                  onClick={() => setViewMode("stacked")}
                  className={`px-2.5 py-1 text-xs ${viewMode === "stacked" ? "bg-zinc-100 dark:bg-zinc-800 font-medium" : "hover:bg-zinc-50 dark:hover:bg-zinc-900"}`}
                >
                  Stacked
                </button>
              </div>
              <div className="flex items-center gap-1.5">
                <ArrowButton direction="left" disabled={selected === 0} onClick={() => go(-1)} />
                <span className="text-xs text-zinc-400 font-mono min-w-[4ch] text-center">{selected + 1}/{samples.length}</span>
                <ArrowButton direction="right" disabled={selected === samples.length - 1} onClick={() => go(1)} />
              </div>
            </div>
          </div>
          <p className="text-xs text-zinc-500 leading-snug mt-1">{sample.risk_subdomain}</p>
        </header>

        {/* Panels */}
        <div className={`flex-1 ${viewMode === "side-by-side" ? "flex gap-4" : "flex flex-col gap-4 overflow-y-auto"} p-4 ${viewMode === "side-by-side" ? "overflow-hidden min-h-0" : ""}`}>
          <PromptPanel
            title="Adversarial Prompt"
            content={sample.adversarial_prompt}
            color="bg-orange-50 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300"
          />
          {activeModels.map((model, i) => {
            const plan = sample.plans[model];
            if (!plan) return null;
            return (
              <PromptPanel
                key={model}
                title={shortModel(model)}
                content={plan}
                color={MODEL_COLORS[models.indexOf(model) % MODEL_COLORS.length]}
              />
            );
          })}
        </div>
      </main>
    </div>
  );
}
