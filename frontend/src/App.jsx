import { useState, useEffect, useCallback } from "react";
import {
  Sparkles,
  ArrowLeft,
  ArrowRight,
  Equal,
  RotateCcw,
  Download,
  ChevronDown,
  ChevronUp,
  Loader2,
  Database,
  Clock,
  Hash,
  KeyRound,
  Settings,
  BookOpen,
  BarChart3,
  Zap,
  Brain,
  Award,
  ThumbsUp,
  BarChart2,
  GitCompareArrows,
  MessageSquare,
  FlaskConical,
  Vote,
} from "lucide-react";
import axios from "axios";
import { diffWords } from "diff";

const API = "/api";

function App() {
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gpt-4.1-mini");
  const [models, setModels] = useState([]);
  const [temperature, setTemperature] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(512);

  const [prompts, setPrompts] = useState([]);
  const [promptText, setPromptText] = useState("");
  const [promptCategory, setPromptCategory] = useState(null);
  const [showPromptBank, setShowPromptBank] = useState(false);

  const [responseA, setResponseA] = useState(null);
  const [responseB, setResponseB] = useState(null);
  const [fineTuned, setFineTuned] = useState(null);
  const [generating, setGenerating] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const [reasoning, setReasoning] = useState("");
  const [showDiff, setShowDiff] = useState(false);
  const [page, setPage] = useState("collect"); // "collect" or "train"

  const [recordCount, setRecordCount] = useState(0);
  const [sessionId] = useState(() => crypto.randomUUID());

  // Load prompts & models on mount
  useEffect(() => {
    axios.get(`${API}/prompts`).then((r) => setPrompts(r.data));
    axios.get(`${API}/models`).then((r) => {
      setModels(r.data);
      setModel(r.data[0]);
    });
    refreshCount();
  }, []);

  const refreshCount = () => {
    axios.get(`${API}/stats`).then((r) => setRecordCount(r.data.count));
  };

  const handleGenerate = async () => {
    if (!apiKey) return setError("Enter your OpenAI API key.");
    if (!promptText.trim()) return setError("Enter a prompt first.");
    setError(null);
    setGenerating(true);
    try {
      const { data } = await axios.post(`${API}/generate`, {
        api_key: apiKey,
        prompt: promptText,
        model,
        temperature,
        max_tokens: maxTokens,
      });
      setResponseA(data.response_a);
      setResponseB(data.response_b);
      setFineTuned(data.fine_tuned || null);
    } catch (e) {
      setError(e.response?.data?.detail || "Generation failed.");
    } finally {
      setGenerating(false);
    }
  };

  const handlePreference = async (pref) => {
    setSubmitting(true);
    try {
      await axios.post(`${API}/preference`, {
        prompt: promptText,
        response_a: responseA,
        response_b: responseB,
        preference: pref,
        model_name: model,
        temperature,
        max_tokens: maxTokens,
        prompt_category: promptCategory,
        session_id: sessionId,
        reasoning: reasoning || null,
      });
      setSubmitted(true);
      refreshCount();
    } catch (e) {
      setError(e.response?.data?.detail || "Failed to save preference.");
    } finally {
      setSubmitting(false);
    }
  };

  const resetAll = () => {
    setPromptText("");
    setPromptCategory(null);
    setResponseA(null);
    setResponseB(null);
    setFineTuned(null);
    setSubmitted(false);
    setReasoning("");
    setShowDiff(false);
    setError(null);
  };

  const selectPrompt = (p) => {
    setPromptText(p.prompt);
    setPromptCategory(p.category);
    setResponseA(null);
    setResponseB(null);
    setFineTuned(null);
    setSubmitted(false);
    setReasoning("");
    setShowDiff(false);
    setError(null);
    setShowPromptBank(false);
  };

  const hasResponses = responseA && responseB;

  return (
    <>
    <div className={`${page === "train" ? "hidden" : ""} min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white`}>
      {/* Header */}
      <header className="border-b border-white/[0.06] glass sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="h-11 w-11 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 ring-1 ring-white/10">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white to-slate-300 text-gradient">Preference Collector</h1>
              <p className="text-xs text-slate-500">Human-in-the-loop RLHF data collection</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 glass border border-white/[0.06] rounded-xl px-3.5 py-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
              </span>
              <span className="text-sm font-mono font-semibold text-emerald-400">{recordCount}</span>
              <span className="text-xs text-slate-500">records</span>
            </div>
            <button
              onClick={() => setPage("train")}
              className="flex items-center gap-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 shadow-lg shadow-amber-600/20 transition-all rounded-xl px-4 py-2.5 text-sm font-semibold"
            >
              <FlaskConical className="h-4 w-4" />
              Training Lab
            </button>
            <a
              href={`${API}/export/training-pairs`}
              target="_blank"
              className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 shadow-lg shadow-indigo-600/20 transition-all rounded-xl px-4 py-2.5 text-sm font-semibold"
            >
              <Download className="h-4 w-4" />
              Export
            </a>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sidebar */}
        <aside className="lg:col-span-1 space-y-6">
          {/* API Key */}
          <div className="glass border border-white/[0.06] rounded-2xl p-5 space-y-4 ring-1 ring-inset ring-white/[0.04]">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-300">
              <KeyRound className="h-4 w-4 text-indigo-400" />
              API Key
            </div>
            <input
              type="password"
              placeholder="sk-..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="w-full bg-black/20 border border-white/[0.06] rounded-xl px-3.5 py-2.5 text-sm placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500/30 transition-all"
            />
          </div>

          {/* Model Settings */}
          <div className="glass border border-white/[0.06] rounded-2xl p-5 space-y-4 ring-1 ring-inset ring-white/[0.04]">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-300">
              <Settings className="h-4 w-4 text-indigo-400" />
              Model Settings
            </div>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Model</label>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  disabled={hasResponses}
                  className="w-full bg-black/20 border border-white/[0.06] rounded-xl px-3.5 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/40 disabled:opacity-50 transition-all"
                >
                  {models.map((m) => (
                    <option key={m} value={m} className="bg-slate-900">
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">
                  Temperature: <span className="text-indigo-400 font-mono">{temperature.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  disabled={hasResponses}
                  className="w-full accent-indigo-500"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">
                  Max tokens: <span className="text-indigo-400 font-mono">{maxTokens}</span>
                </label>
                <input
                  type="range"
                  min="64"
                  max="1024"
                  step="64"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  disabled={hasResponses}
                  className="w-full accent-indigo-500"
                />
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="glass border border-white/[0.06] rounded-2xl p-5 space-y-3 ring-1 ring-inset ring-white/[0.04]">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-300">
              <BarChart3 className="h-4 w-4 text-indigo-400" />
              Dataset
            </div>
            <div className="grid grid-cols-1 gap-2">
              <div className="bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border border-indigo-500/20 rounded-xl p-4 text-center">
                <div className="text-3xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 text-gradient font-mono">{recordCount}</div>
                <div className="text-xs text-slate-500 mt-1">preferences collected</div>
              </div>
            </div>
            <div className="flex gap-2">
              <a
                href={`${API}/export/training-pairs`}
                target="_blank"
                className="flex-1 flex items-center justify-center gap-1.5 bg-indigo-500/10 border border-indigo-500/20 hover:bg-indigo-500/20 transition-all rounded-xl px-2 py-2.5 text-xs font-semibold text-indigo-300"
              >
                <Download className="h-3 w-3" />
                JSON
              </a>
              <a
                href={`${API}/export/csv`}
                download="preferences.csv"
                className="flex-1 flex items-center justify-center gap-1.5 bg-emerald-500/10 border border-emerald-500/20 hover:bg-emerald-500/20 transition-all rounded-xl px-2 py-2.5 text-xs font-semibold text-emerald-300"
              >
                <Download className="h-3 w-3" />
                CSV
              </a>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="lg:col-span-3 space-y-6">
          {/* Prompt Input */}
          <div className="glass border border-white/[0.06] rounded-2xl p-6 space-y-4 ring-1 ring-inset ring-white/[0.04]">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Zap className="h-5 w-5 text-amber-400" />
                <span className="bg-gradient-to-r from-white to-slate-400 text-gradient">Prompt</span>
              </h2>
              <button
                onClick={() => setShowPromptBank(!showPromptBank)}
                className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
              >
                <BookOpen className="h-3.5 w-3.5" />
                Prompt Bank
                {showPromptBank ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              </button>
            </div>

            {/* Prompt bank */}
            {showPromptBank && (
              <div className="max-h-64 overflow-y-auto space-y-2 border border-white/10 rounded-xl p-3 bg-black/20">
                {prompts.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => selectPrompt(p)}
                    className="w-full text-left p-3 rounded-lg bg-white/5 hover:bg-indigo-600/20 border border-transparent hover:border-indigo-500/30 transition-all text-sm text-slate-300 hover:text-white"
                  >
                    <span className="text-xs font-mono text-indigo-400 mr-2">#{p.id}</span>
                    {p.prompt}
                  </button>
                ))}
              </div>
            )}

            <textarea
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              placeholder="Enter your prompt or select from the prompt bank..."
              rows={4}
              disabled={hasResponses}
              className="w-full bg-black/20 border border-white/[0.06] rounded-xl px-4 py-3.5 text-sm leading-relaxed resize-none placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:border-indigo-500/30 disabled:opacity-60 transition-all"
            />

            <div className="flex gap-3">
              <button
                onClick={handleGenerate}
                disabled={generating || hasResponses}
                className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl px-4 py-3.5 text-sm font-semibold shadow-lg shadow-indigo-600/20 hover:shadow-indigo-500/30 transition-all"
              >
                {generating ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Generate Two Responses
                  </>
                )}
              </button>
              <button
                onClick={resetAll}
                className="flex items-center gap-2 glass border border-white/[0.06] hover:bg-white/[0.06] rounded-xl px-4 py-3.5 text-sm font-medium transition-all"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-sm text-red-400">
                {error}
              </div>
            )}
          </div>

          {/* Fine-tuned response from prior preferences */}
          {hasResponses && fineTuned && <FineTunedCard data={fineTuned} />}

          {/* Responses side by side */}
          {hasResponses && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <ResponseCard label="Response A" response={responseA} color="blue" />
                <ResponseCard label="Response B" response={responseB} color="violet" />
              </div>

              {/* Diff toggle */}
              <div className="flex justify-center">
                <button
                  onClick={() => setShowDiff(!showDiff)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium border transition-all ${
                    showDiff
                      ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                      : "bg-white/5 border-white/10 text-slate-400 hover:text-white hover:bg-white/10"
                  }`}
                >
                  <GitCompareArrows className="h-4 w-4" />
                  {showDiff ? "Hide" : "Show"} Word Diff
                </button>
              </div>

              {/* Diff view */}
              {showDiff && <DiffView textA={responseA.text} textB={responseB.text} />}

              {/* Reasoning box */}
              {!submitted && (
                <div className="bg-white/5 border border-white/10 rounded-2xl p-5 space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-300">
                    <MessageSquare className="h-4 w-4" />
                    Reasoning <span className="text-xs font-normal text-slate-500">(optional)</span>
                  </div>
                  <textarea
                    value={reasoning}
                    onChange={(e) => setReasoning(e.target.value)}
                    placeholder="Why do you prefer one response over the other? What made you choose?"
                    rows={3}
                    className="w-full bg-black/20 border border-white/10 rounded-xl px-4 py-3 text-sm resize-none placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all"
                  />
                </div>
              )}

              {/* Preference Buttons */}
              <div className="glass border border-white/[0.06] rounded-2xl p-6 ring-1 ring-inset ring-white/[0.04]">
                {submitted ? (
                  <div className="text-center space-y-3">
                    <div className="inline-flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/30 rounded-full px-5 py-2.5 text-emerald-400 font-semibold shadow-lg shadow-emerald-500/10">
                      <Sparkles className="h-4 w-4" />
                      Preference recorded!
                    </div>
                    <p className="text-sm text-slate-400">
                      Click <span className="font-semibold text-white">Reset</span> to label another pair.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <h3 className="text-center text-lg font-semibold">Which response do you prefer?</h3>
                    <div className="grid grid-cols-3 gap-4">
                      <button
                        onClick={() => handlePreference("A")}
                        disabled={submitting}
                        className="group flex flex-col items-center gap-2.5 bg-blue-600/10 border-2 border-blue-500/20 hover:border-blue-400 hover:bg-blue-600/20 hover:shadow-lg hover:shadow-blue-500/10 rounded-2xl p-5 transition-all disabled:opacity-40 cursor-pointer"
                      >
                        <ArrowLeft className="h-6 w-6 text-blue-400 group-hover:scale-110 transition-transform" />
                        <span className="text-sm font-semibold text-blue-300">Prefer A</span>
                      </button>
                      <button
                        onClick={() => handlePreference("tie")}
                        disabled={submitting}
                        className="group flex flex-col items-center gap-2.5 bg-slate-600/10 border-2 border-slate-500/20 hover:border-slate-400 hover:bg-slate-600/20 hover:shadow-lg hover:shadow-slate-500/10 rounded-2xl p-5 transition-all disabled:opacity-40 cursor-pointer"
                      >
                        <Equal className="h-6 w-6 text-slate-400 group-hover:scale-110 transition-transform" />
                        <span className="text-sm font-semibold text-slate-300">Tie</span>
                      </button>
                      <button
                        onClick={() => handlePreference("B")}
                        disabled={submitting}
                        className="group flex flex-col items-center gap-2.5 bg-violet-600/10 border-2 border-violet-500/20 hover:border-violet-400 hover:bg-violet-600/20 hover:shadow-lg hover:shadow-violet-500/10 rounded-2xl p-5 transition-all disabled:opacity-40 cursor-pointer"
                      >
                        <ArrowRight className="h-6 w-6 text-violet-400 group-hover:scale-110 transition-transform" />
                        <span className="text-sm font-semibold text-violet-300">Prefer B</span>
                      </button>
                    </div>
                    {submitting && (
                      <div className="flex items-center justify-center gap-2 text-sm text-slate-400">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Saving...
                      </div>
                    )}
                  </div>
                )}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
    <div className={page === "train" ? "" : "hidden"}>
      <TrainingPage onBack={() => setPage("collect")} apiKey={apiKey} visible={page === "train"} />
    </div>
    </>
  );
}

function FineTunedCard({ data }) {
  const breakdown = data.preference_breakdown || {};
  const total = data.total_votes || 0;

  return (
    <div className="bg-gradient-to-r from-emerald-500/5 to-teal-500/5 border-2 border-emerald-500/30 rounded-2xl p-6 space-y-4 relative overflow-hidden">
      {/* Glow effect */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />

      <div className="flex items-center justify-between relative">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
            <Brain className="h-5 w-5 text-white" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-emerald-400 flex items-center gap-2">
              <Award className="h-4 w-4" />
              Fine-Tuned Response
            </h3>
            <p className="text-xs text-slate-400">
              Based on {total} prior preference{total !== 1 ? "s" : ""} for this prompt
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-3 py-1">
          <ThumbsUp className="h-3.5 w-3.5 text-emerald-400" />
          <span className="text-xs font-semibold text-emerald-400">Human-Preferred</span>
        </div>
      </div>

      {/* Vote breakdown bar */}
      <div className="space-y-2 relative">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <BarChart2 className="h-3.5 w-3.5" />
          <span>Preference Breakdown</span>
        </div>
        <div className="flex h-2.5 rounded-full overflow-hidden bg-white/5">
          {breakdown.A > 0 && (
            <div
              className="bg-blue-500 transition-all"
              style={{ width: `${(breakdown.A / total) * 100}%` }}
              title={`Prefer A: ${breakdown.A}`}
            />
          )}
          {breakdown.tie > 0 && (
            <div
              className="bg-slate-500 transition-all"
              style={{ width: `${(breakdown.tie / total) * 100}%` }}
              title={`Tie: ${breakdown.tie}`}
            />
          )}
          {breakdown.B > 0 && (
            <div
              className="bg-violet-500 transition-all"
              style={{ width: `${(breakdown.B / total) * 100}%` }}
              title={`Prefer B: ${breakdown.B}`}
            />
          )}
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-blue-400">A: {breakdown.A || 0}</span>
          <span className="text-slate-400">Tie: {breakdown.tie || 0}</span>
          <span className="text-violet-400">B: {breakdown.B || 0}</span>
        </div>
      </div>

      {/* The actual preferred response */}
      <div className="bg-black/20 border border-emerald-500/10 rounded-xl p-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-mono text-emerald-400/70">
            via {data.model_name} &middot; {new Date(data.timestamp).toLocaleDateString()}
          </span>
        </div>
        <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto pr-2">
          {data.text}
        </div>
      </div>

      <p className="text-xs text-slate-500 italic relative">
        This is the response humans previously preferred. Compare it with the fresh generations below.
      </p>
    </div>
  );
}

function DiffView({ textA, textB }) {
  const parts = diffWords(textA, textB);

  return (
    <div className="bg-white/5 border border-amber-500/20 rounded-2xl p-5 space-y-3">
      <div className="flex items-center gap-2 text-sm font-semibold text-amber-400">
        <GitCompareArrows className="h-4 w-4" />
        Word-Level Diff
        <span className="text-xs font-normal text-slate-500 ml-1">
          <span className="text-red-400">red = only in A</span> &middot;{" "}
          <span className="text-emerald-400">green = only in B</span>
        </span>
      </div>
      <div className="text-sm leading-relaxed whitespace-pre-wrap max-h-80 overflow-y-auto pr-2">
        {parts.map((part, i) => {
          if (part.added) {
            return (
              <span key={i} className="bg-emerald-500/20 text-emerald-300 rounded px-0.5">
                {part.value}
              </span>
            );
          }
          if (part.removed) {
            return (
              <span key={i} className="bg-red-500/20 text-red-300 line-through rounded px-0.5">
                {part.value}
              </span>
            );
          }
          return (
            <span key={i} className="text-slate-400">
              {part.value}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function TrainingPage({ onBack, apiKey, visible }) {
  const [tab, setTab] = useState("rlhf"); // "rlhf" or "dpo"
  const [trainingData, setTrainingData] = useState([]);
  const [loading, setLoading] = useState(false);

  // RLHF state
  const [rlhfTraining, setRlhfTraining] = useState(false);
  const [rlhfLog, setRlhfLog] = useState([]);
  const [rmEpochs, setRmEpochs] = useState(3);
  const [ppoEpochs, setPpoEpochs] = useState(3);
  const [rmLossHistory, setRmLossHistory] = useState([]);
  const [rmAccHistory, setRmAccHistory] = useState([]);
  const [ppoRewardHistory, setPpoRewardHistory] = useState([]);
  const [ppoKlHistory, setPpoKlHistory] = useState([]);
  const [rlhfDone, setRlhfDone] = useState(false);

  // DPO state
  const [dpoTraining, setDpoTraining] = useState(false);
  const [dpoLog, setDpoLog] = useState([]);
  const [dpoEpochs, setDpoEpochs] = useState(3);
  const [dpoLossHistory, setDpoLossHistory] = useState([]);
  const [dpoDone, setDpoDone] = useState(false);

  // Unified comparison state (RLHF vs DPO side by side)
  const [comparePrompt, setComparePrompt] = useState("");
  const [comparing, setComparing] = useState(false);
  const [rlhfResponse, setRlhfResponse] = useState(null);
  const [dpoResponse, setDpoResponse] = useState(null);
  const [compareError, setCompareError] = useState(null);

  useEffect(() => {
    if (!visible) return;
    setLoading(true);
    axios
      .get(`${API}/export/training-pairs`)
      .then((r) => setTrainingData(r.data))
      .finally(() => setLoading(false));
  }, [visible]);

  // RLHF training
  const startRLHF = async () => {
    if (trainingData.length === 0) return;
    setRlhfTraining(true);
    setRlhfLog([]);
    setRmLossHistory([]);
    setRmAccHistory([]);
    setPpoRewardHistory([]);
    setPpoKlHistory([]);
    setRlhfDone(false);
    try {
      const { data } = await axios.post(`${API}/rlhf/train`, {
        rm_epochs: rmEpochs,
        ppo_epochs: ppoEpochs,
      });
      setRlhfLog(data.log || []);
      setRmLossHistory(data.rm_loss_history || []);
      setRmAccHistory(data.rm_accuracy_history || []);
      setPpoRewardHistory(data.ppo_reward_history || []);
      setPpoKlHistory(data.ppo_kl_history || []);
      setRlhfDone(true);
    } catch (e) {
      setRlhfLog((prev) => [...prev, `ERROR: ${e.response?.data?.detail || e.message}`]);
    } finally {
      setRlhfTraining(false);
    }
  };

  // DPO training
  const startDPO = async () => {
    if (trainingData.length === 0) return;
    setDpoTraining(true);
    setDpoLog([]);
    setDpoLossHistory([]);
    setDpoDone(false);
    try {
      const { data } = await axios.post(`${API}/dpo/train`, { epochs: dpoEpochs });
      setDpoLog(data.log || []);
      setDpoLossHistory(data.loss_history || []);
      setDpoDone(true);
    } catch (e) {
      setDpoLog((prev) => [...prev, `ERROR: ${e.response?.data?.detail || e.message}`]);
    } finally {
      setDpoTraining(false);
    }
  };

  // Unified comparison: fire both RLHF and DPO in parallel
  const runComparison = async () => {
    if (!comparePrompt.trim()) return;
    setComparing(true);
    setRlhfResponse(null);
    setDpoResponse(null);
    setCompareError(null);
    try {
      const [rlhfRes, dpoRes] = await Promise.all([
        axios.post(`${API}/rlhf/compare`, { prompt: comparePrompt, api_key: apiKey }),
        axios.post(`${API}/dpo/compare`, { prompt: comparePrompt, api_key: apiKey }),
      ]);
      setRlhfResponse(rlhfRes.data.aligned_response);
      setDpoResponse(dpoRes.data.aligned_response);
    } catch (e) {
      setCompareError(e.response?.data?.detail || e.message);
    } finally {
      setComparing(false);
    }
  };

  const decisive = trainingData.filter((d) => d.preference !== "tie").length;
  const uniquePrompts = [...new Set(trainingData.map((d) => d.prompt))].length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-amber-950 text-white">
      <header className="border-b border-white/[0.06] glass sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button onClick={onBack} className="flex items-center gap-2 text-sm text-slate-500 hover:text-white transition-all hover:-translate-x-0.5">
              <ArrowLeft className="h-4 w-4" /> Back
            </button>
            <div className="h-6 w-px bg-white/[0.06]" />
            <div className="flex items-center gap-3">
              <div className="h-11 w-11 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg shadow-amber-500/20 ring-1 ring-white/10">
                <FlaskConical className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white to-slate-300 text-gradient">Training Lab</h1>
                <p className="text-xs text-slate-500">RLHF &amp; DPO training from your preference data</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* RLHF vs DPO Explanation */}
        <div className="glass border border-white/[0.06] rounded-2xl p-6 space-y-5 ring-1 ring-inset ring-white/[0.04]">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="h-5 w-5 text-amber-400" />
            <span className="bg-gradient-to-r from-white to-slate-400 text-gradient">RLHF vs DPO: Two Approaches to Alignment</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gradient-to-br from-indigo-500/[0.04] to-indigo-600/[0.08] border border-indigo-500/20 rounded-xl p-5 space-y-3 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-indigo-400 to-indigo-600 rounded-r" />
              <h3 className="text-sm font-bold text-indigo-400 ml-2">RLHF (Reinforcement Learning from Human Feedback)</h3>
              <div className="text-xs text-slate-400 leading-relaxed space-y-1.5 ml-2">
                <p><strong className="text-slate-300">Step 1:</strong> Supervised Fine-Tuning (SFT) on demonstrations</p>
                <p><strong className="text-slate-300">Step 2:</strong> Train a <strong className="text-indigo-300">Reward Model</strong> from preference pairs to score responses</p>
                <p><strong className="text-slate-300">Step 3:</strong> Optimize the policy with <strong className="text-indigo-300">PPO</strong> to maximize the reward while staying close to the SFT model (KL penalty)</p>
                <p className="pt-1 text-slate-500 italic">More complex, can be unstable, but well-proven (ChatGPT, Claude)</p>
              </div>
            </div>
            <div className="bg-gradient-to-br from-amber-500/[0.04] to-amber-600/[0.08] border border-amber-500/20 rounded-xl p-5 space-y-3 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-amber-400 to-orange-600 rounded-r" />
              <h3 className="text-sm font-bold text-amber-400 ml-2">DPO (Direct Preference Optimization)</h3>
              <div className="text-xs text-slate-400 leading-relaxed space-y-1.5 ml-2">
                <p><strong className="text-slate-300">Step 1:</strong> Supervised Fine-Tuning (SFT) on demonstrations</p>
                <p><strong className="text-slate-300">Step 2:</strong> Directly optimize policy on preference pairs using a <strong className="text-amber-300">classification loss</strong></p>
                <p><strong className="text-slate-300">No RL needed:</strong> Derives the optimal policy <em>analytically</em> from the reward model formulation</p>
                <p className="pt-1 text-slate-500 italic">Simpler, more stable, same theoretical optimum (Llama, Zephyr)</p>
              </div>
            </div>
          </div>
          <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-3.5 text-xs text-slate-400">
            <strong className="text-slate-300">Key difference:</strong> RLHF trains a separate reward model then runs RL (PPO) against it. DPO skips the reward model entirely and optimizes the policy directly on preference pairs. Both use the same human preference data you collected.
          </div>
        </div>

        {/* Dataset Summary */}
        <div className="glass border border-white/[0.06] rounded-2xl p-6 space-y-4 ring-1 ring-inset ring-white/[0.04]">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Database className="h-5 w-5 text-amber-400" />
            <span className="bg-gradient-to-r from-white to-slate-400 text-gradient">Training Dataset</span>
          </h2>
          {loading ? (
            <div className="flex items-center gap-2 text-sm text-slate-400"><Loader2 className="h-4 w-4 animate-spin" /> Loading...</div>
          ) : (
            <>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-amber-500/10 to-amber-600/5 border border-amber-500/20 rounded-xl p-4 text-center">
                  <div className="text-2xl font-bold bg-gradient-to-r from-amber-400 to-orange-400 text-gradient font-mono">{trainingData.length}</div>
                  <div className="text-xs text-slate-500 mt-1">training pairs</div>
                </div>
                <div className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border border-emerald-500/20 rounded-xl p-4 text-center">
                  <div className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-teal-400 text-gradient font-mono">{decisive}</div>
                  <div className="text-xs text-slate-500 mt-1">decisive (non-tie)</div>
                </div>
                <div className="bg-gradient-to-br from-indigo-500/10 to-indigo-600/5 border border-indigo-500/20 rounded-xl p-4 text-center">
                  <div className="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 text-gradient font-mono">{uniquePrompts}</div>
                  <div className="text-xs text-slate-500 mt-1">unique prompts</div>
                </div>
              </div>
              {trainingData.length > 0 && (
                <div className="max-h-40 overflow-y-auto border border-white/[0.06] rounded-xl">
                  <table className="w-full text-xs">
                    <thead className="bg-white/5 sticky top-0">
                      <tr>
                        <th className="text-left p-2 text-slate-400 font-medium">#</th>
                        <th className="text-left p-2 text-slate-400 font-medium">Prompt</th>
                        <th className="text-left p-2 text-slate-400 font-medium">Vote</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trainingData.slice(0, 50).map((d, i) => (
                        <tr key={i} className="border-t border-white/5">
                          <td className="p-2 text-slate-500 font-mono">{i + 1}</td>
                          <td className="p-2 text-slate-300 max-w-md truncate">{d.prompt}</td>
                          <td className="p-2">
                            <span className={`px-2 py-0.5 rounded text-xs font-mono ${d.preference === "A" ? "bg-blue-500/10 text-blue-400" : d.preference === "B" ? "bg-violet-500/10 text-violet-400" : "bg-slate-500/10 text-slate-400"}`}>
                              {d.preference}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>

        {/* Tab Switcher */}
        <div className="flex gap-3">
          <button
            onClick={() => setTab("rlhf")}
            className={`flex-1 flex items-center justify-center gap-2.5 py-3.5 rounded-2xl text-sm font-semibold border-2 transition-all ${
              tab === "rlhf"
                ? "bg-indigo-500/10 border-indigo-500/30 text-indigo-400 shadow-lg shadow-indigo-500/10"
                : "glass border-white/[0.06] text-slate-500 hover:text-white hover:bg-white/[0.04]"
            }`}
          >
            <Zap className="h-4 w-4" />
            RLHF Pipeline
            <span className="text-xs opacity-50">(RM + PPO)</span>
          </button>
          <button
            onClick={() => setTab("dpo")}
            className={`flex-1 flex items-center justify-center gap-2.5 py-3.5 rounded-2xl text-sm font-semibold border-2 transition-all ${
              tab === "dpo"
                ? "bg-amber-500/10 border-amber-500/30 text-amber-400 shadow-lg shadow-amber-500/10"
                : "glass border-white/[0.06] text-slate-500 hover:text-white hover:bg-white/[0.04]"
            }`}
          >
            <FlaskConical className="h-4 w-4" />
            DPO
            <span className="text-xs opacity-50">(Direct Preference)</span>
          </button>
        </div>

        {/* ============ RLHF TAB ============ */}
        <div className={`space-y-6 ${tab === "rlhf" ? "" : "hidden"}`}>
            <div className="bg-gradient-to-br from-indigo-500/[0.04] to-indigo-600/[0.08] border border-indigo-500/20 rounded-2xl p-6 space-y-4 ring-1 ring-inset ring-indigo-500/[0.06]">
              <h2 className="text-lg font-semibold flex items-center gap-2 text-indigo-400">
                <Zap className="h-5 w-5" />
                RLHF Training
              </h2>
              <div className="flex items-center gap-4 flex-wrap">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">RM Epochs</label>
                  <input type="number" min="1" max="10" value={rmEpochs} onChange={(e) => setRmEpochs(parseInt(e.target.value) || 3)} disabled={rlhfTraining}
                    className="w-20 bg-black/20 border border-white/[0.06] rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/40 disabled:opacity-50" />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">PPO Epochs</label>
                  <input type="number" min="1" max="10" value={ppoEpochs} onChange={(e) => setPpoEpochs(parseInt(e.target.value) || 3)} disabled={rlhfTraining}
                    className="w-20 bg-black/20 border border-white/[0.06] rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/40 disabled:opacity-50" />
                </div>
                <button onClick={startRLHF} disabled={rlhfTraining || trainingData.length === 0}
                  className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl px-6 py-3 text-sm font-semibold shadow-lg shadow-indigo-600/20 transition-all mt-4 sm:mt-0">
                  {rlhfTraining ? <><Loader2 className="h-4 w-4 animate-spin" /> Training...</> : <><Zap className="h-4 w-4" /> Start RLHF</>}
                </button>
              </div>

              {/* RM Loss Chart */}
              {rmLossHistory.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-slate-300">Phase 1: Reward Model Loss</h3>
                  <BarChart data={rmLossHistory} valueKey="loss" color="indigo" />
                </div>
              )}

              {/* RM Accuracy Chart */}
              {rmAccHistory.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-slate-300">Phase 1: Reward Model Accuracy</h3>
                  <BarChart data={rmAccHistory} valueKey="accuracy" color="emerald" label="acc" />
                </div>
              )}

              {/* PPO Reward Chart */}
              {ppoRewardHistory.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-slate-300">Phase 2: PPO Average Reward</h3>
                  <BarChart data={ppoRewardHistory} valueKey="reward" color="purple" />
                </div>
              )}

              {/* PPO KL Chart */}
              {ppoKlHistory.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-slate-300">Phase 2: PPO KL Divergence (from SFT)</h3>
                  <BarChart data={ppoKlHistory} valueKey="kl" color="rose" label="KL" />
                </div>
              )}

              <TrainingLog log={rlhfLog} />
            </div>
        </div>

        {/* ============ DPO TAB ============ */}
        <div className={`space-y-6 ${tab === "dpo" ? "" : "hidden"}`}>
            <div className="bg-gradient-to-br from-amber-500/[0.04] to-amber-600/[0.08] border border-amber-500/20 rounded-2xl p-6 space-y-4 ring-1 ring-inset ring-amber-500/[0.06]">
              <h2 className="text-lg font-semibold flex items-center gap-2 text-amber-400">
                <FlaskConical className="h-5 w-5" />
                DPO Training
              </h2>
              <div className="flex items-center gap-4">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Epochs</label>
                  <input type="number" min="1" max="10" value={dpoEpochs} onChange={(e) => setDpoEpochs(parseInt(e.target.value) || 3)} disabled={dpoTraining}
                    className="w-20 bg-black/20 border border-white/[0.06] rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/40 disabled:opacity-50" />
                </div>
                <button onClick={startDPO} disabled={dpoTraining || trainingData.length === 0}
                  className="flex items-center gap-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl px-6 py-3 text-sm font-semibold shadow-lg shadow-amber-600/20 transition-all">
                  {dpoTraining ? <><Loader2 className="h-4 w-4 animate-spin" /> Training...</> : <><Zap className="h-4 w-4" /> Start DPO</>}
                </button>
              </div>

              {/* DPO Loss Chart */}
              {dpoLossHistory.length > 0 && (
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold text-slate-300">DPO Loss (implicit reward margin)</h3>
                  <BarChart data={dpoLossHistory} valueKey="loss" color="amber" />
                </div>
              )}

              <TrainingLog log={dpoLog} />
            </div>
        </div>

        {/* ============ RLHF vs DPO COMPARISON ============ */}
        {rlhfDone && dpoDone && (
          <div className="glass border border-white/[0.06] rounded-2xl p-6 space-y-5 ring-1 ring-inset ring-white/[0.04]">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Vote className="h-5 w-5 text-amber-400" />
              <span className="bg-gradient-to-r from-indigo-400 to-amber-400 text-gradient">RLHF vs DPO Comparison</span>
            </h2>
            <p className="text-sm text-slate-500">
              Enter a prompt to see how RLHF-aligned and DPO-aligned responses differ.
            </p>
            <div className="flex gap-3">
              <input value={comparePrompt} onChange={(e) => setComparePrompt(e.target.value)} placeholder="Enter a test prompt..."
                className="flex-1 bg-black/20 border border-white/[0.06] rounded-xl px-4 py-3 text-sm placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-amber-500/40 transition-all" />
              <button onClick={runComparison} disabled={comparing || !comparePrompt.trim() || !apiKey}
                className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-amber-600 hover:from-indigo-500 hover:to-amber-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl px-6 py-3 text-sm font-semibold shadow-lg shadow-indigo-500/10 transition-all">
                {comparing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />} Compare
              </button>
            </div>
            {!apiKey && <p className="text-xs text-red-400">Go back and enter your OpenAI API key first.</p>}
            {compareError && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-sm text-red-400">{compareError}</div>
            )}
            {rlhfResponse && dpoResponse && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gradient-to-br from-indigo-500/[0.04] to-indigo-600/[0.08] border border-indigo-500/20 rounded-2xl p-5 space-y-3 relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-indigo-400 to-purple-600 rounded-r" />
                  <div className="flex items-center gap-2 ml-1">
                    <div className="h-2 w-2 rounded-full bg-indigo-400" />
                    <span className="text-sm font-bold text-indigo-400">RLHF-Aligned</span>
                    <span className="text-xs bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 rounded-lg px-2 py-0.5">RM + PPO</span>
                  </div>
                  <div className="h-px bg-gradient-to-r from-transparent via-indigo-500/10 to-transparent" />
                  <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap max-h-80 overflow-y-auto">{rlhfResponse}</div>
                </div>
                <div className="bg-gradient-to-br from-amber-500/[0.04] to-amber-600/[0.08] border border-amber-500/20 rounded-2xl p-5 space-y-3 relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-amber-400 to-orange-600 rounded-r" />
                  <div className="flex items-center gap-2 ml-1">
                    <div className="h-2 w-2 rounded-full bg-amber-400" />
                    <span className="text-sm font-bold text-amber-400">DPO-Aligned</span>
                    <span className="text-xs bg-amber-500/10 border border-amber-500/20 text-amber-400 rounded-lg px-2 py-0.5">Direct Preference</span>
                  </div>
                  <div className="h-px bg-gradient-to-r from-transparent via-amber-500/10 to-transparent" />
                  <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap max-h-80 overflow-y-auto">{dpoResponse}</div>
                </div>
              </div>
            )}
          </div>
        )}

        {(rlhfDone && !dpoDone) && (
          <div className="glass border border-amber-500/20 rounded-xl px-4 py-3.5 text-sm text-amber-400/70 flex items-center gap-2">
            <FlaskConical className="h-4 w-4" />
            Run DPO training too to unlock the RLHF vs DPO comparison.
          </div>
        )}
        {(!rlhfDone && dpoDone) && (
          <div className="glass border border-indigo-500/20 rounded-xl px-4 py-3.5 text-sm text-indigo-400/70 flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Run RLHF training too to unlock the RLHF vs DPO comparison.
          </div>
        )}
      </div>
    </div>
  );
}

/* Reusable bar chart for training metrics */
function BarChart({ data, valueKey, color, label }) {
  if (!data || data.length === 0) return null;
  const values = data.map((d) => d[valueKey]);
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  const displayLabel = label || valueKey;

  const colorMap = {
    indigo: "from-indigo-500 to-indigo-400",
    amber: "from-amber-500 to-amber-400",
    emerald: "from-emerald-500 to-emerald-400",
    purple: "from-purple-500 to-purple-400",
    rose: "from-rose-500 to-rose-400",
  };
  const textColorMap = {
    indigo: "text-indigo-400",
    amber: "text-amber-400",
    emerald: "text-emerald-400",
    purple: "text-purple-400",
    rose: "text-rose-400",
  };

  return (
    <>
      <div className="bg-black/20 border border-white/10 rounded-xl p-4 h-40 flex items-end gap-px">
        {data.map((point, i) => {
          const height = ((point[valueKey] - min) / range) * 100;
          return (
            <div
              key={i}
              className={`flex-1 bg-gradient-to-t ${colorMap[color]} rounded-t-sm opacity-80 hover:opacity-100 transition-opacity relative group`}
              style={{ height: `${Math.max(height, 2)}%` }}
            >
              <div className={`absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 border border-white/10 rounded px-2 py-0.5 text-xs font-mono ${textColorMap[color]} hidden group-hover:block whitespace-nowrap z-10`}>
                {displayLabel} {point[valueKey].toFixed(4)}
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-xs text-slate-500 font-mono">
        <span>step 0</span>
        <span>step {data.length - 1}</span>
      </div>
    </>
  );
}

/* Reusable training log display */
function TrainingLog({ log }) {
  if (!log || log.length === 0) return null;
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-slate-300">Training Log</h3>
      <div className="bg-black/30 border border-white/10 rounded-xl p-4 max-h-48 overflow-y-auto font-mono text-xs space-y-0.5">
        {log.map((line, i) => (
          <div key={i} className={line.startsWith("ERROR") ? "text-red-400" : line.startsWith("\u2713") || line.includes("complete") ? "text-emerald-400" : line.startsWith("=") ? "text-slate-300 font-bold" : "text-slate-400"}>
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}


function ResponseCard({ label, response, color }) {
  const isBlue = color === "blue";
  const borderColor = isBlue ? "border-blue-500/20" : "border-violet-500/20";
  const bgColor = isBlue ? "bg-gradient-to-br from-blue-500/[0.03] to-blue-600/[0.06]" : "bg-gradient-to-br from-violet-500/[0.03] to-violet-600/[0.06]";
  const labelColor = isBlue ? "text-blue-400" : "text-violet-400";
  const badgeBg = isBlue ? "bg-blue-500/10 border-blue-500/20" : "bg-violet-500/10 border-violet-500/20";
  const glowColor = isBlue ? "shadow-blue-500/5" : "shadow-violet-500/5";
  const ringColor = isBlue ? "ring-blue-500/[0.06]" : "ring-violet-500/[0.06]";

  return (
    <div className={`${bgColor} border ${borderColor} rounded-2xl p-5 space-y-3 shadow-lg ${glowColor} ring-1 ring-inset ${ringColor} hover-float`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${isBlue ? "bg-blue-400" : "bg-violet-400"}`} />
          <span className={`text-sm font-bold ${labelColor}`}>{label}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`${badgeBg} ${labelColor} text-xs font-mono rounded-lg border px-2 py-0.5 flex items-center gap-1`}>
            <Hash className="h-3 w-3" />
            {response.tokens} tok
          </span>
          <span className={`${badgeBg} ${labelColor} text-xs font-mono rounded-lg border px-2 py-0.5 flex items-center gap-1`}>
            <Clock className="h-3 w-3" />
            {response.time}s
          </span>
        </div>
      </div>
      <div className="h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />
      <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap max-h-96 overflow-y-auto pr-2">
        {response.text}
      </div>
    </div>
  );
}

export default App;
