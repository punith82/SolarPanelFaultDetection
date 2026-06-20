import { useState } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:8000/analyze';

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const healthyCount = results.filter((item) => item.status === 'normal').length;
  const faultyCount = results.length - healthyCount;

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setResults([]);
    setError('');
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please upload an image first.');
      return;
    }

    setLoading(true);
    setError('');
    setResults([]);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults(response.data.results || []);
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to analyze image.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 px-4 py-8 text-slate-100">
      <div className="mx-auto max-w-6xl rounded-3xl border border-slate-700 bg-slate-900/90 p-8 shadow-2xl shadow-slate-950/40">
        <header className="mb-8 text-center">
          <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Dashboard</p>
          <h1 className="mt-4 text-4xl font-semibold text-white">Solar Panel Fault Detection System</h1>
          <p className="mx-auto mt-3 max-w-2xl text-slate-400">Upload a top-down panel image, run the analysis pipeline, and inspect panel health instantly.</p>
        </header>

        <section className="grid gap-8 md:grid-cols-[1.2fr_0.8fr]">
          <div className="space-y-6 rounded-3xl border border-slate-800 bg-slate-950/80 p-6">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <label className="block text-sm font-medium text-slate-300">Upload Image</label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="mt-2 block w-full cursor-pointer rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 file:mr-4 file:rounded-full file:border-0 file:bg-cyan-500 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-950"
                />
              </div>
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={loading || !file}
                className="inline-flex items-center justify-center rounded-2xl bg-cyan-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700"
              >
                {loading ? 'Analyzing...' : 'Analyze'}
              </button>
            </div>

            {error && <div className="rounded-2xl bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{error}</div>}

            <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-4">
              <p className="mb-3 text-sm uppercase tracking-[0.24em] text-slate-500">Image preview</p>
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Uploaded preview"
                  className="h-[380px] w-full rounded-3xl object-contain border border-slate-700 bg-slate-950"
                />
              ) : (
                <div className="flex h-[380px] items-center justify-center rounded-3xl border border-dashed border-slate-700 bg-slate-950/40 text-slate-500">
                  Upload a top-view solar panel image to preview it here.
                </div>
              )}
            </div>
          </div>

          <div className="space-y-6">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-6">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Total Panels</p>
                <p className="mt-4 text-4xl font-semibold text-white">{results.length}</p>
              </div>
              <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-6">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Healthy Panels</p>
                <p className="mt-4 text-4xl font-semibold text-emerald-400">{healthyCount}</p>
              </div>
              <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-6 sm:col-span-2">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Faulty Panels</p>
                <p className="mt-4 text-4xl font-semibold text-rose-400">{faultyCount}</p>
              </div>
            </div>

            <div className="rounded-3xl border border-slate-800 bg-slate-900/90 p-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Panel Results</p>
                  <h2 className="mt-2 text-2xl font-semibold text-white">Detection output</h2>
                </div>
                <p className="text-sm text-slate-400">{loading ? 'Running inference...' : 'Results from backend API'}</p>
              </div>

              <div className="overflow-x-auto rounded-3xl bg-slate-950/80">
                <table className="min-w-full divide-y divide-slate-800 text-left text-sm text-slate-300">
                  <thead className="bg-slate-900 text-slate-400">
                    <tr>
                      <th className="px-4 py-4">Panel ID</th>
                      <th className="px-4 py-4">Fault Type</th>
                      <th className="px-4 py-4">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.length === 0 ? (
                      <tr>
                        <td colSpan="3" className="px-4 py-6 text-center text-slate-500">
                          No results yet. Upload an image and press Analyze.
                        </td>
                      </tr>
                    ) : (
                      results.map((item) => (
                        <tr key={item.panel_id} className="border-t border-slate-800 hover:bg-slate-900/80">
                          <td className="px-4 py-4 font-medium text-white">{item.panel_id}</td>
                          <td className="px-4 py-4 capitalize">{item.status.replace('_', ' ')}</td>
                          <td className="px-4 py-4">{(item.confidence * 100).toFixed(1)}%</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
