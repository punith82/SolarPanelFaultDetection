import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Sun,
  ShieldCheck,
  TriangleAlert,
  Activity,
  Upload,
  ScanSearch,
  Cpu,
  Clock,
  Image as ImageIcon,
} from "lucide-react";

const API_URL = "http://localhost:8000/analyze";

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [results, setResults] = useState([]);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  const healthyCount = useMemo(
    () => results.filter((item) => item.status === "normal").length,
    [results],
  );

  const faultyCount = results.length - healthyCount;

  const avgConfidence = useMemo(() => {
    if (!results.length) return 0;

    return (
      results.reduce((sum, item) => sum + item.confidence, 0) / results.length
    );
  }, [results]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const processFile = (selectedFile) => {
    if (!selectedFile) return;

    setFile(selectedFile);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setPreviewUrl(URL.createObjectURL(selectedFile));
    setAnnotatedImage(null);
    setResults([]);
    setInferenceTime(null);
    setError("");
  };

  const handleFileChange = (event) => {
    processFile(event.target.files?.[0]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);

    const droppedFile = event.dataTransfer.files?.[0];
    processFile(droppedFile);
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResults([]);
    setAnnotatedImage(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await axios.post(API_URL, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResults(response.data.results || []);
      setAnnotatedImage(response.data.annotated_image || null);
      setInferenceTime(response.data.inference_time || null);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to analyze image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-8">
        <header className="mb-8 rounded-3xl border border-cyan-500/20 bg-gradient-to-r from-slate-900 via-slate-950 to-cyan-950/20 p-8 shadow-2xl">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex items-center gap-4">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-cyan-500/10">
                  <Sun className="h-8 w-8 text-cyan-400" />
                </div>

                <div>
                  <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">
                    AI Inspection Platform
                  </p>

                  <h1 className="mt-2 text-4xl font-bold text-white">
                    Solar Panel Fault Detection
                  </h1>
                </div>
              </div>

              <p className="mt-4 max-w-3xl text-slate-400">
                Automated photovoltaic inspection using YOLO-based panel
                detection and EfficientNet fault classification.
              </p>
            </div>

            <div className="flex items-center gap-2 rounded-2xl border border-emerald-500/20 bg-emerald-500/10 px-5 py-3">
              <span className="h-3 w-3 animate-pulse rounded-full bg-emerald-400" />
              <span className="font-medium text-emerald-300">
                Inference Service Online
              </span>
            </div>
          </div>
        </header>

        <div className="mb-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            icon={<ScanSearch className="h-6 w-6 text-cyan-400" />}
            title="Total Panels"
            value={results.length}
          />

          <MetricCard
            icon={<ShieldCheck className="h-6 w-6 text-emerald-400" />}
            title="Healthy Panels"
            value={healthyCount}
          />

          <MetricCard
            icon={<TriangleAlert className="h-6 w-6 text-rose-400" />}
            title="Faulty Panels"
            value={faultyCount}
          />

          <MetricCard
            icon={<Activity className="h-6 w-6 text-violet-400" />}
            title="Avg Confidence"
            value={`${(avgConfidence * 100).toFixed(1)}%`}
          />
        </div>

        <div className="mb-8 rounded-3xl border border-slate-800 bg-slate-900 p-6">
          <h2 className="mb-6 text-xl font-semibold text-white">AI Pipeline</h2>

          <div className="flex flex-wrap items-center gap-3">
            <PipelineStep label="Upload Image" />

            <span className="text-slate-600">→</span>

            <PipelineStep label="YOLO Detection" active />

            <span className="text-slate-600">→</span>

            <PipelineStep label="EfficientNet Classification" accent="violet" />

            <span className="text-slate-600">→</span>

            <PipelineStep label="Fault Report" accent="emerald" />
          </div>
        </div>

        <div className="grid gap-8 xl:grid-cols-[1.5fr_0.5fr]">
          <div className="space-y-8">
            <div className="rounded-3xl border border-slate-800 bg-slate-900 p-6">
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                className={`rounded-3xl border-2 border-dashed p-8 text-center transition ${
                  dragging
                    ? "border-cyan-400 bg-cyan-500/10"
                    : "border-slate-700"
                }`}
              >
                <Upload className="mx-auto mb-4 h-12 w-12 text-cyan-400" />

                <h3 className="text-xl font-semibold text-white">
                  Upload Inspection Image
                </h3>

                <p className="mt-2 text-slate-400">
                  Drag and drop an image or browse files
                </p>

                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="mt-6 block w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 file:mr-4 file:rounded-full file:border-0 file:bg-cyan-500 file:px-4 file:py-2 file:font-semibold file:text-slate-950"
                />

                <button
                  onClick={handleAnalyze}
                  disabled={loading || !file}
                  className="mt-6 rounded-2xl bg-cyan-500 px-6 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-700"
                >
                  {loading ? "Running Inference..." : "Analyze Image"}
                </button>
              </div>

              {error && (
                <div className="mt-6 rounded-2xl border border-rose-500/20 bg-rose-500/10 p-4 text-rose-300">
                  {error}
                </div>
              )}
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <ImageCard title="Original Image" image={previewUrl} />

              <ImageCard
                title="YOLO Detection Overlay"
                image={annotatedImage}
              />
            </div>

            <div className="rounded-3xl border border-slate-800 bg-slate-900 p-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-semibold text-white">
                    Detection Results
                  </h2>

                  <p className="mt-2 text-slate-400">
                    Panel-level classification output
                  </p>
                </div>

                {inferenceTime && (
                  <div className="flex items-center gap-2 text-slate-400">
                    <Clock className="h-4 w-4" />
                    {inferenceTime}s
                  </div>
                )}
              </div>

              <div className="overflow-x-auto rounded-2xl">
                <table className="min-w-full text-left">
                  <thead className="border-b border-slate-800 text-sm text-slate-400">
                    <tr>
                      <th className="px-4 py-4">Panel ID</th>
                      <th className="px-4 py-4">Status</th>
                      <th className="px-4 py-4">Confidence</th>
                    </tr>
                  </thead>

                  <tbody>
                    {!results.length ? (
                      <tr>
                        <td
                          colSpan="3"
                          className="px-4 py-12 text-center text-slate-500"
                        >
                          {loading
                            ? "Running model inference..."
                            : "No results available"}
                        </td>
                      </tr>
                    ) : (
                      results.map((item) => (
                        <tr
                          key={item.panel_id}
                          className="border-b border-slate-800/60 hover:bg-slate-800/30"
                        >
                          <td className="px-4 py-4 font-medium text-white">
                            {item.panel_id}
                          </td>

                          <td className="px-4 py-4">
                            <span
                              className={`rounded-full px-3 py-1 text-xs font-semibold ${
                                item.status === "normal"
                                  ? "bg-emerald-500/10 text-emerald-400"
                                  : "bg-rose-500/10 text-rose-400"
                              }`}
                            >
                              {item.status.replace("_", " ")}
                            </span>
                          </td>

                          <td className="px-4 py-4">
                            <div className="flex items-center gap-3">
                              <div className="h-2 w-28 rounded-full bg-slate-800">
                                <div
                                  className="h-2 rounded-full bg-cyan-400"
                                  style={{
                                    width: `${item.confidence * 100}%`,
                                  }}
                                />
                              </div>

                              <span>{(item.confidence * 100).toFixed(1)}%</span>
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="space-y-8">
            <div className="rounded-3xl border border-slate-800 bg-slate-900 p-6">
              <div className="mb-6 flex items-center gap-3">
                <Cpu className="h-6 w-6 text-cyan-400" />

                <h2 className="text-xl font-semibold text-white">
                  Model Information
                </h2>
              </div>

              <div className="space-y-5">
                <InfoRow label="Detector" value="YOLOv8" />

                <InfoRow label="Classifier" value="EfficientNet-B0" />

                <InfoRow label="Framework" value="TensorFlow + Ultralytics" />

                <InfoRow label="Input Size" value="640 × 640" />

                <InfoRow label="Deployment" value="FastAPI" />
              </div>
            </div>

            <div className="rounded-3xl border border-slate-800 bg-slate-900 p-6">
              <div className="mb-6 flex items-center gap-3">
                <ImageIcon className="h-6 w-6 text-violet-400" />

                <h2 className="text-xl font-semibold text-white">
                  Inspection Summary
                </h2>
              </div>

              <div className="space-y-4 text-sm">
                <SummaryRow
                  label="Uploaded File"
                  value={file?.name || "No file selected"}
                />

                <SummaryRow label="Panels Detected" value={results.length} />

                <SummaryRow
                  label="Fault Rate"
                  value={
                    results.length
                      ? `${((faultyCount / results.length) * 100).toFixed(1)}%`
                      : "0%"
                  }
                />

                <SummaryRow
                  label="Last Analysis"
                  value={
                    results.length ? new Date().toLocaleTimeString() : "--"
                  }
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ icon, title, value }) {
  return (
    <div className="rounded-3xl border border-slate-800 bg-slate-900 p-6">
      <div className="flex items-center justify-between">
        {icon}

        <p className="text-3xl font-bold text-white">{value}</p>
      </div>

      <p className="mt-4 text-sm uppercase tracking-wider text-slate-400">
        {title}
      </p>
    </div>
  );
}

function PipelineStep({ label, active = false, accent }) {
  const styles = {
    default: "bg-slate-800 text-slate-300",
    cyan: "bg-cyan-500/10 text-cyan-300",
    violet: "bg-violet-500/10 text-violet-300",
    emerald: "bg-emerald-500/10 text-emerald-300",
  };

  let color = styles.default;

  if (active) color = styles.cyan;
  if (accent === "violet") color = styles.violet;
  if (accent === "emerald") color = styles.emerald;

  return <div className={`rounded-full px-4 py-2 ${color}`}>{label}</div>;
}

function ImageCard({ title, image }) {
  return (
    <div className="rounded-3xl border border-slate-800 bg-slate-900 p-4">
      <h3 className="mb-4 text-lg font-semibold text-white">{title}</h3>

      {image ? (
        <img
          src={image}
          alt={title}
          className="h-[380px] w-full rounded-2xl border border-slate-700 object-contain bg-slate-950"
        />
      ) : (
        <div className="flex h-[380px] items-center justify-center rounded-2xl border border-dashed border-slate-700 bg-slate-950 text-slate-500">
          No image available
        </div>
      )}
    </div>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="flex items-center justify-between border-b border-slate-800 pb-3">
      <span className="text-slate-400">{label}</span>

      <span className="font-medium text-white">{value}</span>
    </div>
  );
}

function SummaryRow({ label, value }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-400">{label}</span>

      <span className="max-w-[150px] truncate text-right text-white">
        {value}
      </span>
    </div>
  );
}

export default App;
